"""
============================================================
  GNN Boundary Extraction + Dataset
  For Kobayashi Crystal Growth Surrogate

  Converts Kobayashi phase field frames into graphs
  representing the crystal boundary. The GNN learns to
  predict how the boundary moves from one timestep to the
  next, rather than predicting the full dense field.

  This directly addresses the FNO failure on sparse fields:
  instead of 256x256=65536 pixels (95% empty), we work with
  300-800 boundary points where all the physics happens.

  Ayush Shah & Tobias Li
  Georgia Institute of Technology
  NSF IRES Physical AI Design Program — Prof. Bo Zhu
============================================================

Pipeline:
    phase field (256x256) 
        → extract boundary pixels (skimage.find_contours)
        → subsample to fixed N_nodes points
        → compute node features (position, curvature, 
                                  gradient, distance to center)
        → build k-nearest-neighbor graph
        → GNN predicts displacement to next boundary
        → reconstruct phase field from new boundary

Usage:
    # Extract and visualise boundary from one frame
    python gnn_boundary.py --npz path/to/traj.npz --frame 20 --plot

    # Build full graph dataset from all trajectories
    python gnn_boundary.py --build_dataset \
        --data_dir data/kobayashi \
        --out_dir data/kobayashi_graphs
"""

import argparse
import os
import glob

import numpy as np
from scipy import ndimage
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

import torch
from torch_geometric.data import Data, Dataset


# ── Constants ─────────────────────────────────────────────────────────────────
N_NODES    = 512    # fixed number of boundary points per frame
                    # subsampled to this after contour extraction
K_NEIGHBORS = 8     # each node connects to its K nearest neighbours
THRESHOLD   = 0.5   # phase field threshold for solid/liquid boundary


# ── Boundary extraction ───────────────────────────────────────────────────────

def extract_boundary_points(phase_field: np.ndarray,
                             n_points: int = N_NODES,
                             threshold: float = THRESHOLD) -> np.ndarray | None:
    """
    Extract the crystal boundary as an ordered set of 2D points.

    Uses marching squares (skimage.measure.find_contours) to find the
    iso-contour at `threshold`. If multiple contours exist (multi-seed
    case), we take the longest one.

    Returns
    -------
    points : (n_points, 2) float32 array of (x, y) positions
             normalised to [0, 1] in both dimensions.
             Returns None if no boundary found (field fully liquid).
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image required: pip install scikit-image")

    # Find contours at the solid/liquid threshold
    contours = measure.find_contours(phase_field, threshold)

    if len(contours) == 0:
        return None

    # Take the longest contour (main crystal boundary)
    contour = max(contours, key=len)   # shape (L, 2) in (row, col) = (y, x)

    if len(contour) < 8:
        return None   # too small to be meaningful

    # Convert to (x, y) and normalise to [0, 1]
    N = phase_field.shape[0]
    xy = contour[:, ::-1].astype(np.float32) / N   # flip to (x, y), normalise

    # Subsample or upsample to exactly n_points
    xy = _resample_contour(xy, n_points)

    return xy


def _resample_contour(xy: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a closed contour to exactly n evenly-spaced points
    using linear interpolation along arc length.
    """
    # Compute cumulative arc length
    diffs  = np.diff(xy, axis=0, prepend=xy[-1:])
    dists  = np.sqrt((diffs ** 2).sum(axis=1))
    cumlen = np.cumsum(dists)
    total  = cumlen[-1]

    if total < 1e-8:
        # Degenerate contour — just tile
        idx = np.linspace(0, len(xy) - 1, n, dtype=int)
        return xy[idx]

    # Target arc-length positions
    target = np.linspace(0, total, n, endpoint=False)

    # Interpolate
    new_xy = np.zeros((n, 2), dtype=np.float32)
    for i, t in enumerate(target):
        idx   = np.searchsorted(cumlen, t)
        idx   = min(idx, len(xy) - 1)
        idx_p = (idx - 1) % len(xy)
        seg   = cumlen[idx] - cumlen[idx_p]
        if seg < 1e-10:
            new_xy[i] = xy[idx]
        else:
            alpha     = (t - cumlen[idx_p]) / seg
            new_xy[i] = (1 - alpha) * xy[idx_p] + alpha * xy[idx]

    return new_xy


# ── Node feature computation ──────────────────────────────────────────────────

def compute_node_features(xy: np.ndarray,
                           phase_field: np.ndarray) -> np.ndarray:
    """
    Compute per-node features for the GNN.

    Features (8 per node):
        0-1  : (x, y) position, normalised [0,1]
        2-3  : displacement from domain centre (dx, dy)
        4    : distance from domain centre
        5    : local curvature (second derivative of arc position)
        6-7  : phase gradient (dphi/dx, dphi/dy) at boundary point,
               normalised by field range

    Parameters
    ----------
    xy          : (N, 2) boundary points in [0,1] coordinates
    phase_field : (H, W) raw phase field

    Returns
    -------
    features : (N, 8) float32
    """
    N_pts = xy.shape[0]
    H, W  = phase_field.shape
    feat  = np.zeros((N_pts, 8), dtype=np.float32)

    # 0-1: position
    feat[:, 0:2] = xy

    # 2-3: displacement from centre
    cx, cy       = 0.5, 0.5
    feat[:, 2]   = xy[:, 0] - cx
    feat[:, 3]   = xy[:, 1] - cy

    # 4: distance from centre
    feat[:, 4]   = np.sqrt(feat[:, 2]**2 + feat[:, 3]**2)

    # 5: local curvature via finite differences on arc
    # κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    dx  = np.gradient(xy[:, 0])
    dy  = np.gradient(xy[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2) ** 1.5 + 1e-10
    curv = np.abs(dx * ddy - dy * ddx) / denom
    feat[:, 5] = np.log1p(curv)   # log(1 + curv) compresses the range
    # 6-7: phase gradient at each boundary point
    # sample the gradient field at the boundary pixel locations
    grad_y, grad_x = np.gradient(phase_field)
    prange = phase_field.max() - phase_field.min() + 1e-8
    for i in range(N_pts):
        px = int(np.clip(xy[i, 0] * W, 0, W - 1))
        py = int(np.clip(xy[i, 1] * H, 0, H - 1))
        feat[i, 6] = grad_x[py, px] / prange
        feat[i, 7] = grad_y[py, px] / prange

    return feat


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(xy: np.ndarray,
                features: np.ndarray,
                target_xy: np.ndarray | None = None,
                k: int = K_NEIGHBORS) -> Data:
    """
    Build a PyTorch Geometric Data object from boundary points.

    Connectivity: k-nearest-neighbour graph in 2D position space.
    Each node is connected to its k closest boundary neighbours.
    We also connect adjacent points along the contour (ring edges)
    to preserve topological ordering.

    Parameters
    ----------
    xy          : (N, 2)  current boundary positions
    features    : (N, F)  node features
    target_xy   : (N, 2)  next-frame boundary positions (training target)
                          None at inference time
    k           : int     number of nearest neighbours

    Returns
    -------
    torch_geometric.data.Data with:
        x      : (N, F)   node features
        pos    : (N, 2)   node positions
        edge_index : (2, E)  graph connectivity
        y      : (N, 2)  displacement target (if target_xy provided)
    """
    N = xy.shape[0]

    # ── KNN edges ─────────────────────────────────────────────────────────────
    # Pairwise distances
    diff  = xy[:, None, :] - xy[None, :, :]    # (N, N, 2)
    dists = np.sqrt((diff ** 2).sum(axis=-1))  # (N, N)
    np.fill_diagonal(dists, np.inf)

    # K nearest neighbours for each node
    knn_idx = np.argsort(dists, axis=1)[:, :k]  # (N, k)
    src = np.repeat(np.arange(N), k)
    dst = knn_idx.ravel()

    # ── Ring edges (adjacent contour points) ──────────────────────────────────
    ring_src = np.arange(N)
    ring_dst = (np.arange(N) + 1) % N
    ring_src2 = ring_dst.copy()
    ring_dst2 = ring_src.copy()

    # Combine and deduplicate
    all_src = np.concatenate([src, ring_src, ring_src2])
    all_dst = np.concatenate([dst, ring_dst, ring_dst2])
    edges   = np.unique(np.stack([all_src, all_dst], axis=1), axis=0)
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    # ── Build Data object ─────────────────────────────────────────────────────
    data = Data(
        x          = torch.tensor(features, dtype=torch.float32),
        pos        = torch.tensor(xy, dtype=torch.float32),
        edge_index = edge_index,
    )

    if target_xy is not None:
        # Target: displacement from current to next boundary
        displacement = target_xy - xy    # (N, 2)
        data.y = torch.tensor(displacement, dtype=torch.float32)

    return data


# ── Full frame pair → graph ───────────────────────────────────────────────────

def frames_to_graph(frame_t: np.ndarray,
                     frame_t1: np.ndarray | None = None,
                     n_nodes: int = N_NODES,
                     k: int = K_NEIGHBORS) -> Data | None:
    """
    Convert a pair of phase field frames to a graph.

    Parameters
    ----------
    frame_t  : (H, W) current phase field
    frame_t1 : (H, W) next phase field (None at inference)
    n_nodes  : number of boundary points to extract
    k        : KNN connectivity

    Returns
    -------
    Data object or None if no boundary found in frame_t
    """
    xy = extract_boundary_points(frame_t, n_points=n_nodes)
    if xy is None:
        return None

    features = compute_node_features(xy, frame_t)

    target_xy = None
    if frame_t1 is not None:
        target_xy = extract_boundary_points(frame_t1, n_points=n_nodes)
        if target_xy is None:
            target_xy = xy.copy()   # no growth — zero displacement

    return build_graph(xy, features, target_xy, k=k)


# ── Dataset ───────────────────────────────────────────────────────────────────

class KobayashiGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset of crystal boundary graphs.

    Each item is a graph representing the boundary at frame t,
    with target displacements to frame t+1.

    Parameters
    ----------
    data_dir   : directory of .npz Kobayashi trajectory files
    n_nodes    : number of boundary points per graph
    k          : KNN connectivity
    min_phi    : skip frames where solid fraction < min_phi
                 (too sparse to extract a meaningful boundary)
    """
    def __init__(self, data_dir: str,
                 n_nodes: int = N_NODES,
                 k: int = K_NEIGHBORS,
                 min_phi: float = 0.005):
        super().__init__()
        self.graphs  = []
        self.n_nodes = n_nodes
        self.k       = k

        files = sorted(glob.glob(os.path.join(data_dir, 'traj_*.npz')))
        print(f"Building graph dataset from {len(files)} trajectories...")

        skipped = 0
        for fpath in files:
            d      = np.load(fpath, allow_pickle=True)
            frames = d['frames']   # (T, H, W)
            T      = frames.shape[0]

            for t in range(T - 1):
                phi = (frames[t] > THRESHOLD).mean()
                if phi < min_phi:
                    skipped += 1
                    continue

                graph = frames_to_graph(frames[t], frames[t + 1],
                                        n_nodes=n_nodes, k=k)
                if graph is not None:
                    self.graphs.append(graph)

        print(f"  Built {len(self.graphs)} graphs  "
              f"({skipped} frames skipped — too sparse)")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


# ── Reconstruction: boundary → phase field ───────────────────────────────────

def boundary_to_field(xy: np.ndarray, N: int = 256) -> np.ndarray:
    """
    Reconstruct a phase field from boundary points.
    Fills the interior of the closed boundary with 1.0.

    Uses OpenCV polygon fill if available, otherwise falls back
    to scipy binary fill holes on a rasterised boundary.

    Parameters
    ----------
    xy : (n_points, 2) boundary points in [0,1] coordinates
    N  : output grid size

    Returns
    -------
    field : (N, N) float32 with 1.0 inside boundary, 0.0 outside
    """
    field  = np.zeros((N, N), dtype=np.float32)
    pixels = (xy * N).astype(np.int32)
    pixels = np.clip(pixels, 0, N - 1)

    try:
        import cv2
        cv2.fillPoly(field, [pixels[:, ::-1]], 1.0)  # cv2 uses (col, row)
    except ImportError:
        # Fallback: rasterise boundary then fill holes
        for i in range(len(pixels)):
            x0, y0 = pixels[i]
            x1, y1 = pixels[(i + 1) % len(pixels)]
            # Bresenham line rasterisation
            pts = _bresenham(x0, y0, x1, y1)
            for px, py in pts:
                if 0 <= px < N and 0 <= py < N:
                    field[py, px] = 1.0
        # Fill interior
        field = ndimage.binary_fill_holes(field).astype(np.float32)

    return field


def _bresenham(x0, y0, x1, y1):
    """Bresenham's line algorithm."""
    pts = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        pts.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x0 += sx
        if e2 < dx:
            err += dx; y0 += sy
    return pts


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz',           default=None,
                    help='Single .npz file to inspect')
    ap.add_argument('--frame',         type=int, default=20,
                    help='Frame index to extract boundary from')
    ap.add_argument('--plot',          action='store_true',
                    help='Show visualisation')
    ap.add_argument('--build_dataset', action='store_true',
                    help='Build full graph dataset')
    ap.add_argument('--data_dir',      default='data/kobayashi')
    ap.add_argument('--out_dir',       default='data/kobayashi_graphs')
    args = ap.parse_args()

    if args.npz:
        # Inspect a single trajectory
        d      = np.load(args.npz, allow_pickle=True)
        frames = d['frames']
        frame  = frames[min(args.frame, len(frames)-1)]

        print(f"Frame {args.frame}: shape={frame.shape}  "
              f"phi={(frame>THRESHOLD).mean():.4f}")

        xy = extract_boundary_points(frame)
        if xy is None:
            print("No boundary found — crystal too small at this frame.")
            return

        print(f"Boundary points: {len(xy)}")
        feat  = compute_node_features(xy, frame)
        print(f"Feature shape: {feat.shape}")
        print(f"Feature ranges:")
        for i, name in enumerate(['x','y','dx','dy','dist','curv','gx','gy']):
            print(f"  {name}: [{feat[:,i].min():.3f}, {feat[:,i].max():.3f}]")

        graph = frames_to_graph(frame,
                                frames[min(args.frame+1, len(frames)-1)])
        print(f"\nGraph: {graph.num_nodes} nodes  "
              f"{graph.num_edges} edges  "
              f"avg degree={graph.num_edges/graph.num_nodes:.1f}")

        if args.plot:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(frame, cmap='viridis', origin='lower')
            axes[0].scatter(xy[:,0]*frame.shape[1],
                           xy[:,1]*frame.shape[0],
                           c='red', s=4, zorder=5)
            axes[0].set_title(f'Phase field + boundary (frame {args.frame})')

            axes[1].scatter(xy[:,0], xy[:,1], c=feat[:,5],
                           cmap='hot', s=8)
            axes[1].set_title('Boundary curvature')
            axes[1].set_aspect('equal')
            axes[1].invert_yaxis()

            # Draw graph edges
            ei = graph.edge_index.numpy()
            for e in range(0, min(ei.shape[1], 500)):
                s, t_ = ei[0,e], ei[1,e]
                axes[2].plot([xy[s,0], xy[t_,0]],
                            [xy[s,1], xy[t_,1]], 'b-', alpha=0.2, lw=0.5)
            axes[2].scatter(xy[:,0], xy[:,1], c='red', s=8, zorder=5)
            axes[2].set_title(f'Graph ({graph.num_edges} edges)')
            axes[2].set_aspect('equal')
            axes[2].invert_yaxis()

            plt.suptitle(os.path.basename(args.npz), fontweight='bold')
            plt.tight_layout()
            plt.show()

    elif args.build_dataset:
        os.makedirs(args.out_dir, exist_ok=True)
        dataset = KobayashiGraphDataset(args.data_dir)
        print(f"\nDataset size: {len(dataset)} graphs")
        print(f"Sample graph: {dataset[0]}")

        # Save as list of Data objects
        import torch
        out_path = os.path.join(args.out_dir, 'graphs.pt')
        torch.save(dataset.graphs, out_path)
        print(f"Saved → {out_path}")

    else:
        print("Use --npz path/to/traj.npz --plot to inspect a trajectory,")
        print("or --build_dataset to build the full graph dataset.")


if __name__ == '__main__':
    main()
