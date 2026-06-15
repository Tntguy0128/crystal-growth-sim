"""
============================================================
  GNN Boundary Extraction + Dataset  (v2, multi-contour)
  For Kobayashi Crystal Growth Surrogate

  CHANGES FROM v1:
  - extract_boundary_points() now extracts ALL contours
    (not just the longest), allocating nodes proportionally
    to arc length. Each node gets a `contour_id` feature.
  - build_graph() builds ring edges WITHIN each contour
    component (not a single global ring), plus global KNN
    edges (which can connect nearby separate crystals).
  - frames_to_graph() matches contours between frame t and
    frame t+1 by nearest-centroid assignment, so multi-seed
    crystals are tracked as separate components instead of
    collapsing into one blob.

  This fixes the "merged blob" issue from week 1 where
  multiple crystal seeds were predicted as a single growing
  region.

  Ayush Shah & Tobias Li
  Georgia Institute of Technology
  NSF IRES Physical AI Design Program — Prof. Bo Zhu
============================================================
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
N_NODES        = 512    # total boundary points across ALL contours
K_NEIGHBORS    = 8       # global KNN connectivity
THRESHOLD      = 0.5     # phase field threshold for solid/liquid boundary
MIN_CONTOUR_LEN = 8       # ignore contours shorter than this (noise)
MIN_NODES_PER_CONTOUR = 16  # minimum points allocated to any contour


# ── Multi-contour boundary extraction ─────────────────────────────────────────

def extract_all_contours(phase_field: np.ndarray,
                          threshold: float = THRESHOLD) -> list[np.ndarray]:
    """
    Find ALL iso-contours at `threshold`, filter tiny ones, return as a
    list of (L_i, 2) arrays in (x, y) normalised-[0,1] coordinates.
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image required: pip install scikit-image")

    contours = measure.find_contours(phase_field, threshold)
    N = phase_field.shape[0]

    out = []
    for c in contours:
        if len(c) < MIN_CONTOUR_LEN:
            continue
        xy = c[:, ::-1].astype(np.float32) / N   # (row,col)->(x,y), normalise
        out.append(xy)

    return out


def extract_boundary_points(phase_field: np.ndarray,
                             n_points: int = N_NODES,
                             threshold: float = THRESHOLD
                             ) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract boundary points from ALL contours, allocating the node
    budget `n_points` proportionally to each contour's arc length
    (with a minimum per contour so small crystals aren't dropped).

    Returns
    -------
    (xy, contour_id) or None if no boundary found.
        xy         : (n_points, 2) float32, positions in [0,1]
        contour_id : (n_points,)   int32, which contour each point
                      belongs to (0, 1, 2, ...)
    """
    contours = extract_all_contours(phase_field, threshold)
    if len(contours) == 0:
        return None

    # Arc length of each contour
    lengths = []
    for c in contours:
        diffs = np.diff(c, axis=0, prepend=c[-1:])
        lengths.append(np.sqrt((diffs ** 2).sum(axis=1)).sum())
    lengths = np.array(lengths)

    n_contours = len(contours)

    # Allocate at least MIN_NODES_PER_CONTOUR to each, distribute the
    # remainder proportionally to arc length.
    base = MIN_NODES_PER_CONTOUR
    remaining = max(0, n_points - base * n_contours)
    weights = lengths / lengths.sum()
    extra = np.floor(weights * remaining).astype(int)

    # Fix rounding so total == n_points exactly
    counts = base + extra
    diff = n_points - counts.sum()
    if diff > 0:
        # give leftover to the largest contour
        counts[np.argmax(lengths)] += diff
    elif diff < 0:
        # trim from the largest contours first
        order = np.argsort(-counts)
        i = 0
        while diff < 0:
            c = order[i % n_contours]
            if counts[c] > MIN_NODES_PER_CONTOUR:
                counts[c] -= 1
                diff += 1
            i += 1

    all_xy  = []
    all_cid = []
    for i, (c, n) in enumerate(zip(contours, counts)):
        if n <= 0:
            continue
        resampled = _resample_contour(c, n)
        all_xy.append(resampled)
        all_cid.append(np.full(n, i, dtype=np.int32))

    xy  = np.concatenate(all_xy, axis=0)
    cid = np.concatenate(all_cid, axis=0)

    return xy, cid


def _resample_contour(xy: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed contour to exactly n evenly-spaced points."""
    diffs  = np.diff(xy, axis=0, prepend=xy[-1:])
    dists  = np.sqrt((diffs ** 2).sum(axis=1))
    cumlen = np.cumsum(dists)
    total  = cumlen[-1]

    if total < 1e-8:
        idx = np.linspace(0, len(xy) - 1, n, dtype=int)
        return xy[idx]

    target = np.linspace(0, total, n, endpoint=False)
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
                           phase_field: np.ndarray,
                           contour_id: np.ndarray) -> np.ndarray:
    """
    Per-node features for the GNN. 9 features total (was 8 in v1):
        0-1  : (x, y) position, normalised [0,1]
        2-3  : displacement from domain centre (dx, dy)
        4    : distance from domain centre
        5    : local curvature, log1p-compressed
        6-7  : phase gradient (dphi/dx, dphi/dy), normalised
        8    : normalised contour id (which crystal cluster)

    Curvature and gradients are computed PER CONTOUR so that
    finite-difference stencils don't wrap across different
    crystal clusters.
    """
    N_pts = xy.shape[0]
    H, W  = phase_field.shape
    feat  = np.zeros((N_pts, 9), dtype=np.float32)

    feat[:, 0:2] = xy
    cx, cy       = 0.5, 0.5
    feat[:, 2]   = xy[:, 0] - cx
    feat[:, 3]   = xy[:, 1] - cy
    feat[:, 4]   = np.sqrt(feat[:, 2]**2 + feat[:, 3]**2)

    grad_y, grad_x = np.gradient(phase_field)
    prange = phase_field.max() - phase_field.min() + 1e-8

    n_contours = int(contour_id.max()) + 1 if N_pts > 0 else 0
    for cid in range(n_contours):
        mask = contour_id == cid
        sub_xy = xy[mask]
        if len(sub_xy) < 3:
            continue

        dx  = np.gradient(sub_xy[:, 0])
        dy  = np.gradient(sub_xy[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denom = (dx**2 + dy**2) ** 1.5 + 1e-10
        curv  = np.abs(dx * ddy - dy * ddx) / denom
        feat[mask, 5] = np.log1p(curv)   # compress huge curvature range

    # phase gradient sampled at each boundary pixel
    for i in range(N_pts):
        px = int(np.clip(xy[i, 0] * W, 0, W - 1))
        py = int(np.clip(xy[i, 1] * H, 0, H - 1))
        feat[i, 6] = grad_x[py, px] / prange
        feat[i, 7] = grad_y[py, px] / prange

    # normalised contour id (helps GNN distinguish clusters)
    if n_contours > 1:
        feat[:, 8] = contour_id.astype(np.float32) / (n_contours - 1)
    else:
        feat[:, 8] = 0.0

    return feat


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(xy: np.ndarray,
                features: np.ndarray,
                contour_id: np.ndarray,
                target_xy: np.ndarray | None = None,
                k: int = K_NEIGHBORS) -> Data:
    """
    Build a PyTorch Geometric Data object.

    Connectivity:
      - Ring edges WITHIN each contour (connects point i to i+1,
        wrapping at the end of each contour's own point range).
      - Global KNN edges (k nearest neighbours in 2D, regardless
        of which contour they belong to). This lets the GNN learn
        interactions between nearby separate crystals — e.g. two
        grains that are about to merge.

    Parameters
    ----------
    xy         : (N, 2)  positions
    features   : (N, F)  node features
    contour_id : (N,)    which contour each node belongs to
    target_xy  : (N, 2)  next-frame positions (training target), or None
    k          : KNN count
    """
    N = xy.shape[0]

    # ── Global KNN edges ──────────────────────────────────────────────────────
    diff  = xy[:, None, :] - xy[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    knn_idx = np.argsort(dists, axis=1)[:, :k]
    src = np.repeat(np.arange(N), k)
    dst = knn_idx.ravel()

    # ── Ring edges WITHIN each contour ───────────────────────────────────────
    ring_src_list, ring_dst_list = [], []
    for cid in np.unique(contour_id):
        idx = np.where(contour_id == cid)[0]
        if len(idx) < 2:
            continue
        nxt = np.roll(idx, -1)
        ring_src_list.append(idx)
        ring_dst_list.append(nxt)
        ring_src_list.append(nxt)
        ring_dst_list.append(idx)

    if ring_src_list:
        ring_src = np.concatenate(ring_src_list)
        ring_dst = np.concatenate(ring_dst_list)
    else:
        ring_src = np.array([], dtype=int)
        ring_dst = np.array([], dtype=int)

    all_src = np.concatenate([src, ring_src])
    all_dst = np.concatenate([dst, ring_dst])
    edges   = np.unique(np.stack([all_src, all_dst], axis=1), axis=0)
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    data = Data(
        x          = torch.tensor(features, dtype=torch.float32),
        pos        = torch.tensor(xy, dtype=torch.float32),
        edge_index = edge_index,
        contour_id = torch.tensor(contour_id, dtype=torch.long),
    )

    if target_xy is not None:
        displacement = target_xy - xy
        data.y = torch.tensor(displacement, dtype=torch.float32)

    return data


# ── Contour matching between consecutive frames ──────────────────────────────

def _contour_centroids(contours: list[np.ndarray]) -> np.ndarray:
    return np.stack([c.mean(axis=0) for c in contours])


def match_contours(contours_t: list[np.ndarray],
                    contours_t1: list[np.ndarray]) -> dict[int, int]:
    """
    Greedily match each contour in frame t to the nearest-centroid
    contour in frame t+1. Returns dict {t_index: t1_index}.

    If frame t+1 has fewer contours (two crystals merged), multiple
    t-contours may map to the same t1-contour — that's fine, it
    correctly represents merging.

    If frame t+1 has a contour with no match in frame t (new
    nucleation event), it is simply not used as a target — the
    corresponding t-contour will have displacement 0 if nothing
    matches it either, which is rare.
    """
    if len(contours_t) == 0 or len(contours_t1) == 0:
        return {}

    cent_t  = _contour_centroids(contours_t)
    cent_t1 = _contour_centroids(contours_t1)

    matches = {}
    for i, ct in enumerate(cent_t):
        dists = np.sqrt(((cent_t1 - ct) ** 2).sum(axis=1))
        matches[i] = int(np.argmin(dists))
    return matches


# ── Full frame pair → graph ───────────────────────────────────────────────────

def frames_to_graph(frame_t: np.ndarray,
                     frame_t1: np.ndarray | None = None,
                     n_nodes: int = N_NODES,
                     k: int = K_NEIGHBORS) -> Data | None:
    """
    Convert a pair of phase field frames to a multi-contour graph.

    If frame_t1 is provided, contours are matched between frames and
    the per-contour target is resampled to the SAME number of points
    as the corresponding source contour (so displacement is well
    defined point-for-point).
    """
    result = extract_boundary_points(frame_t, n_points=n_nodes)
    if result is None:
        return None
    xy, cid = result

    features = compute_node_features(xy, frame_t, cid)

    target_xy = None
    if frame_t1 is not None:
        contours_t  = extract_all_contours(frame_t)
        contours_t1 = extract_all_contours(frame_t1)

        if len(contours_t1) == 0:
            # crystal vanished — zero displacement (shouldn't really happen)
            target_xy = xy.copy()
        else:
            matches = match_contours(contours_t, contours_t1)
            target_parts = []
            for c_id in np.unique(cid):
                idx = np.where(cid == c_id)[0]
                n_pts = len(idx)
                t1_idx = matches.get(int(c_id), 0)
                t1_idx = min(t1_idx, len(contours_t1) - 1)
                resampled = _resample_contour(contours_t1[t1_idx], n_pts)
                target_parts.append((idx, resampled))

            target_xy = np.zeros_like(xy)
            for idx, resampled in target_parts:
                target_xy[idx] = resampled

    return build_graph(xy, features, cid, target_xy, k=k)


# ── Dataset ───────────────────────────────────────────────────────────────────

class KobayashiGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset of multi-contour crystal boundary graphs.
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
            frames = d['frames']
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

def boundary_to_field(xy: np.ndarray, contour_id: np.ndarray,
                       N: int = 256) -> np.ndarray:
    """
    Reconstruct a phase field from multi-contour boundary points.
    Each contour is filled separately (so separate crystals stay
    separate instead of one filled polygon spanning all of them).
    """
    field = np.zeros((N, N), dtype=np.float32)

    for cid in np.unique(contour_id):
        idx = np.where(contour_id == cid)[0]
        if len(idx) < 3:
            continue
        pixels = (xy[idx] * N).astype(np.int32)
        pixels = np.clip(pixels, 0, N - 1)

        sub = np.zeros((N, N), dtype=np.float32)
        try:
            import cv2
            cv2.fillPoly(sub, [pixels[:, ::-1]], 1.0)
        except ImportError:
            for i in range(len(pixels)):
                x0, y0 = pixels[i]
                x1, y1 = pixels[(i + 1) % len(pixels)]
                for px, py in _bresenham(x0, y0, x1, y1):
                    if 0 <= px < N and 0 <= py < N:
                        sub[py, px] = 1.0
            sub = ndimage.binary_fill_holes(sub).astype(np.float32)

        field = np.maximum(field, sub)

    return field


def _bresenham(x0, y0, x1, y1):
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
    ap.add_argument('--npz',           default=None)
    ap.add_argument('--frame',         type=int, default=20)
    ap.add_argument('--plot',          action='store_true')
    ap.add_argument('--build_dataset', action='store_true')
    ap.add_argument('--data_dir',      default='data/kobayashi')
    ap.add_argument('--out_dir',       default='data/kobayashi_graphs')
    args = ap.parse_args()

    if args.npz:
        d      = np.load(args.npz, allow_pickle=True)
        frames = d['frames']
        frame  = frames[min(args.frame, len(frames)-1)]

        print(f"Frame {args.frame}: shape={frame.shape}  "
              f"phi={(frame>THRESHOLD).mean():.4f}")

        result = extract_boundary_points(frame)
        if result is None:
            print("No boundary found — crystal too small at this frame.")
            return
        xy, cid = result
        n_contours = int(cid.max()) + 1
        print(f"Contours found: {n_contours}")
        for c in range(n_contours):
            print(f"  contour {c}: {np.sum(cid==c)} points")

        feat = compute_node_features(xy, frame, cid)
        print(f"Feature shape: {feat.shape}")
        for i, name in enumerate(['x','y','dx','dy','dist','curv','gx','gy','cid']):
            print(f"  {name}: [{feat[:,i].min():.3f}, {feat[:,i].max():.3f}]")

        graph = frames_to_graph(frame, frames[min(args.frame+1, len(frames)-1)])
        print(f"\nGraph: {graph.num_nodes} nodes  {graph.num_edges} edges")

        if args.plot:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(frame, cmap='viridis', origin='lower')
            colors = plt.cm.tab10(cid % 10)
            axes[0].scatter(xy[:,0]*frame.shape[1], xy[:,1]*frame.shape[0],
                           c=colors, s=4, zorder=5)
            axes[0].set_title(f'Phase field + {n_contours} contours (frame {args.frame})')

            axes[1].scatter(xy[:,0], xy[:,1], c=feat[:,5], cmap='hot', s=8)
            axes[1].set_title('Boundary curvature (log1p)')
            axes[1].set_aspect('equal'); axes[1].invert_yaxis()

            ei = graph.edge_index.numpy()
            for e in range(0, min(ei.shape[1], 800)):
                sN, tN = ei[0,e], ei[1,e]
                axes[2].plot([xy[sN,0],xy[tN,0]],[xy[sN,1],xy[tN,1]],
                             'b-', alpha=0.15, lw=0.5)
            axes[2].scatter(xy[:,0], xy[:,1], c=colors, s=8, zorder=5)
            axes[2].set_title(f'Graph ({graph.num_edges} edges, '
                             f'{n_contours} components)')
            axes[2].set_aspect('equal'); axes[2].invert_yaxis()

            plt.suptitle(os.path.basename(args.npz), fontweight='bold')
            plt.tight_layout()
            plt.show()

    elif args.build_dataset:
        os.makedirs(args.out_dir, exist_ok=True)
        dataset = KobayashiGraphDataset(args.data_dir)
        print(f"\nDataset size: {len(dataset)} graphs")
        print(f"Sample graph: {dataset[0]}")

        out_path = os.path.join(args.out_dir, 'graphs.pt')
        torch.save(dataset.graphs, out_path)
        print(f"Saved → {out_path}")

    else:
        print("Use --npz path/to/traj.npz --plot to inspect a trajectory,")
        print("or --build_dataset to build the full graph dataset.")


if __name__ == '__main__':
    main()
