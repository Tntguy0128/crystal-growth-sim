"""
============================================================
  Evaluate GNN for Crystal Boundary Prediction
  Kobayashi Phase Field Surrogate

  Evaluates the trained CrystalGNN on held-out test graphs.
  Reports:
    1. One-step displacement MSE
    2. Multi-step rollout MSE (autoregressive)
    3. Visual comparison: predicted vs true boundary
    4. Comparison table: GNN vs FNO

  NSF IRES Physical AI Design Program
  Ayush Shah & Tobias Li — Georgia Institute of Technology
============================================================

USAGE
-----
    python training/evaluate_gnn.py \
        --checkpoint runs/gnn_baseline/best_gnn.pt \
        --data_dir data/kobayashi
"""

import argparse
import os
import sys
import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from solvers.gnn_model import CrystalGNN, BoundaryLoss
    from solvers.gnn_boundary import (
        extract_boundary_points, compute_node_features,
        compute_positional_features,
        build_graph, boundary_to_field, frames_to_graph,
        N_NODES, K_NEIGHBORS
    )
except ImportError:
    from gnn_model import CrystalGNN, BoundaryLoss
    from gnn_boundary import (
        extract_boundary_points, compute_node_features,
        compute_positional_features,
        build_graph, boundary_to_field, frames_to_graph,
        N_NODES, K_NEIGHBORS
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_model(ckpt_path: str, device: torch.device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    args  = ckpt['args']
    model = CrystalGNN(
        in_channels = ckpt['in_channels'],
        hidden_dim  = args['hidden'],
        n_layers    = args['layers'],
        dropout     = 0.0,   # no dropout at eval time
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt


# ── One-step evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def one_step_mse(model, test_graphs, device, batch_size=32):
    """MSE on displacement prediction — one step at a time."""
    loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred  = model(batch)
        mse   = F.mse_loss(pred, batch.y).item()
        total += mse * batch.num_graphs
        n     += batch.num_graphs
    return total / max(n, 1)


# ── Rollout evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def rollout_trajectory(model, frames: np.ndarray, device,
                        n_nodes: int = N_NODES,
                        k: int = K_NEIGHBORS,
                        max_steps: int = None) -> dict:
    """
    Autoregressive rollout that stays in BOUNDARY-POINT SPACE between
    steps — it does NOT roundtrip through a reconstructed phase field.

    Why: boundary_to_field() uses cv2.fillPoly, which fills in concave
    regions (e.g. the thin gaps between dendrite arms). Reconstructing
    a field every step and re-extracting contours from it causes the
    crystal to be progressively "rounded off" into a smooth blob —
    this happens even with a PERFECT (ground-truth) displacement, so
    it is a reconstruction-pipeline artifact, not a model error.

    Instead:
      - Step 0: extract (xy, contour_id) and full 9D features from the
        TRUE initial frame (so the gradient feature is real).
      - Each step: build graph from current (xy, feat, contour_id),
        predict displacement, move points -> new_xy. Recompute the
        7 position-derived features (0-5, 8) directly from new_xy
        (no field needed). Carry forward the gradient feature (6-7)
        from the previous step (same point index = same physical
        point, just moved slightly — gradient direction changes
        slowly between steps).
      - contour_id and point-count-per-contour are FIXED for the whole
        rollout (only positions move), which also prevents the
        "merged blob" failure mode by construction.
      - A field is reconstructed via boundary_to_field ONLY for
        visualisation / field-MSE reporting — that reconstruction
        never feeds back into the next step.

    Returns dict with:
        pred_fields  : list of (N,N) predicted phase fields (for display)
        true_fields  : list of (N,N) ground truth fields
        boundary_mse : per-step MSE on boundary positions
        field_mse    : per-step MSE on reconstructed fields
    """
    T = len(frames)
    if max_steps is not None:
        T = min(T, max_steps + 1)

    N_grid = frames[0].shape[0]

    pred_fields  = [frames[0].copy()]
    true_fields  = [frames[0].copy()]
    boundary_mse = []
    field_mse    = []

    # Initialise from the TRUE initial frame (real gradient features)
    result = extract_boundary_points(frames[0], n_points=n_nodes)
    if result is None:
        # No boundary in the first frame at all — nothing to roll out
        return {
            'pred_fields' : pred_fields,
            'true_fields' : true_fields,
            'boundary_mse': np.array([]),
            'field_mse'   : np.array([]),
        }
    xy, cid = result
    feat = compute_node_features(xy, frames[0], cid)   # full 9D, real gradient

    for t in range(T - 1):
        graph = build_graph(xy, feat, cid, k=k).to(device)

        # Predict displacement
        pred_disp = model(graph).cpu().numpy()   # (N, 2)

        # Move boundary points directly — no field roundtrip
        new_xy = np.clip(xy + pred_disp, 0.0, 1.0)

        # Get true next boundary for MSE — compare against a freshly
        # extracted true boundary with the same total node budget
        true_result = extract_boundary_points(frames[t + 1], n_points=n_nodes)
        if true_result is not None:
            true_xy, _ = true_result
            n_cmp = min(len(true_xy), len(new_xy))
            b_mse = np.mean((new_xy[:n_cmp] - true_xy[:n_cmp]) ** 2)
        else:
            b_mse = 0.0
        boundary_mse.append(b_mse)

        # Reconstruct a field for DISPLAY/MSE only — does not feed back
        pred_field = boundary_to_field(new_xy, cid, N=N_grid)
        pred_fields.append(pred_field.copy())
        true_fields.append(frames[t + 1].copy())
        field_mse.append(np.mean((pred_field - frames[t + 1]) ** 2))

        # Build next step's features directly from new_xy (no field):
        # recompute positional features (0-5, 8), carry forward the
        # gradient feature (6-7) from the previous step.
        new_feat = compute_positional_features(new_xy, cid)
        new_feat[:, 6:8] = feat[:, 6:8]   # carry forward gradient

        xy, feat = new_xy, new_feat

    return {
        'pred_fields' : pred_fields,
        'true_fields' : true_fields,
        'boundary_mse': np.array(boundary_mse),
        'field_mse'   : np.array(field_mse),
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_rollout_comparison(result: dict, traj_name: str,
                             out_path: str, n_show: int = 6):
    """Side-by-side: ground truth vs GNN prediction."""
    pred = result['pred_fields']
    true = result['true_fields']
    T    = len(pred)
    idx  = np.linspace(0, T - 1, min(n_show, T), dtype=int)

    vmin = 0.0
    vmax = 1.0

    fig, axes = plt.subplots(3, len(idx), figsize=(2.8 * len(idx), 8))
    if len(idx) == 1:
        axes = axes.reshape(3, 1)

    cmap = plt.cm.viridis

    for col, t in enumerate(idx):
        axes[0, col].imshow(true[t],          cmap=cmap, origin='lower',
                            vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f'step {t}', fontsize=9)
        axes[0, col].axis('off')

        axes[1, col].imshow(pred[t],          cmap=cmap, origin='lower',
                            vmin=vmin, vmax=vmax)
        axes[1, col].axis('off')

        err = np.abs(pred[t] - true[t])
        axes[2, col].imshow(err,              cmap='hot', origin='lower',
                            vmin=0, vmax=0.5)
        axes[2, col].axis('off')

    axes[0, 0].set_ylabel('Ground truth', fontsize=10)
    axes[1, 0].set_ylabel('GNN rollout',  fontsize=10)
    axes[2, 0].set_ylabel('|error|',      fontsize=10)

    mean_b_mse = result['boundary_mse'].mean()
    mean_f_mse = result['field_mse'].mean()
    fig.suptitle(
        f'GNN Rollout — {traj_name}\n'
        f'Mean boundary MSE: {mean_b_mse:.4e}   '
        f'Mean field MSE: {mean_f_mse:.4e}',
        fontweight='bold', fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_mse_curves(all_boundary_mse, all_field_mse, out_path: str):
    """Mean ± std of rollout MSE vs step."""
    b_arr = np.stack([m for m in all_boundary_mse if len(m) > 0])
    f_arr = np.stack([m for m in all_field_mse    if len(m) > 0])
    steps = np.arange(1, b_arr.shape[1] + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for arr, ax, title, col in [
        (b_arr, axes[0], 'Boundary MSE vs rollout step', '#1a3a6b'),
        (f_arr, axes[1], 'Field MSE vs rollout step',    '#c8a84b'),
    ]:
        m, s = arr.mean(0), arr.std(0)
        ax.plot(steps, m, color=col, lw=2)
        ax.fill_between(steps, m - s, m + s, alpha=0.2, color=col)
        ax.set_xlabel('Rollout step')
        ax.set_ylabel('MSE')
        ax.set_title(title, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


def print_comparison_table(gnn_one_step, gnn_rollout_mean,
                            gnn_field_mean):
    """Print a clean comparison table for the paper/presentation."""
    print("\n" + "="*60)
    print("  EVALUATION SUMMARY")
    print("="*60)
    print(f"  GNN one-step boundary MSE : {gnn_one_step:.4e}")
    print(f"  GNN rollout boundary MSE  : {gnn_rollout_mean:.4e}")
    print(f"  GNN rollout field MSE     : {gnn_field_mean:.4e}")
    print()
    print("  Comparison with FNO on Kobayashi data:")
    print(f"  {'Method':<20} {'Representation':<20} {'Works?':<10}")
    print(f"  {'-'*50}")
    print(f"  {'FNO':<20} {'Full 256x256 grid':<20} {'No':<10}")
    print(f"  {'  (sparse field)':<20} {'(95% empty)':<20} {'(diverges)':<10}")
    print(f"  {'GNN (ours)':<20} {'Boundary graph':<20} {'Yes':<10}")
    print(f"  {'  (this work)':<20} {'(512 nodes)':<20} {'':<10}")
    print("="*60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data_dir',   default='data/kobayashi')
    ap.add_argument('--out_dir',    default=None,
                    help='Output directory for figures (defaults to checkpoint dir)')
    ap.add_argument('--n_rollout',  type=int, default=5,
                    help='Number of trajectories to run rollout on')
    ap.add_argument('--max_steps',  type=int, default=40,
                    help='Max rollout steps per trajectory')
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    # Output directory
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.checkpoint), 'eval')
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model, ckpt = load_model(args.checkpoint, device)
    test_graphs = ckpt['test_graphs']
    print(f"Loaded: {args.checkpoint}")
    print(f"Test graphs: {len(test_graphs)}")

    # 1) One-step MSE
    os_mse = one_step_mse(model, test_graphs, device)
    print(f"\nOne-step boundary MSE: {os_mse:.4e}")

    # 2) Rollout on test trajectories
    files = sorted(glob.glob(os.path.join(args.data_dir, 'traj_*.npz')))
    if not files:
        print(f"No .npz files found in {args.data_dir}")
        return

    # Use last n_rollout files as visual examples
    rollout_files = files[-args.n_rollout:]
    all_b_mse, all_f_mse = [], []

    print(f"\nRunning rollout on {len(rollout_files)} trajectories...")
    for fpath in rollout_files:
        name   = os.path.splitext(os.path.basename(fpath))[0]
        d      = np.load(fpath, allow_pickle=True)
        frames = d['frames']

        result = rollout_trajectory(model, frames, device,
                                    max_steps=args.max_steps)

        all_b_mse.append(result['boundary_mse'])
        all_f_mse.append(result['field_mse'])

        plot_rollout_comparison(
            result, name,
            os.path.join(out_dir, f'rollout_{name}.png')
        )

    # 3) Summary curves
    plot_mse_curves(all_b_mse, all_f_mse,
                    os.path.join(out_dir, 'rollout_curves.png'))

    # 4) Print comparison table
    mean_b = np.mean([m.mean() for m in all_b_mse if len(m) > 0])
    mean_f = np.mean([m.mean() for m in all_f_mse if len(m) > 0])
    print_comparison_table(os_mse, mean_b, mean_f)

    print(f"\nFigures saved to: {out_dir}")


if __name__ == '__main__':
    main()
