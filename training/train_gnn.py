"""
============================================================
  Train GNN for Crystal Boundary Prediction
  Kobayashi Phase Field Surrogate

  Trains the CrystalGNN to predict boundary displacement
  from one timestep to the next.

  NSF IRES Physical AI Design Program
  Ayush Shah & Tobias Li — Georgia Institute of Technology
============================================================

USAGE
-----
    python training/train_gnn.py \
        --graphs  data/kobayashi_graphs/graphs.pt \
        --out_dir runs/gnn_baseline \
        --epochs  100

On Colab:
    python train_gnn.py \
        --graphs  /content/drive/MyDrive/kobayashi_graphs/graphs.pt \
        --out_dir runs/gnn_baseline \
        --epochs  100
"""

import argparse
import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

# Allow running from repo root or training/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from solvers.gnn_model import CrystalGNN, BoundaryLoss
except ImportError:
    from gnn_model import CrystalGNN, BoundaryLoss


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ── Train / val split ─────────────────────────────────────────────────────────

def split_graphs(graphs, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Split graph list into train / val / test.
    Shuffles before splitting so different parameter combos
    are spread across all splits.
    """
    rng = random.Random(seed)
    idx = list(range(len(graphs)))
    rng.shuffle(idx)

    n       = len(graphs)
    n_test  = max(1, int(n * test_frac))
    n_val   = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    train = [graphs[i] for i in idx[:n_train]]
    val   = [graphs[i] for i in idx[n_train:n_train + n_val]]
    test  = [graphs[i] for i in idx[n_train + n_val:]]

    return train, val, test


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, loss_fn, device,
              optimizer=None) -> float:
    """Train or evaluate one epoch. Returns mean loss."""
    is_train = optimizer is not None
    model.train(is_train)

    total, n = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch in loader:
            batch = batch.to(device)
            pred  = model(batch)        # (N_total, 2)

            loss = loss_fn(pred, batch.y, batch.pos)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total += loss.item() * batch.num_graphs
            n     += batch.num_graphs

    return total / max(n, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs',   required=True,
                    help='Path to graphs.pt built by gnn_boundary.py')
    ap.add_argument('--out_dir',  default='runs/gnn_baseline')
    ap.add_argument('--epochs',   type=int,   default=100)
    ap.add_argument('--batch',    type=int,   default=32)
    ap.add_argument('--lr',       type=float, default=1e-3)
    ap.add_argument('--hidden',   type=int,   default=64)
    ap.add_argument('--layers',   type=int,   default=3)
    ap.add_argument('--dropout',  type=float, default=0.1)
    ap.add_argument('--smooth_w', type=float, default=0.10,
                    help='Smoothness loss weight')
    ap.add_argument('--length_w', type=float, default=0.05,
                    help='Length preservation loss weight')
    ap.add_argument('--patience', type=int,   default=20,
                    help='Early stopping patience')
    ap.add_argument('--seed',     type=int,   default=42)
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--test_frac',type=float, default=0.15)
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nCrystal GNN Training")
    print(f"  Device:  {device}")
    print(f"  Graphs:  {args.graphs}")
    print(f"  Out dir: {args.out_dir}")

    # ── Load graphs ───────────────────────────────────────────────────────────
    print(f"\nLoading graphs...")
    graphs = torch.load(args.graphs, map_location='cpu', weights_only=False)
    print(f"  Total graphs: {len(graphs)}")
    print(f"  Sample: {graphs[0]}")

    train_g, val_g, test_g = split_graphs(
        graphs, args.val_frac, args.test_frac, args.seed)
    print(f"  Split: train={len(train_g)}  val={len(val_g)}  test={len(test_g)}")

    train_loader = DataLoader(train_g, batch_size=args.batch,
                              shuffle=True)
    val_loader   = DataLoader(val_g,   batch_size=args.batch,
                              shuffle=False)
    test_loader  = DataLoader(test_g,  batch_size=args.batch,
                              shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    in_channels = graphs[0].x.shape[1]   # 8 features
    model = CrystalGNN(
        in_channels = in_channels,
        hidden_dim  = args.hidden,
        n_layers    = args.layers,
        dropout     = args.dropout,
    ).to(device)
    print(f"\n  Model: CrystalGNN  hidden={args.hidden}  "
          f"layers={args.layers}  "
          f"params={model.count_params():,}")

    # ── Optimizer / loss ──────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn   = BoundaryLoss(smooth_weight=args.smooth_w,
                             length_weight=args.length_w)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val      = float('inf')
    no_improve    = 0
    ckpt_path     = os.path.join(args.out_dir, 'best_gnn.pt')
    history       = {'train': [], 'val': []}

    print(f"\nStarting training for {args.epochs} epochs...\n")
    print("-" * 65)

    sweep_t0 = time.time()

    for epoch in range(args.epochs):
        t0         = time.time()
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss   = run_epoch(model, val_loader,   loss_fn, device)
        scheduler.step()

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        lr_now = optimizer.param_groups[0]['lr']
        dt     = time.time() - t0
        flag   = ''

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_loss'   : val_loss,
                'args'       : vars(args),
                'in_channels': in_channels,
                'test_graphs': test_g,
            }, ckpt_path)
            flag = '  <- best (saved)'
        else:
            no_improve += 1

        print(f"  epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4e}  val={val_loss:.4e}  "
              f"lr={lr_now:.1e}  {dt:.1f}s{flag}")

        if no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    total_time = (time.time() - sweep_t0) / 60
    print(f"\n{'='*65}")
    print(f"Training complete in {total_time:.1f} min")
    print(f"Best val loss: {best_val:.4e}")
    print(f"Checkpoint: {ckpt_path}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    test_loss = run_epoch(model, test_loader, loss_fn, device)
    print(f"Test loss:  {test_loss:.4e}")

    # ── Save history ──────────────────────────────────────────────────────────
    import json
    with open(os.path.join(args.out_dir, 'history.json'), 'w') as f:
        json.dump({**history, 'test_loss': test_loss,
                   'best_val': best_val}, f, indent=2)

    print(f"\nEvaluate with:")
    print(f"  python training/evaluate_gnn.py "
          f"--checkpoint {ckpt_path}")


if __name__ == '__main__':
    main()
