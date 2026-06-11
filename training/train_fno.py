"""
============================================================
  Train a Fourier Neural Operator on PFC trajectories

  Objective:  minimize  MSE( FNO(n_t) , n_t+1 )
  (Pure data loss. NO physics-informed terms yet -- that is
   deliberate; we want a clean baseline before adding mass /
   energy / rollout penalties.)

  NSF IRES Physical AI Design Program
============================================================

USAGE
-----
    python train_fno.py --config config.yaml

What it does:
  * loads + splits the dataset (by trajectory, no leakage)
  * trains the FNO with Adam + (optional) cosine LR schedule
  * validates every epoch
  * checkpoints the best-val and the most-recent model
  * early-stops when validation stops improving
  * logs scalars to TensorBoard (if installed)

The checkpoint stores the model weights AND the normalization stats + config,
so evaluate_fno.py can reproduce the exact preprocessing.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml

from dataset import build_datasets
from fno_model import build_model, count_parameters


# ----------------------------------------------------------------------------
#  Utilities
# ----------------------------------------------------------------------------
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def pick_device(requested):
    """Resolve 'auto' to cuda -> mps -> cpu, or honor an explicit choice."""
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer, cfg, steps_per_epoch):
    t = cfg["train"]
    kind = t.get("scheduler", "none")
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t["epochs"])
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=t["step_size"], gamma=t["gamma"])
    return None


# ----------------------------------------------------------------------------
#  One epoch of training / validation
# ----------------------------------------------------------------------------
def run_epoch(model, loader, loss_fn, device, optimizer=None,
              grad_clip=0.0, writer=None, epoch=0, log_every=20, tag="train"):
    """
    Run a single pass over `loader`. If `optimizer` is given we train;
    otherwise we evaluate under torch.no_grad(). Returns the mean MSE.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total = 0.0
    n = 0
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for it, (x, y) in enumerate(loader):
            x = x.to(device)          # (B, C, H, W)
            y = y.to(device)          # (B, 1, H, W)

            pred = model(x)           # (B, 1, H, W)
            loss = loss_fn(pred, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bs = x.size(0)
            total += loss.item() * bs
            n += bs

            if is_train and writer is not None and (it % log_every == 0):
                step = epoch * len(loader) + it
                writer.add_scalar("batch/train_mse", loss.item(), step)

    return total / max(n, 1)


# ----------------------------------------------------------------------------
#  Main training entry point
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"].get("seed", 0))
    device = pick_device(cfg["train"].get("device", "auto"))
    print(f"Device: {device}")

    # --- data ---
    train_ds, val_ds, test_ds, info = build_datasets(cfg)
    print(f"Trajectories  -> train {len(info['train_files'])} | "
          f"val {len(info['val_files'])} | test {len(info['test_files'])}")
    print(f"Frame pairs   -> train {len(train_ds)} | "
          f"val {len(val_ds)} | test {len(test_ds)}")
    print(f"Normalization -> mean {info['norm_mean']:.5f}  std {info['norm_std']:.5f}")
    print(f"Conditioning  -> {info['include_conditioning']} "
          f"(in_channels={info['in_channels']})")

    nw = cfg["train"].get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, num_workers=nw, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                            shuffle=False, num_workers=nw)

    # --- model ---
    # Keep model in_channels in sync with whether conditioning is enabled.
    cfg["model"]["in_channels"] = info["in_channels"]
    model = build_model(cfg).to(device)
    print(f"Model: FNO2d  modes={cfg['model']['modes']} width={cfg['model']['width']} "
          f"layers={cfg['model']['layers']}  |  {count_parameters(model):,} params")

    # --- optimizer / schedule / loss ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    scheduler = make_scheduler(optimizer, cfg, len(train_loader))
class WeightedMSE(nn.Module):
    def __init__(self, crystal_weight=50.0):
        super().__init__()
        self.w = crystal_weight

    def forward(self, pred, target):
        weights = 1.0 + (self.w - 1.0) * (target > 0.1).float()
        return (weights * (pred - target)**2).mean()

loss_fn = WeightedMSE(crystal_weight=50.0)
    # --- logging / checkpoint dirs ---
    out_dir = cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    writer = None
    if cfg["logging"].get("tensorboard", True):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
            print(f"TensorBoard logging to {os.path.join(out_dir, 'tb')}")
        except Exception as e:                      # pragma: no cover
            print(f"TensorBoard unavailable ({e}); continuing without it.")

    ckpt_path = os.path.join(out_dir, cfg["logging"].get("ckpt_name", "best.pt"))
    last_path = os.path.join(out_dir, cfg["logging"].get("last_name", "last.pt"))

    def save_ckpt(path, epoch, val_loss):
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "val_loss": val_loss,
            "config": cfg,
            "norm_mean": info["norm_mean"],
            "norm_std": info["norm_std"],
            "include_conditioning": info["include_conditioning"],
            "cond_stats": info["cond_stats"],
            "in_channels": info["in_channels"],
            "train_files": info["train_files"],
            "val_files": info["val_files"],
            "test_files": info["test_files"],
        }, path)

    # --- training loop ---
    best_val = float("inf")
    epochs_no_improve = 0
    es = cfg["train"]
    print("\nStarting training...\n")
    for epoch in range(es["epochs"]):
        t0 = time.time()
        train_mse = run_epoch(
            model, train_loader, loss_fn, device, optimizer=optimizer,
            grad_clip=es.get("grad_clip", 0.0), writer=writer, epoch=epoch,
            log_every=cfg["logging"].get("log_every", 20), tag="train")
        val_mse = run_epoch(model, val_loader, loss_fn, device, tag="val")

        if scheduler is not None:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        if writer is not None:
            writer.add_scalar("epoch/train_mse", train_mse, epoch)
            writer.add_scalar("epoch/val_mse", val_mse, epoch)
            writer.add_scalar("epoch/lr", lr_now, epoch)

        dt = time.time() - t0
        improved = val_mse < best_val - es.get("min_delta", 0.0)
        flag = ""
        if improved:
            best_val = val_mse
            epochs_no_improve = 0
            save_ckpt(ckpt_path, epoch, val_mse)
            flag = "  <- best (saved)"
        else:
            epochs_no_improve += 1

        save_ckpt(last_path, epoch, val_mse)
        print(f"epoch {epoch:3d}/{es['epochs']}  "
              f"train {train_mse:.3e}  val {val_mse:.3e}  "
              f"lr {lr_now:.2e}  {dt:.1f}s{flag}")

        # --- early stopping ---
        if es.get("early_stopping", False) and epochs_no_improve >= es["patience"]:
            print(f"\nEarly stopping: no val improvement for {es['patience']} epochs.")
            break

    if writer is not None:
        writer.close()
    print(f"\nDone. Best val MSE: {best_val:.3e}")
    print(f"Best checkpoint:  {ckpt_path}")
    print(f"Last checkpoint:  {last_path}")
    print("Evaluate with:  python evaluate_fno.py --config config.yaml "
          f"--checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
