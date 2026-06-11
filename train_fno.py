"""
============================================================
  Train a Fourier Neural Operator on PFC trajectories
                                              (PFC ONLY)

  This is the PHASE FIELD CRYSTAL trainer that lives at the
  repo root, next to the PFC solver/pipeline. It is separate
  from training/train_fno.py, which is the Kobayashi-oriented
  trainer (crystal-weighted loss) maintained on its own track.

  Objective:  minimize  MSE( FNO(n_t) , n_t+1 )
  Pure data loss -- no physics penalty terms. Physical
  structure enters only through the ARCHITECTURE flags in
  config.yaml (model.predict_delta, model.enforce_mass),
  which keep the optimization target a plain MSE.

  NSF IRES Physical AI Design Program
============================================================

USAGE
-----
    python train_fno.py --config config.yaml

What it does:
  * loads + splits the dataset (by trajectory or by config; no leakage)
  * trains the FNO with Adam + cosine LR schedule
  * validates every epoch (one-step MSE)
  * every `rollout_val_every` epochs, also runs a free-running
    AUTOREGRESSIVE ROLLOUT on validation trajectories and logs the
    multi-step MSE + final mass drift -- the metrics that actually
    matter for a surrogate, watched while training instead of only
    discovered at evaluation time
  * checkpoints best-val and most-recent models
  * early-stops when one-step validation stops improving
  * logs everything to TensorBoard (if installed)

The checkpoint stores the model weights AND the normalization stats +
config + file splits, so evaluate_fno.py reproduces the exact
preprocessing and evaluates on the held-out test trajectories.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml

from dataset import (build_datasets, inspect_trajectory, _load_manifest,
                     _run_id_from_path, _COND_COLUMNS)
from fno_model import build_model, count_parameters


# ----------------------------------------------------------------------------
#  Setup utilities
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
    if getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer, cfg):
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
#  Gradient clipping that works with the FNO's complex spectral weights
# ----------------------------------------------------------------------------
def clip_grad_norm_(parameters, max_norm):
    """
    Drop-in replacement for nn.utils.clip_grad_norm_. The spectral conv
    weights are complex (cfloat), and some backends (notably MPS) do not
    implement norm ops for complex tensors. Viewing each complex gradient as
    a (..., 2) real tensor gives the identical norm (|z|^2 = re^2 + im^2)
    and works everywhere.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    norms = [torch.linalg.vector_norm(
        torch.view_as_real(g) if g.is_complex() else g) for g in grads]
    total = torch.linalg.vector_norm(torch.stack(norms))
    scale = max_norm / (total + 1e-12)
    if scale < 1.0:
        for g in grads:
            g.mul_(scale)        # real scalar x complex grad broadcasts fine
    return total


# ----------------------------------------------------------------------------
#  One epoch of training / validation (one-step, teacher-forced)
# ----------------------------------------------------------------------------
def run_epoch(model, loader, loss_fn, device, optimizer=None,
              grad_clip=0.0, writer=None, epoch=0, log_every=20):
    """
    One pass over `loader`. With `optimizer` we train; without it we
    evaluate under no_grad. Returns the mean MSE over all pairs.

    Prints live in-epoch progress (batch counter + throughput) so a slow
    device or stalled run is visible immediately instead of looking hung.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total, n = 0.0, 0
    t0 = time.perf_counter()
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for it, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)   # (B, C, H, W)  frame_t
            y = y.to(device, non_blocking=True)   # (B, 1, H, W)  frame_t+1

            pred = model(x)           # (B, 1, H, W)
            loss = loss_fn(pred, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bs = x.size(0)
            total += loss.item() * bs
            n += bs

            if is_train and (it + 1) % log_every == 0:
                rate = (it + 1) / (time.perf_counter() - t0)
                eta = (len(loader) - it - 1) / max(rate, 1e-9)
                print(f"\r    batch {it + 1:4d}/{len(loader)}  "
                      f"mse {loss.item():.3e}  "
                      f"{rate:4.1f} batch/s  eta {eta:4.0f}s ", end="", flush=True)
                if writer is not None:
                    writer.add_scalar("batch/train_mse", loss.item(),
                                      epoch * len(loader) + it)
    if is_train:
        print("\r" + " " * 70 + "\r", end="", flush=True)   # clear progress line
    return total / max(n, 1)


# ----------------------------------------------------------------------------
#  Rollout validation (autoregressive, the honest surrogate metric)
# ----------------------------------------------------------------------------
def _conditioning_for(path, manifest, cond_stats):
    """[r, n0] for a trajectory (manifest first, then npz meta), normalized."""
    rec_meta = inspect_trajectory(path)["meta"]
    row = manifest.get(_run_id_from_path(path), {})
    vals = []
    for col in _COND_COLUMNS:
        if col in row:
            vals.append(float(row[col]))
        elif col in rec_meta:
            vals.append(float(rec_meta[col]))
        else:
            vals.append(0.0)
    cond = np.asarray(vals, dtype=np.float32)
    if cond_stats is not None:
        cond = (cond - cond_stats["mean"]) / (cond_stats["std"] + 1e-8)
    return cond


@torch.no_grad()
def rollout_validation(model, files, info, manifest, device, max_traj=5):
    """
    Free-running rollout from frame_0 on up to `max_traj` validation
    trajectories:  n0 -> n1_hat -> n2_hat -> ...  (model eats its own output).

    All trajectories are rolled out together as one batch -- T-1 forward
    passes total instead of (T-1) * n_traj, so this stays cheap even when
    called every few epochs.

    Returns (mean rollout MSE over all steps/trajectories in PHYSICAL units,
             mean relative mass drift at the final step).
    """
    model.eval()
    mean, std = info["norm_mean"], info["norm_std"]

    # Stack the ground-truth trajectories: (B, T, H, W) in physical units.
    trajs, conds = [], []
    for path in files[:max_traj]:
        rec = inspect_trajectory(path)
        trajs.append(np.concatenate([rec["inputs"], rec["targets"][-1:]], axis=0))
        conds.append(_conditioning_for(path, manifest, info["cond_stats"])
                     if info["include_conditioning"] else None)
    T = min(t.shape[0] for t in trajs)                  # common horizon
    true_phys = np.stack([t[:T] for t in trajs])        # (B, T, H, W)
    B, _, H, W = true_phys.shape

    cond_maps = None
    if info["include_conditioning"]:
        c = torch.from_numpy(np.stack(conds)).float()                  # (B, n_cond)
        cond_maps = c.view(B, -1, 1, 1).expand(B, c.shape[1], H, W).to(device)

    cur = torch.from_numpy(
        (true_phys[:, 0] - mean) / (std + 1e-8)).float().view(B, 1, H, W).to(device)
    preds = [true_phys[:, 0]]
    for _ in range(T - 1):
        x = cur if cond_maps is None else torch.cat([cur, cond_maps], dim=1)
        cur = model(x)
        preds.append(cur.cpu().numpy()[:, 0] * (std + 1e-8) + mean)   # physical
    preds = np.stack(preds, axis=1)                     # (B, T, H, W)

    mse = float(np.mean((preds[:, 1:] - true_phys[:, 1:]) ** 2))
    m_true = true_phys[:, -1].mean(axis=(1, 2))
    m_pred = preds[:, -1].mean(axis=(1, 2))
    drift = float(np.mean(np.abs(m_pred - m_true) / (np.abs(m_true) + 1e-12)))
    return mse, drift


# ----------------------------------------------------------------------------
#  Main training entry point
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train the PFC FNO surrogate")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"].get("seed", 0))
    device = pick_device(cfg["train"].get("device", "auto"))
    print(f"Device: {device}")
    if device.type == "cpu":
        print("\n" + "!" * 64)
        print("! WARNING: no GPU detected -- training on CPU is 10-30x slower.")
        print("! In Colab: Runtime -> Change runtime type -> T4 GPU, then")
        print("! restart and rerun the notebook from the top.")
        print("!" * 64 + "\n")

    # --- data ---
    train_ds, val_ds, test_ds, info = build_datasets(cfg)
    manifest = _load_manifest(cfg["data"]["data_dir"])
    print(f"Trajectories  -> train {len(info['train_files'])} | "
          f"val {len(info['val_files'])} | test {len(info['test_files'])} "
          f"(split_by={info.get('split_by', 'trajectory')})")
    print(f"Frame pairs   -> train {len(train_ds)} | "
          f"val {len(val_ds)} | test {len(test_ds)}")
    print(f"Normalization -> mean {info['norm_mean']:.5f}  std {info['norm_std']:.5f}")
    print(f"Conditioning  -> {info['include_conditioning']} "
          f"(in_channels={info['in_channels']})")

    nw = cfg["train"].get("num_workers", 0)
    pin = device.type == "cuda"          # faster host->GPU copies on CUDA
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, num_workers=nw, drop_last=False,
                              pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                            shuffle=False, num_workers=nw, pin_memory=pin)

    # --- model ---
    cfg["model"]["in_channels"] = info["in_channels"]
    model = build_model(cfg).to(device)
    print(f"Model: FNO2d  modes={cfg['model']['modes']} "
          f"width={cfg['model']['width']} layers={cfg['model']['layers']}  "
          f"predict_delta={cfg['model'].get('predict_delta', False)}  "
          f"enforce_mass={cfg['model'].get('enforce_mass', False)}  |  "
          f"{count_parameters(model):,} params")

    # --- optimizer / schedule / loss ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    scheduler = make_scheduler(optimizer, cfg)
    loss_fn = nn.MSELoss()           # pure MSE -- the clean PFC baseline

    # --- logging / checkpoints ---
    out_dir = cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    writer = None
    if cfg["logging"].get("tensorboard", True):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
            print(f"TensorBoard logging to {os.path.join(out_dir, 'tb')}")
        except Exception as e:
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
    es = cfg["train"]
    rollout_every = es.get("rollout_val_every", 10)
    rollout_trajs = es.get("rollout_val_trajs", 5)
    best_val = float("inf")
    epochs_no_improve = 0

    print("\nStarting training...\n")
    for epoch in range(es["epochs"]):
        t0 = time.time()
        train_mse = run_epoch(
            model, train_loader, loss_fn, device, optimizer=optimizer,
            grad_clip=es.get("grad_clip", 0.0), writer=writer, epoch=epoch,
            log_every=cfg["logging"].get("log_every", 20))
        val_mse = run_epoch(model, val_loader, loss_fn, device)

        if scheduler is not None:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        if writer is not None:
            writer.add_scalar("epoch/train_mse", train_mse, epoch)
            writer.add_scalar("epoch/val_mse", val_mse, epoch)
            writer.add_scalar("epoch/lr", lr_now, epoch)

        # Periodic autoregressive rollout on val trajectories: the metric a
        # surrogate actually lives or dies by, watched during training.
        roll_note = ""
        if rollout_every > 0 and (epoch % rollout_every == 0
                                  or epoch == es["epochs"] - 1):
            r_mse, r_drift = rollout_validation(
                model, info["val_files"], info, manifest, device,
                max_traj=rollout_trajs)
            roll_note = f"  rollout {r_mse:.3e} (mass drift {r_drift:.1e})"
            if writer is not None:
                writer.add_scalar("epoch/val_rollout_mse", r_mse, epoch)
                writer.add_scalar("epoch/val_rollout_mass_drift", r_drift, epoch)

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
              f"lr {lr_now:.2e}  {dt:.1f}s{roll_note}{flag}")

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
