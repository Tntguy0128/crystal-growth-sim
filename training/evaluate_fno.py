"""
============================================================
  Evaluate a trained FNO surrogate against the PFC solver

  Reports:
    1. One-step prediction MSE         (teacher-forced, per pair)
    2. Multi-step rollout MSE          (autoregressive, free-running)
    3. Relative mass-conservation error (PFC conserves total mass exactly)
    4. Rollout visualizations          (predicted vs. true, side by side)

  NSF IRES Physical AI Design Program
============================================================

USAGE
-----
    python evaluate_fno.py --config config.yaml --checkpoint runs/fno_baseline/best.pt

ONE-STEP vs ROLLOUT (why both matter)
-------------------------------------
* One-step MSE feeds the network the TRUE frame_t and scores its prediction of
  frame_t+1. This is the easy regime -- the input is always on the data manifold.
* Rollout MSE feeds the network its OWN previous prediction:
        n0 -> n1_hat -> n2_hat -> ...
  Errors compound, so rollout MSE grows with horizon and is the honest test of
  a surrogate. We compare the free-running rollout against the true PFC
  trajectory frame by frame.

MASS CONSERVATION. The PFC dynamics conserve the spatial mean of n (it is a
conserved order parameter -- see the nabla^2 in PCF_Baseline.py). A good
surrogate should keep mean(n) roughly constant during rollout. We report the
relative drift of the predicted mean versus the true mean at each step, in
PHYSICAL units (after undoing normalization).
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")               # headless: write figures to disk
import matplotlib.pyplot as plt

import yaml

from dataset import inspect_trajectory, _load_manifest, _run_id_from_path, _COND_COLUMNS
from fno_model import build_model


# ----------------------------------------------------------------------------
#  Setup helpers
# ----------------------------------------------------------------------------
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def pick_device(requested):
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(ckpt_path, device):
    """Load weights + saved preprocessing metadata, rebuild the model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cfg["model"]["in_channels"] = ckpt["in_channels"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


# ----------------------------------------------------------------------------
#  Normalization (mirrors dataset.py, using the checkpoint's stored stats)
# ----------------------------------------------------------------------------
class Normalizer:
    def __init__(self, ckpt, manifest):
        self.mean = ckpt["norm_mean"]
        self.std = ckpt["norm_std"]
        self.include_conditioning = ckpt["include_conditioning"]
        self.cond_stats = ckpt["cond_stats"]
        self.manifest = manifest

    def norm(self, a):
        return (a - self.mean) / (self.std + 1e-8)

    def denorm(self, a):
        return a * (self.std + 1e-8) + self.mean

    def conditioning(self, path, traj_meta):
        """Resolve [r, n0] for a trajectory (manifest first, then npz meta)."""
        run = _run_id_from_path(path)
        mrow = self.manifest.get(run, {})
        vals = []
        for col in _COND_COLUMNS:
            if col in mrow:
                vals.append(float(mrow[col]))
            elif col in traj_meta:
                vals.append(float(traj_meta[col]))
            else:
                vals.append(0.0)
        cond = np.asarray(vals, dtype=np.float32)
        if self.cond_stats is not None:
            cond = (cond - self.cond_stats["mean"]) / (self.cond_stats["std"] + 1e-8)
        return cond

    def make_input(self, field_norm, cond, H, W, device):
        """Build a (1, C, H, W) model input from a normalized field (H, W)."""
        x = torch.from_numpy(np.ascontiguousarray(field_norm)).view(1, 1, H, W)
        if self.include_conditioning and cond is not None:
            chans = [x] + [torch.full((1, 1, H, W), float(c)) for c in cond]
            x = torch.cat(chans, dim=1)
        return x.to(device)


# ----------------------------------------------------------------------------
#  1) One-step prediction MSE (teacher-forced)
# ----------------------------------------------------------------------------
@torch.no_grad()
def one_step_mse(model, files, normo, device):
    """Mean MSE over every (frame_t -> frame_t+1) pair in `files`, normalized space."""
    total, n = 0.0, 0
    for path in files:
        rec = inspect_trajectory(path)
        H, W = rec["inputs"].shape[-2:]
        cond = normo.conditioning(path, rec["meta"]) if normo.include_conditioning else None
        xs = normo.norm(rec["inputs"])           # (P, H, W)
        ys = normo.norm(rec["targets"])          # (P, H, W)
        for t in range(xs.shape[0]):
            inp = normo.make_input(xs[t], cond, H, W, device)
            pred = model(inp).cpu().numpy()[0, 0]
            total += float(np.mean((pred - ys[t]) ** 2))
            n += 1
    return total / max(n, 1)


# ----------------------------------------------------------------------------
#  2 + 3) Autoregressive rollout: MSE vs horizon and mass drift
# ----------------------------------------------------------------------------
@torch.no_grad()
def rollout(model, path, normo, device, length=0):
    """
    Free-running rollout from frame_0.

    Returns dict with:
        pred   : (L+1, H, W) physical-unit predicted fields (incl. frame_0)
        true   : (L+1, H, W) physical-unit ground-truth fields
        mse    : (L,) per-step MSE in physical units, pred vs true
        mass_err : (L+1,) relative |mean(pred)-mean(true)| / |mean(true)|
    """
    rec = inspect_trajectory(path)
    # Rebuild the full trajectory in physical units: inputs[0..P-1] + last target.
    true_phys = np.concatenate(
        [rec["inputs"], rec["targets"][-1:]], axis=0).astype(np.float32)  # (T, H, W)
    T, H, W = true_phys.shape
    L = T - 1 if (length is None or length <= 0) else min(length, T - 1)

    cond = normo.conditioning(path, rec["meta"]) if normo.include_conditioning else None

    # Start from the true initial frame (normalized).
    cur = normo.norm(true_phys[0])
    preds_phys = [true_phys[0].copy()]
    for _ in range(L):
        inp = normo.make_input(cur, cond, H, W, device)
        nxt = model(inp).cpu().numpy()[0, 0]      # normalized prediction
        preds_phys.append(normo.denorm(nxt))
        cur = nxt                                  # feed prediction back in

    preds_phys = np.stack(preds_phys, axis=0)      # (L+1, H, W)
    true_cut = true_phys[:L + 1]

    mse = np.mean((preds_phys[1:] - true_cut[1:]) ** 2, axis=(1, 2))   # (L,)
    pred_mean = preds_phys.mean(axis=(1, 2))
    true_mean = true_cut.mean(axis=(1, 2))
    mass_err = np.abs(pred_mean - true_mean) / (np.abs(true_mean) + 1e-12)

    return {"pred": preds_phys, "true": true_cut, "mse": mse, "mass_err": mass_err}


# ----------------------------------------------------------------------------
#  4) Visualization
# ----------------------------------------------------------------------------
def plot_rollout(roll, path, out_png, n_frames=6):
    """Top row: true PFC frames. Middle: FNO rollout. Bottom: abs error."""
    pred, true = roll["pred"], roll["true"]
    Lp1 = pred.shape[0]
    idx = np.linspace(0, Lp1 - 1, min(n_frames, Lp1), dtype=int)

    vmin = np.percentile(true[-1], 2)
    vmax = np.percentile(true[-1], 98)
    err = np.abs(pred - true)
    emax = max(err[idx].max(), 1e-8)

    fig, axes = plt.subplots(3, len(idx), figsize=(2.4 * len(idx), 7.2))
    if len(idx) == 1:
        axes = axes.reshape(3, 1)
    for c, t in enumerate(idx):
        axes[0, c].imshow(true[t], cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        axes[0, c].set_title(f"step {t}", fontsize=9)
        axes[1, c].imshow(pred[t], cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        axes[2, c].imshow(err[t], cmap="magma", origin="lower", vmin=0, vmax=emax)
        for r in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    axes[0, 0].set_ylabel("PFC (true)", fontsize=10)
    axes[1, 0].set_ylabel("FNO rollout", fontsize=10)
    axes[2, 0].set_ylabel("|error|", fontsize=10)
    fig.suptitle(f"Rollout vs. ground truth — {os.path.basename(path)}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_curves(all_mse, all_mass, out_png):
    """Mean +/- std of rollout MSE and mass error vs. step, across trajectories."""
    mse = np.stack(all_mse, axis=0)        # (n_traj, L)
    mass = np.stack(all_mass, axis=0)      # (n_traj, L+1)
    steps_mse = np.arange(1, mse.shape[1] + 1)
    steps_mass = np.arange(mass.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    m, s = mse.mean(0), mse.std(0)
    axes[0].plot(steps_mse, m, color="#1a3a6b", lw=2)
    axes[0].fill_between(steps_mse, m - s, m + s, alpha=0.2, color="#1a3a6b")
    axes[0].set_xlabel("rollout step"); axes[0].set_ylabel("MSE (physical units)")
    axes[0].set_title("Rollout MSE vs. horizon", fontweight="bold")
    axes[0].set_yscale("log"); axes[0].grid(True, alpha=0.3)

    m2, s2 = mass.mean(0), mass.std(0)
    axes[1].plot(steps_mass, m2, color="#c8a84b", lw=2)
    axes[1].fill_between(steps_mass, m2 - s2, m2 + s2, alpha=0.2, color="#c8a84b")
    axes[1].set_xlabel("rollout step"); axes[1].set_ylabel("relative mass error")
    axes[1].set_title("Mass-conservation drift", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="defaults to <logging.out_dir>/<logging.ckpt_name>")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(cfg["train"].get("device", "auto"))
    print(f"Device: {device}")

    ckpt_path = args.checkpoint or os.path.join(
        cfg["logging"]["out_dir"], cfg["logging"].get("ckpt_name", "best.pt"))
    model, ckpt = load_checkpoint(ckpt_path, device)
    print(f"Loaded checkpoint: {ckpt_path}  (val_loss={ckpt.get('val_loss'):.3e})")

    manifest = _load_manifest(cfg["data"]["data_dir"])
    normo = Normalizer(ckpt, manifest)

    # Evaluate on the SAME test split the checkpoint was trained against.
    test_files = ckpt.get("test_files") or []
    if not test_files:
        raise RuntimeError("Checkpoint has no stored test split.")
    print(f"Test trajectories: {len(test_files)}")

    out_dir = cfg["eval"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # 1) One-step MSE
    os_mse = one_step_mse(model, test_files, normo, device)

    # 2+3) Rollouts over the whole test set
    L = cfg["eval"].get("rollout_length", 0)
    all_mse, all_mass, final_mse = [], [], []
    rolls = []
    for path in test_files:
        roll = rollout(model, path, normo, device, length=L)
        all_mse.append(roll["mse"])
        all_mass.append(roll["mass_err"])
        final_mse.append(roll["mse"][-1])
        rolls.append((path, roll))

    rollout_mse_mean = float(np.mean([m.mean() for m in all_mse]))
    final_step_mse = float(np.mean(final_mse))
    mass_err_final = float(np.mean([m[-1] for m in all_mass]))

    # 4) Visualizations
    n_vis = min(cfg["eval"].get("num_vis_trajectories", 3), len(rolls))
    for path, roll in rolls[:n_vis]:
        name = os.path.splitext(os.path.basename(path))[0]
        plot_rollout(roll, path, os.path.join(out_dir, f"rollout_{name}.png"),
                     n_frames=cfg["eval"].get("vis_frames", 6))
    plot_curves(all_mse, all_mass, os.path.join(out_dir, "rollout_curves.png"))

    # --- report ---
    print("\n" + "=" * 56)
    print("  EVALUATION SUMMARY  (test split, physical units)")
    print("=" * 56)
    print(f"  1) One-step prediction MSE (normalized) : {os_mse:.4e}")
    print(f"  2) Rollout MSE  (mean over horizon)     : {rollout_mse_mean:.4e}")
    print(f"     Rollout MSE  (final step)            : {final_step_mse:.4e}")
    print(f"  3) Relative mass error (final step)     : {mass_err_final:.4e}")
    print(f"  4) Figures written to                   : {out_dir}/")
    print("=" * 56)


if __name__ == "__main__":
    main()
