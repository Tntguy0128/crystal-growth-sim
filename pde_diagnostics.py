"""
============================================================
  PDE-residual diagnostics for the FNO surrogate (step 2)

  Localizes WHERE and WHEN an FNO rollout leaves the physical
  manifold, using the differentiable PFC solver as a yardstick.

  For each test trajectory we run the free-running autoregressive
  rollout and, at every step, measure the relative PDE residual
  between the model's PREDICTED frame and the SOLVER applied to the
  model's previous frame:

      res(t) = || n_pred(t) - solver(n_pred(t-1)) ||^2
               / ( || solver(n_pred(t-1)) - n_pred(t-1) ||^2 + eps )

  Unlike rollout MSE, this needs NO ground truth -- it is a pure
  physics check, so it flags unphysical predictions even on the
  unlabeled states the model drifts into, and its SPATIAL map points
  at exactly which grains / boundaries the prediction got wrong.

  Outputs (under out_dir):
    pde_residual_curve.png   residual vs rollout step (mean over test trajs)
    pde_residual_map.png     4-panel spatial map on the single worst frame
    pde_residual.csv         per-traj, per-step residual + rollout MSE

  Usage:
    python pde_diagnostics.py --checkpoint runs/.../best.pt --out diag/
  or import run_pde_diagnostics(...) from a notebook.

  NSF IRES Physical AI Design Program
============================================================
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import inspect_trajectory, _load_manifest, _run_id_from_path
from fno_model import build_model
from train_fno import DifferentiablePFCStep, pick_device


def _load_model(ck, device):
    cfg = ck["config"]
    cfg["model"]["in_channels"] = ck.get("in_channels", 1)
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, cfg


def _traj_r(path, manifest, meta):
    """Physical temperature r for a trajectory (manifest first, then npz)."""
    row = manifest.get(_run_id_from_path(path), {})
    if "r" in row:
        return float(row["r"])
    return float(meta.get("r", -0.30))


@torch.no_grad()
def _rollout(model, frames_norm, device):
    """frames_norm: (T,H,W) normalized -> preds_norm: (T,H,W) (density only)."""
    T, H, W = frames_norm.shape
    cur = torch.from_numpy(frames_norm[0]).view(1, 1, H, W).float().to(device)
    preds = [cur]
    for _ in range(T - 1):
        cur = model(cur)
        preds.append(cur)
    return torch.cat(preds, 0)[:, 0].cpu().numpy()      # (T,H,W)


def run_pde_diagnostics(ckpt_path, out_dir, max_traj=8, device=None, label=""):
    """
    Returns a dict with the mean residual-vs-step curve and the worst frame,
    and writes the figures + CSV to out_dir. `label` tags the curve/title.
    """
    device = device or pick_device("auto")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, _ = _load_model(ck, device)
    mean, std = float(ck["norm_mean"]), float(ck["norm_std"])
    test_files = ck["test_files"]
    if ck.get("include_conditioning"):
        print("WARNING: conditioning models are not supported by this diagnostic "
              "(density-only rollout); results approximate.")

    # Differentiable stepper from the data's own metadata (matches the generator).
    rec0 = inspect_trajectory(test_files[0])
    meta0, H0 = rec0["meta"], rec0["inputs"].shape[-1]
    dx = float(meta0.get("dx", float(meta0.get("L", 16 * np.pi)) / H0))
    dt = float(meta0.get("dt", 0.25))
    M = float(meta0.get("M", 1.0))
    substeps = int(meta0.get("save_every", 25))
    stepper = DifferentiablePFCStep(dx, dt, M, substeps).to(device)
    manifest = {}
    try:
        manifest = _load_manifest(os.path.dirname(test_files[0]))
    except Exception:
        pass

    os.makedirs(out_dir, exist_ok=True)
    rows, curves = [], []
    worst = None        # (res, name, t, prev, pred, solver) all normalized maps

    for path in test_files[:max_traj]:
        rec = inspect_trajectory(path)
        name = os.path.basename(path)
        frames = np.concatenate([rec["inputs"], rec["targets"][-1:]], 0).astype(np.float32)
        T, H, W = frames.shape
        frames_norm = (frames - mean) / (std + 1e-8)
        preds_norm = _rollout(model, frames_norm, device)               # (T,H,W)
        r = _traj_r(path, manifest, rec["meta"])
        rr = torch.tensor([r], device=device)

        res_steps = []
        for t in range(1, T):
            prev = torch.from_numpy(preds_norm[t - 1]).view(1, 1, H, W).float().to(device)
            pred = torch.from_numpy(preds_norm[t]).view(1, 1, H, W).float().to(device)
            tgt_phys = stepper.frame(prev * std + mean, rr)             # solver(prev)
            tgt = (tgt_phys - mean) / (std + 1e-8)                      # normalized
            err_map = (pred - tgt) ** 2
            den = float(((tgt - prev) ** 2).mean()) + 1e-8
            rel = float(err_map.mean()) / den
            roll_mse = float(((preds_norm[t] - frames_norm[t]) ** 2).mean())
            res_steps.append(rel)
            rows.append({"trajectory": name, "step": t, "r": r,
                         "pde_residual": rel, "rollout_mse": roll_mse})
            if worst is None or rel > worst[0]:
                worst = (rel, name, t,
                         prev.cpu()[0, 0].numpy(), pred.cpu()[0, 0].numpy(),
                         tgt.cpu()[0, 0].numpy())
        curves.append(res_steps)

    # --- residual-vs-step curve (mean +/- spread across trajectories) ---
    K = min(len(c) for c in curves)
    arr = np.array([c[:K] for c in curves])                            # (n_traj, K)
    mean_curve, std_curve = arr.mean(0), arr.std(0)
    steps = np.arange(1, K + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, mean_curve, "-o", lw=2, label=f"mean PDE residual{(' '+label) if label else ''}")
    plt.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
    plt.xlabel("rollout step"); plt.ylabel("relative PDE residual")
    plt.title(f"Off-manifold drift along rollout{(' — '+label) if label else ''}")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    curve_png = os.path.join(out_dir, "pde_residual_curve.png")
    plt.savefig(curve_png, dpi=120); plt.close()

    # --- worst-frame spatial map: where the physics breaks ---
    _, wname, wt, wprev, wpred, wsolver = worst
    resid_map = (wpred - wsolver) ** 2
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    for a, img, ttl in zip(
            ax, [wprev, wpred, wsolver, resid_map],
            [f"prev pred (t={wt-1})", f"model pred (t={wt})",
             "solver(prev) = physics target", "residual (pred - target)^2"]):
        cmap = "inferno" if ttl.startswith("residual") else "viridis"
        im = a.imshow(img, cmap=cmap); a.set_title(ttl, fontsize=9); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"Worst physics violation: {wname} step {wt} "
                 f"(residual {worst[0]:.2e}){(' — '+label) if label else ''}")
    fig.tight_layout()
    map_png = os.path.join(out_dir, "pde_residual_map.png")
    fig.savefig(map_png, dpi=120); plt.close(fig)

    # --- CSV ---
    csv_path = os.path.join(out_dir, "pde_residual.csv")
    import csv as _csv
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["trajectory", "step", "r",
                                           "pde_residual", "rollout_mse"])
        w.writeheader(); w.writerows(rows)

    print(f"[pde_diagnostics{(' '+label) if label else ''}] "
          f"mean residual over rollout: {float(mean_curve.mean()):.3e}  |  "
          f"final-step: {float(mean_curve[-1]):.3e}  |  "
          f"worst: {wname} step {wt} ({worst[0]:.2e})")
    print(f"  wrote {curve_png}, {map_png}, {csv_path}")
    return {"steps": steps, "mean_curve": mean_curve, "std_curve": std_curve,
            "worst": {"trajectory": wname, "step": wt, "residual": worst[0]},
            "curve_png": curve_png, "map_png": map_png, "csv": csv_path}


def main():
    ap = argparse.ArgumentParser(description="PDE-residual rollout diagnostics")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="pde_diag")
    ap.add_argument("--max-traj", type=int, default=8)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    run_pde_diagnostics(args.checkpoint, args.out,
                        max_traj=args.max_traj, label=args.label)


if __name__ == "__main__":
    main()
