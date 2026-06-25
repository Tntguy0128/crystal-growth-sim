"""
============================================================
  Grow a crystal from a drawn seed — the model layer

  Wraps the existing PFC pipeline so a (drawn) mask becomes a
  predicted crystal trajectory:

      mask  ->  custom_mask initial field  ->  FNO rollout  ->  frames

  Also exposes the two things that make this a TOOL rather than a
  demo:
    * grow_solver(): the exact PFC solver on the same seed, for
      ground-truth verification (slower).
    * pde_confidence(): how physical the FNO rollout is, via the
      differentiable solver — a trust score for the analysis.

  Inference is the fast path; the solver is opt-in verification.

  NSF IRES Physical AI Design Program
============================================================
"""

import os
import sys
from dataclasses import replace

import numpy as np

# repo root on the path so we can reuse the existing modules
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch                                            # noqa: E402
from pfc_solver import PFCConfig, PFCSolver, make_initial_condition  # noqa: E402
from fno_model import build_model                       # noqa: E402
from train_fno import DifferentiablePFCStep, pick_device  # noqa: E402


# ----------------------------------------------------------------------------
#  Model loading
# ----------------------------------------------------------------------------
def load_fno(ckpt_path, device=None):
    """Load a trained FNO checkpoint. Returns (model, checkpoint_dict, device)."""
    device = device or pick_device("auto")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    cfg["model"]["in_channels"] = ck.get("in_channels", 1)
    if ck.get("include_conditioning"):
        raise ValueError("This tool supports density-only checkpoints "
                         "(include_conditioning=False).")
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck, device


# ----------------------------------------------------------------------------
#  Seed mask -> physical initial field
# ----------------------------------------------------------------------------
def prep_mask(mask, N=128):
    """Coerce an arbitrary drawn array to an (N, N) float mask in [0, 1]."""
    m = np.asarray(mask, dtype=float)
    if m.ndim == 3:                                     # RGBA/RGB -> intensity
        m = m[..., :3].mean(axis=-1) if m.shape[-1] >= 3 else m[..., 0]
    if m.shape != (N, N):
        from scipy.ndimage import zoom
        m = zoom(m, (N / m.shape[0], N / m.shape[1]), order=1)
    mx = np.abs(m).max()
    return (m / mx) if mx > 0 else m


def mask_to_field(mask, r=-0.30, n0=-0.285, N=128, seed_k0=1.0, rng_seed=0):
    """
    Build the PFC initial density field from a drawn mask via the solver's
    custom_mask seed path. Returns (field0, cfg) — cfg carries r/n0/dx/dt/etc.
    """
    cfg = PFCConfig(N=N, r=r, n0=n0, seed_type="custom_mask",
                    custom_mask=prep_mask(mask, N), seed_k0=seed_k0,
                    rng_seed=rng_seed)
    cfg.validate()
    field0 = make_initial_condition(cfg, np.random.default_rng(rng_seed))
    return field0.astype(np.float32), cfg


# ----------------------------------------------------------------------------
#  Rollouts
# ----------------------------------------------------------------------------
@torch.no_grad()
def grow_fno(model, ck, device, field0, steps=39):
    """Autoregressive FNO rollout from field0. Returns (steps+1, N, N) physical."""
    mean, std = float(ck["norm_mean"]), float(ck["norm_std"])
    H, W = field0.shape
    cur = torch.from_numpy((field0 - mean) / (std + 1e-8)).float().view(1, 1, H, W).to(device)
    frames = [np.asarray(field0, dtype=np.float32)]
    for _ in range(steps):
        cur = model(cur)
        frames.append(cur.cpu().numpy()[0, 0] * (std + 1e-8) + mean)
    return np.stack(frames).astype(np.float32)


def grow_solver(cfg, steps=39):
    """
    Ground-truth: run the real PFC solver on the same seed for `steps` saved
    frames. Returns (<=steps+1, N, N). Slower than the FNO — opt-in verification.
    """
    save_every = cfg.save_every
    T = (steps + 1) * save_every * cfg.dt              # enough steps for steps+1 frames
    cfg2 = replace(cfg, T=T)
    res = PFCSolver(cfg2).run()
    return res.n_all.astype(np.float32)


# ----------------------------------------------------------------------------
#  Trust: is the FNO rollout physical? (differentiable-solver residual)
# ----------------------------------------------------------------------------
def pde_confidence(frames_phys, cfg, device="cpu", scale=0.5):
    """
    Mean relative PDE residual of the FNO rollout, and a 0..1 trust score
    (1 = perfectly physical). Uses the differentiable solver as the yardstick;
    no ground truth required.
    """
    stepper = DifferentiablePFCStep(cfg.dx, cfg.dt, cfg.M, cfg.save_every).to(device)
    rr = torch.tensor([cfg.r], device=device, dtype=torch.float32)
    res = []
    with torch.no_grad():
        for t in range(1, len(frames_phys)):
            H, W = frames_phys[t].shape
            prev = torch.from_numpy(frames_phys[t - 1]).float().view(1, 1, H, W).to(device)
            pred = torch.from_numpy(frames_phys[t]).float().view(1, 1, H, W).to(device)
            tgt = stepper.frame(prev, rr)
            num = float(((pred - tgt) ** 2).mean())
            den = float(((tgt - prev) ** 2).mean()) + 1e-8
            res.append(num / den)
    mean_res = float(np.mean(res)) if res else float("nan")
    trust = float(np.exp(-mean_res / scale)) if res else 0.0
    return mean_res, trust, res
