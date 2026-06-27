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
from pfc_solver import (PFCConfig, PFCSolver,            # noqa: E402
                        make_initial_condition, _hex_pattern)
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


def preset_field(seed_type, r=-0.30, n0=-0.285, n_seeds=14, N=128,
                 seed_k0=1.0, rng_seed=7):
    """
    A built-in seed (e.g. 'multi' -> polycrystal, 'point' -> single crystal)
    as a reliable live-demo fallback when freehand drawing is awkward. Returns
    (field0, cfg) like mask_to_field.
    """
    cfg = PFCConfig(N=N, r=r, n0=n0, seed_type=seed_type, n_seeds=n_seeds,
                    seed_k0=seed_k0, rng_seed=rng_seed)
    cfg.validate()
    field0 = make_initial_condition(cfg, np.random.default_rng(rng_seed))
    return field0.astype(np.float32), cfg


def mask_to_field(mask, r=-0.30, n0=-0.285, N=128, seed_k0=1.0, rng_seed=None,
                  orientation="random", per_region=False):
    """
    Build the PFC initial density field from a drawn mask.

    The lattice ORIENTATION is what makes results diverse: the built-in
    custom_mask path always plants 0deg, so every drawing grows the same crystal.
    Here:
      orientation : "random" (a fresh angle each call -> different every grow),
                    or a number in degrees (0..60) for a fixed angle.
      per_region  : if True, each connected blob/stroke of the drawing gets its
                    OWN random orientation -> a polycrystalline seed with grain
                    boundaries where regions meet.
      rng_seed    : None -> nondeterministic (varies each call); int -> repeatable.

    Returns (field0, cfg).
    """
    m = prep_mask(mask, N)
    cfg = PFCConfig(N=N, r=r, n0=n0, seed_type="custom_mask", custom_mask=m,
                    seed_k0=seed_k0, rng_seed=(rng_seed or 0))
    cfg.validate()
    rng = np.random.default_rng(rng_seed)
    x = np.linspace(0.0, cfg.L, N, endpoint=False)
    PX, PY = np.meshgrid(x, x, indexing="ij")
    field = cfg.n0 + cfg.noise_amplitude * rng.standard_normal((N, N))

    if per_region:
        from scipy.ndimage import label
        lab, nlab = label(m > 0.15)
        for i in range(1, nlab + 1):
            comp = (lab == i).astype(float) * m
            th = rng.uniform(0.0, np.pi / 3.0)
            field += cfg.seed_amplitude * comp * _hex_pattern(PX, PY, seed_k0, th)
        if nlab == 0:                                    # nothing drawn -> single patch
            th = rng.uniform(0.0, np.pi / 3.0)
            field += cfg.seed_amplitude * m * _hex_pattern(PX, PY, seed_k0, th)
    else:
        th = (rng.uniform(0.0, np.pi / 3.0) if orientation == "random"
              else np.deg2rad(float(orientation)))
        field += cfg.seed_amplitude * m * _hex_pattern(PX, PY, seed_k0, th)

    return field.astype(np.float32), cfg


# ----------------------------------------------------------------------------
#  Rollouts
# ----------------------------------------------------------------------------
def _cond_maps(ck, cfg, H, W, device):
    """Normalized (r, n0) conditioning channels for a PARAMETRIC checkpoint."""
    if not ck.get("include_conditioning") or cfg is None:
        return None
    vals = np.array([cfg.r, cfg.n0], dtype=np.float32)
    cs = ck.get("cond_stats")
    if cs is not None:
        vals = (vals - np.asarray(cs["mean"])) / (np.asarray(cs["std"]) + 1e-8)
    t = torch.from_numpy(vals).float().view(1, -1, 1, 1)
    return t.expand(1, vals.shape[0], H, W).to(device)


@torch.no_grad()
def grow_fno(model, ck, device, field0, steps=39, cfg=None):
    """
    Autoregressive FNO rollout from field0. Returns (steps+1, N, N) physical.
    For a PARAMETRIC checkpoint (include_conditioning), pass `cfg` so r and n0
    are appended as conditioning channels — moving them changes the phase.
    """
    mean, std = float(ck["norm_mean"]), float(ck["norm_std"])
    H, W = field0.shape
    cur = torch.from_numpy((field0 - mean) / (std + 1e-8)).float().view(1, 1, H, W).to(device)
    cond = _cond_maps(ck, cfg, H, W, device)
    frames = [np.asarray(field0, dtype=np.float32)]
    for _ in range(steps):
        x = cur if cond is None else torch.cat([cur, cond], dim=1)
        cur = model(x)
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
def pde_confidence(frames_phys, cfg, device="cpu", scale=3.0):
    """
    Mean relative PDE residual of the FNO rollout, and a 0..1 trust score
    (1 = perfectly physical). Uses the differentiable solver as the yardstick;
    no ground truth required.

    `scale` calibrates residual -> trust via exp(-residual/scale). A well-behaved
    rollout sits around residual ~0.6-1.0 (the relative residual converges near
    1 as the crystal approaches equilibrium), so scale=3.0 maps a normal
    prediction to ~70-85% and only flags markedly off-manifold rollouts.
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
