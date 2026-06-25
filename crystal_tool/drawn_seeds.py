"""
============================================================
  Drawn-style seed synthesis for the custom_mask fine-tune

  The FNO was trained on hex/point/multi seeds (centered Gaussian
  envelopes). A user draws arbitrary blobs and strokes — out of
  distribution. This module synthesizes random "drawn" initial
  fields so we can generate matching PFC trajectories and fine-tune.

  Two things make these useful training seeds:
    * arbitrary SHAPE — random blobs + thick strokes, 1..n regions,
      anywhere on the grid (the shape gap), and
    * per-region random ORIENTATION — different drawn regions nucleate
      at different lattice angles, so multi-region drawings grow into
      POLYCRYSTALS with grain boundaries (directly exercises the
      orientation weakness, not just the shape gap).

  Pure numpy. Reuses pfc_solver._hex_pattern so the planted lattice is
  identical to the rest of the pipeline.

  NSF IRES Physical AI Design Program
============================================================
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from pfc_solver import _hex_pattern              # noqa: E402


def _blob_envelope(N, rng):
    """A round Gaussian patch (periodic-aware). Returns (env, cx, cy)."""
    cx, cy = rng.integers(0, N, size=2)
    sigma = rng.integers(N // 14, N // 6)
    SX, SY = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    ddx = np.minimum(np.abs(SX - cx), N - np.abs(SX - cx))
    ddy = np.minimum(np.abs(SY - cy), N - np.abs(SY - cy))
    env = np.exp(-(ddx ** 2 + ddy ** 2) / (2.0 * sigma ** 2))
    return env, int(cx), int(cy)


def _stroke_envelope(N, rng):
    """A thick poly-line / curve (what a pen draws). Returns (env, cx, cy)."""
    n_pts = rng.integers(2, 5)
    pts = rng.integers(N // 6, 5 * N // 6, size=(n_pts, 2)).astype(float)
    width = rng.uniform(N / 22, N / 12)
    SX, SY = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    grid = np.stack([SX, SY], axis=-1).astype(float)       # (N,N,2)
    dist = np.full((N, N), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        L2 = float(ab @ ab) + 1e-9
        t = np.clip(((grid - a) @ ab) / L2, 0.0, 1.0)       # (N,N)
        proj = a + t[..., None] * ab                        # (N,N,2)
        d = np.linalg.norm(grid - proj, axis=-1)
        dist = np.minimum(dist, d)
    env = np.exp(-(dist ** 2) / (2.0 * width ** 2))
    cx, cy = pts.mean(axis=0)
    return env, int(cx), int(cy)


def random_drawn_field(cfg, rng, max_regions=4):
    """
    Build a drawn-style PFC initial field for config `cfg`.

    Returns (field0, mask) where field0 is the (N, N) physical density and
    mask is the union of region envelopes in [0, 1] (the "drawing").
    """
    N, L = cfg.N, cfg.L
    x = np.linspace(0.0, L, N, endpoint=False)
    PX, PY = np.meshgrid(x, x, indexing="ij")
    k0 = cfg.seed_k0 if cfg.seed_k0 is not None else 2.0 * np.pi / (4.0 * cfg.dx)

    n = cfg.n0 + cfg.noise_amplitude * rng.standard_normal((N, N))
    mask = np.zeros((N, N))

    n_regions = int(rng.integers(1, max_regions + 1))
    for _ in range(n_regions):
        env, cx, cy = (_blob_envelope(N, rng) if rng.random() < 0.5
                       else _stroke_envelope(N, rng))
        theta = rng.uniform(0.0, np.pi / 3.0)              # hex 60-deg symmetry
        n += cfg.seed_amplitude * env * _hex_pattern(
            PX - x[cx], PY - x[cy], k0, theta)
        mask = np.maximum(mask, env)

    return n.astype(np.float64), mask.astype(np.float32)


if __name__ == "__main__":
    # quick visual-free sanity: fields vary, masks are in range, regions present
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pfc_solver import PFCConfig
    rng = np.random.default_rng(0)
    cfg = PFCConfig(seed_k0=1.0, r=-0.30, n0=-0.285)
    for i in range(3):
        f, m = random_drawn_field(cfg, rng)
        print(f"seed {i}: field range [{f.min():+.3f}, {f.max():+.3f}]  "
              f"mask coverage {float((m > 0.1).mean())*100:5.1f}%  "
              f"mean {f.mean():+.4f}")
    print("drawn_seeds OK")
