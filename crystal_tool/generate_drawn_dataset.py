"""
============================================================
  Drawn-seed PFC dataset for the custom_mask fine-tune

  For each random drawn seed (drawn_seeds.random_drawn_field) we run
  the EXACT PFC solver and save a trajectory in the same .npz format
  the FNO pipeline already consumes (save_trajectory embeds r/n0/
  seed_type, so split_by: config + conditioning work with no manifest).

  Files are traj_{1000+i}.npz so their run ids never collide with the
  main sweep (ids < 1000). seed_type is recorded as "custom_mask" so
  these runs form their own configuration group and are easy to weight.

  Run (CPU is fine — this is data gen, not training):
    python crystal_tool/generate_drawn_dataset.py --n 80 --output-dir data_pfc_drawn

  NSF IRES Physical AI Design Program
============================================================
"""

import argparse
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np
from scipy.fft import fft2

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pfc_solver import (PFCConfig, PFCSolver, PFCResult,        # noqa: E402
                        compute_free_energy, save_trajectory)
from sanity_checks import run_all_checks                        # noqa: E402

import drawn_seeds                                              # noqa: E402

# Stable r only (deeper quench = crystalline; r=-0.20 melts) — matches the
# grain-rich set so the fine-tune data lives in the same physical regime.
DRAWN_R_VALUES = [-0.25, -0.28, -0.30, -0.33, -0.40]
DRAWN_N0_VALUES = [-0.27, -0.285, -0.30]
BASE_RUN_ID = 1000          # drawn files = traj_1000.. ; sweep stays < 1000
BASE_RNG_SEED = 5000


def run_from_field(cfg: PFCConfig, field0: np.ndarray) -> PFCResult:
    """
    Mirror PFCSolver.run() but start from a supplied initial field (so we can
    inject the multi-orientation drawn seed instead of cfg's built-in IC).
    Reuses the solver's own step() and free-energy machinery.
    """
    solver = PFCSolver(cfg)                      # validates cfg, builds operators
    n = field0.astype(np.float64)
    mass_initial = float(n.mean())

    n_frames = cfg.n_frames
    n_all = np.zeros((n_frames, cfg.N, cfg.N), dtype=np.float32)
    n_all[0] = n.astype(np.float32)
    n_hat = fft2(n)
    aborted, frame_idx = False, 1

    t0 = time.perf_counter()
    steps_done = 0
    for i in range(1, cfg.n_steps):
        n, n_hat = solver.step(n, n_hat)
        steps_done += 1
        if i % cfg.save_every == 0 and frame_idx < n_frames:
            if not np.isfinite(n).all():
                aborted = True
                n_all = n_all[:frame_idx]
                break
            n_all[frame_idx] = n.astype(np.float32)
            frame_idx += 1
    runtime = time.perf_counter() - t0

    n_all = n_all[:frame_idx]
    t_vals = np.arange(frame_idx) * cfg.save_every * cfg.dt
    free_energy = compute_free_energy(n_all, solver.ops, cfg.r, cfg.dx)

    cfg_dict = asdict(cfg)
    if isinstance(cfg_dict.get("custom_mask"), np.ndarray):
        cfg_dict["custom_mask"] = f"<array {cfg.custom_mask.shape}>"

    return PFCResult(
        n_all=n_all, t_vals=t_vals, free_energy=free_energy,
        mass_initial=mass_initial,
        mass_final=float(n.mean()) if not aborted else float("nan"),
        runtime_seconds=runtime,
        steps_per_second=steps_done / runtime if runtime > 0 else float("inf"),
        aborted=aborted, config=cfg_dict,
    )


def generate(n_traj, output_dir, T=250.0, seed_k0=1.0,
             rng_base=BASE_RNG_SEED, id_base=BASE_RUN_ID, verbose=True):
    """
    Generate `n_traj` drawn-seed trajectories into output_dir. `rng_base` and
    `id_base` let callers make a DISJOINT set (e.g. a held-out drawn test set)
    that shares no seeds or filenames with the training set.
    """
    os.makedirs(output_dir, exist_ok=True)
    ok = warn = fail = 0
    for i in range(n_traj):
        rng = np.random.default_rng(rng_base + i)
        r = DRAWN_R_VALUES[i % len(DRAWN_R_VALUES)]
        n0 = DRAWN_N0_VALUES[(i // len(DRAWN_R_VALUES)) % len(DRAWN_N0_VALUES)]
        run_id = f"{id_base + i:04d}"

        base = PFCConfig(r=r, n0=n0, seed_type="custom_mask", seed_k0=seed_k0,
                         T=T, output_dir=output_dir, rng_seed=rng_base + i)
        field0, mask = drawn_seeds.random_drawn_field(base, rng)
        cfg = replace(base, custom_mask=mask)            # satisfies validate()

        result = run_from_field(cfg, field0)
        report = run_all_checks(
            result.n_all, result.free_energy,
            mass_drift_tol=cfg.mass_drift_tol,
            energy_increase_tol=cfg.energy_increase_tol,
            aborted=result.aborted, verbose=False)
        save_trajectory(result, cfg, os.path.join(output_dir, f"traj_{run_id}.npz"))

        status = report["status"]
        ok += status == "ok"; warn += status == "warn"; fail += status == "fail"
        if verbose:
            print(f"[{run_id}] r={r:+.2f} n0={n0:+.3f}  "
                  f"frames={result.n_all.shape[0]:2d}  "
                  f"Edrop={result.energy_drop_percent:5.1f}%  "
                  f"mass_drift={result.mass_relative_drift:.1e}  {status}", flush=True)

    print(f"\nGenerated {n_traj} drawn trajectories -> {output_dir}/  "
          f"({ok} ok / {warn} warn / {fail} fail)")


def main():
    ap = argparse.ArgumentParser(description="Generate drawn-seed PFC trajectories")
    ap.add_argument("--n", type=int, default=80, help="number of trajectories")
    ap.add_argument("--output-dir", default="data_pfc_drawn")
    ap.add_argument("--T", type=float, default=250.0)
    ap.add_argument("--seed-k0", type=float, default=1.0)
    ap.add_argument("--rng-base", type=int, default=BASE_RNG_SEED,
                    help="base RNG seed (use a different value for a disjoint set)")
    ap.add_argument("--id-base", type=int, default=BASE_RUN_ID,
                    help="base run id / filename index (keep sets non-colliding)")
    args = ap.parse_args()
    generate(args.n, args.output_dir, T=args.T, seed_k0=args.seed_k0,
             rng_base=args.rng_base, id_base=args.id_base)


if __name__ == "__main__":
    main()
