"""
============================================================
  PFC dataset generation — single runs and parameter sweeps

  Runs the PFC teacher solver over a parameter grid, screens
  every trajectory with sanity checks, and writes ML-ready
  .npz files + a manifest.csv that dataset.py / the FNO
  pipeline consume directly.

  NSF IRES Physical AI Design Program
============================================================

USAGE
-----
  # one simulation with the config-file (or default) parameters:
  python generate_dataset.py --single

  # one simulation, overriding parameters from the CLI:
  python generate_dataset.py --single --seed-type multi --r -0.30 --rng-seed 7

  # the full 36-run sweep (4 r-values x 3 n0-values x 3 seed types):
  python generate_dataset.py --sweep

  # the sweep in parallel:
  python generate_dataset.py --sweep --parallel --max-workers 4

  # a LARGER dataset: every config run 3x with different noise realizations
  # (run ids and rng seeds stay globally unique across ensemble members):
  python generate_dataset.py --sweep --ensemble 3 --parallel

  # resume an interrupted big sweep (skips runs whose .npz already exists):
  python generate_dataset.py --sweep --ensemble 3 --skip-existing

  # with a config file (overrides defaults; CLI overrides the file):
  python generate_dataset.py --sweep --config pfc_config_large.yaml

In Colab / notebooks, import and call directly:

    from generate_dataset import run_single, run_sweep
    from pfc_solver import PFCConfig
    row = run_single(PFCConfig(seed_type="point"), run_id="demo")
    rows = run_sweep(PFCConfig(output_dir="data_pfc"), parallel=False)

DETERMINISM. Every sweep run gets rng_seed = base_rng_seed + run_index, where
run_index counts across ALL ensemble members, so the whole dataset is
reproducible from the config alone and no two runs share a seed or filename.
Filenames are traj_{run_id:04d}.npz, matching the existing data/ convention.
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from pfc_solver import PFCConfig, PFCSolver, save_trajectory
from sanity_checks import run_all_checks

# ----------------------------------------------------------------------------
#  The sweep grid (36 runs) — same grid and ordering as the existing dataset
# ----------------------------------------------------------------------------
SWEEP_R_VALUES = [-0.25, -0.28, -0.30, -0.33]
SWEEP_N0_VALUES = [-0.27, -0.285, -0.30]
SWEEP_SEED_TYPES = ["hex", "point", "multi"]
BASE_RNG_SEED = 1000          # run i uses rng_seed = BASE_RNG_SEED + i

MANIFEST_COLUMNS = [
    "run_id", "r", "n0", "seed_type", "ensemble", "rng_seed",
    "mass_relative_drift", "energy_drop_percent",
    "runtime_seconds", "n_frames", "status", "warnings",
]


# ----------------------------------------------------------------------------
#  Single run
# ----------------------------------------------------------------------------
def run_single(cfg: PFCConfig, run_id: str, verbose: bool = True) -> dict:
    """
    Run one simulation end-to-end:
      solve -> sanity-check -> save .npz (+ optional plots) -> manifest row.
    """
    if verbose:
        print(f"[{run_id}] r={cfg.r:+.3f}  n0={cfg.n0:+.3f}  "
              f"seed={cfg.seed_type:<12s} rng_seed={cfg.rng_seed}")

    result = PFCSolver(cfg).run()

    report = run_all_checks(
        result.n_all, result.free_energy,
        mass_drift_tol=cfg.mass_drift_tol,
        energy_increase_tol=cfg.energy_increase_tol,
        aborted=result.aborted, verbose=verbose,
    )

    stem = os.path.join(cfg.output_dir, f"traj_{run_id}")
    save_trajectory(result, cfg, stem + ".npz")

    if cfg.save_plots or cfg.save_animation:
        from visualization import render_outputs   # lazy: matplotlib optional
        render_outputs(result, cfg, stem)

    return {
        "run_id": run_id,
        "r": cfg.r,
        "n0": cfg.n0,
        "seed_type": cfg.seed_type,
        "rng_seed": cfg.rng_seed,
        "mass_relative_drift": f"{result.mass_relative_drift:.3e}",
        "energy_drop_percent": f"{result.energy_drop_percent:.2f}",
        "runtime_seconds": f"{result.runtime_seconds:.2f}",
        "n_frames": result.n_all.shape[0],
        "status": report["status"],
        "warnings": " | ".join(report["warnings"]),
    }


# Top-level (picklable) worker for ProcessPoolExecutor.
def _sweep_worker(args):
    cfg, run_id, ensemble = args
    row = run_single(cfg, run_id, verbose=True)
    row["ensemble"] = ensemble
    return row


# ----------------------------------------------------------------------------
#  Manifest handling: read-merge-rewrite so reruns update rather than duplicate
# ----------------------------------------------------------------------------
def update_manifest(output_dir: str, rows: list) -> str:
    path = os.path.join(output_dir, "manifest.csv")
    existing = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                existing[row.get("run_id", "")] = row
    for row in rows:
        existing[str(row["run_id"])] = {k: str(v) for k, v in row.items()}

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rid in sorted(existing):
            # Backfill any columns an older manifest didn't have.
            writer.writerow({c: existing[rid].get(c, "") for c in MANIFEST_COLUMNS})
    return path


# ----------------------------------------------------------------------------
#  Parameter sweep
# ----------------------------------------------------------------------------
def build_sweep_configs(base_cfg: PFCConfig,
                        r_values=None, n0_values=None, seed_types=None,
                        base_rng_seed: int = BASE_RNG_SEED,
                        n_ensemble: int = 1):
    """
    Cartesian product (ensemble x r x n0 x seed_type) with deterministic,
    globally unique per-run seeds and run ids.

    `n_ensemble` repeats the whole grid with fresh noise realizations: member
    e of config c gets run_index = e * grid_size + c, so rng_seed and the
    traj_XXXX filename never collide between members. More ensemble members =
    more independent trajectories per physical configuration, which teaches
    the FNO the dynamics rather than one specific noise history.
    """
    r_values = r_values or SWEEP_R_VALUES
    n0_values = n0_values or SWEEP_N0_VALUES
    seed_types = seed_types or SWEEP_SEED_TYPES

    jobs = []
    idx = 0
    for ensemble in range(max(1, n_ensemble)):
        for r in r_values:              # outer-to-inner order matches data/manifest.csv
            for n0 in n0_values:
                for seed_type in seed_types:
                    cfg = replace(base_cfg, r=r, n0=n0, seed_type=seed_type,
                                  rng_seed=base_rng_seed + idx)
                    jobs.append((cfg, f"{idx:04d}", ensemble))
                    idx += 1
    return jobs


def run_sweep(base_cfg: PFCConfig, parallel: bool = False,
              max_workers: int = None, skip_existing: bool = False,
              **grid_kwargs) -> list:
    """Run the full sweep; returns the manifest rows for the runs executed."""
    jobs = build_sweep_configs(base_cfg, **grid_kwargs)
    os.makedirs(base_cfg.output_dir, exist_ok=True)

    if skip_existing:
        # Resume support for big sweeps: a run is done iff its .npz exists.
        # (Its manifest row from the earlier session is preserved by the
        # read-merge-rewrite in update_manifest.)
        remaining = [j for j in jobs if not os.path.exists(
            os.path.join(base_cfg.output_dir, f"traj_{j[1]}.npz"))]
        if len(remaining) < len(jobs):
            print(f"Skipping {len(jobs) - len(remaining)} already-completed runs.")
        jobs = remaining

    print(f"Sweep: {len(jobs)} runs -> {base_cfg.output_dir}/  "
          f"(parallel={parallel}"
          + (f", max_workers={max_workers}" if parallel else "") + ")\n")
    if not jobs:
        print("Nothing to do.")
        return []

    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            rows = list(pool.map(_sweep_worker, jobs))
    else:
        rows = [_sweep_worker(job) for job in jobs]

    manifest = update_manifest(base_cfg.output_dir, rows)

    n_ok = sum(r["status"] == "ok" for r in rows)
    n_warn = sum(r["status"] == "warn" for r in rows)
    n_fail = sum(r["status"] == "fail" for r in rows)
    total_t = sum(float(r["runtime_seconds"]) for r in rows)
    print(f"\nSweep complete: {n_ok} ok / {n_warn} warn / {n_fail} fail "
          f"| total solver time {total_t:.1f}s")
    print(f"Manifest: {manifest}")
    return rows


# ----------------------------------------------------------------------------
#  Config-file + CLI plumbing
# ----------------------------------------------------------------------------
def load_config_file(path: str) -> dict:
    """Load YAML config (returns {} if no file given). yaml import is lazy so
    the script also works in bare environments without pyyaml."""
    if not path:
        return {}
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser(description="PFC dataset generation")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--single", action="store_true", help="run one simulation")
    mode.add_argument("--sweep", action="store_true", help="run the parameter sweep")
    ap.add_argument("--config", default=None, help="YAML config file (optional)")
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--ensemble", type=int, default=None,
                    help="repeat the sweep grid this many times with "
                         "different noise realizations (default 1)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip runs whose .npz already exists (resume a sweep)")
    # Common overrides (CLI > config file > dataclass defaults):
    ap.add_argument("--r", type=float)
    ap.add_argument("--n0", type=float)
    ap.add_argument("--seed-type", choices=["hex", "point", "multi",
                                            "random_noise", "custom_mask"])
    ap.add_argument("--rng-seed", type=int)
    ap.add_argument("--T", type=float)
    ap.add_argument("--N", type=int)
    ap.add_argument("--output-dir")
    ap.add_argument("--save-plots", action="store_true")
    ap.add_argument("--save-animation", action="store_true")
    ap.add_argument("--custom-mask", help="path to .npy mask for seed_type=custom_mask")
    args = ap.parse_args()

    file_cfg = load_config_file(args.config)
    run_section = dict(file_cfg.get("run", {}))
    sweep_section = dict(file_cfg.get("sweep", {}))

    # Apply CLI overrides onto the run section.
    cli_overrides = {
        "r": args.r, "n0": args.n0, "seed_type": args.seed_type,
        "rng_seed": args.rng_seed, "T": args.T, "N": args.N,
        "output_dir": args.output_dir, "custom_mask": args.custom_mask,
    }
    run_section.update({k: v for k, v in cli_overrides.items() if v is not None})
    if args.save_plots:
        run_section["save_plots"] = True
    if args.save_animation:
        run_section["save_animation"] = True

    base_cfg = PFCConfig(**run_section)

    if args.single:
        os.makedirs(base_cfg.output_dir, exist_ok=True)
        # Pick the next free traj_XXXX index so single runs never collide.
        existing = [f for f in os.listdir(base_cfg.output_dir)
                    if f.startswith("traj_") and f.endswith(".npz")]
        nums = [int(f[5:9]) for f in existing if f[5:9].isdigit()]
        run_id = f"{(max(nums) + 1 if nums else 0):04d}"
        row = run_single(base_cfg, run_id)
        update_manifest(base_cfg.output_dir, [row])
        print(f"\nSaved {base_cfg.output_dir}/traj_{run_id}.npz  "
              f"(status: {row['status']})")
    else:
        parallel = args.parallel or bool(sweep_section.get("parallel", False))
        max_workers = args.max_workers or sweep_section.get("max_workers")
        n_ensemble = args.ensemble or sweep_section.get("n_ensemble", 1)
        skip_existing = args.skip_existing or bool(
            sweep_section.get("skip_existing", False))
        run_sweep(
            base_cfg, parallel=parallel, max_workers=max_workers,
            skip_existing=skip_existing,
            r_values=sweep_section.get("r_values"),
            n0_values=sweep_section.get("n0_values"),
            seed_types=sweep_section.get("seed_types"),
            base_rng_seed=sweep_section.get("base_rng_seed", BASE_RNG_SEED),
            n_ensemble=n_ensemble,
        )


if __name__ == "__main__":
    main()
