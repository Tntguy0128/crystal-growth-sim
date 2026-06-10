"""
============================================================
  Kobayashi Dataset Generation — Parameter Sweep
  Generates training data for the FNO surrogate model

  Ayush Shah & Tobias Li
  Georgia Institute of Technology
  NSF IRES Physical AI Design Program — Prof. Bo Zhu

  Usage (run from repo root):
      python dataset/generate_kobayashi.py \
          --out_dir data/kobayashi \
          --steps 4000 \
          --save_every 50

  This produces ~72 trajectory .npz files covering the
  parameter space of the Kobayashi snowflake model.
  Each file is ~8MB uncompressed, ~2MB compressed.
  Total dataset: ~150MB on disk, ~2-3 hours on CPU,
  ~20-30 min on Colab GPU (numpy still CPU-bound but
  Colab has faster CPUs).

  Upload to Google Drive after generation — the paths
  in config_kobayashi.yaml point there.
============================================================
"""

import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np

# Allow running from repo root or from dataset/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from solvers.kobayashi import run_kobayashi

# ── Parameter grid ────────────────────────────────────────────────────────────
#
# We sweep the four most visually important parameters.
# Total: 3 × 3 × 2 × 4 = 72 trajectories
#
# Te (undercooling): controls growth speed and sparsity
#   Low  (0.40) = slow, very sparse open arms
#   Mid  (0.46) = balanced — best visual result
#   High (0.54) = fast, arms fill in quickly
#
# delta (anisotropy): controls arm sharpness
#   Low  (0.04) = rounded, organic-looking arms
#   Mid  (0.06) = balanced
#   High (0.09) = very sharp needle-like dendrites
#
# noise_amp: controls natural disorder
#   Low  (0.08) = nearly perfect symmetry
#   High (0.18) = clearly asymmetric, natural-looking
#
# seed_type: controls initial crystal geometry
#   hex6  = 6-arm star (standard snowflake)
#   hex4  = 4-arm cross (square crystal)
#   point = single point (spontaneous nucleation)
#   multi = multiple seeds (polycrystalline)

SWEEP = dict(
    Te         = [0.40, 0.46, 0.54],
    delta      = [0.04, 0.06, 0.09],
    noise_amp  = [0.08, 0.18],
    seed_type  = ['hex6', 'hex4', 'point', 'multi'],
)

# Fixed parameters — not swept
FIXED = dict(
    j          = 6,       # 6-fold symmetry for all snowflake runs
    pulse      = 0.30,
    nuc_rate   = 0.10,
    N          = 256,
)


def build_configs(sweep: dict, fixed: dict) -> list[dict]:
    """Expand the sweep grid into a flat list of config dicts."""
    keys   = list(sweep.keys())
    values = list(sweep.values())
    configs = []
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        cfg.update(fixed)
        configs.append(cfg)
    return configs


def already_done(out_dir: str, run_id: str) -> bool:
    return os.path.exists(os.path.join(out_dir, f'traj_{run_id}.npz'))


def log_manifest(out_dir: str, run_id: str, cfg: dict, phi_c: float,
                 n_frames: int, elapsed: float):
    path       = os.path.join(out_dir, 'manifest.csv')
    write_hdr  = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_hdr:
            w.writerow(['run_id', 'Te', 'delta', 'noise_amp', 'seed_type',
                        'j', 'pulse', 'nuc_rate', 'phi_c', 'n_frames',
                        'elapsed_s'])
        w.writerow([run_id, cfg['Te'], cfg['delta'], cfg['noise_amp'],
                    cfg['seed_type'], cfg['j'], cfg['pulse'], cfg['nuc_rate'],
                    f'{phi_c:.4f}', n_frames, f'{elapsed:.1f}'])


def main():
    ap = argparse.ArgumentParser(
        description='Generate Kobayashi training dataset.')
    ap.add_argument('--out_dir',    default='data/kobayashi',
                    help='Output directory for .npz files')
    ap.add_argument('--steps',      type=int, default=4000,
                    help='Simulation steps per run (4000 = ~60 frames at save_every=50)')
    ap.add_argument('--save_every', type=int, default=50,
                    help='Save one frame every N steps')
    ap.add_argument('--start_idx',  type=int, default=0,
                    help='Resume from this run index (skip earlier runs)')
    ap.add_argument('--dry_run',    action='store_true',
                    help='Print configs without running anything')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    configs    = build_configs(SWEEP, FIXED)
    total      = len(configs)
    print(f"\nKobayashi Dataset Generation")
    print(f"  Total runs:  {total}")
    print(f"  Steps/run:   {args.steps}")
    print(f"  Frames/run:  {args.steps // args.save_every}")
    print(f"  Output dir:  {args.out_dir}")
    print(f"  Start index: {args.start_idx}\n")

    if args.dry_run:
        for i, cfg in enumerate(configs):
            print(f"  {i:03d}  Te={cfg['Te']}  delta={cfg['delta']}  "
                  f"noise={cfg['noise_amp']}  seed={cfg['seed_type']}")
        return

    sweep_t0   = time.time()
    runs_done  = 0

    for i, cfg in enumerate(configs[args.start_idx:], start=args.start_idx):
        run_id = f'{i:04d}'

        if already_done(args.out_dir, run_id):
            print(f"  [skip] {run_id} already exists")
            continue

        print(f"\n{'='*56}")
        print(f"  RUN {run_id}/{total-1}  "
              f"Te={cfg['Te']}  delta={cfg['delta']}  "
              f"noise={cfg['noise_amp']}  seed={cfg['seed_type']}")
        print(f"{'='*56}")

        t0     = time.time()
        result = run_kobayashi(
            Te         = cfg['Te'],
            delta      = cfg['delta'],
            j          = cfg['j'],
            noise_amp  = cfg['noise_amp'],
            pulse      = cfg['pulse'],
            nuc_rate   = cfg['nuc_rate'],
            seed_type  = cfg['seed_type'],
            N          = cfg['N'],
            steps      = args.steps,
            save_every = args.save_every,
            seed       = i,          # different RNG seed per run
            verbose    = True,
        )
        elapsed = time.time() - t0

        # Save trajectory
        out_path = os.path.join(args.out_dir, f'traj_{run_id}.npz')
        np.savez_compressed(
            out_path,
            frames      = result['frames'],
            T_field     = result['T_field'],
            fno_inputs  = result['fno_inputs'],
            fno_targets = result['fno_targets'],
            phi_c       = result['phi_c'],
            t_values    = result['t_values'],
            params      = np.array([
                cfg['Te'], cfg['delta'],
                float(cfg['j']), cfg['noise_amp'],
                cfg['pulse'],
            ], dtype=np.float32),
            seed_type   = np.bytes_(cfg['seed_type']),
        )

        n_frames = result['frames'].shape[0]
        log_manifest(args.out_dir, run_id, cfg,
                     float(result['phi_c']), n_frames, elapsed)

        runs_done += 1
        elapsed_total = (time.time() - sweep_t0) / 60
        runs_left     = total - args.start_idx - runs_done
        eta           = (elapsed_total / runs_done) * runs_left if runs_done else 0

        print(f"\n  Saved → {out_path}")
        print(f"  frames={n_frames}  phi_c={result['phi_c']:.3f}  "
              f"time={elapsed:.1f}s")
        print(f"  Progress: {runs_done}/{total - args.start_idx}  |  "
              f"Elapsed: {elapsed_total:.1f}min  |  ETA: {eta:.1f}min")

    print(f"\n{'='*56}")
    print(f"Sweep complete in {(time.time()-sweep_t0)/60:.1f} min")
    print(f"Files saved to: {args.out_dir}")
    files = [f for f in os.listdir(args.out_dir) if f.endswith('.npz')]
    print(f"Total .npz files: {len(files)}")


if __name__ == '__main__':
    main()
