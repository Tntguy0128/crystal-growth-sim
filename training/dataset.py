"""
============================================================
  PFC trajectory dataset for FNO training

  Loads .npz trajectories, auto-discovers their layout, and
  produces (frame_t -> frame_t+1) training pairs.

  NSF IRES Physical AI Design Program
============================================================

WHAT THIS FILE DOES
-------------------
The FNO learns the one-step map  n(t) -> n(t + Dt).  To train it we need pairs

        input  = frame_t
        target = frame_t+1

drawn from every saved trajectory.

The dataset format is AUTO-INSPECTED rather than hard-coded, so this loader
works on the shipped PFC dataset (keys: frames / fno_inputs / fno_targets /
params / seed_type / ...) and also on the bare `n_all` array produced by
PCF_Baseline.py. The discovery logic:

  * If a trajectory npz already contains explicit pair arrays (e.g.
    `fno_inputs` and `fno_targets`), use them directly.
  * Otherwise find the 3D field array (shape (T, H, W)) -- by trying common key
    names first, then by shape -- and derive pairs as frames[:-1], frames[1:].
  * Scalar / vector metadata (r, n0, seed_type, ...) is collected from the npz
    and, when present, from a sibling manifest.csv keyed by run id.

SPLITS ARE BY TRAJECTORY, NOT BY FRAME. All frame pairs from one trajectory go
entirely into train OR val OR test. This prevents leakage (adjacent frames from
the same run are highly correlated; mixing them across splits inflates scores).

NORMALIZATION. We z-score the density field using mean/std computed on the
TRAIN split only, then store those stats so evaluation/rollout can invert them.
Targets are normalized with the SAME stats as inputs (same physical quantity).
"""

import csv
import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset


# Candidate key names, tried in order during auto-inspection.
_TRAJ_KEYS = ["frames", "n_all", "trajectory", "traj", "density", "n", "fields"]
_INPUT_KEYS = ["fno_inputs", "inputs", "x", "input"]
_TARGET_KEYS = ["fno_targets", "targets", "y", "target"]
# Manifest columns we recognize as numeric conditioning variables.
_COND_COLUMNS = ["r", "n0"]


# ----------------------------------------------------------------------------
#  Low-level npz inspection helpers
# ----------------------------------------------------------------------------
def _decode_scalar(v):
    """Turn a 0-d numpy array / bytes into a plain Python scalar or str."""
    if isinstance(v, np.ndarray) and v.shape == ():
        v = v.item()
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


def _find_3d_array(npz, prefer_keys):
    """Return (key, array) for the first 3D float array, preferring known names."""
    for k in prefer_keys:
        if k in npz and npz[k].ndim == 3:
            return k, npz[k]
    # Fall back: any 3D array, choosing the one with the most frames.
    best = None
    for k in npz.files:
        a = npz[k]
        if isinstance(a, np.ndarray) and a.ndim == 3:
            if best is None or a.shape[0] > best[1].shape[0]:
                best = (k, a)
    if best is None:
        raise ValueError(
            f"No 3D (T, H, W) field array found. Keys present: {list(npz.files)}"
        )
    return best


def inspect_trajectory(path):
    """
    Open one .npz and return a normalized record:

        {
          'inputs' : (P, H, W) float32,   # frame_t
          'targets': (P, H, W) float32,   # frame_t+1
          'meta'   : { 'r':..., 'n0':..., 'seed_type':..., ... },
          'path'   : str,
        }

    where P is the number of pairs (= T - 1 when derived from a full trajectory).
    """
    npz = np.load(path, allow_pickle=True)
    keys = set(npz.files)

    # 1) Prefer explicit pre-built pairs if both are present.
    in_key = next((k for k in _INPUT_KEYS if k in keys), None)
    tg_key = next((k for k in _TARGET_KEYS if k in keys), None)
    if in_key is not None and tg_key is not None:
        inputs = np.asarray(npz[in_key], dtype=np.float32)
        targets = np.asarray(npz[tg_key], dtype=np.float32)
    else:
        # 2) Derive pairs from the full trajectory array.
        _, frames = _find_3d_array(npz, _TRAJ_KEYS)
        frames = np.asarray(frames, dtype=np.float32)
        inputs = frames[:-1]
        targets = frames[1:]

    # 3) Collect scalar / small-vector metadata.
    meta = {}
    for k in keys:
        a = npz[k]
        if isinstance(a, np.ndarray) and a.ndim == 0:
            meta[k] = _decode_scalar(a)
    # A `params` vector (r, n0, noise, dt_save) -> name the first two if absent.
    if "params" in keys and npz["params"].ndim == 1:
        p = np.asarray(npz["params"], dtype=np.float32)
        meta.setdefault("params", p)
        if len(p) >= 1:
            meta.setdefault("r", float(p[0]))
        if len(p) >= 2:
            meta.setdefault("n0", float(p[1]))

    return {"inputs": inputs, "targets": targets, "meta": meta, "path": path}


def _load_manifest(data_dir):
    """Read manifest.csv (if present) into {run_id_str: {col: value}}."""
    path = os.path.join(data_dir, "manifest.csv")
    if not os.path.exists(path):
        return {}
    table = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rid = row.get("run_id") or row.get("id") or row.get("run")
            if rid is not None:
                table[str(rid)] = row
    return table


def _run_id_from_path(path):
    """Extract the numeric run id from a filename like traj_0007.npz -> '0007'."""
    m = re.search(r"(\d+)", os.path.basename(path))
    return m.group(1) if m else None


# ----------------------------------------------------------------------------
#  Torch Dataset
# ----------------------------------------------------------------------------
class PFCPairDataset(Dataset):
    """
    A flat dataset of (input_frame, target_frame) pairs gathered from a list of
    trajectory files.

    Each item is a tuple (x, y):
        x : (C, H, W)   input  -- density (+ optional conditioning channels)
        y : (1, H, W)   target -- next density field (always 1 channel)

    Args:
        files               : list of .npz paths belonging to this split.
        manifest            : {run_id: {col: val}} from _load_manifest (may be {}).
        norm_mean, norm_std : z-score stats (floats). If None, no normalization.
        include_conditioning: if True, append r and n0 as constant-valued
                              channels broadcast over the grid. OFF by default
                              (clean MSE baseline first).
        cond_stats          : optional {'mean': [...], 'std': [...]} to normalize
                              the conditioning channels. If None they are passed raw.
    """

    def __init__(self, files, manifest=None, norm_mean=None, norm_std=None,
                 include_conditioning=False, cond_stats=None):
        self.files = files
        self.manifest = manifest or {}
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.include_conditioning = include_conditioning
        self.cond_stats = cond_stats

        # Build a flat index of (file_idx, frame_idx) and cache loaded records.
        self._records = []          # one inspect_trajectory() dict per file
        self.index = []             # list of (file_idx, pair_idx)
        self.cond = []              # per-file conditioning vector [r, n0]
        for fi, path in enumerate(files):
            rec = inspect_trajectory(path)
            self._records.append(rec)
            n_pairs = rec["inputs"].shape[0]
            for pi in range(n_pairs):
                self.index.append((fi, pi))
            self.cond.append(self._conditioning_vector(rec, path))

    def _conditioning_vector(self, rec, path):
        """Resolve [r, n0] for a trajectory from manifest first, then npz meta."""
        vals = []
        run = _run_id_from_path(path)
        mrow = self.manifest.get(run, {})
        for col in _COND_COLUMNS:
            if col in mrow:
                vals.append(float(mrow[col]))
            elif col in rec["meta"]:
                vals.append(float(rec["meta"][col]))
            else:
                vals.append(0.0)
        return np.asarray(vals, dtype=np.float32)

    def __len__(self):
        return len(self.index)

    def _normalize(self, a):
        if self.norm_mean is None or self.norm_std is None:
            return a
        return (a - self.norm_mean) / (self.norm_std + 1e-8)

    def __getitem__(self, idx):
        fi, pi = self.index[idx]
        rec = self._records[fi]

        x = self._normalize(rec["inputs"][pi])              # (H, W)
        y = self._normalize(rec["targets"][pi])             # (H, W)

        x = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)  # (1, H, W)
        y = torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0)  # (1, H, W)

        if self.include_conditioning:
            H, W = x.shape[-2:]
            cond = self.cond[fi].copy()
            if self.cond_stats is not None:
                cond = (cond - self.cond_stats["mean"]) / (self.cond_stats["std"] + 1e-8)
            # Broadcast each scalar to a full (1, H, W) channel and stack.
            cond_channels = [
                torch.full((1, H, W), float(c), dtype=x.dtype) for c in cond
            ]
            x = torch.cat([x] + cond_channels, dim=0)       # (1 + n_cond, H, W)

        return x, y


# ----------------------------------------------------------------------------
#  Split + stats orchestration
# ----------------------------------------------------------------------------
def list_trajectory_files(data_dir):
    """All .npz trajectory files in a directory, sorted for reproducibility."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz trajectories found in {data_dir!r}")
    return files


def split_files(files, val_frac=0.15, test_frac=0.15, seed=0):
    """
    Shuffle and split the file list BY TRAJECTORY into (train, val, test).
    Deterministic given `seed`.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    files = [files[i] for i in perm]

    n = len(files)
    n_test = max(1, int(round(test_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"Not enough trajectories ({n}) for the requested split fractions."
        )
    return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]


def compute_norm_stats(files):
    """
    Mean / std of the density field over a set of trajectories (train only).
    Computed in a streaming fashion so we never hold all data in memory.
    """
    total = 0.0
    total_sq = 0.0
    count = 0
    for path in files:
        rec = inspect_trajectory(path)
        a = rec["inputs"].astype(np.float64)
        total += a.sum()
        total_sq += (a ** 2).sum()
        count += a.size
    mean = total / count
    var = max(total_sq / count - mean ** 2, 0.0)
    std = float(np.sqrt(var))
    return float(mean), std


def compute_cond_stats(files, manifest):
    """Mean/std of the conditioning variables (r, n0) across trajectories."""
    rows = []
    for path in files:
        run = _run_id_from_path(path)
        mrow = manifest.get(run, {})
        rec = inspect_trajectory(path)
        vals = []
        for col in _COND_COLUMNS:
            if col in mrow:
                vals.append(float(mrow[col]))
            elif col in rec["meta"]:
                vals.append(float(rec["meta"][col]))
            else:
                vals.append(0.0)
        rows.append(vals)
    rows = np.asarray(rows, dtype=np.float64)
    return {
        "mean": rows.mean(axis=0).astype(np.float32),
        "std": rows.std(axis=0).astype(np.float32),
    }


def build_datasets(cfg):
    """
    Top-level helper used by train_fno.py / evaluate_fno.py.

    Returns (train_ds, val_ds, test_ds, info) where `info` carries the
    normalization stats, conditioning settings, and the file splits so the
    exact same configuration can be reconstructed at evaluation time.
    """
    d = cfg["data"]
    data_dir = d["data_dir"]
    include_conditioning = d.get("include_conditioning", False)

    files = list_trajectory_files(data_dir)
    manifest = _load_manifest(data_dir)
    train_files, val_files, test_files = split_files(
        files,
        val_frac=d.get("val_frac", 0.15),
        test_frac=d.get("test_frac", 0.15),
        seed=d.get("split_seed", 0),
    )

    # Normalization stats from TRAIN ONLY (or disabled).
    if d.get("normalize", True):
        norm_mean, norm_std = compute_norm_stats(train_files)
    else:
        norm_mean, norm_std = 0.0, 1.0

    cond_stats = None
    if include_conditioning and d.get("normalize_conditioning", True):
        cond_stats = compute_cond_stats(train_files, manifest)

    def make(fs):
        return PFCPairDataset(
            fs, manifest=manifest,
            norm_mean=norm_mean, norm_std=norm_std,
            include_conditioning=include_conditioning,
            cond_stats=cond_stats,
        )

    info = {
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "include_conditioning": include_conditioning,
        "cond_stats": cond_stats,
        "in_channels": 1 + (len(_COND_COLUMNS) if include_conditioning else 0),
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
    }
    return make(train_files), make(val_files), make(test_files), info


# ----------------------------------------------------------------------------
#  Self-test: run `python dataset.py [data_dir]` to inspect the format.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    ddir = sys.argv[1] if len(sys.argv) > 1 else "data"
    print(f"Inspecting dataset in: {ddir}\n")

    files = list_trajectory_files(ddir)
    print(f"Found {len(files)} trajectory files.")
    manifest = _load_manifest(ddir)
    print(f"Manifest rows: {len(manifest)}\n")

    rec = inspect_trajectory(files[0])
    print(f"First trajectory: {os.path.basename(rec['path'])}")
    print(f"  inputs  shape: {rec['inputs'].shape}  dtype: {rec['inputs'].dtype}")
    print(f"  targets shape: {rec['targets'].shape}")
    print(f"  metadata keys: {list(rec['meta'].keys())}")
    print(f"  r = {rec['meta'].get('r')}   n0 = {rec['meta'].get('n0')}   "
          f"seed_type = {rec['meta'].get('seed_type')}\n")

    tr, va, te = split_files(files)
    print(f"Split: train={len(tr)}  val={len(va)}  test={len(te)} trajectories")
    mean, std = compute_norm_stats(tr)
    print(f"Train density stats: mean={mean:.5f}  std={std:.5f}")

    total_pairs = sum(inspect_trajectory(f)["inputs"].shape[0] for f in files)
    print(f"Total (input, target) pairs across all trajectories: {total_pairs}")
