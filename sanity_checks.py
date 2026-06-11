"""
============================================================
  Sanity checks for PFC trajectories

  The whole point of the dataset is to be a TRUSTWORTHY
  teacher for ML surrogates. Every run is screened here
  before it lands in the training set.

  NSF IRES Physical AI Design Program
============================================================

Checks, and the physics behind them:

  1. FINITE      — no NaN / Inf anywhere. A blow-up means the run is garbage.
  2. MASS        — PFC dynamics are conserved (dn/dt = lap(...)), so the
                   spatial mean of n must stay constant to ~ float precision.
                   Drift beyond tolerance signals a numerical bug.
  3. ENERGY      — PFC is a gradient flow: F[n] must (essentially) decrease
                   monotonically. Small uphill blips are tolerated because the
                   nonlinear term is integrated explicitly; large increases
                   mean dt is too big or something is broken.
  4. FIELD RANGE — track min/max of n. The one-mode hexagonal phase keeps
                   |n| order ~1; values far outside indicate instability.

Each check returns a list of human-readable warnings; `run_all_checks`
aggregates them into {"status": ok|warn|fail, "warnings": [...], "stats": {...}}.
"""

import numpy as np


# ----------------------------------------------------------------------------
#  Individual checks
# ----------------------------------------------------------------------------
def check_finite(n_all: np.ndarray):
    """FAIL-level check: any NaN or Inf in the saved frames."""
    warnings = []
    if not np.isfinite(n_all).all():
        bad_frames = [int(i) for i in range(n_all.shape[0])
                      if not np.isfinite(n_all[i]).all()]
        warnings.append(
            f"FAIL: non-finite values (NaN/Inf) in frames {bad_frames[:5]}"
            + ("..." if len(bad_frames) > 5 else "")
        )
    return warnings


def check_mass(n_all: np.ndarray, tol: float = 1e-5):
    """
    Relative drift of mean(n) across ALL frames vs frame 0.
    (Frames are stored float32, so ~1e-7 relative noise is expected;
    the default tolerance leaves comfortable headroom above that.)
    """
    warnings = []
    means = n_all.reshape(n_all.shape[0], -1).mean(axis=1, dtype=np.float64)
    drift = np.abs(means / means[0] - 1.0)
    worst = float(drift.max())
    if worst > tol:
        warnings.append(
            f"WARN: mass drift {worst:.3e} exceeds tolerance {tol:.1e} "
            f"(worst at frame {int(drift.argmax())})"
        )
    return warnings, worst


def check_energy(free_energy: np.ndarray, rel_tol: float = 1e-3):
    """
    Free energy should decrease. Flag any step where F increases by more than
    rel_tol * |F| (relative to the local energy scale).
    """
    warnings = []
    F = np.asarray(free_energy, dtype=np.float64)
    if len(F) < 2:
        return warnings, 0.0
    dF = np.diff(F)
    scale = np.maximum(np.abs(F[:-1]), 1e-30)
    rel_increase = dF / scale                       # positive = energy went UP
    worst = float(rel_increase.max())
    if worst > rel_tol:
        idx = int(rel_increase.argmax())
        warnings.append(
            f"WARN: free energy increased by {worst:.3e} (relative) "
            f"between frames {idx} and {idx + 1}; tolerance {rel_tol:.1e}"
        )
    if F[-1] > F[0]:
        warnings.append(
            f"WARN: net free-energy INCREASE over the run "
            f"({F[0]:.5g} -> {F[-1]:.5g}); gradient flow should dissipate"
        )
    return warnings, worst


def check_field_range(n_all: np.ndarray, limit: float = 3.0):
    """
    Track global min/max of the field. |n| beyond `limit` is far outside the
    physical one-mode amplitude and usually precedes a blow-up.
    """
    warnings = []
    finite = n_all[np.isfinite(n_all)]
    if finite.size == 0:
        return warnings, (float("nan"), float("nan"))
    vmin, vmax = float(finite.min()), float(finite.max())
    if max(abs(vmin), abs(vmax)) > limit:
        warnings.append(
            f"WARN: field range [{vmin:.3f}, {vmax:.3f}] exceeds |n| <= {limit}"
        )
    return warnings, (vmin, vmax)


# ----------------------------------------------------------------------------
#  Aggregate
# ----------------------------------------------------------------------------
def run_all_checks(n_all: np.ndarray, free_energy: np.ndarray,
                   mass_drift_tol: float = 1e-5,
                   energy_increase_tol: float = 1e-3,
                   field_limit: float = 3.0,
                   aborted: bool = False,
                   verbose: bool = True) -> dict:
    """
    Run every check on one trajectory.

    Returns:
        {
          "status"  : "ok" | "warn" | "fail",
          "warnings": [str, ...],
          "stats"   : {mass_drift, energy_worst_increase, field_min, field_max}
        }
    """
    warnings = []

    if aborted:
        warnings.append("FAIL: run aborted early (NaN/Inf during stepping)")

    warnings += check_finite(n_all)
    w_mass, mass_drift = check_mass(n_all, tol=mass_drift_tol)
    warnings += w_mass
    w_en, worst_increase = check_energy(free_energy, rel_tol=energy_increase_tol)
    warnings += w_en
    w_rng, (vmin, vmax) = check_field_range(n_all, limit=field_limit)
    warnings += w_rng

    if any(w.startswith("FAIL") for w in warnings):
        status = "fail"
    elif warnings:
        status = "warn"
    else:
        status = "ok"

    report = {
        "status": status,
        "warnings": warnings,
        "stats": {
            "mass_drift": mass_drift,
            "energy_worst_relative_increase": worst_increase,
            "field_min": vmin,
            "field_max": vmax,
        },
    }

    if verbose:
        tag = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}[status]
        print(f"  [{tag}] mass drift {mass_drift:.2e} | "
              f"field [{vmin:+.3f}, {vmax:+.3f}] | "
              f"worst dF/|F| {worst_increase:+.2e}")
        for w in warnings:
            print(f"         {w}")

    return report
