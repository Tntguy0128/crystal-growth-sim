"""
============================================================
  PFC phase-diagram explorer

  Sweeps the Phase Field Crystal control parameters (quench
  depth r, mean density n0) with NEUTRAL (random-noise) initial
  conditions, lets each system spontaneously order, and
  classifies the resulting phase — hexagonal / stripe / liquid —
  from its diffraction pattern (analyze.classify_phase).

  This is the research capability the surrogate is meant to
  accelerate: a parametric map of which microstructure forms
  where. The solver builds it in seconds here; a *parametric*
  FNO would reproduce any point on it in a single forward pass.

  python crystal_tool/phase_diagram.py
============================================================
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze as A
from pfc_solver import PFCConfig, PFCSolver

PHASE_COLOR = {"hexagonal": "#0E9594", "stripe": "#E0913A",
               "liquid": "#9FB0C0", "unstable": "#D85A30"}


def scan(r_values, n0_values, T=300.0, noise=0.10, rng_seed=3):
    """Return a list of {r, n0, phase, crystallinity, field} over the grid."""
    out = []
    for r in r_values:
        for n0 in n0_values:
            cfg = PFCConfig(r=r, n0=n0, seed_type="random_noise",
                            noise_amplitude=noise, T=T, rng_seed=rng_seed)
            res = PFCSolver(cfg).run()
            f = res.n_all[-1]
            if res.aborted or not np.isfinite(f).all():
                phase, c = "unstable", 0.0
            else:
                phase = A.classify_phase(f, cfg.dx)
                c = A.crystallinity(f, A.lattice_wavelength(f, cfg.dx)[1])
            out.append({"r": r, "n0": n0, "phase": phase,
                        "crystallinity": float(c), "field": f})
    return out


def render(results, r_values, n0_values, out_path):
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(3, 5, width_ratios=[2.3, 1, 1, 0.05, 0.9])

    # --- phase map (left) ---
    axm = fig.add_subplot(gs[:, 0])
    for d in results:
        axm.scatter(d["n0"], d["r"], s=520, marker="s",
                    color=PHASE_COLOR[d["phase"]], edgecolors="white", linewidths=1.5)
    axm.set_xlabel("mean density  n₀", fontsize=12)
    axm.set_ylabel("quench depth  r", fontsize=12)
    axm.set_title("PFC phase diagram (solver + classifier)", fontsize=13, fontweight="bold")
    axm.grid(alpha=0.2)
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
                          markersize=12, label=p)
               for p, c in PHASE_COLOR.items() if p != "unstable"]
    axm.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.9)

    # --- representative thumbnails (right): one of each phase ---
    picks = {}
    for d in results:
        if d["phase"] not in picks and d["phase"] in ("hexagonal", "stripe", "liquid"):
            picks[d["phase"]] = d
    order = ["hexagonal", "stripe", "liquid"]
    thumb_cells = [gs[0, 1], gs[0, 2], gs[1, 1]]
    for ph, cell in zip(order, thumb_cells):
        ax = fig.add_subplot(cell); ax.axis("off")
        if ph in picks:
            d = picks[ph]
            ax.imshow(d["field"], cmap="magma")
            ax.set_title(f"{ph}\nr={d['r']:+.2f} n₀={d['n0']:+.2f}", fontsize=9,
                         color=PHASE_COLOR[ph])
        else:
            ax.set_title(f"{ph}\n(not found)", fontsize=9)

    fig.suptitle("Different parameters → genuinely different microstructure phases",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=95); plt.close(fig)
    print("wrote", out_path)


def main():
    r_values = [-0.20, -0.30, -0.40]
    n0_values = [0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.42]
    t0 = time.time()
    res = scan(r_values, n0_values)
    # tally
    from collections import Counter
    tally = Counter(d["phase"] for d in res)
    print(f"scanned {len(res)} (r,n0) points in {time.time()-t0:.1f}s  ->  {dict(tally)}")
    for d in res:
        print(f"  r={d['r']:+.2f} n0={d['n0']:+.2f} : {d['phase']}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_out", "phase_diagram.png")
    render(res, r_values, n0_values, out)


if __name__ == "__main__":
    main()
