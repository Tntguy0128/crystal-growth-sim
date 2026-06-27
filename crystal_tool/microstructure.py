"""
============================================================
  Microstructure reveal — show that crystals which look
  identical as density fields are physically different.

  Three materials-science views that expose the differences a
  raw density image hides:
    * orientation map   — atoms colored by local lattice angle
                          (the EBSD / inverse-pole-figure analog);
                          different grains / global angles = different colors
    * diffraction (FFT) — the six Bragg peaks rotate with orientation
                          and broaden with disorder
    * quantitative      — free energy, defect count, bond-orientational
                          order (psi6), orientation spread

  Two crystals that are visually indistinguishable show clearly
  different orientation maps, rotated diffraction spots, and
  measurably different free energy / defect counts.

  python crystal_tool/microstructure.py        # builds a comparison figure
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze as A
from pfc_solver import FourierOperators, compute_free_energy


def orientation_per_atom(field, dx):
    """Atoms + per-atom local lattice angle (deg, 0..60) and order magnitude."""
    lam, _ = A.lattice_wavelength(field, dx)
    lam_px = lam / dx if np.isfinite(lam) else 16.0
    atoms = A.find_atoms(field, lam_px)
    if len(atoms) < 4:
        return atoms, np.array([]), np.array([])
    _, nbrs = A._delaunay_adjacency(atoms[:, ::-1].astype(float))
    theta, mag = A._local_orientation(atoms, nbrs)
    return atoms, theta, mag


def free_energy(field, r, dx, M=1.0, dt=0.25):
    """PFC free energy of a single field (lower = more relaxed/stable)."""
    N = field.shape[0]
    ops = FourierOperators(N, dx, r, M, dt)
    return float(compute_free_energy(field[None].astype(np.float64), ops, r, dx)[0])


def diffraction(field):
    """Log power spectrum (centered) — the crystal's diffraction pattern."""
    F = np.abs(np.fft.fftshift(np.fft.fft2(field - field.mean())))
    return np.log1p(F)


def psi6(theta, mag):
    """Global bond-orientational order |<psi6>| in [0,1] (1 = single perfect orient)."""
    if len(theta) == 0:
        return 0.0
    good = ~np.isnan(theta)
    return float(np.abs(np.mean(np.exp(6j * np.deg2rad(theta[good])))))


def compare(fields, labels, r, dx, out_path):
    n = len(fields)
    fig, axes = plt.subplots(3, n, figsize=(4.3 * n, 12))
    for j, (f, lab) in enumerate(zip(fields, labels)):
        p = A.analyze(f, dx)
        atoms, theta, mag = orientation_per_atom(f, dx)
        Fe = free_energy(f, r, dx)
        o6 = psi6(theta, mag)

        # row 0: raw density (the "looks identical" view)
        ax = axes[0, j]; ax.imshow(f, cmap="magma"); ax.axis("off")
        ax.set_title(f"{lab}\nDENSITY  (looks ~identical)", fontsize=10)

        # row 1: orientation map (atoms colored by local lattice angle)
        ax = axes[1, j]; ax.imshow(f, cmap="gray", alpha=0.25)
        if len(atoms):
            sc = ax.scatter(atoms[:, 1], atoms[:, 0], c=theta, cmap="hsv",
                            vmin=0, vmax=60, s=34, edgecolors="k", linewidths=0.2)
        ax.axis("off")
        ax.set_title(f"ORIENTATION MAP\nmean {np.nanmean(theta):.0f}°  spread {p['orientation_spread']:.2f}",
                     fontsize=10)

        # row 2: diffraction pattern (FFT) + the quantitative microstructure
        ax = axes[2, j]; D = diffraction(f); c = D.shape[0] // 2; w = 22
        ax.imshow(D[c - w:c + w, c - w:c + w], cmap="inferno"); ax.axis("off")
        ax.set_title(f"DIFFRACTION (FFT)\nF={Fe:.1f}   defects={p['n_defects']}   ψ6={o6:.2f}",
                     fontsize=10)
    fig.suptitle("Same-looking density  →  different microstructure "
                 "(orientation, diffraction, free energy, defects)",
                 fontsize=14, fontweight="bold")
    # shared colorbar for the orientation row
    cax = fig.add_axes([0.92, 0.40, 0.012, 0.2])
    import matplotlib.cm as cm
    sm = cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(0, 60)); sm.set_array([])
    fig.colorbar(sm, cax=cax, label="lattice angle (°)")
    fig.tight_layout(rect=[0, 0, 0.91, 0.97])
    fig.savefig(out_path, dpi=95); plt.close(fig)
    print("wrote", out_path)


def _demo():
    import grow
    m, ck, device = grow.load_fno(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fno_demo.pt"))

    def disk(N, cx, cy, rad):
        Y, X = np.mgrid[0:N, 0:N]
        return (np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) < rad).astype(float)

    blob = disk(128, 64, 64, 16)
    fields, labels, R = [], [], -0.30
    # two single-crystal grows at different random orientations (look identical)
    for s in (2, 5):
        f0, cfg = grow.mask_to_field(blob, r=R, n0=-0.285, orientation="random", rng_seed=s)
        fields.append(grow.grow_fno(m, ck, device, f0, steps=30)[-1]); labels.append(f"single #{s}")
    # one per-region polycrystal (genuinely different microstructure)
    mask = np.zeros((128, 128)); rng = np.random.default_rng(3)
    for _ in range(3):
        cx, cy = rng.integers(30, 98, 2); mask = np.maximum(mask, disk(128, cx, cy, 12))
    f0, cfg = grow.mask_to_field(mask, r=R, n0=-0.285, per_region=True, rng_seed=3)
    fields.append(grow.grow_fno(m, ck, device, f0, steps=20)[-1]); labels.append("polycrystal")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_out", "microstructure.png")
    compare(fields, labels, R, cfg.dx, out)


if __name__ == "__main__":
    _demo()
