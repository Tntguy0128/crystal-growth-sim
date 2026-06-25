"""
============================================================
  Crystal field analysis — the measurement engine

  Pure numpy/scipy. Given a PFC density field n(x, y), report
  the physical characteristics a materials person asks about:

    * lattice wavelength      (FFT)
    * crystallinity           (spectral concentration, 0..1)
    * dominant orientation    (angle of the hex spectral peaks)
    * atom sites + defects    (peaks -> Delaunay -> coordination)
    * grain count             (orientation clustering over the lattice)

  Density peaks in a one-mode PFC field ARE atom positions, so the
  standard crystallography pipeline (peaks -> triangulate -> analyze
  coordination and bond orientation) applies directly.

  This module is deliberately MODEL-FREE: it runs on any field, so it
  can be validated on exact solver output before it is ever pointed at
  an FNO prediction.

  NSF IRES Physical AI Design Program
============================================================
"""

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import Delaunay


# ----------------------------------------------------------------------------
#  Spectral descriptors (robust, no peak detection needed)
# ----------------------------------------------------------------------------
def _power_spectrum(field):
    f = field - field.mean()
    F = np.fft.fftshift(np.fft.fft2(f))
    return np.abs(F) ** 2


def _radial_profile(P):
    """Azimuthally-averaged power vs integer radius (in k-pixels)."""
    N, M = P.shape
    cy, cx = N // 2, M // 2
    y, x = np.indices(P.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), P.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def lattice_wavelength(field, dx):
    """
    Dominant modulation wavelength lambda (physical units). For a field on an
    N-pixel grid, a spectral ring at radius R k-pixels corresponds to
    lambda = N * dx / R.  Returns (lambda, R0) with R0 the dominant ring radius.
    """
    P = _power_spectrum(field)
    prof = _radial_profile(P)
    prof[:2] = 0.0                       # kill DC / lowest bins
    R0 = int(np.argmax(prof))
    N = field.shape[0]
    lam = (N * dx) / R0 if R0 > 0 else np.nan
    return lam, R0


def crystallinity(field, R0=None, band=2):
    """
    Fraction of (non-DC) spectral power concentrated in the dominant ring.
    ~1 for a sharp single lattice, low for liquid / disordered. In [0, 1].
    """
    P = _power_spectrum(field)
    prof = _radial_profile(P)
    prof[:2] = 0.0
    if R0 is None:
        R0 = int(np.argmax(prof))
    N, M = P.shape
    cy, cx = N // 2, M // 2
    y, x = np.indices(P.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ring = (r >= R0 - band) & (r <= R0 + band)
    total = P.copy()
    total[r < 2] = 0.0                    # exclude DC core
    denom = total.sum()
    return float(P[ring].sum() / denom) if denom > 0 else 0.0


def _ring_angles(field, R0, band=2, nbins=180):
    """Angular power distribution on the dominant ring (for orientation)."""
    P = _power_spectrum(field)
    N, M = P.shape
    cy, cx = N // 2, M // 2
    y, x = np.indices(P.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = (r >= R0 - band) & (r <= R0 + band)
    ang = (np.arctan2(y - cy, x - cx)) % np.pi      # mod 180: peaks come in pairs
    hist = np.zeros(nbins)
    bins = (ang[mask] / np.pi * nbins).astype(int) % nbins
    np.add.at(hist, bins, P[mask])
    return hist


def dominant_orientation(field, R0):
    """
    Dominant lattice orientation in degrees, folded into [0, 60) by the hex
    6-fold symmetry. Also returns an angular spread (0 = single sharp
    orientation, larger = polycrystalline / smeared).
    """
    hist = _ring_angles(field, R0)
    nbins = len(hist)
    k = int(np.argmax(hist))
    theta = (k / nbins * 180.0) % 60.0
    # spread: normalized circular dispersion of the ring power (mod 60 deg)
    ang = np.arange(nbins) / nbins * 180.0
    w = hist / (hist.sum() + 1e-12)
    z = np.sum(w * np.exp(1j * np.deg2rad(ang * 6.0)))   # 6-fold folding
    spread = float(1.0 - np.abs(z))                       # 0 sharp .. 1 smeared
    return theta, spread


# ----------------------------------------------------------------------------
#  Atom sites, defects, grains (peaks -> Delaunay -> geometry)
# ----------------------------------------------------------------------------
def find_atoms(field, lam_px, rel_thresh=0.2):
    """
    Atom positions = local maxima of the density. `lam_px` is the modulation
    wavelength in pixels (sets the min-separation footprint). Returns an
    (n, 2) array of (row, col) coordinates.
    """
    a_px = 1.1547 * lam_px                       # hex nn spacing = (2/sqrt3) lambda
    size = max(3, int(round(0.75 * a_px)))
    if size % 2 == 0:
        size += 1
    mx = maximum_filter(field, size=size, mode="nearest")
    thr = field.mean() + rel_thresh * (field.max() - field.mean())
    peaks = (field == mx) & (field > thr)
    return np.argwhere(peaks)


def _delaunay_adjacency(pts):
    """Neighbor sets + the Delaunay object for a point set."""
    tri = Delaunay(pts)
    nbrs = [set() for _ in range(len(pts))]
    for s in tri.simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    nbrs[s[i]].add(s[j])
    return tri, nbrs


def defect_atoms(atoms, shape, margin_px):
    """
    Interior atoms whose Delaunay coordination != 6 (dislocations / disclination
    cores). Border atoms (within margin_px of an edge) and convex-hull vertices
    are excluded, where coordination is unreliable. Returns (defect_idx, coord).
    """
    if len(atoms) < 4:
        return np.array([], dtype=int), np.array([])
    pts = atoms[:, ::-1].astype(float)            # (x, y)
    tri, nbrs = _delaunay_adjacency(pts)
    coord = np.array([len(n) for n in nbrs])
    hull = set(np.unique(tri.convex_hull))
    H, W = shape
    r, c = atoms[:, 0], atoms[:, 1]
    interior = ((r > margin_px) & (r < H - margin_px) &
                (c > margin_px) & (c < W - margin_px))
    is_defect = interior & np.array([i not in hull for i in range(len(atoms))]) \
        & (coord != 6)
    return np.where(is_defect)[0], coord


def _local_orientation(atoms, nbrs):
    """Per-atom bond-orientational angle theta = arg(psi6)/6, in degrees [0,60)."""
    pts = atoms[:, ::-1].astype(float)
    theta = np.full(len(atoms), np.nan)
    mag = np.zeros(len(atoms))
    for i, ns in enumerate(nbrs):
        if not ns:
            continue
        d = pts[list(ns)] - pts[i]
        ang = np.arctan2(d[:, 1], d[:, 0])
        psi6 = np.mean(np.exp(6j * ang))
        theta[i] = (np.rad2deg(np.angle(psi6)) / 6.0) % 60.0
        mag[i] = np.abs(psi6)
    return theta, mag


def _ang_diff60(a, b):
    d = abs(a - b) % 60.0
    return min(d, 60.0 - d)


def grain_count(atoms, tol_deg=10.0, order_thresh=0.6, min_grain=8):
    """
    Number of grains: connected components of the lattice where neighbouring
    well-ordered atoms share a local orientation (within tol_deg, mod 60).
    Returns (n_grains, labels) with labels[-1]-style -1 for unassigned atoms.
    """
    n = len(atoms)
    if n < min_grain:
        return 0, np.full(n, -1)
    _, nbrs = _delaunay_adjacency(atoms[:, ::-1].astype(float))
    theta, mag = _local_orientation(atoms, nbrs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    good = mag >= order_thresh
    for i in range(n):
        if not good[i]:
            continue
        for j in nbrs[i]:
            if good[j] and not np.isnan(theta[i]) and not np.isnan(theta[j]) \
                    and _ang_diff60(theta[i], theta[j]) <= tol_deg:
                union(i, j)

    labels = np.full(n, -1)
    sizes = {}
    for i in range(n):
        if good[i]:
            sizes.setdefault(find(i), []).append(i)
    g = 0
    for root, members in sizes.items():
        if len(members) >= min_grain:
            for m in members:
                labels[m] = g
            g += 1
    return g, labels


# ----------------------------------------------------------------------------
#  Top-level: one call -> the starter-set characteristics
# ----------------------------------------------------------------------------
def analyze(field, dx, rel_thresh=0.2):
    """
    Measure the starter set on a density field. Returns a dict with scalar
    characteristics plus atom/defect coordinates for visualization.
    """
    field = np.asarray(field, dtype=float)
    lam, R0 = lattice_wavelength(field, dx)
    cryst = crystallinity(field, R0)
    lam_px = lam / dx if np.isfinite(lam) else 16.0
    theta, spread = dominant_orientation(field, R0)

    atoms = find_atoms(field, lam_px, rel_thresh)
    margin = max(2, int(round(0.6 * 1.1547 * lam_px)))
    defects, coord = defect_atoms(atoms, field.shape, margin)
    n_grains, labels = grain_count(atoms)

    n_int = int(((atoms[:, 0] > margin) & (atoms[:, 0] < field.shape[0] - margin) &
                 (atoms[:, 1] > margin) & (atoms[:, 1] < field.shape[1] - margin)).sum()) \
        if len(atoms) else 0
    defect_density = (len(defects) / n_int) if n_int > 0 else 0.0

    return {
        "lattice_wavelength": float(lam),
        "crystallinity": float(cryst),
        "dominant_orientation_deg": float(theta),
        "orientation_spread": float(spread),
        "n_atoms": int(len(atoms)),
        "n_defects": int(len(defects)),
        "defect_density": float(defect_density),
        "n_grains": int(n_grains),
        # arrays for plotting (row, col)
        "atoms": atoms,
        "defect_coords": atoms[defects] if len(defects) else np.empty((0, 2)),
        "grain_labels": labels,
    }


def summary_lines(props):
    """Human-readable one-liners for the UI / console."""
    poly = props["n_grains"] if props["n_grains"] > 0 else "—"
    kind = ("liquid / disordered" if props["crystallinity"] < 0.25
            else "single crystal" if props["n_grains"] <= 1
            else f"polycrystal ({props['n_grains']} grains)")
    return [
        f"Structure         : {kind}",
        f"Crystallinity     : {props['crystallinity']:.2f}  (0 liquid → 1 perfect)",
        f"Grains            : {poly}",
        f"Dominant orient.  : {props['dominant_orientation_deg']:.1f}°  "
        f"(spread {props['orientation_spread']:.2f})",
        f"Defects           : {props['n_defects']}  "
        f"(density {props['defect_density']*100:.1f}% of interior atoms)",
        f"Lattice wavelength: {props['lattice_wavelength']:.2f}  ({props['n_atoms']} atoms)",
    ]


# ----------------------------------------------------------------------------
#  Self-test: synthetic single crystal + bicrystal (no solver / torch needed)
# ----------------------------------------------------------------------------
def _hex_field(N, dx, q, theta_deg, x0=0.0, y0=0.0):
    x = (np.arange(N) - x0) * dx
    y = (np.arange(N) - y0) * dx
    X, Y = np.meshgrid(x, y, indexing="ij")
    th = np.deg2rad(theta_deg)
    f = np.zeros((N, N))
    for j in range(3):
        a = th + j * (2 * np.pi / 3)
        f += np.cos(q * (np.cos(a) * X + np.sin(a) * Y))
    return f


if __name__ == "__main__":
    N, dx, q = 128, 16 * np.pi / 128, 1.0
    lam_true = 2 * np.pi / q
    print(f"grid N={N} dx={dx:.4f}  q={q}  true wavelength={lam_true:.3f}\n")

    print("=== single crystal (theta = 15 deg) ===")
    f1 = _hex_field(N, dx, q, 15.0)
    p1 = analyze(f1, dx)
    for ln in summary_lines(p1):
        print("  " + ln)
    assert p1["crystallinity"] > 0.5, "single crystal should be highly crystalline"
    assert p1["n_grains"] == 1, f"expected 1 grain, got {p1['n_grains']}"
    assert abs(p1["lattice_wavelength"] - lam_true) / lam_true < 0.1, "wavelength off"
    assert _ang_diff60(p1["dominant_orientation_deg"], 15.0) < 6.0, "orientation off"

    print("\n=== bicrystal (left 10 deg | right 35 deg) ===")
    fa = _hex_field(N, dx, q, 10.0)
    fb = _hex_field(N, dx, q, 35.0)
    f2 = np.where(np.arange(N)[None, :] < N // 2, fa, fb)
    p2 = analyze(f2, dx)
    for ln in summary_lines(p2):
        print("  " + ln)
    assert p2["n_grains"] >= 2, f"expected >=2 grains, got {p2['n_grains']}"

    print("\n=== liquid (noise) ===")
    rng = np.random.default_rng(0)
    f3 = 0.05 * rng.standard_normal((N, N))
    p3 = analyze(f3, dx)
    print(f"  crystallinity={p3['crystallinity']:.2f}  grains={p3['n_grains']}")
    assert p3["crystallinity"] < 0.4, "noise should read as low crystallinity"

    print("\nAll analyze() self-tests passed.")
