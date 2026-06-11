"""
============================================================
  Phase Field Crystal (PFC) solver — refactored, reusable
  Teacher model for FNO surrogate training data

  Numerics are mathematically identical to PCF_Baseline.py
  (pseudo-spectral, semi-implicit time stepping); only the
  code organization changed.

  NSF IRES Physical AI Design Program — Georgia Tech
============================================================

THE PHYSICS / NUMERICS (what this solver integrates)
----------------------------------------------------
The PFC model evolves a conserved density field n(x, y, t):

    dn/dt = M * lap( dF/dn )                       (conserved dynamics)
    dF/dn = (r + (1 + lap)^2) n + n^3

so the full PDE is

    dn/dt = M * lap[ (r + (1 + lap)^2) n + n^3 ].

In Fourier space lap -> -k^2, so the LINEAR part is diagonal:

    L(k) = -M k^2 (k^4 - 2k^2 + 1 + r)             (note (1-k^2)^2 = k^4-2k^2+1)

and the NONLINEAR part is

    N_hat = -M k^2 * fft(n^3)                       (lap -> -k^2)

(The original baseline file had a sign typo in its *comment* here; its code,
which we reproduce, uses the correct -M k^2.)

TIME STEPPING: semi-implicit (IMEX). Treat the stiff linear term implicitly
and the nonlinear term explicitly:

    n_hat(t+dt) = ( n_hat(t) + dt * N_hat(t) ) / ( 1 - dt * L(k) )

The denominator is precomputed once. This allows the large dt = 0.25 used
throughout. A 2/3-rule dealiasing mask zeroes the highest third of modes in
the nonlinear term to prevent aliasing instability (n^3 triples wavenumbers).

CONSERVATION PROPERTIES (used as sanity checks downstream):
  * Mass: the k = 0 mode has L(0) = 0 and N_hat(0) = 0, so mean(n) is
    conserved to floating-point precision.
  * Free energy F[n] = Int[ 1/2 n (r + (1+lap)^2) n + 1/4 n^4 ] dr decreases
    monotonically (gradient flow), up to small explicit-nonlinearity errors.
"""

import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.fft import fft2, ifft2

VALID_SEED_TYPES = ("hex", "point", "multi", "random_noise", "custom_mask")


# ----------------------------------------------------------------------------
#  Configuration
# ----------------------------------------------------------------------------
@dataclass
class PFCConfig:
    """All knobs for one PFC run. Defaults reproduce PCF_Baseline.py."""
    # --- physics ---
    r: float = -0.28          # temperature parameter (more negative = deeper quench)
    n0: float = -0.285        # mean density (sits in the hexagonal phase)
    M: float = 1.0            # mobility

    # --- grid ---
    N: int = 128              # grid points per side
    L: float = 16.0 * np.pi   # physical domain size

    # --- time ---
    dt: float = 0.25          # solver timestep
    T: float = 250.0          # total simulated time
    save_every: int = 25      # save a frame every this many steps

    # --- initial condition ---
    seed_type: str = "hex"    # one of VALID_SEED_TYPES
    rng_seed: int = 42        # RNG seed (noise + random seed placement)
    noise_amplitude: float = 0.01   # std of the initial liquid noise
    seed_amplitude: float = 0.20    # amplitude of the planted seed pattern
    n_seeds: int = 5                # number of seeds for seed_type="multi"
    # Wavenumber of the planted crystal pattern. None reproduces the baseline
    # (2*pi / (4*dx) = 4.0 on the default grid). NOTE: the PFC equilibrium
    # lattice wavenumber is k ~ 1, so the baseline's k0=4 pattern actually
    # DECAYS within a few steps and crystallization nucleates from the noise +
    # seed remnant. Set seed_k0: 1.0 to plant a pattern that survives and
    # grows directly. Default stays baseline-faithful.
    seed_k0: float = None
    custom_mask: object = None      # (N, N) array or path to .npy, for "custom_mask"

    # --- output ---
    output_dir: str = "data_pfc"
    save_plots: bool = False
    save_animation: bool = False

    # --- sanity-check tolerances (consumed by sanity_checks.py) ---
    mass_drift_tol: float = 1e-5        # relative drift of mean(n) across frames
    energy_increase_tol: float = 1e-3   # allowed relative free-energy uptick

    # ----- derived quantities -----
    @property
    def dx(self) -> float:
        return self.L / self.N            # == np.linspace(0, L, N, endpoint=False) spacing

    @property
    def n_steps(self) -> int:
        return int(self.T / self.dt)

    @property
    def n_frames(self) -> int:
        return self.n_steps // self.save_every

    def validate(self):
        if self.seed_type not in VALID_SEED_TYPES:
            raise ValueError(f"seed_type must be one of {VALID_SEED_TYPES}, "
                             f"got {self.seed_type!r}")
        if self.n_frames < 1:
            raise ValueError("T / dt / save_every give zero saved frames")
        if self.seed_type == "custom_mask" and self.custom_mask is None:
            raise ValueError("seed_type='custom_mask' requires custom_mask "
                             "(an (N, N) array or a path to a .npy file)")
        return self


# ----------------------------------------------------------------------------
#  Fourier operators — built once per (grid, physics) and reused every step
# ----------------------------------------------------------------------------
class FourierOperators:
    """
    Precomputed spectral quantities:

        k2           : |k|^2 on the (N, N) grid
        dealias      : 2/3-rule boolean mask for the nonlinear term
        L_op         : linear PFC eigenvalue  -M k^2 (k^4 - 2k^2 + 1 + r)
        linear_denom : 1 / (1 - dt * L_op), the semi-implicit denominator
    """

    def __init__(self, N: int, dx: float, r: float, M: float, dt: float):
        kx = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
        KX, KY = np.meshgrid(kx, kx, indexing="ij")
        self.k2 = KX ** 2 + KY ** 2

        kmax_dealias = kx.max() * 2.0 / 3.0
        self.dealias = (np.abs(KX) < kmax_dealias) & (np.abs(KY) < kmax_dealias)

        # (1 - k^2)^2 expanded: k^4 - 2 k^2 + 1
        self.L_op = -M * self.k2 * (self.k2 ** 2 - 2.0 * self.k2 + 1.0 + r)
        self.linear_denom = 1.0 / (1.0 - dt * self.L_op)
        self.M = M


# ----------------------------------------------------------------------------
#  Initial conditions (seed library)
# ----------------------------------------------------------------------------
def _hex_pattern(PX, PY, k0: float, theta: float = 0.0):
    """
    One-mode hexagonal pattern: sum of three plane waves whose wavevectors are
    120 degrees apart (rotated by `theta`), all with |k| = k0.

        theta = 0 reduces to the baseline form
        cos(k0 x) + 2 cos(k0 x / 2) cos(sqrt(3)/2 k0 y).

    PX, PY are PHYSICAL coordinates (possibly shifted to a seed center).
    """
    pattern = np.zeros_like(PX)
    for j in range(3):
        a = theta + j * (2.0 * np.pi / 3.0)
        pattern += np.cos(k0 * (np.cos(a) * PX + np.sin(a) * PY))
    return pattern


def make_initial_condition(cfg: PFCConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Build n(x, y, t=0): uniform liquid at n0 + small noise + the chosen seed.

    Seed types:
        hex          : centered Gaussian-enveloped hexagonal patch
                       (numerically identical to PCF_Baseline.py)
        point        : single localized Gaussian density bump
        multi        : n_seeds hexagonal patches at random positions/rotations
        random_noise : liquid + noise only (homogeneous nucleation)
        custom_mask  : caller-supplied (N, N) mask in [0, 1]; crystal pattern is
                       planted wherever the mask is > 0 (drawn-seed interface)
    """
    N, dx = cfg.N, cfg.dx
    x = np.linspace(0.0, cfg.L, N, endpoint=False)

    # Base liquid + noise (same call order as the baseline so seed_type="hex"
    # with rng_seed=42 reproduces it bit-for-bit at default settings).
    n = cfg.n0 + cfg.noise_amplitude * rng.standard_normal((N, N))

    # Index grids (baseline used index-space Gaussians) and physical grids.
    SX, SY = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    PX, PY = x[SX], x[SY]

    # Lattice wavenumber: baseline default is ~4 grid points per stripe
    # (k0 = 4.0 on the default grid); see seed_k0 in PFCConfig for the caveat.
    k0 = cfg.seed_k0 if cfg.seed_k0 is not None else 2.0 * np.pi / (4.0 * dx)

    if cfg.seed_type == "hex":
        cx, cy = N // 2, N // 2
        sigma2 = (N // 10) ** 2                              # baseline envelope width
        envelope = np.exp(-((SX - cx) ** 2 + (SY - cy) ** 2) / (2.0 * sigma2))
        n += cfg.seed_amplitude * envelope * _hex_pattern(PX, PY, k0)

    elif cfg.seed_type == "point":
        # A featureless Gaussian bump: nucleation happens because the bump
        # locally pushes the density into the unstable region.
        cx, cy = N // 2, N // 2
        sigma2 = (N // 16) ** 2                              # tighter than "hex"
        n += cfg.seed_amplitude * np.exp(
            -((SX - cx) ** 2 + (SY - cy) ** 2) / (2.0 * sigma2))

    elif cfg.seed_type == "multi":
        # Several hexagonal patches with random centers and random lattice
        # orientations -> grain boundaries form where they meet.
        for _ in range(cfg.n_seeds):
            cx, cy = rng.integers(0, N, size=2)
            theta = rng.uniform(0.0, np.pi / 3.0)            # hex symmetry: 60 deg
            sigma2 = (rng.integers(N // 16, N // 8)) ** 2
            # Periodic (wrapped) distance so seeds near edges stay round.
            ddx = np.minimum(np.abs(SX - cx), N - np.abs(SX - cx))
            ddy = np.minimum(np.abs(SY - cy), N - np.abs(SY - cy))
            envelope = np.exp(-(ddx ** 2 + ddy ** 2) / (2.0 * sigma2))
            n += cfg.seed_amplitude * envelope * _hex_pattern(
                PX - x[cx], PY - x[cy], k0, theta)

    elif cfg.seed_type == "random_noise":
        # Nothing extra: liquid + the noise already added above.
        pass

    elif cfg.seed_type == "custom_mask":
        mask = cfg.custom_mask
        if isinstance(mask, str):
            mask = np.load(mask)
        mask = np.asarray(mask, dtype=np.float64)
        if mask.shape != (N, N):
            raise ValueError(f"custom_mask shape {mask.shape} != grid ({N}, {N})")
        mmax = np.abs(mask).max()
        if mmax > 0:
            mask = mask / mmax                               # normalize to [0, 1]
        # The mask acts as the seed envelope -- crystal grows from wherever
        # the user drew. This is the hook for the future drawing interface.
        n += cfg.seed_amplitude * mask * _hex_pattern(PX, PY, k0)

    return n


# ----------------------------------------------------------------------------
#  Free energy (diagnostic, identical formula to the baseline)
# ----------------------------------------------------------------------------
def compute_free_energy(frames: np.ndarray, ops: FourierOperators,
                        r: float, dx: float) -> np.ndarray:
    """
    F[n] = Sum[ n (lap n + 1/2 lap^2 n) + 1/2 (1+r) n^2 + 1/4 n^4 ] dx^2
    evaluated per saved frame, using spectral derivatives
    (lap -> -k^2, lap^2 -> k^4 in Fourier space).
    """
    F = np.zeros(frames.shape[0])
    for i, n in enumerate(frames):
        nh = fft2(n)
        lap_n = ifft2(-ops.k2 * nh).real
        laplap_n = ifft2(ops.k2 ** 2 * nh).real
        F[i] = np.sum(
            n * (lap_n + 0.5 * laplap_n)
            + 0.5 * (1.0 + r) * n ** 2
            + 0.25 * n ** 4
        ) * dx ** 2
    return F


# ----------------------------------------------------------------------------
#  Results container
# ----------------------------------------------------------------------------
@dataclass
class PFCResult:
    n_all: np.ndarray            # (n_frames, N, N) float32 saved trajectory
    t_vals: np.ndarray           # (n_frames,) physical times of the frames
    free_energy: np.ndarray      # (n_frames,) F[n] per frame
    mass_initial: float          # mean(n) at t = 0   (float64, from the solver state)
    mass_final: float            # mean(n) at t = T
    runtime_seconds: float       # wall time of the stepping loop only
    steps_per_second: float      # solver throughput (for FNO speed comparison)
    aborted: bool = False        # True if NaN/Inf appeared and the run stopped early
    config: dict = field(default_factory=dict)

    @property
    def mass_relative_drift(self) -> float:
        return abs(self.mass_final / self.mass_initial - 1.0)

    @property
    def energy_initial(self) -> float:
        return float(self.free_energy[0])

    @property
    def energy_final(self) -> float:
        return float(self.free_energy[-1])

    @property
    def energy_drop_percent(self) -> float:
        f0, f1 = self.energy_initial, self.energy_final
        return 100.0 * (f0 - f1) / abs(f0) if f0 != 0 else 0.0


# ----------------------------------------------------------------------------
#  The solver
# ----------------------------------------------------------------------------
class PFCSolver:
    """
    Pseudo-spectral semi-implicit PFC integrator.

    Usage:
        cfg = PFCConfig(r=-0.28, n0=-0.285, seed_type="multi", rng_seed=7)
        result = PFCSolver(cfg).run()
    """

    def __init__(self, cfg: PFCConfig):
        self.cfg = cfg.validate()
        self.ops = FourierOperators(cfg.N, cfg.dx, cfg.r, cfg.M, cfg.dt)

    # -- one semi-implicit step (kept separate so it is testable/benchmarkable)
    def step(self, n: np.ndarray, n_hat: np.ndarray):
        """
        Advance one dt. Takes and returns BOTH the real field and its FFT so
        we never do redundant transforms (2 FFTs per step: fft of n^3, ifft).
        """
        ops, dt = self.ops, self.cfg.dt
        # Nonlinear term in Fourier space, dealiased:  -M k^2 fft(n^3)
        NL_hat = -(ops.M * ops.k2 * fft2(n ** 3)) * ops.dealias
        # Semi-implicit update (linear implicit, nonlinear explicit).
        n_hat = (n_hat + dt * NL_hat) * ops.linear_denom
        n = ifft2(n_hat).real
        return n, n_hat

    def run(self, progress: bool = False) -> PFCResult:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.rng_seed)

        n = make_initial_condition(cfg, rng)        # float64 internal state
        mass_initial = float(n.mean())

        n_frames = cfg.n_frames
        n_all = np.zeros((n_frames, cfg.N, cfg.N), dtype=np.float32)
        n_all[0] = n.astype(np.float32)

        n_hat = fft2(n)
        aborted = False
        frame_idx = 1

        # Time only the stepping loop -- this is the number we will compare
        # against FNO inference speed (the FNO replaces save_every steps
        # with a single forward pass).
        t0 = time.perf_counter()
        steps_done = 0
        for i in range(1, cfg.n_steps):             # same loop bounds as baseline
            n, n_hat = self.step(n, n_hat)
            steps_done += 1

            if i % cfg.save_every == 0 and frame_idx < n_frames:
                if not np.isfinite(n).all():
                    # Blow-up: stop early, keep the frames we have.
                    aborted = True
                    n_all = n_all[:frame_idx]
                    break
                n_all[frame_idx] = n.astype(np.float32)
                frame_idx += 1
                if progress:
                    print(f"  step {i:6d}/{cfg.n_steps}  "
                          f"range [{n.min():+.3f}, {n.max():+.3f}]", end="\r")
        runtime = time.perf_counter() - t0
        if progress:
            print()

        n_all = n_all[:frame_idx]
        t_vals = np.arange(frame_idx) * cfg.save_every * cfg.dt
        free_energy = compute_free_energy(n_all, self.ops, cfg.r, cfg.dx)

        # Serializable copy of the config (custom_mask arrays summarized).
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

    def benchmark(self, n_steps: int = 200) -> dict:
        """
        Time the raw stepping loop (no I/O, no diagnostics). Use this to
        compare against FNO inference: one FNO call spans save_every solver
        steps, so the fair ratio is
            (seconds_per_step * save_every)  vs  seconds_per_FNO_forward.
        """
        cfg = self.cfg
        rng = np.random.default_rng(cfg.rng_seed)
        n = make_initial_condition(cfg, rng)
        n_hat = fft2(n)
        n, n_hat = self.step(n, n_hat)              # warm-up (FFT plan caching)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            n, n_hat = self.step(n, n_hat)
        dt_wall = time.perf_counter() - t0
        per_step = dt_wall / n_steps
        return {
            "seconds_per_step": per_step,
            "steps_per_second": 1.0 / per_step,
            "seconds_per_frame_interval": per_step * cfg.save_every,
            "grid": f"{cfg.N}x{cfg.N}",
        }


# ----------------------------------------------------------------------------
#  ML-ready trajectory output
# ----------------------------------------------------------------------------
def save_trajectory(result: PFCResult, cfg: PFCConfig, path: str) -> str:
    """
    Write one .npz with the full trajectory plus all metadata and diagnostics.

    The key `n_all` (T, H, W) is what dataset.py auto-discovers for FNO
    training; everything else is provenance + sanity-check material.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        # --- trajectory ---
        n_all=result.n_all,
        t_vals=result.t_vals.astype(np.float32),
        # --- physics / numerics provenance ---
        r=np.float64(cfg.r), n0=np.float64(cfg.n0), M=np.float64(cfg.M),
        N=np.int64(cfg.N), L=np.float64(cfg.L), dx=np.float64(cfg.dx),
        dt=np.float64(cfg.dt), T=np.float64(cfg.T),
        save_every=np.int64(cfg.save_every),
        seed_type=np.str_(cfg.seed_type),
        rng_seed=np.int64(cfg.rng_seed),
        # Seed-construction provenance (matters once sweeps vary these):
        noise_amplitude=np.float64(cfg.noise_amplitude),
        seed_amplitude=np.float64(cfg.seed_amplitude),
        n_seeds=np.int64(cfg.n_seeds),
        # Effective planted wavenumber actually used (resolves the None default).
        seed_k0=np.float64(cfg.seed_k0 if cfg.seed_k0 is not None
                           else 2.0 * np.pi / (4.0 * cfg.dx)),
        # --- conservation diagnostics ---
        mass_initial=np.float64(result.mass_initial),
        mass_final=np.float64(result.mass_final),
        mass_relative_drift=np.float64(result.mass_relative_drift),
        free_energy=result.free_energy.astype(np.float32),
        energy_initial=np.float64(result.energy_initial),
        energy_final=np.float64(result.energy_final),
        energy_drop_percent=np.float64(result.energy_drop_percent),
        # --- benchmarking ---
        runtime_seconds=np.float64(result.runtime_seconds),
    )
    return path


# ----------------------------------------------------------------------------
#  Quick demo:  python pfc_solver.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = PFCConfig(T=50.0)                          # short demo run
    solver = PFCSolver(cfg)
    print(f"Running demo: {cfg.N}x{cfg.N}, seed_type={cfg.seed_type}, "
          f"{cfg.n_steps} steps -> {cfg.n_frames} frames")
    res = solver.run(progress=True)
    print(f"runtime          : {res.runtime_seconds:.2f}s "
          f"({res.steps_per_second:.0f} steps/s)")
    print(f"mass drift       : {res.mass_relative_drift:.3e}")
    print(f"energy drop      : {res.energy_drop_percent:.2f}%")
    bench = solver.benchmark()
    print(f"benchmark        : {bench['seconds_per_step']*1e3:.2f} ms/step, "
          f"{bench['seconds_per_frame_interval']*1e3:.1f} ms per saved-frame interval")
