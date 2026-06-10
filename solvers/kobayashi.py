"""
============================================================
  Kobayashi Phase Field Solver
  Dendritic crystal growth simulation

  Matches the JavaScript physics in crystal_simulator.html
  exactly so the FNO learns the same dynamics the GUI shows.

  Reference:
    R. Kobayashi, "Modeling and numerical simulations of
    dendritic crystal growth," Physica D 63 (1993) 410-423.

  Ayush Shah & Tobias Li
  Georgia Institute of Technology
  NSF IRES Physical AI Design Program — Prof. Bo Zhu
============================================================

Usage
-----
Run a single simulation and save frames:

    python kobayashi.py --te 0.44 --delta 0.06 --j 6 \
                        --noise 0.12 --seed hex6 \
                        --steps 2000 --save_every 50 \
                        --out output.npz

Or import and call run_kobayashi() directly from the
dataset generation script.
"""

import argparse
import time
import numpy as np
from scipy.fft import fft2, ifft2


# ── Default simulation constants (match JavaScript exactly) ──────────────────
DEFAULTS = dict(
    N          = 256,       # grid size (must match JS: N=256)
    dt         = 0.008,     # timestep
    tau        = 0.0003,    # phase field relaxation time
    K          = 1.8,       # latent heat coupling
    alpha      = 0.9,       # reaction coefficient
    gamma      = 10.0,      # sharpness of m(T) sigmoid
    eps0       = 0.010,     # base anisotropy magnitude
    # --- sweep parameters ---
    Te         = 0.44,      # undercooling (equilibrium temperature)
    delta      = 0.06,      # anisotropy strength
    j          = 6,         # symmetry order (6 = snowflake, 4 = square)
    noise_amp  = 0.12,      # spatial noise amplitude on anisotropy direction
    pulse      = 0.30,      # growth speed pulse amplitude
    nuc_rate   = 0.10,      # secondary nucleation rate
    # --- run length ---
    steps      = 3000,      # total simulation steps
    save_every = 50,        # save one frame every N steps
    # --- seed ---
    seed_type  = 'hex6',    # 'hex6' | 'hex4' | 'point' | 'ring' | 'multi'
    seed       = 42,        # random seed for reproducibility
)


# ── Noise field construction (matches JS buildNoise()) ───────────────────────

def _smooth_octave(N: int, scale: int, rng: np.random.Generator) -> np.ndarray:
    """
    One octave of smooth value noise at the given grid scale.
    Uses Hermite (smoothstep) interpolation between random grid points,
    exactly matching the JS implementation.
    """
    # Random values at coarse grid corners
    seeds = rng.random((N, N)) - 0.5

    result = np.zeros((N, N), dtype=np.float32)
    yy, xx = np.mgrid[0:N, 0:N]

    gx  = (xx // scale) * scale
    gy  = (yy // scale) * scale
    gx2 = np.minimum(gx + scale, N - 1)
    gy2 = np.minimum(gy + scale, N - 1)

    u = (xx - gx) / scale   # fractional position [0,1]
    v = (yy - gy) / scale

    # Hermite smoothstep: 3t² - 2t³
    su = u * u * (3 - 2 * u)
    sv = v * v * (3 - 2 * v)

    # Bilinear interpolation between four corners
    result = (
        seeds[gy,  gx]  * (1 - su) * (1 - sv) +
        seeds[gy,  gx2] * su       * (1 - sv) +
        seeds[gy2, gx]  * (1 - su) * sv       +
        seeds[gy2, gx2] * su       * sv
    ).astype(np.float32)

    return result


def build_noise_field(N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Multi-octave smooth noise field for spatial anisotropy variation.
    Weights match JS: 0.50, 0.28, 0.14, 0.08 for scales 32, 16, 8, 4.
    Normalised to [-1, 1].
    """
    o1 = _smooth_octave(N, 32, rng)
    o2 = _smooth_octave(N, 16, rng)
    o3 = _smooth_octave(N,  8, rng)
    o4 = _smooth_octave(N,  4, rng)

    noise = 0.50 * o1 + 0.28 * o2 + 0.14 * o3 + 0.08 * o4
    mx    = np.abs(noise).max() + 1e-10
    return (noise / mx).astype(np.float32)


# ── Seed initial conditions ───────────────────────────────────────────────────

def make_seed(seed_type: str, N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build the initial phase field with a planted crystal seed.
    Returns a (N, N) float32 array with values in [0, 1].
    Matches the JS plantSeed() presets exactly.
    """
    p = np.zeros((N, N), dtype=np.float32)
    cx, cy = N // 2, N // 2

    if seed_type == 'hex6':
        arm_len = int(N * 0.12)
        arm_w   = 3
        for a in range(6):
            ang = a * np.pi / 3
            for r in range(arm_len):
                x = int(round(cx + r * np.cos(ang)))
                y = int(round(cy + r * np.sin(ang)))
                for dy in range(-arm_w, arm_w + 1):
                    for dx in range(-arm_w, arm_w + 1):
                        px, py = x + dx, y + dy
                        if 0 <= px < N and 0 <= py < N:
                            d = np.sqrt(dx*dx + dy*dy)
                            p[py, px] = min(1.0, p[py, px] +
                                            float(np.exp(-d*d / (arm_w * 0.4))))
        # Centre disk
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if dx*dx + dy*dy <= 16:
                    p[cy + dy, cx + dx] = 1.0

    elif seed_type == 'hex4':
        arm_len = int(N * 0.11)
        arm_w   = 3
        for a in range(4):
            ang = a * np.pi / 2
            for r in range(arm_len):
                x = int(round(cx + r * np.cos(ang)))
                y = int(round(cy + r * np.sin(ang)))
                for dh in range(-arm_w, arm_w + 1):
                    for dw in range(-arm_w, arm_w + 1):
                        px, py = x + dw, y + dh
                        if 0 <= px < N and 0 <= py < N:
                            d = np.sqrt(dw*dw + dh*dh)
                            p[py, px] = min(1.0, p[py, px] +
                                            float(np.exp(-d*d / (arm_w * 0.4))))
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if dx*dx + dy*dy <= 16:
                    p[cy + dy, cx + dx] = 1.0

    elif seed_type == 'point':
        r = 5
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx*dx + dy*dy <= r*r:
                    p[cy + dy, cx + dx] = 1.0

    elif seed_type == 'ring':
        R = N * 0.10
        w = 3.0
        yy, xx = np.mgrid[0:N, 0:N]
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        diff = np.abs(dist - R)
        mask = diff < w
        p[mask] = np.exp(-diff[mask]**2 / (w * 0.8)).astype(np.float32)

    elif seed_type == 'multi':
        # 6 seeds: centre + 5 random positions in inner half
        positions = [(cx, cy)]
        for _ in range(5):
            ang = rng.random() * 2 * np.pi
            r   = N * (0.08 + rng.random() * 0.14)
            positions.append((
                int(round(cx + r * np.cos(ang))),
                int(round(cy + r * np.sin(ang))),
            ))
        r = 4
        for (sx, sy) in positions:
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx*dx + dy*dy > r*r:
                        continue
                    px, py = sx + dx, sy + dy
                    if 0 <= px < N and 0 <= py < N:
                        d = np.sqrt(dx*dx + dy*dy)
                        p[py, px] = min(1.0, p[py, px] +
                                        float(np.exp(-d*d / (r * 0.6))))
    else:
        raise ValueError(f"Unknown seed_type: {seed_type!r}. "
                         f"Choose from hex6, hex4, point, ring, multi.")

    return np.clip(p, 0.0, 1.0)


# ── Core Kobayashi time-stepper ───────────────────────────────────────────────

def run_kobayashi(
    Te:         float = DEFAULTS['Te'],
    delta:      float = DEFAULTS['delta'],
    j:          int   = DEFAULTS['j'],
    noise_amp:  float = DEFAULTS['noise_amp'],
    pulse:      float = DEFAULTS['pulse'],
    nuc_rate:   float = DEFAULTS['nuc_rate'],
    seed_type:  str   = DEFAULTS['seed_type'],
    N:          int   = DEFAULTS['N'],
    steps:      int   = DEFAULTS['steps'],
    save_every: int   = DEFAULTS['save_every'],
    dt:         float = DEFAULTS['dt'],
    tau:        float = DEFAULTS['tau'],
    K:          float = DEFAULTS['K'],
    alpha:      float = DEFAULTS['alpha'],
    gamma:      float = DEFAULTS['gamma'],
    eps0:       float = DEFAULTS['eps0'],
    seed:       int   = DEFAULTS['seed'],
    verbose:    bool  = True,
) -> dict:
    """
    Run a Kobayashi phase field simulation and return all saved frames.

    Returns
    -------
    dict with keys:
        frames      (T, N, N) float32  full phase field trajectory
        T_field     (T, N, N) float32  temperature field trajectory
        fno_inputs  (T-1, N, N) float32  frame_t   for FNO training
        fno_targets (T-1, N, N) float32  frame_t+1 for FNO training
        params      dict of simulation parameters
        phi_c       float  percolation threshold (-1 if not found)
        t_values    (T,) float32  simulation time at each saved frame
    """
    rng    = np.random.default_rng(seed)
    nf     = build_noise_field(N, rng)
    p      = make_seed(seed_type, N, rng)
    T_arr  = np.zeros((N, N), dtype=np.float32)

    n_frames = steps // save_every
    frames   = np.zeros((n_frames, N, N), dtype=np.float32)
    T_frames = np.zeros((n_frames, N, N), dtype=np.float32)

    frames[0]   = p.copy()
    T_frames[0] = T_arr.copy()
    frame_idx   = 1

    nuc_prob = nuc_rate * 0.0012   # matches JS scaling

    t0 = time.time()

    for step in range(1, steps):

        # Pulse modulation of eps (matches JS: pulseV)
        pulse_v = 1.0 + pulse * np.sin(step * 0.10)

        # ── Phase gradient for anisotropy direction ───────────────────────
        # Finite differences with clamped boundary (matches JS xp/xm/yp/ym)
        dpx = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) * 0.5
        dpy = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) * 0.5

        # Fix boundary artefacts from roll
        dpx[:, 0]    = p[:, 1]  - p[:, 0]
        dpx[:, -1]   = p[:, -1] - p[:, -2]
        dpy[0, :]    = p[1, :]  - p[0, :]
        dpy[-1, :]   = p[-1, :] - p[-2, :]

        theta = np.arctan2(dpy, dpx)

        # Spatially-varying anisotropy direction (our novel stochastic extension)
        noise_angle = nf * noise_amp * np.pi
        eps = eps0 * (1.0 + delta * np.cos(j * (theta + noise_angle))) * pulse_v

        # ── Anisotropic diffusion ─────────────────────────────────────────
        # Laplacian with clamped boundaries
        lap_p = (
            np.roll(p, -1, axis=1) + np.roll(p, 1, axis=1) +
            np.roll(p, -1, axis=0) + np.roll(p, 1, axis=0) - 4.0 * p
        )
        # Clamp boundary rows/cols (no periodic BC)
        lap_p[:, 0]  = lap_p[:, 1]
        lap_p[:, -1] = lap_p[:, -2]
        lap_p[0, :]  = lap_p[1, :]
        lap_p[-1, :] = lap_p[-2, :]

        diff_p = eps * eps * lap_p   # simplified anisotropic diffusion

        # ── Reaction term ─────────────────────────────────────────────────
        m      = (alpha / np.pi) * np.arctan(gamma * (Te - T_arr))
        react  = p * (1.0 - p) * (p - 0.5 + m)
        dp_dt  = (diff_p + react) / tau

        # ── Temperature field ─────────────────────────────────────────────
        lap_T = (
            np.roll(T_arr, -1, axis=1) + np.roll(T_arr, 1, axis=1) +
            np.roll(T_arr, -1, axis=0) + np.roll(T_arr, 1, axis=0) - 4.0 * T_arr
        )
        lap_T[:, 0]  = lap_T[:, 1]
        lap_T[:, -1] = lap_T[:, -2]
        lap_T[0, :]  = lap_T[1, :]
        lap_T[-1, :] = lap_T[-2, :]

        dT_dt = lap_T + K * dp_dt * tau

        # ── Update fields ─────────────────────────────────────────────────
        p_new = np.clip(p + dt * dp_dt, 0.0, 1.0)
        T_new = T_arr + dt * dT_dt

        # ── Stochastic secondary nucleation ───────────────────────────────
        # Near the growth front (low p, high gradient) spontaneous seeds form.
        # This is our key novel extension over the original Kobayashi model.
        if nuc_rate > 0:
            grad_mag = np.sqrt(dpx**2 + dpy**2)
            nuc_mask = (
                (p_new < 0.06) &
                (grad_mag > 0.004) &
                (rng.random((N, N)) < nuc_prob * grad_mag * 25)
            )
            p_new[nuc_mask] = 0.20 + rng.random(nuc_mask.sum()) * 0.25

        p     = p_new.astype(np.float32)
        T_arr = T_new.astype(np.float32)

        # ── Save frame ────────────────────────────────────────────────────
        if step % save_every == 0 and frame_idx < n_frames:
            frames[frame_idx]   = p.copy()
            T_frames[frame_idx] = T_arr.copy()
            if verbose:
                elapsed = time.time() - t0
                pct     = 100.0 * step / steps
                phi     = (p > 0.5).mean()
                print(f"  {pct:5.1f}%  step {step:5d}/{steps}  "
                      f"phi={phi:.3f}  {elapsed:.0f}s elapsed", end='\r')
            frame_idx += 1

    if verbose:
        print(f"\n  Done in {time.time()-t0:.1f}s  |  frames={frame_idx}")

    # ── Percolation threshold ─────────────────────────────────────────────────
    phi_c = _find_percolation(frames[frame_idx - 1])

    # ── Build FNO training pairs ──────────────────────────────────────────────
    fno_inputs  = frames[:frame_idx - 1].copy()   # frame_t
    fno_targets = frames[1:frame_idx].copy()       # frame_t+1

    t_vals = np.arange(frame_idx, dtype=np.float32) * save_every * dt

    return {
        'frames':      frames[:frame_idx],
        'T_field':     T_frames[:frame_idx],
        'fno_inputs':  fno_inputs,
        'fno_targets': fno_targets,
        'phi_c':       np.float32(phi_c),
        't_values':    t_vals,
        'params': dict(
            Te=Te, delta=delta, j=j, noise_amp=noise_amp,
            pulse=pulse, nuc_rate=nuc_rate, seed_type=seed_type,
            N=N, dt=dt, tau=tau, K=K, alpha=alpha,
            gamma=gamma, eps0=eps0,
        ),
    }


def _find_percolation(field: np.ndarray, threshold: float = 0.5) -> float:
    """
    BFS check: does a connected solid cluster span left-to-right?
    Returns the solid fraction at detection, or -1 if not found.
    """
    N    = field.shape[0]
    grid = (field > threshold).astype(np.uint8)
    phi  = grid.mean()

    visited = np.zeros(N * N, dtype=np.uint8)
    queue   = []
    flat    = grid.ravel()

    for y in range(N):
        idx = y * N
        if flat[idx]:
            queue.append(idx)
            visited[idx] = 1

    qi = 0
    while qi < len(queue):
        idx = queue[qi]; qi += 1
        x   = idx % N
        if x == N - 1:
            return float(phi)
        for nb in (idx - N, idx + N,
                   idx - 1 if x > 0   else -1,
                   idx + 1 if x < N-1 else -1):
            if 0 <= nb < N * N and not visited[nb] and flat[nb]:
                visited[nb] = 1
                queue.append(nb)

    return -1.0


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Run one Kobayashi crystal growth simulation.')
    ap.add_argument('--te',         type=float, default=DEFAULTS['Te'],
                    help='Undercooling (equilibrium temperature)')
    ap.add_argument('--delta',      type=float, default=DEFAULTS['delta'],
                    help='Anisotropy strength')
    ap.add_argument('--j',          type=int,   default=DEFAULTS['j'],
                    help='Symmetry order (6=snowflake, 4=square)')
    ap.add_argument('--noise',      type=float, default=DEFAULTS['noise_amp'],
                    help='Spatial noise amplitude')
    ap.add_argument('--pulse',      type=float, default=DEFAULTS['pulse'],
                    help='Growth speed pulse amplitude')
    ap.add_argument('--nuc_rate',   type=float, default=DEFAULTS['nuc_rate'],
                    help='Secondary nucleation rate')
    ap.add_argument('--seed',       type=str,   default=DEFAULTS['seed_type'],
                    help='Seed type: hex6 | hex4 | point | ring | multi')
    ap.add_argument('--steps',      type=int,   default=DEFAULTS['steps'],
                    help='Total simulation steps')
    ap.add_argument('--save_every', type=int,   default=DEFAULTS['save_every'],
                    help='Save one frame every N steps')
    ap.add_argument('--N',          type=int,   default=DEFAULTS['N'],
                    help='Grid size (NxN)')
    ap.add_argument('--rng_seed',   type=int,   default=DEFAULTS['seed'],
                    help='Random seed for reproducibility')
    ap.add_argument('--out',        type=str,   default='kobayashi_test.npz',
                    help='Output .npz file path')
    ap.add_argument('--plot',       action='store_true',
                    help='Show summary plot after simulation')
    args = ap.parse_args()

    print(f"\nKobayashi Solver")
    print(f"  Te={args.te}  delta={args.delta}  j={args.j}")
    print(f"  noise={args.noise}  pulse={args.pulse}  nuc_rate={args.nuc_rate}")
    print(f"  seed={args.seed}  steps={args.steps}  N={args.N}\n")

    result = run_kobayashi(
        Te        = args.te,
        delta     = args.delta,
        j         = args.j,
        noise_amp = args.noise,
        pulse     = args.pulse,
        nuc_rate  = args.nuc_rate,
        seed_type = args.seed,
        steps     = args.steps,
        save_every= args.save_every,
        N         = args.N,
        seed      = args.rng_seed,
        verbose   = True,
    )

    # Save
    np.savez_compressed(
        args.out,
        frames      = result['frames'],
        T_field     = result['T_field'],
        fno_inputs  = result['fno_inputs'],
        fno_targets = result['fno_targets'],
        phi_c       = result['phi_c'],
        t_values    = result['t_values'],
        params      = np.array([
            result['params']['Te'],
            result['params']['delta'],
            float(result['params']['j']),
            result['params']['noise_amp'],
            result['params']['pulse'],
        ], dtype=np.float32),
        seed_type   = np.bytes_(args.seed),
    )
    print(f"\nSaved → {args.out}")
    print(f"  frames:      {result['frames'].shape}")
    print(f"  fno_inputs:  {result['fno_inputs'].shape}")
    print(f"  fno_targets: {result['fno_targets'].shape}")
    print(f"  phi_c:       {result['phi_c']:.4f}")

    if args.plot:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('kob', [
            '#07090f', '#0a2040', '#0ea5e9', '#f59e0b', '#ffffff'])
        frames = result['frames']
        T      = result['T_field']
        show   = np.linspace(0, len(frames)-1, 6, dtype=int)
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        for col, idx in enumerate(show):
            t_val = result['t_values'][idx]
            axes[0, col].imshow(frames[idx], cmap=cmap, origin='lower',
                                vmin=0, vmax=1)
            axes[0, col].set_title(f't={t_val:.1f}', fontsize=9)
            axes[0, col].axis('off')
            axes[1, col].imshow(T[idx], cmap='coolwarm', origin='lower')
            axes[1, col].set_title('T field', fontsize=9)
            axes[1, col].axis('off')
        fig.suptitle(
            f'Kobayashi  Te={args.te}  δ={args.delta}  '
            f'j={args.j}  noise={args.noise}',
            fontweight='bold')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
