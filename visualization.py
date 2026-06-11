"""
============================================================
  Optional visualization for PFC runs

  Deliberately minimal: the pipeline's job is clean data
  generation, not figures. Everything here is gated by
  cfg.save_plots / cfg.save_animation and saved to disk
  (Agg backend, safe in headless runs and worker processes).

  NSF IRES Physical AI Design Program
============================================================
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless-safe (Colab, ProcessPoolExecutor)
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap

# Same navy-(liquid) -> gold-(crystal) map as the baseline, for continuity.
CMAP_PFC = LinearSegmentedColormap.from_list("pfc", [
    (0.00, "#0a1628"), (0.35, "#1a3a6b"), (0.58, "#c8a84b"),
    (0.78, "#e8d5a3"), (1.00, "#ffffff"),
])


def _color_limits(n_all):
    """Robust color limits from the final frame (2nd/98th percentile)."""
    return np.percentile(n_all[-1], 2), np.percentile(n_all[-1], 98)


def plot_growth_sequence(result, cfg, path, n_show: int = 6):
    """2x3 grid of frames spanning the run (like the baseline's Cell 7)."""
    n_all, t_vals = result.n_all, result.t_vals
    idx = np.linspace(0, n_all.shape[0] - 1, min(n_show, n_all.shape[0]), dtype=int)
    vmin, vmax = _color_limits(n_all)

    rows, cols = 2, (len(idx) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 5.4))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, idx):
        ax.imshow(n_all[i], cmap=CMAP_PFC, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"t = {t_vals[i]:.0f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(idx):]:
        ax.axis("off")
    fig.suptitle(f"PFC growth — r={cfg.r}, n0={cfg.n0}, seed={cfg.seed_type}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_free_energy(result, cfg, path):
    """Free energy per unit area vs time — should monotonically decrease."""
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    F_per_area = result.free_energy / cfg.L ** 2
    ax.plot(result.t_vals, F_per_area, color="#1a3a6b", lw=2)
    ax.fill_between(result.t_vals, F_per_area, alpha=0.15, color="#1a3a6b")
    ax.set_xlabel("time t"); ax.set_ylabel(r"$\mathcal{F}[n] / L^2$")
    ax.set_title(f"Free energy dissipation (drop {result.energy_drop_percent:.1f}%)",
                 fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def save_growth_animation(result, cfg, path, fps: int = 10):
    """GIF of the full trajectory (slow to render — off by default)."""
    n_all, t_vals = result.n_all, result.t_vals
    vmin, vmax = _color_limits(n_all)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(n_all[0], cmap=CMAP_PFC, vmin=vmin, vmax=vmax, origin="lower")
    txt = ax.text(0.03, 0.95, "t = 0", transform=ax.transAxes, color="white",
                  fontweight="bold", va="top",
                  bbox=dict(boxstyle="round", fc="#1a3a6b", alpha=0.7))
    ax.set_title(f"r={cfg.r}, n0={cfg.n0}, seed={cfg.seed_type}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    def update(i):
        im.set_data(n_all[i])
        txt.set_text(f"t = {t_vals[i]:.0f}")
        return im, txt

    ani = animation.FuncAnimation(fig, update, frames=n_all.shape[0],
                                  interval=1000 // fps, blit=True)
    ani.save(path, writer="pillow", fps=fps, dpi=100)
    plt.close(fig)
    return path


def render_outputs(result, cfg, stem: str):
    """Write whatever cfg asks for; `stem` is the path without extension."""
    written = []
    if cfg.save_plots:
        written.append(plot_growth_sequence(result, cfg, stem + "_growth.png"))
        written.append(plot_free_energy(result, cfg, stem + "_energy.png"))
    if cfg.save_animation:
        written.append(save_growth_animation(result, cfg, stem + ".gif"))
    return written
