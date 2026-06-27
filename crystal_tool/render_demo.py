"""
Render tangible Crystal Studio results headlessly with a real checkpoint.
Produces one summary panel per seed (growth filmstrip + final crystal with
detected atoms/defects + measured characteristics + target check) into
crystal_tool/demo_out/. Mirrors what the Streamlit app shows.

    python crystal_tool/render_demo.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grow as G
import analyze as A
import check as C

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fno_demo.pt")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_out")
os.makedirs(OUT, exist_ok=True)
STEPS = 30


def disk(N, cx, cy, rad):
    Y, X = np.mgrid[0:N, 0:N]
    return (np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) < rad).astype(float)


def panel(title, frames, props, rows, ok, trust, fname):
    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.15])
    idx = np.linspace(0, len(frames) - 1, 6).astype(int)
    for k, fi in enumerate(idx):
        ax = fig.add_subplot(gs[0, k]); ax.imshow(frames[fi], cmap="magma")
        ax.axis("off"); ax.set_title(f"frame {fi}", fontsize=9)
    axf = fig.add_subplot(gs[1, 0:2]); axf.imshow(frames[-1], cmap="magma")
    if len(props["atoms"]):
        axf.scatter(props["atoms"][:, 1], props["atoms"][:, 0], s=4, c="cyan", alpha=0.5)
    if len(props["defect_coords"]):
        axf.scatter(props["defect_coords"][:, 1], props["defect_coords"][:, 0],
                    s=45, facecolors="none", edgecolors="lime", linewidths=1.3)
    axf.set_title("final crystal — atoms (cyan), defects (green)", fontsize=9); axf.axis("off")
    axt = fig.add_subplot(gs[1, 2:4]); axt.axis("off")
    axt.text(0, 1, "MEASURED\n" + "\n".join(A.summary_lines(props)),
             va="top", family="monospace", fontsize=10)
    axc = fig.add_subplot(gs[1, 4:6]); axc.axis("off")
    lines = ["TARGET CHECK"]
    for r in rows:
        lines.append(f"  [{'OK ' if r['ok'] else 'NO '}] {r['criterion']}: "
                     f"want {r['desired']}  got {r['actual']}")
    lines.append(f"\n  MATCH: {'YES' if ok else 'NO'}    confidence {trust*100:.0f}%")
    axc.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10,
             color="darkgreen" if ok else "firebrick")
    fig.suptitle(title, fontsize=14, fontweight="bold"); fig.tight_layout()
    path = os.path.join(OUT, fname); fig.savefig(path, dpi=110); plt.close(fig)
    print(f"  wrote {path}")
    print("   " + " | ".join(A.summary_lines(props)[:3]))


def main():
    m, ck, device = G.load_fno(CKPT)
    print(f"model: fno_demo.pt on {device}  "
          f"(pde_weight={ck['config']['train'].get('pde_weight')})\n")

    seeds = [
        ("Drawn blob -> single crystal", dict(kind="mask", mask=disk(128, 64, 64, 16),
         r=-0.30, n0=-0.285), {"structure": "single", "max_defects": 10, "min_crystallinity": 0.7}),
        ("Multi-seed -> crystal with measured defects", dict(kind="preset",
         seed_type="multi", n_seeds=16, r=-0.30, n0=-0.285),
         {"max_defects": 8, "min_crystallinity": 0.9}),
        ("Two drawn blobs vs strict target", dict(kind="mask",
         mask=np.maximum(disk(128, 44, 50, 13), disk(128, 84, 80, 13)), r=-0.33, n0=-0.285),
         {"structure": "single", "max_defects": 2, "min_crystallinity": 0.8}),
    ]
    for i, (title, seed, target) in enumerate(seeds, 1):
        if seed["kind"] == "mask":
            field0, cfg = G.mask_to_field(seed["mask"], r=seed["r"], n0=seed["n0"], N=128)
        else:
            field0, cfg = G.preset_field(seed["seed_type"], r=seed["r"], n0=seed["n0"],
                                         n_seeds=seed.get("n_seeds", 14))
        frames = G.grow_fno(m, ck, device, field0, steps=STEPS, cfg=cfg)
        props = A.analyze(frames[-1], cfg.dx)
        rows, ok = C.check(props, target)
        _, trust, _ = G.pde_confidence(frames, cfg, device="cpu")
        print(f"[{i}] {title}")
        panel(title, frames, props, rows, ok, trust, f"demo_{i}.png")
        print()


if __name__ == "__main__":
    main()
