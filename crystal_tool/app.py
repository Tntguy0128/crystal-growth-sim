"""
============================================================
  Crystal Studio — draw a seed, predict the crystal, measure it

  A Streamlit tool that ties the pieces together:
    draw a seed  ->  FNO grows the crystal (fast)  ->  analyze()
    measures it  ->  check() compares it to your target spec.

  Plus the two things that make it trustworthy:
    * a confidence score (how physical the prediction is), and
    * "Verify with the real solver" for ground truth.

  Run:
    pip install -r crystal_tool/requirements.txt
    streamlit run crystal_tool/app.py

  Point the sidebar at a trained density-only checkpoint (best.pt).

  NSF IRES Physical AI Design Program
============================================================
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import analyze as A
import check as C
import grow as G


st.set_page_config(page_title="Crystal Studio", layout="wide")
N = 128


# ----------------------------------------------------------------------------
#  helpers
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def get_model(ckpt_path):
    return G.load_fno(ckpt_path)


def render(field, atoms=None, defects=None, title=""):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(field, cmap="magma", interpolation="nearest")
    if atoms is not None and len(atoms):
        ax.scatter(atoms[:, 1], atoms[:, 0], s=6, c="cyan", alpha=0.6, linewidths=0)
    if defects is not None and len(defects):
        ax.scatter(defects[:, 1], defects[:, 0], s=40, facecolors="none",
                   edgecolors="lime", linewidths=1.5)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


def show_props(props, container):
    container.markdown("\n".join("- " + ln for ln in A.summary_lines(props)))


# ----------------------------------------------------------------------------
#  sidebar — model + physics controls
# ----------------------------------------------------------------------------
st.sidebar.header("Model & physics")
ckpt = st.sidebar.text_input(
    "Checkpoint (best.pt)", value="fno_demo.pt",
    help="Path to a trained density-only checkpoint (repo root). "
         "fno_demo.pt is the grain-trained model.")
r = st.sidebar.slider("r  (quench depth)", -0.45, -0.20, -0.30, 0.01)
n0 = st.sidebar.slider("n0 (mean density)", -0.35, -0.20, -0.285, 0.005)
steps = st.sidebar.slider("rollout steps", 5, 60, 39, 1)
seed_k0 = st.sidebar.number_input("seed wavenumber k0", value=1.0, step=0.1)

st.sidebar.markdown("**Seed diversity**")
orient_mode = st.sidebar.radio(
    "Lattice orientation", ["Random each grow", "Fixed angle"],
    help="The crystal angle is what makes each drawing look different. "
         "Random gives a fresh orientation every time you grow.")
fixed_angle = st.sidebar.slider("Fixed angle (deg)", 0, 60, 0,
                                disabled=(orient_mode != "Fixed angle"))
per_region = st.sidebar.checkbox(
    "Per-region orientations (polycrystal seed)", value=False,
    help="Each separate blob/stroke gets its own orientation, so multi-part "
         "drawings seed grain boundaries.")

st.title("Crystal Studio")
st.caption("Draw a seed → predict the crystal → measure it → check it against your target.")

col_draw, col_out = st.columns([1, 1])

# ----------------------------------------------------------------------------
#  left — draw the seed
# ----------------------------------------------------------------------------
with col_draw:
    st.subheader("1 · Choose a seed")
    seed_source = st.radio(
        "Seed source",
        ["Draw", "Preset · multi (polycrystal)", "Preset · point (single)"],
        help="Draw your own, or use a built-in seed as a reliable fallback.")
    st.caption("White = where crystal nucleates. Draw blobs, lines, patterns.")
    canvas = st_canvas(
        fill_color="#FFFFFF", stroke_width=14, stroke_color="#FFFFFF",
        background_color="#000000", height=320, width=320,
        drawing_mode="freedraw", key="seed_canvas",
    )
    grow_btn = st.button("Grow crystal (FNO)", type="primary")
    verify_btn = st.button("Verify with real solver")

# ----------------------------------------------------------------------------
#  growth + analysis
# ----------------------------------------------------------------------------
def run_growth(use_solver):
    if not use_solver and not os.path.exists(ckpt):
        st.error(f"Checkpoint not found: `{ckpt}`. Download "
                 "**fno_grains_tuned.pt** from Drive into the repo root "
                 "(or fix the sidebar path).")
        return
    # Build the initial field from the chosen seed source.
    if seed_source == "Draw":
        if canvas.image_data is None or canvas.image_data[..., :3].sum() == 0:
            st.warning("Draw a seed first, or pick a preset.")
            return
        orientation = fixed_angle if orient_mode == "Fixed angle" else "random"
        field0, cfg = G.mask_to_field(canvas.image_data, r=r, n0=n0, N=N,
                                      seed_k0=seed_k0, orientation=orientation,
                                      per_region=per_region)
    elif "multi" in seed_source:
        field0, cfg = G.preset_field("multi", r=r, n0=n0, N=N, seed_k0=seed_k0)
    else:
        field0, cfg = G.preset_field("point", r=r, n0=n0, N=N, seed_k0=seed_k0)
    with col_out:
        st.subheader("2 · Crystal growth")
        ph = st.empty()
        if use_solver:
            frames = G.grow_solver(cfg, steps=steps)
            tag = "solver (ground truth)"
        else:
            model, ck, device = get_model(ckpt)
            frames = G.grow_fno(model, ck, device, field0, steps=steps, cfg=cfg)
            tag = "FNO prediction"
        for f in frames:
            ph.pyplot(render(f, title=tag))
            time.sleep(0.04)
        final = frames[-1]
        props = A.analyze(final, cfg.dx)
        ph.pyplot(render(final, atoms=props["atoms"],
                         defects=props["defect_coords"],
                         title=f"{tag} — final frame"))
        st.session_state["props"] = props
        st.session_state["cfg_dx"] = cfg.dx
        st.session_state["tag"] = tag

        st.subheader("3 · Measured characteristics")
        show_props(props, st)

        if not use_solver:
            mean_res, trust, _ = G.pde_confidence(frames, cfg, device="cpu")
            badge = "🟢 high" if trust > 0.7 else "🟡 medium" if trust > 0.4 else "🔴 low"
            st.metric("Prediction confidence (physics-consistency)",
                      f"{trust*100:.0f}%  {badge}",
                      help="From the differentiable-solver PDE residual; "
                           "low means the FNO drifted off the physics — "
                           "verify with the real solver.")


if grow_btn:
    run_growth(use_solver=False)
elif verify_btn:
    run_growth(use_solver=True)

# ----------------------------------------------------------------------------
#  target check (uses the last analysis in session)
# ----------------------------------------------------------------------------
st.divider()
st.subheader("4 · Does it match what you wanted?")
tc = st.columns(5)
want_structure = tc[0].selectbox("Structure", ["(any)", "single", "poly"])
want_maxdef = tc[1].number_input("Max defects", value=10, step=1)
want_mincryst = tc[2].slider("Min crystallinity", 0.0, 1.0, 0.7, 0.05)
want_orient = tc[3].number_input("Orientation °", value=-1.0, step=1.0,
                                 help="-1 to ignore")
want_lambda = tc[4].number_input("Wavelength", value=-1.0, step=0.1,
                                  help="-1 to ignore")

if st.button("Check against target"):
    if "props" not in st.session_state:
        st.warning("Grow a crystal first.")
    else:
        target = {"max_defects": int(want_maxdef),
                  "min_crystallinity": float(want_mincryst)}
        if want_structure != "(any)":
            target["structure"] = want_structure
        if want_orient >= 0:
            target["orientation_deg"] = float(want_orient)
        if want_lambda > 0:
            target["wavelength"] = float(want_lambda)
        rows, all_pass = C.check(st.session_state["props"], target)
        st.table(rows)
        if all_pass:
            st.success("✓ The predicted crystal matches your target.")
        else:
            st.error("✗ The predicted crystal does not match your target.")
