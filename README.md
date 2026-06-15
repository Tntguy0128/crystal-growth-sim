# crystal-growth-sim
Base GUI(Need to heavily improve): file:///C:/Users/tntgu/Downloads/crystal_simulator_3.html
# Crystal Growth Simulation — Project Notes

**NSF IRES Physical AI Design Program, Georgia Institute of Technology**
**Researchers:** Ayush Shah & Tobias Li
**Advisor:** Prof. Bo Zhu
**Repo:** crystal-growth-sim

This document summarizes the research so far so that work on the
frontend/GUI (and later, backend integration) can proceed with full
context. It is written for a coding assistant picking up this project
without prior conversation history.

---

## 1. Project Goal

Build an ML-accelerated, computer-graphics-quality crystal growth
simulator. Three things distinguish this project from prior work
(Kim & Lin 2003, Ichimura et al. 2026):

1. **Stochastic naturalism** — prior phase-field crystal simulations
   (including Kim & Lin and Ichimura) are fully deterministic: the same
   seed always produces the exact same crystal, with perfect symmetry.
   We added a spatially-varying anisotropy noise field so every
   simulation produces a unique, naturally-asymmetric crystal — closer
   to how real snowflakes look.

2. **ML surrogate for real-time simulation** — classical phase-field
   solvers are too slow for interactive use. We want a trained model
   that can predict crystal growth in milliseconds so a user can draw
   a seed and watch it grow in real time in a browser.

3. **Browser-based, zero-install delivery** — a single HTML file with
   WebGL rendering, no server or installation required. Ichimura's
   system requires a C++ desktop application.

---

## 2. Physics Model: Kobayashi Phase Field

`solvers/kobayashi.py` implements the Kobayashi phase-field model — the
same family of equations used in Kim & Lin 2003 for dendritic ice/snow
crystal growth. Core equation:

```
tau * dpsi/dt = div(eps(theta)^2 * grad(psi)) + psi(1-psi)(psi - 0.5 + m(T))
eps(theta) = eps0 * (1 + delta * cos(j * (theta + noise(x,y))))
```

- `psi` = phase field (0 = liquid, 1 = solid)
- `j` = number of symmetry arms (j=6 for snowflakes)
- `noise(x,y)` = **our novel addition** — a smooth, multi-octave noise
  field that locally rotates the preferred growth direction, breaking
  perfect symmetry

`dataset/generate_kobayashi.py` runs parameter sweeps (undercooling,
anisotropy strength, noise amplitude, seed type/count) and saves
trajectories as `.npz` files (`traj_NNNN.npz`, each containing a
`frames` array of shape `(T, 256, 256)`).

Dataset generated so far: 72 trajectories, 57 clean (15 failed to
nucleate with low-undercooling point seeds), 80 frames each.

---

## 3. ML Surrogate Attempt #1: FNO (FAILED — important negative result)

`train_fno.py` / `evaluate_fno.py` — we first tried a Fourier Neural
Operator (the same architecture that worked well for a different,
denser phase-field model in week 1).

**Result: complete failure, and the reason is a genuine research
finding.** The Kobayashi field is only ~1.7% solid at any given time
(95-98% empty liquid). The FNO operates on the full 256×256=65,536
pixel grid. The trivial loss-minimizing strategy is "predict zero
everywhere," which the FNO converged to. Attempts to fix this with
weighted MSE (50x penalty on crystal pixels) caused the opposite
failure: the FNO filled the *entire* domain with a smeared, wrong
pattern. Rollout mass-conservation error reached 40-60x.

**Conclusion:** FNO (and grid-based neural operators generally) are
unsuited to sparse-interface problems. This motivated the pivot to a
graph-based representation operating only on the crystal *boundary*.

---

## 4. ML Surrogate Attempt #2: GNN on Boundary Graph (PARTIAL SUCCESS)

### 4.1 Core idea

Instead of the full 65,536-pixel grid, extract the crystal's boundary
as a set of ~512 points (a graph), and train a Graph Neural Network to
predict the per-point displacement `(dx, dy)` from one timestep to the
next. Every node is on the boundary — no wasted capacity on empty
liquid.

Files:
- `solvers/gnn_boundary.py` — boundary extraction, graph construction,
  feature computation, field reconstruction (for visualization)
- `solvers/gnn_model.py` — `CrystalGNN`: 3-layer message-passing
  network, ~82k params (hidden=64, layers=3)
- `training/train_gnn.py` — training script
- `training/evaluate_gnn.py` — evaluation + autoregressive rollout

### 4.2 Multi-contour tracking (v2)

Early version only tracked the single longest contour, so multi-seed
trajectories (multiple separate crystals growing in one frame)
collapsed into one blob. Fixed by:
- Extracting ALL contours (capped at `MAX_CONTOURS=24`), allocating the
  512-point budget proportionally to arc length (min 16 pts/contour,
  falling back to pure-proportional allocation with no minimum if
  `n_contours * 16 > 512` — this also fixed an infinite-loop bug in the
  node-allocation rounding logic)
- 9D node features (was 8D): added `contour_id` as feature 8
- Per-contour curvature computation (log1p-compressed — raw curvature
  range was [0.03, 1896], wildly unstable)
- `match_contours()` — nearest-centroid matching between frame t and
  t+1 contours, so each contour gets its own correctly-tracked
  displacement target
- Result: 86% of training graphs (3493/4075) have multiple contours,
  mean 12.96 contours/graph — v1 was discarding most of this structure

### 4.3 Training results

GPU training (T4, ~3.5s/epoch, 100 epochs, ~6 min total):
- One-step boundary MSE: **9.2e-4** (excellent — proves the
  architecture is fundamentally correct)
- 82k params (hidden=64, layers=3), up from 14.8k (hidden=32, layers=2)

### 4.4 Autoregressive rollout: the hard part

40-step rollout (predict frame, feed prediction back as input, repeat)
revealed a sequence of compounding-error failure modes. **Each was
diagnosed and fixed with a verified before/after improvement — this
progression is itself a useful methods narrative:**

1. **Field-roundtrip inflation.** Original rollout reconstructed a
   phase field each step (via `cv2.fillPoly`) and re-extracted contours
   from it. `fillPoly` rounds off concave dendrite notches, inflating
   area ~6-8% per step even with a *perfect* oracle prediction — proven
   with a synthetic test. **Fix:** rollout now stays entirely in
   boundary-point space; `contour_id` and points-per-contour are FIXED
   for the whole rollout (field reconstruction happens only at the end,
   for display/MSE, never fed back). This also structurally guarantees
   separate seeds can never merge.

2. **Translation drift.** Even with merging fixed, multi-seed crystals
   visibly slid across the canvas / collapsed toward each other over 40
   steps. Cause: small systematic non-zero *mean* displacement per
   contour (true radial growth has ~zero mean by symmetry, so any
   non-zero mean is rollout error). **Fix:** subtract each contour's
   mean predicted displacement before applying it (pins each crystal's
   centroid). Verified: a deliberately-biased synthetic model showed
   exactly 0.00000 centroid drift after the fix.

3. **Arm collapse / shrinking.** Single-seed snowflakes collapsed from
   6 arms down to 1-2 dominant arms; multi-seed crystals shrank back
   down after initial growth. **Fix:** decompose each point's
   displacement into radial (relative to its contour's centroid) +
   tangential components; clip radial component to >= 0. This
   guarantees "solid never becomes liquid" — each contour's minimum
   radius is monotonically non-decreasing. Verified against both an
   adversarial shrink-model (radius held exactly flat, never negative)
   and normal growth (proceeds correctly, matches expected ~8%/step
   compounding to within numerical precision).

4. **Runaway spike.** With shrinkage blocked, error concentrated into
   single points growing into long unbounded spikes (one arm extending
   far beyond the others). **Fix:** bound each point's per-step
   displacement by its contour's local point spacing (ring-edge
   length), self-calibrating to contour size. First attempt used MEAN
   edge length with multiplier=2.0 — barely helped, because the spike
   itself inflates the mean (positive feedback loop). **Second
   attempt** uses MEDIAN edge length (robust to the one outlier) with
   multiplier=1.0 — verified 7x reduction in spike magnitude over 40
   steps in a synthetic test (0.378 -> 0.056).

### 4.5 Current status (end of week 2, Monday)

After all four fixes: traj_0071 (multi-seed) improved from field MSE
1.33e-2 -> 6.88e-3 (2x). traj_0067 (multi-seed, contains a hard case)
improved from 3.34e-2 -> 2.40e-2 (~30%), but still shows a visible
elongated spike on one seed — the *direction* of that seed's growth is
wrong, which clamping can bound but not correct.

**Conclusion: the GNN-on-boundary-graph architecture is proven correct
(one-step MSE 9.2e-4, beats FNO which fails completely), and we
documented + fixed four distinct rollout-stability failure modes with
verified improvements. Full 40-step open-ended rollout remains
imperfect — this is consistent with known long-horizon instability in
one-step-trained autoregressive GNN simulators in the literature (e.g.
Sanchez-Gonzalez et al. 2020 needed millions of training steps and
still documented degradation).**

`runs/gnn_v3/best_gnn.pt` is the current best checkpoint (trained with
noise injection, `--noise_std 0.004`). Graph dataset:
`kobayashi_graphs_v2/graphs.pt` (4075 graphs).

---

## 5. Planned ML Surrogate Attempt #3: Local-Crop CNN (NOT YET STARTED)

**Hypothesis:** a small CNN/U-Net operating on a cropped window (e.g.
48x48 or 64x64 px) of the phase field, centered on one growing seed,
should avoid BOTH prior failure modes:

- Avoids the FNO sparsity problem: a crop centered on a seed is mostly
  crystal + near-boundary, not 95% empty
- Avoids the GNN's "independent point can run away" problem:
  convolution enforces local spatial coherence by construction — a
  thin unsupported spike is a high-frequency pattern that conv nets are
  bad at producing

**Scope if pursued:** new dataset (per-seed crops + crop-center
tracking across frames), new training script, new
rollout/reconstruction logic (paste predicted crop back into full
field at the tracked center). Comparable effort to the GNN's week-2
arc. Multi-seed inference = one forward pass per active seed.

**Sequencing decision:** GUI work first (this document's handoff
point), CNN second, integration third. The GUI's model-calling
interface should be a single function:

```
predict_next_frame(current_state) -> next_state
```

so that swapping GNN -> CNN later (or running both, or falling back to
the classical solver) is a backend change, not a GUI rewrite.

---

## 6. GUI / Frontend — Current State and Goals

### 6.1 Current state

`crystal_simulator.html` (v3) — single self-contained file, WebGL
rendering, runs the Kobayashi solver live in JavaScript (not yet using
any trained model). Features:
- 3 color palettes (thin-film interference, thermal ice, deep ocean)
- Seed presets: 6-arm star, 4-arm cross, ring, multi-seed; freehand
  draw/erase
- 7 physics sliders: undercooling, symmetry arms, anisotropy, disorder,
  nucleation, pulse rhythm, speed
- Percolation detection (BFS) with flash animation + solid-fraction
  display
- GLSL fragment shader: grain-boundary glow via gradient magnitude,
  interior ridge detection via Laplacian, motion highlights

### 6.2 Goals for this round of frontend work

From the week 3 plan (game-quality GUI):
- **Normal mapping** from the phase-field gradient, to make the flat
  2D simulation read as pseudo-3D (proper highlights/shadows on
  dendrite arms)
- **Subsurface scattering approximation** for translucent-ice look
- **Bokeh background** + specular highlights
- **Three crystal types**: snowflake (Kobayashi, existing), coral
  (diffusion-limited aggregation, new), honeycomb (competitive grain
  growth, new)
- **Crystal max-size / growth-cap**: real snowflakes stop growing
  (finite supercooled-water supply). Proposed fix (NOT YET
  IMPLEMENTED): radial undercooling falloff — make `Te` (equilibrium
  temperature) decrease smoothly with distance from the seed center,
  so growth naturally tapers near `max_radius` instead of hitting a
  hard wall. One-line change to the JS `kobStep()` function. This is a
  GUI-only change, independent of the trained models — should NOT
  require touching the Python solver or retraining.

### 6.3 Backend integration (future, after CNN)

Plan: FastAPI server wrapping `predict_next_frame()`. GUI sends current
boundary/state, gets back predicted next state, renders it. Until the
CNN is ready and the GNN's rollout stability is acceptable, the GUI
should continue using the JS Kobayashi solver as the "ground truth"
real-time simulation — ML surrogates are an additive "fast preview"
feature, not (yet) a replacement.

---

## 7. Key File Reference

| File | Purpose |
|---|---|
| `solvers/kobayashi.py` | Python phase-field solver (matches JS GUI physics) |
| `dataset/generate_kobayashi.py` | Trajectory dataset generator / parameter sweep |
| `solvers/gnn_boundary.py` | Boundary extraction, graph construction, field reconstruction |
| `solvers/gnn_model.py` | `CrystalGNN` model + `BoundaryLoss` |
| `training/train_gnn.py` | GNN training (supports `--noise_std` for rollout robustness) |
| `training/evaluate_gnn.py` | Evaluation + autoregressive rollout (centroid-pinning, monotonic-radius, bounded-step) |
| `train_fno.py` / `evaluate_fno.py` | FNO attempt (documented failure, kept for the negative result) |
| `crystal_simulator.html` | WebGL frontend (this round's focus) |
| `runs/gnn_v3/best_gnn.pt` | Current best GNN checkpoint |
| `kobayashi_graphs_v2/graphs.pt` | Training graph dataset (4075 graphs) |

---

## 8. Immediate Next Steps (in order)

1. **GUI improvements** (this handoff): normal mapping, subsurface
   scattering, bokeh, three crystal types, growth-cap via radial
   undercooling falloff
2. **Local-crop CNN**: new ML surrogate attempt, as described in
   Section 5
3. **Backend integration**: FastAPI wrapper, GUI <-> model connection,
   speed benchmark (GNN/CNN vs classical solver)
4. **Write-up**: methods section documenting the FNO negative result
   and the four-stage GNN rollout-stabilization process — both are
   genuine findings worth presenting in full, not just the final numbers
