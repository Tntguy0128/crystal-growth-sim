# Crystal Studio

Draw a seed → predict the crystal with the FNO → **measure its physical
characteristics** → check them against the structure you wanted.

This is the interactive *tool* layer on top of the PFC/FNO project. It is kept
in its own folder so it never collides with the partner `gui/`.

## Layers (each usable on its own)

| File | What it does | Deps | Status |
|------|--------------|------|--------|
| `analyze.py` | Measure a field: grain count, orientation, defects, lattice wavelength, crystallinity | numpy, scipy | **tested** (synthetic single-/bi-crystal + liquid) |
| `check.py` | Compare measured characteristics to a target spec | python | **tested** |
| `grow.py` | mask → FNO rollout, solver verification, PDE-residual confidence | torch + repo | written, runs with a checkpoint |
| `app.py` | Streamlit UI tying it together | streamlit + above | written, runs with deps |

The measurement engine is **model-free** — it runs on any field, so it was
validated on synthetic crystals (and can be validated on exact solver output)
*before* it is ever pointed at an FNO prediction.

## Run

```bash
pip install -r crystal_tool/requirements.txt
streamlit run crystal_tool/app.py
```

In the sidebar, point "Checkpoint" at a trained **density-only** `best.pt`
(e.g. from the grain-rich runs). Draw a seed, click **Grow crystal (FNO)**,
read the measured characteristics, then **Check against target**. Use
**Verify with real solver** for ground truth on the same seed.

## What works today vs. what's next

- **Works now:** the full pipeline — drawn mask → custom_mask initial field →
  FNO rollout → analysis → target check → optional solver verification, with a
  physics-consistency confidence score.
- **Known limitation:** freehand seeds are partly out-of-distribution for a
  model trained on `hex`/`point`/`multi` seeds, and the model's orientation
  errors carry into orientation/grain readings. The fix is a `custom_mask`
  fine-tune (generate drawn-style seeds with `grow.mask_to_field` + the solver,
  retrain). The "Verify with real solver" button is the honest backstop in the
  meantime.
- **Later:** inverse design — because the FNO is differentiable, backprop from a
  desired characteristic to a seed.

## Self-tests

```bash
python crystal_tool/analyze.py   # synthetic crystal measurements
python crystal_tool/check.py     # target-matching logic
```
