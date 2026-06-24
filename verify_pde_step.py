"""
============================================================
  Verify the differentiable PFC stepper (physics step 2)

  The PDE-residual penalty in train_fno.py is only meaningful
  if train_fno.DifferentiablePFCStep reproduces the numpy
  solver in pfc_solver.py -- that is what makes the residual
  ~0 on true data (the solver IS the data-generating process).

  This script runs a short PFC trajectory, then checks that
  one frame interval of the torch stepper matches:
    (a) re-stepping the numpy solver save_every steps, and
    (b) the next SAVED frame of the trajectory.

  Run anywhere torch is installed (locally or in Colab):
      python verify_pde_step.py

  NSF IRES Physical AI Design Program
============================================================
"""

import numpy as np
import torch
from scipy.fft import fft2 as np_fft2

from pfc_solver import PFCConfig, PFCSolver
from train_fno import DifferentiablePFCStep, PDEResidual


def main():
    torch.set_grad_enabled(False)

    # A small multi-grain run (the regime the residual is meant to help).
    cfg = PFCConfig(seed_type="multi", n_seeds=10, r=-0.30,
                    seed_k0=1.0, T=100.0, rng_seed=3)
    res = PFCSolver(cfg).run()
    frames = res.n_all                          # (F, N, N) float32, physical
    print(f"trajectory: {frames.shape[0]} frames, "
          f"dt={cfg.dt} save_every={cfg.save_every} M={cfg.M} r={cfg.r}")
    assert frames.shape[0] >= 3, "need a few frames"

    i = 1                                        # advance frame i -> i+1
    n0 = frames[i].astype(np.float64)

    # (1) numpy ground truth: re-step the solver save_every times from frame i.
    solver = PFCSolver(cfg)
    n, nh = n0.copy(), np_fft2(n0)
    for _ in range(cfg.save_every):
        n, nh = solver.step(n, nh)
    np_next = n                                  # float64

    # (2) torch differentiable stepper, float64 for a tight math comparison.
    stepper = DifferentiablePFCStep(cfg.dx, cfg.dt, cfg.M, cfg.save_every)
    nt = torch.from_numpy(n0).view(1, 1, *n0.shape)        # float64
    rt = torch.tensor([cfg.r], dtype=torch.float64)
    torch_next = stepper.frame(nt, rt)[0, 0].numpy()

    e_solver = np.abs(torch_next - np_next).max()
    e_saved = np.abs(torch_next - frames[i + 1].astype(np.float64)).max()
    print(f"\nmax |torch stepper - numpy re-step| : {e_solver:.2e}   "
          f"(expect < 1e-6: same math)")
    print(f"max |torch stepper - saved frame_i+1|: {e_saved:.2e}   "
          f"(expect < 1e-4: float32 storage)")

    # (3) The residual itself: ~0 on a TRUE pair, large on a wrong prediction.
    mean, std = float(frames.mean()), float(frames.std())
    pen = PDEResidual(stepper, mean, std).double()
    n_in = ((frames[i] - mean) / std)
    n_true = ((frames[i + 1] - mean) / std)
    n_in_t = torch.from_numpy(n_in).view(1, 1, *n_in.shape).double()
    n_true_t = torch.from_numpy(n_true).view(1, 1, *n_true.shape).double()
    rt64 = torch.tensor([cfg.r], dtype=torch.float64)

    res_true = float(pen(n_in_t, n_true_t, rt64))
    # A deliberately wrong "prediction": the input unchanged (identity step).
    res_identity = float(pen(n_in_t, n_in_t, rt64))
    # Another wrong one: a shuffled (orientation-scrambled) field.
    n_wrong_t = torch.from_numpy(np.rot90(n_true, 1).copy()).view(
        1, 1, *n_true.shape).double()
    res_rot = float(pen(n_in_t, n_wrong_t, rt64))

    print(f"\nPDE residual on TRUE next frame   : {res_true:.3e}   (want ~0)")
    print(f"PDE residual on identity (no step): {res_identity:.3e}   (want >> true)")
    print(f"PDE residual on rotated target    : {res_rot:.3e}   (want >> true)")

    ok = (e_solver < 1e-6 and e_saved < 1e-4
          and res_true < 1e-2 and res_identity > 10 * res_true)
    print("\n" + ("PASS: stepper matches solver, residual behaves as intended."
                  if ok else "FAIL: see numbers above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
