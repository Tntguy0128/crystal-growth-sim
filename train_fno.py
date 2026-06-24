"""
============================================================
  Train a Fourier Neural Operator on PFC trajectories
                                              (PFC ONLY)

  This is the PHASE FIELD CRYSTAL trainer that lives at the
  repo root, next to the PFC solver/pipeline. It is separate
  from training/train_fno.py, which is the Kobayashi-oriented
  trainer (crystal-weighted loss) maintained on its own track.

  Objective:  minimize  MSE( FNO(n_t) , n_t+1 )
  Pure data loss -- no physics penalty terms. Physical
  structure enters only through the ARCHITECTURE flags in
  config.yaml (model.predict_delta, model.enforce_mass),
  which keep the optimization target a plain MSE.

  NSF IRES Physical AI Design Program
============================================================

USAGE
-----
    python train_fno.py --config config.yaml

What it does:
  * loads + splits the dataset (by trajectory or by config; no leakage)
  * trains the FNO with Adam + cosine LR schedule
  * validates every epoch (one-step MSE)
  * every `rollout_val_every` epochs, also runs a free-running
    AUTOREGRESSIVE ROLLOUT on validation trajectories and logs the
    multi-step MSE + final mass drift -- the metrics that actually
    matter for a surrogate, watched while training instead of only
    discovered at evaluation time
  * checkpoints best-val and most-recent models
  * early-stops when one-step validation stops improving
  * logs everything to TensorBoard (if installed)

The checkpoint stores the model weights AND the normalization stats +
config + file splits, so evaluate_fno.py reproduces the exact
preprocessing and evaluates on the held-out test trajectories.
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import yaml

from dataset import (build_datasets, inspect_trajectory, _load_manifest,
                     _run_id_from_path, _COND_COLUMNS)
from fno_model import build_model, count_parameters


# ----------------------------------------------------------------------------
#  Setup utilities
# ----------------------------------------------------------------------------
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def pick_device(requested):
    """Resolve 'auto' to cuda -> mps -> cpu, or honor an explicit choice."""
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer, cfg):
    t = cfg["train"]
    kind = t.get("scheduler", "none")
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t["epochs"])
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=t["step_size"], gamma=t["gamma"])
    return None


# ----------------------------------------------------------------------------
#  Loss: MSE, optionally weighted toward dynamically active / complex regions
# ----------------------------------------------------------------------------
class PFCLoss(nn.Module):
    """
    Mean squared error with an optional per-pixel weight that emphasizes the
    regions where the surrogate actually fails (growth fronts, grain
    boundaries) -- a plain MSE barely notices a thin defect line because it is
    a tiny fraction of the pixels.

    weight_mode:
      "none"     -- plain MSE (the clean baseline).
      "delta"    -- weight by |target - prev|, the per-pixel CHANGE from the
                    previous frame. Large where the crystal is actively
                    growing / reorganizing; ~0 in settled regions. Directly
                    targets the "locks in the wrong structure too early"
                    failure. Needs `prev` (falls back to plain MSE without it).
      "gradient" -- weight by the spatial gradient magnitude of the target
                    (structure-rich regions). Note: a hex crystal oscillates
                    everywhere, so this is less selective than "delta".

    The weight is w = 1 + alpha * (indicator / per-sample max indicator), so
    alpha=0 recovers plain MSE and the weighting is scale-free.
    """

    def __init__(self, weight_mode="none", alpha=5.0):
        super().__init__()
        self.weight_mode = weight_mode
        self.alpha = float(alpha)

    @staticmethod
    def _grad_mag(f):
        gx = F.pad(f[..., 1:, :] - f[..., :-1, :], (0, 0, 0, 1))
        gy = F.pad(f[..., :, 1:] - f[..., :, :-1], (0, 1, 0, 0))
        return torch.sqrt(gx * gx + gy * gy + 1e-12)

    def forward(self, pred, target, prev=None):
        se = (pred - target) ** 2
        if self.weight_mode == "none" or self.alpha == 0.0 \
                or (self.weight_mode == "delta" and prev is None):
            return se.mean()
        ind = self._grad_mag(target) if self.weight_mode == "gradient" \
            else (target - prev).abs()
        w = 1.0 + self.alpha * ind / (ind.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        return (w * se).mean()


# ----------------------------------------------------------------------------
#  Free-energy dissipation penalty (physics-informed, step 1)
# ----------------------------------------------------------------------------
class FreeEnergyDissipation(nn.Module):
    """
    A thermodynamic soft constraint: PFC is a gradient flow, so its free
    energy F[n] can only DECREASE in time. This penalizes any predicted step
    whose free energy is higher than the frame it evolved from.

    The free energy is the same functional the solver/sanity-checks use
    (see pfc_solver.compute_free_energy), evaluated spectrally:

        F[n] = sum( n(lap n + 1/2 lap^2 n) + 1/2 (1+r) n^2 + 1/4 n^4 ) dx^2

    Two practical points:
      * The field is z-score normalized during training, but F is defined on
        the PHYSICAL field, so we denormalize (n = n_norm*std + mean) first.
      * We penalize the RELATIVE increase relu((F_pred - F_prev)/|F_prev|),
        which is dimensionless and ~O(1), so the weight is easy to set and the
        prefactor dx^2 cancels out. On clean ground-truth data this term is ~0
        (F genuinely decreases), so it never fights correct predictions -- it
        only activates when the model proposes an energy-increasing, and hence
        unphysical, step. This is an inequality, so it is robust to the coarse
        frame spacing dt (unlike a full PDE residual).
    """

    def __init__(self, norm_mean, norm_std, dx):
        super().__init__()
        self.mean = float(norm_mean)
        self.std = float(norm_std)
        self.dx = float(dx)
        self._k2 = None          # cached |k|^2 grid (built lazily per device/size)

    def _k2_grid(self, H, W, device):
        if (self._k2 is None or tuple(self._k2.shape) != (H, W)
                or self._k2.device != device):
            kx = torch.fft.fftfreq(H, d=self.dx, device=device) * 2 * math.pi
            ky = torch.fft.fftfreq(W, d=self.dx, device=device) * 2 * math.pi
            KX, KY = torch.meshgrid(kx, ky, indexing="ij")
            self._k2 = KX ** 2 + KY ** 2
        return self._k2

    def free_energy(self, n_norm, r):
        """n_norm: (B,1,H,W) normalized field; r: (B,) physical -> F: (B,)."""
        n = n_norm * self.std + self.mean
        B, _, H, W = n.shape
        k2 = self._k2_grid(H, W, n.device)
        nh = torch.fft.fft2(n)
        lap_n = torch.fft.ifft2(-k2 * nh).real
        laplap_n = torch.fft.ifft2(k2 ** 2 * nh).real
        rr = r.view(B, 1, 1, 1)
        integ = n * (lap_n + 0.5 * laplap_n) + 0.5 * (1.0 + rr) * n ** 2 + 0.25 * n ** 4
        return integ.sum(dim=(-2, -1)).squeeze(1) * self.dx ** 2

    def forward(self, pred_norm, prev_norm, r):
        F_pred = self.free_energy(pred_norm, r)
        F_prev = self.free_energy(prev_norm, r)
        return torch.relu((F_pred - F_prev) / (F_prev.abs() + 1e-8)).mean()


# ----------------------------------------------------------------------------
#  Differentiable PFC solver step + PDE-residual penalty (physics-informed,
#  step 2). Step 1 enforced only ONE consequence of the physics (energy must
#  not rise); step 2 enforces the WHOLE governing equation, using the solver
#  itself as a teacher.
# ----------------------------------------------------------------------------
class DifferentiablePFCStep(nn.Module):
    """
    A torch re-implementation of the PFC solver's semi-implicit pseudo-spectral
    update (see pfc_solver.PFCSolver.step), so the solver can run INSIDE the
    training graph.

    One saved frame of the dataset spans `substeps` solver steps of size `dt`
    (the generator's save_every), so `frame()` applies the update that many
    times -- reproducing exactly the integrator that produced the ground-truth
    next frame. Because every op (fft2/ifft2, complex multiply, divide) is
    differentiable, autograd flows through the whole chain; in the loss below we
    nonetheless DETACH the output (use it as a fixed teacher) to avoid
    backpropagating through `substeps` stiff semi-implicit steps.

    The PFC operators depend on the per-sample temperature r, so the linear
    eigenvalue L_op and the semi-implicit denominator are rebuilt per batch from
    r. The |k|^2 grid and the 2/3 dealiasing mask depend only on the grid and
    are cached per device/size (same lazy pattern as FreeEnergyDissipation).
    """

    def __init__(self, dx, dt, M, substeps):
        super().__init__()
        self.dx = float(dx)
        self.dt = float(dt)
        self.M = float(M)
        self.substeps = int(substeps)
        self._k2 = None
        self._dealias = None

    def _grids(self, H, W, device):
        if (self._k2 is None or tuple(self._k2.shape) != (H, W)
                or self._k2.device != device):
            kx = torch.fft.fftfreq(H, d=self.dx, device=device) * 2 * math.pi
            ky = torch.fft.fftfreq(W, d=self.dx, device=device) * 2 * math.pi
            KX, KY = torch.meshgrid(kx, ky, indexing="ij")
            self._k2 = KX ** 2 + KY ** 2
            kmax = kx.abs().max() * 2.0 / 3.0          # 2/3-rule dealiasing cutoff
            self._dealias = ((KX.abs() < kmax) & (KY.abs() < kmax)).to(self._k2.dtype)
        return self._k2, self._dealias

    def frame(self, n, r):
        """
        Advance a PHYSICAL field n: (B,1,H,W) by one saved-frame interval
        (`substeps` semi-implicit steps). r: (B,) physical temperatures.
        Mirrors PFCSolver.step exactly:  NL_hat = -M k^2 fft(n^3) (dealiased),
        n_hat <- (n_hat + dt NL_hat)/(1 - dt L_op),  n <- ifft(n_hat).
        """
        B, _, H, W = n.shape
        k2, dealias = self._grids(H, W, n.device)
        rr = r.view(B, 1, 1, 1)
        # Linear PFC eigenvalue  L = -M k^2 (k^4 - 2k^2 + 1 + r), diagonal in k.
        L_op = -self.M * k2 * (k2 ** 2 - 2.0 * k2 + 1.0 + rr)        # (B,1,H,W)
        denom = 1.0 / (1.0 - self.dt * L_op)                        # semi-implicit
        n_hat = torch.fft.fft2(n)
        for _ in range(self.substeps):
            NL_hat = -(self.M * k2 * torch.fft.fft2(n ** 3)) * dealias
            n_hat = (n_hat + self.dt * NL_hat) * denom
            n = torch.fft.ifft2(n_hat).real
        return n


class PDEResidual(nn.Module):
    """
    Physics-informed consistency loss (step 2): the FULL PFC equation, enforced
    through the differentiable solver rather than the free-energy inequality of
    step 1.

    For a predicted step we ask: if the solver advanced the model's INPUT frame
    by one saved-frame interval, would it land on the model's PREDICTED frame?
    It should -- that solver IS the data-generating process. The residual is the
    relative squared mismatch

        || n_pred - solver(n_in) ||^2  /  ( || solver(n_in) - n_in ||^2 + eps )

    on the normalized field (same units as the data MSE). Dividing by the actual
    frame-to-frame change makes it dimensionless and ~O(1), and ~0 wherever the
    model predicts correctly -- so, like step 1, it only bites when a prediction
    leaves the physical manifold.

    The solver target is DETACHED: gradients flow only through n_pred, so we get
    the teaching signal without backpropagating through `substeps` stiff steps.
    The real payoff is during ROLLOUT, where n_in is an UNLABELED state the model
    drifted into -- the solver supplies a physics target there that the dataset
    cannot. Solver runs in physical units, so we denormalize in / renormalize out.
    """

    def __init__(self, stepper, norm_mean, norm_std):
        super().__init__()
        self.stepper = stepper
        self.mean = float(norm_mean)
        self.std = float(norm_std)

    def forward(self, n_in_norm, n_pred_norm, r):
        n_in = n_in_norm * self.std + self.mean                     # -> physical
        with torch.no_grad():
            target_phys = self.stepper.frame(n_in, r)               # detached teacher
        target_norm = (target_phys - self.mean) / (self.std + 1e-8)
        num = ((n_pred_norm - target_norm) ** 2).mean(dim=(-3, -2, -1))
        den = ((target_norm - n_in_norm) ** 2).mean(dim=(-3, -2, -1)) + 1e-8
        return (num / den).mean()


# ----------------------------------------------------------------------------
#  Gradient clipping that works with the FNO's complex spectral weights
# ----------------------------------------------------------------------------
def clip_grad_norm_(parameters, max_norm):
    """
    Drop-in replacement for nn.utils.clip_grad_norm_. The spectral conv
    weights are complex (cfloat), and some backends (notably MPS) do not
    implement norm ops for complex tensors. Viewing each complex gradient as
    a (..., 2) real tensor gives the identical norm (|z|^2 = re^2 + im^2)
    and works everywhere.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    norms = [torch.linalg.vector_norm(
        torch.view_as_real(g) if g.is_complex() else g) for g in grads]
    total = torch.linalg.vector_norm(torch.stack(norms))
    scale = max_norm / (total + 1e-12)
    if scale < 1.0:
        for g in grads:
            g.mul_(scale)        # real scalar x complex grad broadcasts fine
    return total


# ----------------------------------------------------------------------------
#  One epoch of training / validation (one-step, teacher-forced)
# ----------------------------------------------------------------------------
def run_epoch(model, loader, loss_fn, device, optimizer=None,
              grad_clip=0.0, writer=None, epoch=0, log_every=20):
    """
    One pass over `loader`. With `optimizer` we train; without it we
    evaluate under no_grad. Returns the mean MSE over all pairs.

    Prints live in-epoch progress (batch counter + throughput) so a slow
    device or stalled run is visible immediately instead of looking hung.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total, n = 0.0, 0
    t0 = time.perf_counter()
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for it, batch in enumerate(loader):
            x, y = batch[0], batch[1]             # (third element r is unused here)
            x = x.to(device, non_blocking=True)   # (B, C, H, W)  frame_t
            y = y.to(device, non_blocking=True)   # (B, 1, H, W)  frame_t+1

            pred = model(x)           # (B, 1, H, W)
            loss = loss_fn(pred, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bs = x.size(0)
            total += loss.item() * bs
            n += bs

            if is_train and (it + 1) % log_every == 0:
                rate = (it + 1) / (time.perf_counter() - t0)
                eta = (len(loader) - it - 1) / max(rate, 1e-9)
                print(f"\r    batch {it + 1:4d}/{len(loader)}  "
                      f"mse {loss.item():.3e}  "
                      f"{rate:4.1f} batch/s  eta {eta:4.0f}s ", end="", flush=True)
                if writer is not None:
                    writer.add_scalar("batch/train_mse", loss.item(),
                                      epoch * len(loader) + it)
    if is_train:
        print("\r" + " " * 70 + "\r", end="", flush=True)   # clear progress line
    return total / max(n, 1)


def train_epoch(model, loader, loss_fn, device, optimizer, grad_clip,
                rollout_steps, include_cond, n_cond,
                rollout_detach=True, rollout_discount=1.0,
                fe_penalty=None, physics_weight=0.0,
                pde_penalty=None, pde_weight=0.0,
                writer=None, epoch=0, log_every=20):
    """
    One training epoch. Handles both regimes:

      * one-step  (rollout_steps == 1): batches are (x, y, r); standard
        teacher-forced step. The weighted loss gets prev = x's density channel.

      * multi-step (rollout_steps > 1): batches are (density0, cond, targets, r)
        from PFCSequenceDataset. We unroll the model on its OWN predictions
        for `rollout_steps`, summing a (optionally discounted) per-step loss.
        `rollout_detach` feeds back detached predictions (the "pushforward"
        trick): each step still gets a one-step gradient, but evaluated on the
        model's slightly-off rollout distribution, which trains it to correct
        its own drift without the instability of backprop-through-time.

    If `physics_weight > 0` and `fe_penalty` is given, a free-energy
    dissipation term (relu of the relative energy increase from the previous
    frame) is added (step 1). If `pde_weight > 0` and `pde_penalty` is given, a
    full PDE-residual term is added (step 2): the differentiable solver advances
    the model's INPUT frame and the prediction is penalized for disagreeing with
    it. During rollout this is evaluated at the model's own (unlabeled) states,
    where the solver -- not the dataset -- supplies the physics target.

        total = data_loss + physics_weight * energy_penalty + pde_weight * pde_residual

    Returns (mean data loss, mean energy penalty, mean PDE residual).
    """
    model.train()
    total, total_pen, total_pde, n = 0.0, 0.0, 0.0, 0
    use_fe = fe_penalty is not None and physics_weight > 0
    use_pde = pde_penalty is not None and pde_weight > 0
    t0 = time.perf_counter()
    for it, batch in enumerate(loader):
        if rollout_steps > 1:
            density0, cond, targets, r = batch
            density0 = density0.to(device, non_blocking=True)   # (B,1,H,W)
            targets = targets.to(device, non_blocking=True)     # (B,k,H,W)
            r = r.to(device, non_blocking=True)
            B, _, H, W = density0.shape
            cond_maps = None
            if include_cond and cond.numel() > 0:
                cond_maps = cond.to(device).view(B, n_cond, 1, 1).expand(
                    B, n_cond, H, W)

            cur, prev = density0, density0
            data_loss, pen, pde, wsum = 0.0, 0.0, 0.0, 0.0
            for j in range(targets.shape[1]):
                inp = cur if cond_maps is None else torch.cat([cur, cond_maps], 1)
                pred = model(inp)                               # (B,1,H,W)
                w = rollout_discount ** j
                data_loss = data_loss + w * loss_fn(pred, targets[:, j:j + 1], prev)
                if use_fe:
                    pen = pen + w * fe_penalty(pred, prev, r)
                if use_pde:
                    # Residual against the solver applied to the model's ACTUAL
                    # input `cur` (a true frame at j=0, an unlabeled rollout
                    # state for j>0) -- not the true `prev`.
                    pde = pde + w * pde_penalty(cur, pred, r)
                wsum += w
                prev = targets[:, j:j + 1]                      # true previous frame
                cur = pred.detach() if rollout_detach else pred
            data_loss = data_loss / wsum
            pen = pen / wsum if use_fe else torch.zeros((), device=device)
            pde = pde / wsum if use_pde else torch.zeros((), device=device)
            bs = B
        else:
            x, y, r = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True)
            pred = model(x)
            data_loss = loss_fn(pred, y, x[:, :1])
            pen = fe_penalty(pred, x[:, :1], r) if use_fe \
                else torch.zeros((), device=device)
            pde = pde_penalty(x[:, :1], pred, r) if use_pde \
                else torch.zeros((), device=device)
            bs = x.size(0)

        loss = data_loss + physics_weight * pen + pde_weight * pde
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total += data_loss.item() * bs
        total_pen += float(pen.detach()) * bs
        total_pde += float(pde.detach()) * bs
        n += bs
        if (it + 1) % log_every == 0:
            rate = (it + 1) / (time.perf_counter() - t0)
            eta = (len(loader) - it - 1) / max(rate, 1e-9)
            extra = f"  Epen {float(pen.detach()):.2e}" if use_fe else ""
            extra += f"  PDE {float(pde.detach()):.2e}" if use_pde else ""
            print(f"\r    batch {it + 1:4d}/{len(loader)}  data {data_loss.item():.3e}"
                  f"{extra}  {rate:4.1f} batch/s  eta {eta:4.0f}s ", end="", flush=True)
            if writer is not None:
                writer.add_scalar("batch/train_data_loss", data_loss.item(),
                                  epoch * len(loader) + it)
                if use_fe:
                    writer.add_scalar("batch/train_energy_penalty", float(pen.detach()),
                                      epoch * len(loader) + it)
                if use_pde:
                    writer.add_scalar("batch/train_pde_residual", float(pde.detach()),
                                      epoch * len(loader) + it)
    print("\r" + " " * 70 + "\r", end="", flush=True)
    return total / max(n, 1), total_pen / max(n, 1), total_pde / max(n, 1)


# ----------------------------------------------------------------------------
#  Rollout validation (autoregressive, the honest surrogate metric)
# ----------------------------------------------------------------------------
def _conditioning_for(path, manifest, cond_stats):
    """[r, n0] for a trajectory (manifest first, then npz meta), normalized."""
    rec_meta = inspect_trajectory(path)["meta"]
    row = manifest.get(_run_id_from_path(path), {})
    vals = []
    for col in _COND_COLUMNS:
        if col in row:
            vals.append(float(row[col]))
        elif col in rec_meta:
            vals.append(float(rec_meta[col]))
        else:
            vals.append(0.0)
    cond = np.asarray(vals, dtype=np.float32)
    if cond_stats is not None:
        cond = (cond - cond_stats["mean"]) / (cond_stats["std"] + 1e-8)
    return cond


@torch.no_grad()
def rollout_validation(model, files, info, manifest, device, max_traj=5):
    """
    Free-running rollout from frame_0 on up to `max_traj` validation
    trajectories:  n0 -> n1_hat -> n2_hat -> ...  (model eats its own output).

    All trajectories are rolled out together as one batch -- T-1 forward
    passes total instead of (T-1) * n_traj, so this stays cheap even when
    called every few epochs.

    Returns (mean rollout MSE over all steps/trajectories in PHYSICAL units,
             mean relative mass drift at the final step).
    """
    model.eval()
    mean, std = info["norm_mean"], info["norm_std"]

    # Stack the ground-truth trajectories: (B, T, H, W) in physical units.
    trajs, conds = [], []
    for path in files[:max_traj]:
        rec = inspect_trajectory(path)
        trajs.append(np.concatenate([rec["inputs"], rec["targets"][-1:]], axis=0))
        conds.append(_conditioning_for(path, manifest, info["cond_stats"])
                     if info["include_conditioning"] else None)
    T = min(t.shape[0] for t in trajs)                  # common horizon
    true_phys = np.stack([t[:T] for t in trajs])        # (B, T, H, W)
    B, _, H, W = true_phys.shape

    cond_maps = None
    if info["include_conditioning"]:
        c = torch.from_numpy(np.stack(conds)).float()                  # (B, n_cond)
        cond_maps = c.view(B, -1, 1, 1).expand(B, c.shape[1], H, W).to(device)

    cur = torch.from_numpy(
        (true_phys[:, 0] - mean) / (std + 1e-8)).float().view(B, 1, H, W).to(device)
    preds = [true_phys[:, 0]]
    for _ in range(T - 1):
        x = cur if cond_maps is None else torch.cat([cur, cond_maps], dim=1)
        cur = model(x)
        preds.append(cur.cpu().numpy()[:, 0] * (std + 1e-8) + mean)   # physical
    preds = np.stack(preds, axis=1)                     # (B, T, H, W)

    mse = float(np.mean((preds[:, 1:] - true_phys[:, 1:]) ** 2))
    m_true = true_phys[:, -1].mean(axis=(1, 2))
    m_pred = preds[:, -1].mean(axis=(1, 2))
    drift = float(np.mean(np.abs(m_pred - m_true) / (np.abs(m_true) + 1e-12)))
    return mse, drift


# ----------------------------------------------------------------------------
#  Main training entry point
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train the PFC FNO surrogate")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"].get("seed", 0))
    device = pick_device(cfg["train"].get("device", "auto"))
    print(f"Device: {device}")
    if device.type == "cpu":
        print("\n" + "!" * 64)
        print("! WARNING: no GPU detected -- training on CPU is 10-30x slower.")
        print("! In Colab: Runtime -> Change runtime type -> T4 GPU, then")
        print("! restart and rerun the notebook from the top.")
        print("!" * 64 + "\n")

    # --- data ---
    train_ds, val_ds, test_ds, info = build_datasets(cfg)
    manifest = _load_manifest(cfg["data"]["data_dir"])
    print(f"Trajectories  -> train {len(info['train_files'])} | "
          f"val {len(info['val_files'])} | test {len(info['test_files'])} "
          f"(split_by={info.get('split_by', 'trajectory')})")
    print(f"Frame pairs   -> train {len(train_ds)} | "
          f"val {len(val_ds)} | test {len(test_ds)}")
    print(f"Normalization -> mean {info['norm_mean']:.5f}  std {info['norm_std']:.5f}")
    print(f"Conditioning  -> {info['include_conditioning']} "
          f"(in_channels={info['in_channels']})")

    nw = cfg["train"].get("num_workers", 0)
    pin = device.type == "cuda"          # faster host->GPU copies on CUDA
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, num_workers=nw, drop_last=False,
                              pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                            shuffle=False, num_workers=nw, pin_memory=pin)

    # --- model ---
    cfg["model"]["in_channels"] = info["in_channels"]
    model = build_model(cfg).to(device)
    print(f"Model: FNO2d  modes={cfg['model']['modes']} "
          f"width={cfg['model']['width']} layers={cfg['model']['layers']}  "
          f"predict_delta={cfg['model'].get('predict_delta', False)}  "
          f"enforce_mass={cfg['model'].get('enforce_mass', False)}  |  "
          f"{count_parameters(model):,} params")

    # --- optimizer / schedule / loss ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    scheduler = make_scheduler(optimizer, cfg)
    # Training loss: plain MSE by default; optionally weighted toward growth
    # fronts / boundaries (config train.loss_weight_mode + loss_alpha).
    loss_fn = PFCLoss(
        weight_mode=cfg["train"].get("loss_weight_mode", "none"),
        alpha=cfg["train"].get("loss_alpha", 5.0),
    )
    # Validation is always plain one-step MSE -> a stable, comparable scalar.
    val_loss_fn = nn.MSELoss()
    rollout_steps = info.get("rollout_steps", 1)
    n_cond = info["in_channels"] - 1

    # Physics-informed free-energy dissipation penalty (step 1). physics_weight
    # = 0 recovers the pure-data baseline.
    physics_weight = float(cfg["train"].get("physics_weight", 0.0))
    fe_penalty = None
    if physics_weight > 0:
        fe_penalty = FreeEnergyDissipation(
            info["norm_mean"], info["norm_std"], info["dx"]).to(device)

    # Physics-informed full PDE-residual penalty (step 2): the differentiable
    # PFC solver as a teacher. pde_weight = 0 leaves it off. The solver's dt / M
    # / substeps come from the data's own metadata (so the stepper reproduces
    # the integrator that generated the frames); config can override them.
    pde_weight = float(cfg["train"].get("pde_weight", 0.0))
    pde_penalty = None
    if pde_weight > 0:
        meta0 = inspect_trajectory(info["train_files"][0])["meta"]
        pde_dt = float(cfg["train"].get("pde_dt", meta0.get("dt", 0.25)))
        pde_M = float(cfg["train"].get("pde_M", meta0.get("M", 1.0)))
        pde_substeps = int(cfg["train"].get("pde_substeps",
                                            meta0.get("save_every", 25)))
        stepper = DifferentiablePFCStep(info["dx"], pde_dt, pde_M, pde_substeps)
        pde_penalty = PDEResidual(stepper, info["norm_mean"], info["norm_std"]).to(device)

    print(f"Loss: weight_mode={cfg['train'].get('loss_weight_mode', 'none')} "
          f"alpha={cfg['train'].get('loss_alpha', 5.0)}  |  "
          f"rollout_steps={rollout_steps} "
          f"(detach={cfg['train'].get('rollout_detach', True)})  |  "
          f"physics_weight={physics_weight} (free-energy dissipation, dx={info['dx']:.4f})")
    if pde_weight > 0:
        print(f"      pde_weight={pde_weight} (PDE residual via differentiable "
              f"solver: dt={pde_dt} M={pde_M} substeps={pde_substeps})")

    # --- logging / checkpoints ---
    out_dir = cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    writer = None
    if cfg["logging"].get("tensorboard", True):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
            print(f"TensorBoard logging to {os.path.join(out_dir, 'tb')}")
        except Exception as e:
            print(f"TensorBoard unavailable ({e}); continuing without it.")

    ckpt_path = os.path.join(out_dir, cfg["logging"].get("ckpt_name", "best.pt"))
    last_path = os.path.join(out_dir, cfg["logging"].get("last_name", "last.pt"))

    def save_ckpt(path, epoch, val_loss):
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "val_loss": val_loss,
            "config": cfg,
            "norm_mean": info["norm_mean"],
            "norm_std": info["norm_std"],
            "include_conditioning": info["include_conditioning"],
            "cond_stats": info["cond_stats"],
            "in_channels": info["in_channels"],
            "train_files": info["train_files"],
            "val_files": info["val_files"],
            "test_files": info["test_files"],
        }, path)

    # --- training loop ---
    es = cfg["train"]
    rollout_every = es.get("rollout_val_every", 10)
    rollout_trajs = es.get("rollout_val_trajs", 5)
    best_val = float("inf")
    epochs_no_improve = 0

    print("\nStarting training...\n")
    for epoch in range(es["epochs"]):
        t0 = time.time()
        train_mse, train_pen, train_pde = train_epoch(
            model, train_loader, loss_fn, device, optimizer,
            grad_clip=es.get("grad_clip", 0.0),
            rollout_steps=rollout_steps,
            include_cond=info["include_conditioning"], n_cond=n_cond,
            rollout_detach=es.get("rollout_detach", True),
            rollout_discount=es.get("rollout_discount", 1.0),
            fe_penalty=fe_penalty, physics_weight=physics_weight,
            pde_penalty=pde_penalty, pde_weight=pde_weight,
            writer=writer, epoch=epoch,
            log_every=cfg["logging"].get("log_every", 20))
        val_mse = run_epoch(model, val_loader, val_loss_fn, device)

        if scheduler is not None:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        if writer is not None:
            writer.add_scalar("epoch/train_mse", train_mse, epoch)
            writer.add_scalar("epoch/val_mse", val_mse, epoch)
            writer.add_scalar("epoch/lr", lr_now, epoch)
            if physics_weight > 0:
                writer.add_scalar("epoch/train_energy_penalty", train_pen, epoch)
            if pde_weight > 0:
                writer.add_scalar("epoch/train_pde_residual", train_pde, epoch)

        # Periodic autoregressive rollout on val trajectories: the metric a
        # surrogate actually lives or dies by, watched during training.
        roll_note = ""
        if rollout_every > 0 and (epoch % rollout_every == 0
                                  or epoch == es["epochs"] - 1):
            r_mse, r_drift = rollout_validation(
                model, info["val_files"], info, manifest, device,
                max_traj=rollout_trajs)
            roll_note = f"  rollout {r_mse:.3e} (mass drift {r_drift:.1e})"
            if writer is not None:
                writer.add_scalar("epoch/val_rollout_mse", r_mse, epoch)
                writer.add_scalar("epoch/val_rollout_mass_drift", r_drift, epoch)

        dt = time.time() - t0
        improved = val_mse < best_val - es.get("min_delta", 0.0)
        flag = ""
        if improved:
            best_val = val_mse
            epochs_no_improve = 0
            save_ckpt(ckpt_path, epoch, val_mse)
            flag = "  <- best (saved)"
        else:
            epochs_no_improve += 1

        save_ckpt(last_path, epoch, val_mse)
        pen_note = f"  Epen {train_pen:.2e}" if physics_weight > 0 else ""
        pen_note += f"  PDE {train_pde:.2e}" if pde_weight > 0 else ""
        print(f"epoch {epoch:3d}/{es['epochs']}  "
              f"train {train_mse:.3e}{pen_note}  val {val_mse:.3e}  "
              f"lr {lr_now:.2e}  {dt:.1f}s{roll_note}{flag}")

        if es.get("early_stopping", False) and epochs_no_improve >= es["patience"]:
            print(f"\nEarly stopping: no val improvement for {es['patience']} epochs.")
            break

    if writer is not None:
        writer.close()
    print(f"\nDone. Best val MSE: {best_val:.3e}")
    print(f"Best checkpoint:  {ckpt_path}")
    print(f"Last checkpoint:  {last_path}")
    print("Evaluate with:  python evaluate_fno.py --config config.yaml "
          f"--checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
