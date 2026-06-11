"""
============================================================
  Fourier Neural Operator (FNO) — 2D
  Surrogate model for the Phase Field Crystal (PFC) solver

  Learns the one-step map   n(x, y, t)  ->  n(x, y, t + Dt)
  on the crystal density field n.

  NSF IRES Physical AI Design Program
============================================================

WHAT IS AN FNO (read this first)
--------------------------------
A neural operator learns a mapping between *functions* (here, between the
density field at one time and the density field at the next saved time),
rather than between fixed-size vectors. The Fourier Neural Operator does the
"learning in frequency space" trick:

    1. Lift the input field to a higher-dimensional channel space (a 1x1 conv).
    2. Repeat several "Fourier layers". Each Fourier layer has two branches that
       are summed:
         (a) a SPECTRAL branch: FFT -> keep only the lowest `modes` frequencies
             -> multiply by a learned complex weight per mode -> inverse FFT.
             This is a global convolution: every output point depends on every
             input point. Because PFC dynamics are themselves driven by a few
             low-wavenumber modes (the lattice spacing sets a dominant k), a
             handful of Fourier modes captures most of the physics.
         (b) a LOCAL branch: an ordinary 1x1 convolution (pointwise linear mix
             of channels). This restores the high-frequency detail that the
             truncated spectral branch throws away.
       The two branches are added, then passed through a GELU nonlinearity.
    3. Project back down to a single output channel (the predicted next field).

WHY FOURIER. The PFC PDE itself is integrated in Fourier space (see
PCF_Baseline.py: the linear operator is diagonal in k). Convolutions become
cheap multiplications there, and the operator is resolution-invariant: a model
trained at 128x128 can in principle be evaluated at other resolutions, because
the learned weights live in mode-space, not pixel-space.

TENSOR SHAPE CONVENTION
-----------------------
Throughout this file we use the PyTorch image convention:

    x : (B, C, H, W)
        B = batch size
        C = number of channels (feature maps)
        H = grid height (y), W = grid width (x)

Real-to-complex FFTs (torch.fft.rfft2) collapse the last axis to W//2 + 1,
so spectral tensors are (B, C, H, W//2 + 1) and complex-valued.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
#  Spectral convolution layer  --  the heart of the FNO
# ----------------------------------------------------------------------------
class SpectralConv2d(nn.Module):
    """
    A 2D spectral convolution: a learned, mode-truncated multiplication in
    Fourier space.

    Forward pass, step by step (shapes for a 128x128 grid, modes1=modes2=16):

        x            (B, in_ch, 128, 128)            real
        rfft2(x)  -> (B, in_ch, 128,  65)            complex  (W//2+1 = 65)
        keep only the lowest `modes1` x `modes2` frequencies and multiply each
        by a learned complex weight that mixes input channels into output
        channels (an einsum over the channel axis):
                     (B, out_ch, 128, 65)            complex  (mostly zeros)
        irfft2    -> (B, out_ch, 128, 128)           real

    Only `modes1 * modes2` frequencies per channel pair carry weights; all
    higher frequencies are set to zero. That truncation is exactly what makes
    the operator smooth and resolution-independent.

    Note on rfft layout: rfft2 returns frequencies ordered as
        dim -2 (height): 0, +1, +2, ..., +N/2, -(N/2-1), ..., -1   (full, signed)
        dim -1 (width) : 0, +1, ..., +N/2                          (non-negative)
    So the "lowest modes" along height live in BOTH the first `modes1` rows
    (positive freqs) and the last `modes1` rows (negative freqs); along width
    they live only in the first `modes2` columns. We therefore write the two
    height-corners separately below.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1   # number of Fourier modes kept along height (y)
        self.modes2 = modes2   # number of Fourier modes kept along width  (x)

        # One complex weight tensor per height-corner (positive / negative freq).
        # Shape: (in_ch, out_ch, modes1, modes2), complex.
        # The 1/(in*out) scaling keeps activations at a sane magnitude at init.
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels,
                               self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels,
                               self.modes1, self.modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _mul2d(inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Complex multiply that also mixes channels.

            inp     : (B, in_ch,  modes1, modes2)   complex
            weights : (in_ch, out_ch, modes1, modes2) complex
            return  : (B, out_ch, modes1, modes2)   complex

        The einsum 'bixy, ioxy -> boxy' contracts the input-channel axis i,
        i.e. for every kept frequency (x, y) it applies a learned (in->out)
        complex linear map.
        """
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        # 1) Forward real FFT.  (B, in_ch, H, W) -> (B, in_ch, H, W//2 + 1)
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # 2) Allocate the output spectrum (all zeros = all high modes dropped).
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # 3) Fill the two low-frequency corners with learned interactions.
        #    Top-left corner  = lowest POSITIVE height freqs.
        out_ft[:, :, :self.modes1, :self.modes2] = self._mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        #    Bottom-left corner = lowest NEGATIVE height freqs (wrapped to the end).
        out_ft[:, :, -self.modes1:, :self.modes2] = self._mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # 4) Inverse real FFT back to a real field of the original size.
        #    s=(H, W) tells irfft2 the target spatial size.
        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return x


# ----------------------------------------------------------------------------
#  Full FNO-2d network
# ----------------------------------------------------------------------------
class FNO2d(nn.Module):
    """
    The complete 2D Fourier Neural Operator.

    Pipeline (default in_channels=1, width=32, modes=16, n_layers=4):

        input field        (B, in_channels, H, W)
          + 2 coord chans   -> optionally append normalized (x, y) grids so the
                               network knows absolute position; helps with the
                               (otherwise translation-invariant) spectral conv.
        lift  (1x1 conv)   (B, width, H, W)
        [ Fourier layer ] x n_layers, each:
            spectral_branch = SpectralConv2d(x)        (global, low modes)
            local_branch    = Conv2d_1x1(x)            (pointwise channel mix)
            x = GELU( spectral_branch + local_branch ) + x_residual
          (every hidden layer shares `width`, so the residual skip applies
           on all Fourier layers)
        project  (1x1 conv -> 1x1 conv)  (B, 1, H, W)
        output: predicted next field

    Args:
        modes1, modes2 : Fourier modes kept per spatial axis (e.g. 16).
                         For the PFC data (L = 16*pi, N = 128) mode index m
                         corresponds to wavenumber k = m/8; measurement shows
                         99.999% of crystal spectral energy lies below m = 16,
                         so 16 modes capture essentially everything.
        width          : channel width of the hidden representation (e.g. 32).
        n_layers       : number of Fourier layers (e.g. 4).
        in_channels    : input channels. 1 for the bare density field; add more
                         if you pass conditioning maps (r, n0) as extra channels
                         (see dataset.py `include_conditioning`).
        out_channels   : 1 (single predicted density field).
        use_grid       : append normalized (x, y) coordinate channels to the input.
        predict_delta  : if True, the network outputs the CHANGE dn and the
                         forward pass returns n_t + dn. One PFC frame step
                         changes the field only mildly, so learning the
                         increment is better conditioned than re-predicting
                         the whole field, and rollouts start from the exact
                         identity rather than an approximation of it.
        enforce_mass   : if True, project the prediction so its spatial mean
                         equals the input's spatial mean EXACTLY. PFC dynamics
                         conserve mean(n) (conserved order parameter), and this
                         bakes that into the architecture -- no physics loss
                         term needed, and rollouts cannot drift in mass.
                         (Both options require out_channels == 1.)
    """

    def __init__(self, modes1: int = 16, modes2: int = 16,
                 width: int = 32, n_layers: int = 4,
                 in_channels: int = 1, out_channels: int = 1,
                 use_grid: bool = True,
                 predict_delta: bool = False, enforce_mass: bool = False):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_grid = use_grid
        self.predict_delta = predict_delta
        self.enforce_mass = enforce_mass
        if (predict_delta or enforce_mass) and out_channels != 1:
            raise ValueError("predict_delta / enforce_mass assume out_channels == 1")

        # The lift sees the raw channels plus 2 coordinate channels (if enabled).
        lift_in = in_channels + (2 if use_grid else 0)

        # 1x1 conv that lifts (lift_in -> width). A 1x1 conv == a per-pixel
        # linear layer across channels.
        self.lift = nn.Conv2d(lift_in, width, kernel_size=1)

        # Stacks of spectral convs and matching local 1x1 convs.
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2)
            for _ in range(n_layers)
        ])
        self.local_layers = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1)
            for _ in range(n_layers)
        ])

        # Projection head: width -> 128 -> out_channels, with a GELU between.
        self.project1 = nn.Conv2d(width, 128, kernel_size=1)
        self.project2 = nn.Conv2d(128, out_channels, kernel_size=1)

    def _make_grid(self, shape, device) -> torch.Tensor:
        """
        Build normalized coordinate channels in [0, 1].

            return : (B, 2, H, W)   channel 0 = x grid, channel 1 = y grid

        Giving the network absolute coordinates breaks the pure translation
        invariance of the spectral conv, which matters here because the seed
        sits at a specific location (e.g. the domain centre).
        """
        B, _, H, W = shape
        gx = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        gy = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        return torch.cat([gx, gy], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, in_channels, H, W). Channel 0 is always the density field
        # (conditioning channels, if any, come after it) -- keep a handle on it
        # for the residual / mass-conservation steps at the end.
        density = x[:, :1]                                  # (B, 1, H, W)

        if self.use_grid:
            grid = self._make_grid(x.shape, x.device)      # (B, 2, H, W)
            x = torch.cat([x, grid], dim=1)                 # (B, in+2, H, W)

        x = self.lift(x)                                    # (B, width, H, W)

        for i, (spectral, local) in enumerate(
                zip(self.spectral_layers, self.local_layers)):
            x_in = x
            out = spectral(x) + local(x)                    # global + local
            out = F.gelu(out)
            # Residual connection: add the layer input back. All hidden layers
            # share `width`, so the shapes always match here.
            x = out + x_in

        x = self.project1(x)                                # (B, 128, H, W)
        x = F.gelu(x)
        x = self.project2(x)                                # (B, out_ch, H, W)

        if self.predict_delta:
            # x is the predicted CHANGE dn. Optionally remove its spatial mean
            # so that mean(n_t + dn) == mean(n_t) exactly (mass conservation;
            # the z-score normalization is affine, so preserving the mean in
            # normalized space preserves the physical mean too).
            if self.enforce_mass:
                x = x - x.mean(dim=(-2, -1), keepdim=True)
            x = density + x
        elif self.enforce_mass:
            # Direct prediction: shift the output's mean onto the input's.
            x = x - x.mean(dim=(-2, -1), keepdim=True) \
                  + density.mean(dim=(-2, -1), keepdim=True)
        return x


def build_model(cfg: dict) -> FNO2d:
    """Construct an FNO2d from a config dict (see config.yaml -> 'model')."""
    m = cfg["model"]
    return FNO2d(
        modes1=m.get("modes", 16),
        modes2=m.get("modes", 16),
        width=m.get("width", 32),
        n_layers=m.get("layers", 4),
        in_channels=m.get("in_channels", 1),
        out_channels=m.get("out_channels", 1),
        use_grid=m.get("use_grid", True),
        predict_delta=m.get("predict_delta", False),
        enforce_mass=m.get("enforce_mass", False),
    )


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters (handy for logging)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
