"""
============================================================
  Graph Neural Network for Crystal Boundary Prediction
  Kobayashi Phase Field Surrogate

  Learns to predict how the crystal boundary moves from
  one timestep to the next, operating directly on the
  boundary graph rather than the full dense field.

  Architecture: Message Passing Neural Network (MPNN)
  - Each node aggregates information from its neighbours
  - Predicts (dx, dy) displacement for each boundary point
  - 3 message passing layers with residual connections

  Why GNN beats FNO here:
  - FNO operates on the full 256x256 grid (95% empty)
  - GNN operates only on 512 boundary nodes (100% meaningful)
  - All information is local to the boundary — exactly what
    message passing is designed for

  Ayush Shah & Tobias Li
  Georgia Institute of Technology
  NSF IRES Physical AI Design Program — Prof. Bo Zhu
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import add_self_loops


# ── Message Passing Layer ─────────────────────────────────────────────────────

class CrystalConv(MessagePassing):
    """
    Custom message passing layer for crystal boundary graphs.

    For each node i, aggregates messages from all neighbours j:

        m_ij = MLP([h_i || h_j || e_ij])   (message)
        h_i' = MLP([h_i || mean(m_ij)])     (update)

    where:
        h_i  = node hidden state
        e_ij = edge feature = relative position (xj - xi)
                              and distance ||xj - xi||

    Using relative position as an edge feature is key —
    the GNN needs to know the local geometry of the boundary
    to predict how it moves.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='mean')   # mean aggregation over neighbours
        self.in_ch  = in_channels
        self.out_ch = out_channels

        # Edge feature dimension: (dx, dy, dist) = 3
        edge_dim = 3

        # Message MLP: [h_i || h_j || e_ij] → message
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

        # Update MLP: [h_i || agg_msg] → h_i'
        self.upd_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

        # Residual projection (if dimensions differ)
        self.residual = (nn.Linear(in_channels, out_channels)
                         if in_channels != out_channels else nn.Identity())

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                pos: torch.Tensor) -> torch.Tensor:
        """
        x          : (N, in_channels)  node features
        edge_index : (2, E)            graph connectivity
        pos        : (N, 2)            node positions (for edge features)
        """
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j):
        """
        Compute message from node j to node i.
        x_i, x_j  : (E, in_channels)
        pos_i, pos_j : (E, 2)
        """
        # Relative position as edge feature
        rel_pos  = pos_j - pos_i                                   # (E, 2)
        dist     = rel_pos.norm(dim=-1, keepdim=True)              # (E, 1)
        edge_feat = torch.cat([rel_pos, dist], dim=-1)             # (E, 3)

        msg_input = torch.cat([x_i, x_j, edge_feat], dim=-1)      # (E, 2*in+3)
        return self.msg_mlp(msg_input)                             # (E, out)

    def update(self, aggr_out, x):
        """
        aggr_out : (N, out_channels)  aggregated messages
        x        : (N, in_channels)   original node features
        """
        upd_input = torch.cat([x, aggr_out], dim=-1)  # (N, in+out)
        out       = self.upd_mlp(upd_input)            # (N, out)
        out       = self.norm(out + self.residual(x))  # residual + norm
        return out


# ── Full GNN Model ────────────────────────────────────────────────────────────

class CrystalGNN(nn.Module):
    """
    Full GNN for crystal boundary displacement prediction.

    Input:  graph with node features (N, 8) and positions (N, 2)
    Output: per-node displacement (N, 2) = (dx, dy) to next boundary

    Architecture:
        Input projection  → hidden_dim
        3 × CrystalConv   (message passing layers)
        Output projection → 2 (displacement)

    The model is equivariant to the ordering of boundary nodes
    but not to rotation — it uses absolute positions, which is
    correct here because crystals grow from a fixed seed location.
    """

    def __init__(self, in_channels: int = 8,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message passing layers
        self.conv_layers = nn.ModuleList([
            CrystalConv(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Output projection: hidden → displacement (dx, dy)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        data.x          : (N, in_channels)  node features
        data.pos        : (N, 2)            node positions
        data.edge_index : (2, E)            graph connectivity

        Returns
        -------
        displacement : (N, 2)  predicted (dx, dy) per node
        """
        x          = data.x
        pos        = data.pos
        edge_index = data.edge_index

        # Project input features to hidden dimension
        h = self.input_proj(x)   # (N, hidden_dim)

        # Message passing
        for conv in self.conv_layers:
            h = conv(h, edge_index, pos)
            h = self.dropout(h)

        # Predict displacement
        return self.output_proj(h)   # (N, 2)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Loss function ─────────────────────────────────────────────────────────────

class BoundaryLoss(nn.Module):
    """
    Combined loss for boundary displacement prediction.

    Components:
    1. MSE on displacement vectors (primary)
    2. Smoothness penalty: adjacent boundary points should
       have similar displacements (prevents jagged predictions)
    3. Length preservation: total boundary length shouldn't
       change drastically in one step

    smooth_weight and length_weight can be set to 0 to use
    pure MSE during initial training.
    """

    def __init__(self, smooth_weight: float = 0.1,
                 length_weight: float = 0.05):
        super().__init__()
        self.sw = smooth_weight
        self.lw = length_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                pos: torch.Tensor) -> torch.Tensor:
        """
        pred   : (N, 2) predicted displacements
        target : (N, 2) true displacements
        pos    : (N, 2) current boundary positions
        """
        # Primary: MSE on displacements
        mse = F.mse_loss(pred, target)

        # Smoothness: consecutive points should move similarly
        # (ring structure: point i connects to point i+1)
        if self.sw > 0:
            smooth = F.mse_loss(pred[:-1], pred[1:])
        else:
            smooth = torch.tensor(0.0, device=pred.device)

        # Length preservation: predicted new positions shouldn't
        # drastically change segment lengths
        if self.lw > 0:
            new_pos    = pos + pred
            seg_pred   = (new_pos[1:] - new_pos[:-1]).norm(dim=1)
            seg_orig   = (pos[1:]     - pos[:-1]).norm(dim=1)
            length_pen = F.mse_loss(seg_pred, seg_orig)
        else:
            length_pen = torch.tensor(0.0, device=pred.device)

        return mse + self.sw * smooth + self.lw * length_pen


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import torch
    from torch_geometric.data import Data

    # Fake a small graph to verify forward pass
    N = 512
    model = CrystalGNN(in_channels=8, hidden_dim=64, n_layers=3)
    print(f"CrystalGNN: {model.count_params():,} parameters")

    # Random graph
    x   = torch.randn(N, 8)
    pos = torch.rand(N, 2)

    # Ring edges + some random KNN edges
    ring_src = torch.arange(N)
    ring_dst = (torch.arange(N) + 1) % N
    edge_index = torch.stack([
        torch.cat([ring_src, ring_dst]),
        torch.cat([ring_dst, ring_src])
    ])

    data = Data(x=x, pos=pos, edge_index=edge_index,
                y=torch.randn(N, 2))

    out = model(data)
    print(f"Input:  x={data.x.shape}  pos={data.pos.shape}  "
          f"edges={data.edge_index.shape}")
    print(f"Output: {out.shape}  (expected [{N}, 2])")

    loss_fn = BoundaryLoss()
    loss    = loss_fn(out, data.y, pos)
    print(f"Loss:   {loss.item():.6f}")
    print("Forward pass OK")
