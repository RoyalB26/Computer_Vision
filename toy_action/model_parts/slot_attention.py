"""Slot attention module and 3D positional embeddings for video spatiotemporal representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_3d_grid(resolution):
    """Construct a normalized 3D coordinate grid with complement coordinates."""
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing="ij" if hasattr(torch, "meshgrid") else None)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], resolution[2], -1]).unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


class SoftPositionEmbed3D(nn.Module):
    """Learnable soft position embedding for 3D spatiotemporal feature volumes."""

    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.embedding = nn.Linear(6, hidden_size, bias=True)
        self.register_buffer("grid", build_3d_grid(resolution))

    def forward(self, inputs):
        B, T, H, W, dim = inputs.shape
        if (
            self.grid is None
            or self.grid.shape[1] != T
            or self.grid.shape[2] != H
            or self.grid.shape[3] != W
        ):
            grid = build_3d_grid((T, H, W)).to(device=inputs.device, dtype=inputs.dtype)
        else:
            grid = self.grid.to(device=inputs.device, dtype=inputs.dtype)

        grid = self.embedding(grid)
        return inputs + grid


class SlotAttention(nn.Module):
    """Spatiotemporal 3D Slot Attention mechanism."""

    def __init__(
        self,
        num_slots: int = 24,
        dim: int = 256,
        num_actor_class: int = 48,
        eps: float = 1e-8,
        input_dim: int = 256,
        resolution=(16, 7, 7),
        allocated_slot: bool = True,
        slot_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.num_actor_class = num_actor_class
        self.allocated_slot = allocated_slot
        self.eps = eps
        self.scale = dim**-0.5
        self.resolution = resolution
        self.slot_dropout = slot_dropout

        # Learnable Gaussian parameters for slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim).abs())
        self.register_buffer("slots_eps", torch.randn(1, num_slots, dim))

        self.FC1 = nn.Linear(dim, dim)
        self.FC2 = nn.Linear(dim, dim)
        self.LN = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pe = SoftPositionEmbed3D(dim, list(resolution))

    def get_3d_slot(self, slots, inputs):
        b, l, h, w, d = inputs.shape
        inputs = self.pe(inputs)
        inputs = torch.reshape(inputs, (b, -1, d))

        inputs = inputs + self.FC2(F.relu(self.FC1(self.LN(inputs))))

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
        attn_ori = dots.softmax(dim=1) + self.eps
        attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True)
        slots = torch.einsum("bjd,bij->bid", v, attn)

        slots = slots.reshape(b, -1, d)
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots, attn_ori

    def forward(self, inputs, num_slots=None):
        b, nf, h, w, d = inputs.shape
        mu = self.slots_mu.expand(1, self.num_slots, -1)
        sigma = self.slots_sigma.expand(1, self.num_slots, -1)
        slots = (mu + sigma * self.slots_eps).expand(b, -1, -1)
        slots_out, attns = self.get_3d_slot(slots, inputs)
        return slots_out, attns
