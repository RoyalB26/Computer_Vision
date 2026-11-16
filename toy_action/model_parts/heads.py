"""Deformable Spatial & Mamba Action Classification Heads for Toy Action models."""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_2d_sincos_position_embedding(
    H: int,
    W: int,
    embed_dim: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build 2D sin-cos positional embedding of shape ``[1, embed_dim, H, W]``."""
    assert embed_dim % 4 == 0, f"embed_dim ({embed_dim}) must be divisible by 4."
    dim_per_axis = embed_dim // 2
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)

    def _sincos_1d(grid: torch.Tensor, dim: int) -> torch.Tensor:
        omega = torch.arange(dim // 2, device=device, dtype=dtype)
        omega = omega / (dim / 2.0)
        omega = 1.0 / (10000.0 ** omega)
        out = torch.einsum("n, d -> nd", grid, omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

    emb_x = _sincos_1d(grid_x, dim_per_axis)
    emb_y = _sincos_1d(grid_y, dim_per_axis)
    pos_emb = torch.cat([emb_x, emb_y], dim=-1)
    return pos_emb.view(H, W, embed_dim).permute(2, 0, 1).unsqueeze(0).contiguous()


class DeformableFeatureHead(nn.Module):
    """Deformable Attention Spatial Head adapted from Visdrone's DeformableSlotDetectionHead.

    Operates purely on spatial grids (without slots), sampling high-resolution
    features from ``feat_deform`` onto a target ``grid_size`` (default ``10x10``)
    prediction map and returning the refined ``10x10`` feature image.
    """

    def __init__(
        self,
        dim: int = 256,
        grid_size: Tuple[int, int] = (10, 10),
        num_points: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})."
        self.dim = dim
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads

        # 1. Spatial projection for base grid map (resized to grid_size, e.g. 10x10)
        self.pred_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        # 2. Deformable Attention Engine (same sampling formulation as Visdrone)
        self.value_proj = nn.Linear(dim, dim)
        self.sampling_offsets = nn.Linear(dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(dim, num_heads * num_points)
        self.output_proj = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, 1, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= (i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def _get_reference_grid(
        self,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create or retrieve cached normalized [0, 1] reference coordinates of shape ``[H*W, 2]``."""
        cache_key = (H, W, device, dtype)
        if getattr(self, "_grid_cache_key", None) != cache_key:
            y_grid, x_grid = torch.meshgrid(
                torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device, dtype=dtype),
                torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device, dtype=dtype),
                indexing="ij",
            )
            self._cached_ref_grid = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=-1)
            self._grid_cache_key = cache_key
        return self._cached_ref_grid

    def _get_pos_embed(
        self,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create or retrieve cached 2D sin-cos positional embedding of shape ``[1, dim, H, W]``."""
        cache_key = (H, W, self.dim, device, dtype)
        if getattr(self, "_pos_cache_key", None) != cache_key:
            self._cached_pos_emb = build_2d_sincos_position_embedding(
                H, W, self.dim, device=device, dtype=dtype
            )
            self._pos_cache_key = cache_key
        return self._cached_pos_emb

    def forward(
        self,
        map_pred: torch.Tensor,
        feat_deform: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Refine features onto target ``grid_size`` (e.g. 10x10) using deformable sampling.

        Args:
            map_pred: Base feature map ``[BT, dim, H_in, W_in]``.
            feat_deform: High-resolution feature map ``[BT, dim, H_def, W_def]``.

        Returns:
            out_map: Refined ``10x10`` feature map ``[BT, dim, H_grid, W_grid]``.
            sampling_info: Dict with ``sampling_locations`` and ``attn_weights``.
        """
        bt, _, h_in, w_in = map_pred.shape
        _, _, h_def, w_def = feat_deform.shape
        h_grid, w_grid = self.grid_size
        q_len = h_grid * w_grid
        device = map_pred.device

        # Resize base map cleanly to exact 10x10 grid
        if (h_in, w_in) != (h_grid, w_grid):
            if h_in >= h_grid and w_in >= w_grid:
                map_grid = F.adaptive_avg_pool2d(map_pred, (h_grid, w_grid))
            else:
                map_grid = F.interpolate(map_pred, size=(h_grid, w_grid), mode="bilinear", align_corners=False)
        else:
            map_grid = map_pred

        # Spatial features + 2D pos emb -> query_flat: [BT, 100, dim]
        spatial_map = self.pred_proj(map_grid)
        pos_emb = self._get_pos_embed(h_grid, w_grid, device=device, dtype=spatial_map.dtype)
        query_flat = (spatial_map + pos_emb).flatten(2).permute(0, 2, 1).contiguous()

        # Normalized [0, 1] reference points: [BT, 100, 2]
        ref_grid = self._get_reference_grid(h_grid, w_grid, device=device, dtype=query_flat.dtype)
        ref_flat = ref_grid.unsqueeze(0).expand(bt, -1, -1)

        # Value projection from high-res feat_deform: [BT * num_heads, head_dim, H_def, W_def]
        value = self.value_proj(feat_deform.permute(0, 2, 3, 1))
        value = value.view(bt, h_def, w_def, self.num_heads, self.head_dim)
        value_permuted = value.permute(0, 3, 4, 1, 2).contiguous().view(bt * self.num_heads, self.head_dim, h_def, w_def)

        # Deformable sampling offsets & attention weights
        offsets = self.sampling_offsets(query_flat).view(bt, q_len, self.num_heads, self.num_points, 2)
        spatial_scale = torch.tensor([1.0 / w_def, 1.0 / h_def], device=device, dtype=offsets.dtype).view(1, 1, 1, 1, 2)
        sampling_locations = ref_flat.unsqueeze(2).unsqueeze(3) + offsets * spatial_scale

        sampling_grid = 2.0 * sampling_locations - 1.0
        grid_permuted = sampling_grid.permute(0, 2, 1, 3, 4).contiguous().view(
            bt * self.num_heads, q_len, self.num_points, 2
        )

        attn_weights = self.attention_weights(query_flat).view(bt, q_len, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)

        sampled_feat = F.grid_sample(
            value_permuted,
            grid_permuted,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # [BT * num_heads, head_dim, Q, num_points]

        sampled_feat = sampled_feat.view(bt, self.num_heads, self.head_dim, q_len, self.num_points)
        sampled_feat = sampled_feat.permute(0, 3, 1, 4, 2)  # [BT, Q, num_heads, num_points, head_dim]

        out = (sampled_feat * attn_weights.unsqueeze(-1)).sum(dim=3)
        out = self.output_proj(out.reshape(bt, q_len, self.dim))

        # Residual enhancement + LayerNorm, reshaped back to 10x10 spatial feature image
        enhanced_flat = self.norm_out(query_flat + out)
        out_map = enhanced_flat.permute(0, 2, 1).contiguous().view(bt, self.dim, h_grid, w_grid)

        sampling_info = {
            "sampling_locations": sampling_locations.view(bt, h_grid, w_grid, self.num_heads, self.num_points, 2),
            "attn_weights": attn_weights.view(bt, h_grid, w_grid, self.num_heads, self.num_points),
        }
        return out_map, sampling_info


class MambaActionHead(nn.Module):
    """Multi-label action classification head over flattened Mamba spatial tokens ``[B, 100, D]``."""

    def __init__(
        self,
        in_channel: int = 256,
        num_actor_classes: int = 48,
        grid_size: Tuple[int, int] = (10, 10),
        num_ego_classes: int = 0,
        ego_channel: int = 0,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.num_actor_classes = num_actor_classes
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.num_ego_classes = num_ego_classes

        self.norm = nn.LayerNorm(in_channel)
        self.spatial_attn = nn.Linear(in_channel, 1)
        self.proj = nn.Sequential(
            nn.Linear(in_channel * 3, in_channel),
            nn.LayerNorm(in_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.fc_actor = nn.Linear(in_channel, num_actor_classes)

        if self.num_ego_classes > 0:
            c = ego_channel if ego_channel > 0 else in_channel
            self.fc_ego = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Dropout(p=dropout),
                nn.Linear(c, num_ego_classes),
            )
        else:
            self.fc_ego = None

    def forward(
        self,
        x: torch.Tensor,
        ego_x: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Classify actions from spatial tokens ``[B, 100, D]``.

        Returns:
            y_ego: Optional ego logits ``[B, num_ego_classes]``.
            y_actor: Multi-label actor logits ``[B, num_actor_classes]``.
            spatial_attn_map: Spatial attention map ``[B, H_grid, W_grid]``.
        """
        b, n, _ = x.shape
        h_grid, w_grid = self.grid_size
        norm_x = self.norm(x)

        attn_weights = F.softmax(self.spatial_attn(norm_x), dim=1)  # [B, N, 1]
        attn_feat = (norm_x * attn_weights).sum(dim=1)              # [B, D]
        max_feat = norm_x.max(dim=1).values                         # [B, D]
        mean_feat = norm_x.mean(dim=1)                              # [B, D]

        fused = torch.cat([attn_feat, max_feat, mean_feat], dim=-1)
        feat = self.proj(fused)
        y_actor = self.fc_actor(feat)

        spatial_attn_map = attn_weights.squeeze(-1).view(b, h_grid, w_grid)
        y_ego = self.fc_ego(ego_x) if self.fc_ego is not None and ego_x is not None else None
        return y_ego, y_actor, spatial_attn_map
