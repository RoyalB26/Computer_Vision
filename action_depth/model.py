"""2D Oxz Ground-Plane Map Slot Model (action_depth).

Novel Architecture Highlights:
  1. Lightweight 3D Conv / ResNet Feature Backbone -> (B, T, H=7, W=7, D)
  2. Spatiotemporal Slot Attention: clusters visual tokens into S object slots + bg slot.
  3. Oxz2DMapEncoder (Replaces expensive Conv3d SlotAttentionTubeEncoder):
       - Projects per-slot attention from camera view (H, W) -> 2D ground plane Oxz (Z_res, X_res).
       - Compresses time T into 2D Motion Channels:
           * Channel 0: Max Occupancy max_t(M_t)
           * Channel 1: Chronological Time Ramp sum( (t/T) * M_t )
           * Channel 2: Early Occupancy (t=0..T/2)
           * Channel 3: Late Occupancy (t=T/2..T)
           * Channel 4: Lateral Delta dx
           * Channel 5: Depth Delta dz
           * Channel 6..6+vis_dim: Masked Visual Features
       - Processes each slot's 2D Oxz map using standard, highly-efficient 2D Convolutions (Conv2d)!
  4. Auxiliary Losses:
       - BCE multi-label action loss
       - 2D Oxz BEV heatmap supervision loss (Direct ground-truth 2D tracking loss)
       - Attention negative suppression loss
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from BaseModel.core import MODELS
from BaseModel.model import BaseModel

from .generate_data import ALL_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES, SYMBOLS, ACTIONS
from .dataset import ActionDepthLMDBDataset


class Oxz2DMapEncoder(nn.Module):
    """Encodes Spatiotemporal Slot Attention into 2D Ground-Plane Oxz maps using pure Conv2d.

    Takes (B, T, S, H, W) slot attention and (B, T, H, W, D) visual features,
    lifts them to 2D Oxz grid (Z_bins, X_bins), encodes time T into 2D motion trails,
    and extracts slot motion embeddings via 2D CNN.
    """

    def __init__(
        self,
        slot_dim: int = 128,
        hidden_dim: int = 96,
        z_bins: int = 16,
        x_bins: int = 16,
        vis_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.z_bins = z_bins
        self.x_bins = x_bins
        self.vis_dim = vis_dim

        # 6 motion channels: [max_occ, time_ramp, early_occ, late_occ, dx, dz] + vis_dim
        in_channels = 6 + vis_dim
        self.vis_proj = nn.Linear(slot_dim, vis_dim) if vis_dim > 0 else None

        self.conv2d_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 2, slot_dim),
        )

        # Precompute perspective lifting weight matrix from (H, W) -> (Z_bins, X_bins)
        # In this 2.5D setup, ground starts at horizon (row ~ 0) to bottom (row ~ H-1)
        self.register_buffer("_coords_cached", None, persistent=False)

    def _lift_attn_to_oxz(self, obj_attn: Tensor) -> Tensor:
        """Splat/lift (B*S, T, H, W) attention maps onto 2D ground plane (B*S, T, Z_bins, X_bins)."""
        bs, t, h, w = obj_attn.shape
        device = obj_attn.device

        # Resample to regular (Z_bins, X_bins) grid via bilinear grid_sample
        # Target grid on ground plane:
        # Z in [0.0 (near camera/bottom), 1.0 (far / horizon)]
        # X in [-1.0 (left), 1.0 (right)]
        target_z = torch.linspace(0.0, 1.0, steps=self.z_bins, device=device)
        target_x = torch.linspace(-1.0, 1.0, steps=self.x_bins, device=device)
        tz, tx = torch.meshgrid(target_z, target_x, indexing="ij")

        # Ground-to-Image perspective geometry (matching generate_data.py):
        # Horizon y is at ~0.40 of canvas -> in [-1, 1] normalized coord: v_horizon = -0.20
        # Bottom of canvas is +1.0 (span = 1.20)
        # Vertical: dy_norm = ((1.0 - tz) ** 1.6) * 1.20 -> sample_v = -0.20 + dy_norm
        # Horizontal: sample_u = tx * fan_scale * 0.88, where fan_scale = 0.25 + 0.75 * (1.0 - tz)
        sample_fan = 0.25 + 0.75 * (1.0 - tz)
        sample_u = torch.clamp(tx * sample_fan * 0.88, -1.0, 1.0)
        sample_v = torch.clamp(-0.20 + ((1.0 - tz) ** 1.6) * 1.20, -1.0, 1.0)

        sampling_grid = torch.stack([sample_u, sample_v], dim=-1).unsqueeze(0).expand(bs * t, -1, -1, -1)
        flat_attn = obj_attn.reshape(bs * t, 1, h, w)

        oxz_sampled = F.grid_sample(flat_attn, sampling_grid, mode="bilinear", align_corners=True, padding_mode="zeros")
        oxz_t = oxz_sampled.view(bs, t, self.z_bins, self.x_bins)
        return oxz_t

    def forward(
        self,
        obj_attn: Tensor,
        x_vis: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            obj_attn: (B, T, S, H, W)
            x_vis: (B, T, H, W, slot_dim)
        Returns:
            slot_motion_emb: (B, S, slot_dim)
            oxz_density_maps: (B, T, S, Z_bins, X_bins)
        """
        b, t, s, h, w = obj_attn.shape
        bs = b * s

        attn_bs = obj_attn.permute(0, 2, 1, 3, 4).reshape(bs, t, h, w)
        oxz_t = self._lift_attn_to_oxz(attn_bs)  # (B*S, T, Z_bins, X_bins)

        # 1. Temporal channels on the 2D Oxz ground plane:
        max_occ = oxz_t.amax(dim=1, keepdim=True)  # (B*S, 1, Z, X)
        sum_occ = oxz_t.sum(dim=1, keepdim=True).clamp(min=1e-5)

        t_ramp_weights = torch.linspace(0.1, 1.0, steps=t, device=oxz_t.device).view(1, t, 1, 1)
        time_ramp = (oxz_t * t_ramp_weights).sum(dim=1, keepdim=True) / sum_occ

        mid_t = t // 2
        early_occ = oxz_t[:, :mid_t].amax(dim=1, keepdim=True)
        late_occ = oxz_t[:, mid_t:].amax(dim=1, keepdim=True)

        # Centroid deltas (dx, dz) on Oxz plane
        grid_z = torch.linspace(0.0, 1.0, steps=self.z_bins, device=oxz_t.device).view(1, 1, self.z_bins, 1)
        grid_x = torch.linspace(-1.0, 1.0, steps=self.x_bins, device=oxz_t.device).view(1, 1, 1, self.x_bins)
        mass_t = oxz_t.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-5)

        cx_t = (oxz_t * grid_x).sum(dim=(-2, -1), keepdim=True) / mass_t  # (B*S, T, 1, 1)
        cz_t = (oxz_t * grid_z).sum(dim=(-2, -1), keepdim=True) / mass_t

        dx_map = (cx_t[:, -1:] - cx_t[:, :1]).expand(-1, 1, self.z_bins, self.x_bins) * max_occ
        dz_map = (cz_t[:, -1:] - cz_t[:, :1]).expand(-1, 1, self.z_bins, self.x_bins) * max_occ

        channels = [max_occ, time_ramp, early_occ, late_occ, dx_map, dz_map]

        if self.vis_proj is not None and x_vis is not None:
            # Masked visual features: pool x_vis weighted by obj_attn per slot
            x_v = self.vis_proj(x_vis).permute(0, 4, 1, 2, 3)  # (B, vis_dim, T, H, W)
            attn_weight = obj_attn.permute(0, 2, 1, 3, 4).unsqueeze(2)  # (B, S, 1, T, H, W)
            x_v_slot = (x_v.unsqueeze(1) * attn_weight).mean(dim=(3, 4, 5))  # (B, S, vis_dim)
            x_v_map = x_v_slot.reshape(bs, self.vis_dim, 1, 1).expand(-1, -1, self.z_bins, self.x_bins)
            channels.append(x_v_map * max_occ)

        oxz_2d_tensor = torch.cat(channels, dim=1)  # (B*S, in_channels, Z_bins, X_bins)

        # 2. Pure Conv2d forward pass!
        feat_2d = self.conv2d_net(oxz_2d_tensor)  # (B*S, hidden_dim, Z', X')
        pool_avg = F.adaptive_avg_pool2d(feat_2d, 1).flatten(1)
        pool_max = F.adaptive_max_pool2d(feat_2d, 1).flatten(1)
        motion_emb = self.out_proj(torch.cat([pool_avg, pool_max], dim=-1)).view(b, s, self.slot_dim)

        oxz_density = oxz_t.view(b, s, t, self.z_bins, self.x_bins).permute(0, 2, 1, 3, 4)
        return motion_emb, oxz_density


class SlotAttention(nn.Module):
    def __init__(self, num_slots: int = 7, dim: int = 128, eps: float = 1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.eps = eps
        self.scale = dim ** -0.5

        self.obj_queries = nn.Parameter(torch.randn(1, num_slots, dim) * 0.05)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        b, t, h, w, d = inputs.shape
        slots = self.obj_queries.expand(b, -1, -1)

        inputs_flat = inputs.reshape(b, -1, d)
        inputs_norm = self.norm_input(inputs_flat)
        k, v = self.to_k(inputs_norm), self.to_v(inputs_norm)

        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
        attn = dots.softmax(dim=1) + self.eps

        hw = h * w
        attn_t = attn.view(b, self.num_slots, t, hw)
        attn_norm = attn_t / attn_t.sum(dim=-1, keepdim=True).clamp(min=self.eps)

        v_t = v.view(b, t, hw, d)
        # Compute slot representations per frame
        slots_seq = torch.einsum("bljd,bilj->blid", v_t, attn_norm)
        slots_seq = slots_seq + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots_seq))))
        return slots_seq, attn


@MODELS.register_module(name="ActionDepthModel")
class ActionDepthModel(BaseModel):
    """Action Depth Model with 2D Oxz Ground Plane Map Encoder."""

    def __init__(
        self,
        num_slots: int = 6,
        slot_dim: int = 128,
        num_classes: int = NUM_CLASSES,
        bg_slot: bool = True,
        z_bins: int = 16,
        x_bins: int = 16,
        vis_dim: int = 16,
        bev_loss_weight: float = 2.0,
        attn_suppress_weight: float = 0.5,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_classes = num_classes
        self.bg_slot = bg_slot
        self.total_slots = num_slots + 1 if bg_slot else num_slots
        self.bev_loss_weight = float(bev_loss_weight)
        self.attn_suppress_weight = float(attn_suppress_weight)

        # 1. Lightweight 3D Conv Backbone
        self.backbone = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, slot_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(slot_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((16, 14, 14)),
        )

        # 2. Slot Attention
        self.slot_attention = SlotAttention(num_slots=self.total_slots, dim=slot_dim)

        # 3. Novel 2D Oxz Map Encoder (Conv2d over ground plane)
        self.oxz_encoder = Oxz2DMapEncoder(
            slot_dim=slot_dim,
            hidden_dim=96,
            z_bins=z_bins,
            x_bins=x_bins,
            vis_dim=vis_dim,
        )

        # 4. Action Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(slot_dim * 2),
            nn.Linear(slot_dim * 2, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, num_classes // num_slots),  # 8 classes per symbol slot
        )

    def forward_train(self, inputs: Tensor, targets: Dict[str, Any], **kwargs) -> Dict[str, Tensor]:
        # inputs: (B, 3, 16, 224, 224)
        b = inputs.shape[0]
        feats = self.backbone(inputs)  # (B, slot_dim, 16, H, W)
        x_vis = feats.permute(0, 2, 3, 4, 1).contiguous()  # (B, 16, H, W, slot_dim)
        _, t, h, w, _ = x_vis.shape

        slots_seq, attn = self.slot_attention(x_vis)
        # attn: (B, total_slots, T * H * W) -> (B, T, total_slots, H, W)
        attn_5d = attn.view(b, self.total_slots, t, h, w).permute(0, 2, 1, 3, 4)

        obj_slots = slots_seq[:, :, :self.num_slots, :]  # (B, 16, 6, slot_dim)
        obj_attn = attn_5d[:, :, :self.num_slots, :, :]  # (B, 16, 6, H, W)

        # Encode via 2D Oxz ground plane encoder!
        motion_emb, pred_oxz_bev = self.oxz_encoder(obj_attn, x_vis=x_vis)
        vis_emb = obj_slots.mean(dim=1)  # (B, 6, slot_dim)

        fused_slots = torch.cat([vis_emb, motion_emb], dim=-1)  # (B, 6, slot_dim * 2)
        logits_per_slot = self.classifier(fused_slots)  # (B, 6, 8)
        logits_48 = logits_per_slot.view(b, self.num_classes)

        # Losses:
        gt_actor = targets["actor"].to(inputs.device)
        loss_actor = F.binary_cross_entropy_with_logits(logits_48, gt_actor, pos_weight=torch.tensor(4.0, device=inputs.device))

        # Auxiliary Oxz BEV Map Supervision Loss (direct ground-truth 2D tracking supervision!)
        loss_bev = torch.tensor(0.0, device=inputs.device)
        if "oxz_bev" in targets and targets["oxz_bev"] is not None:
            gt_bev = targets["oxz_bev"].to(inputs.device)  # (B, 16, 6, 32, 32)
            # Downsample gt_bev to match (z_bins, x_bins)
            gt_bev_down = F.interpolate(
                gt_bev.view(b * 16 * self.num_slots, 1, 32, 32),
                size=(self.oxz_encoder.z_bins, self.oxz_encoder.x_bins),
                mode="area",
            ).view(b, 16, self.num_slots, self.oxz_encoder.z_bins, self.oxz_encoder.x_bins)
            loss_bev = self.bev_loss_weight * F.mse_loss(pred_oxz_bev, gt_bev_down)

        # Inactive slot suppression loss
        inactive_mask = (gt_actor.view(b, self.num_slots, -1) <= 0.5).all(dim=-1)  # (B, 6)
        loss_suppress = torch.tensor(0.0, device=inputs.device)
        if inactive_mask.any():
            inactive_expanded = inactive_mask.view(b, 1, self.num_slots, 1, 1).expand_as(obj_attn)
            suppressed = obj_attn[inactive_expanded]
            loss_suppress = self.attn_suppress_weight * F.binary_cross_entropy(
                suppressed.clamp(1e-6, 1.0 - 1e-6), torch.zeros_like(suppressed)
            )

        total_loss = loss_actor + loss_bev + loss_suppress
        return {
            "loss": total_loss,
            "loss_actor": loss_actor,
            "loss_bev_oxz": loss_bev,
            "loss_suppress": loss_suppress,
            "pred_actor": logits_48,
            "target_actor": gt_actor,
        }

    def predict(self, inputs: Tensor, **kwargs) -> Dict[str, Tensor]:
        b = inputs.shape[0]
        feats = self.backbone(inputs)
        x_vis = feats.permute(0, 2, 3, 4, 1).contiguous()
        _, t, h, w, _ = x_vis.shape
        slots_seq, attn = self.slot_attention(x_vis)
        attn_5d = attn.view(b, self.total_slots, t, h, w).permute(0, 2, 1, 3, 4)
        obj_slots = slots_seq[:, :, :self.num_slots, :]
        obj_attn = attn_5d[:, :, :self.num_slots, :, :]
        motion_emb, pred_oxz_bev = self.oxz_encoder(obj_attn, x_vis=x_vis)
        vis_emb = obj_slots.mean(dim=1)
        fused_slots = torch.cat([vis_emb, motion_emb], dim=-1)
        logits_48 = self.classifier(fused_slots).view(b, self.num_classes)
        return {
            "logits": logits_48,
            "probs": torch.sigmoid(logits_48),
            "pred_oxz_bev": pred_oxz_bev,
        }

