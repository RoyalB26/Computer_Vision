"""Slot-Guided Deformable High-Resolution Micro-Patch Injector (Approach 2).

Instead of running the entire Backbone+Neck on full high-resolution images or cropping
5 large rectangular tiles, this module:
  1. Uses the P3 Slot Attention map (`1 - A_bg` + per-class small/rare weights) to select
     only the Top-K most salient foreground P3 locations (e.g. K=192 out of 14,400 = ~1.3%).
  2. Predicts learned deformable center offsets (dx, dy) and adaptive patch scales (sx, sy)
     from P3 features at those K locations (Deformable-DETR style, 100% differentiable).
  3. Samples K high-resolution RGB micro-patches (P x P) directly from the original high-res
     image via a single batched `F.grid_sample` call.
  4. Encodes the K micro-patches with a lightweight GroupNorm Micro-CNN (~1-2 GFLOPs) and
     injects the high-res detail back into P3 via Zero-Init scatter-add + local depthwise spread,
     preserving 100% of pretrained yolo11n.pt behavior at Epoch 0.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroPatchEncoder(nn.Module):
    """Lightweight GroupNorm Depthwise-Separable CNN encoding (B*K, 3, P, P) -> (B*K, out_dim)."""

    def __init__(self, out_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        g1 = min(8, hidden_dim)
        g2 = min(8, out_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(g1, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(g2, out_dim),
            nn.SiLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, groups=out_dim, bias=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(g2, out_dim),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = self.stem(patches)
        x = self.block1(x)
        x = self.block2(x)
        gap = x.mean(dim=(2, 3))
        gmp = x.amax(dim=(2, 3))
        return self.head(torch.cat([gap, gmp], dim=-1))


class DeformableHighResPatchInjector(nn.Module):
    """Slot-Guided Deformable High-Res Micro-Patch Sampling & P3 Residual Injection."""

    def __init__(
        self,
        p3_channels: int = 64,
        num_classes: int = 10,
        num_patches: int = 192,
        patch_size: int = 32,
        base_patch_radius: float = 0.035,   # 45x45px span on 1280x1280 (dynamic 27..63px for Tiny & Small p90)
        max_offset_cells: float = 1.5,      # Max deformable center shift in P3 grid cells
        class_focus_weights: Optional[List[float]] = None,
        spread_local_3x3: bool = True,
        patch_dropout: float = 0.1,
    ):
        super().__init__()
        self.p3_channels = int(p3_channels)
        self.num_classes = int(num_classes)
        self.num_patches = max(1, int(num_patches))
        self.patch_size = max(8, int(patch_size))
        self.base_patch_radius = float(base_patch_radius)
        self.max_offset_cells = float(max_offset_cells)
        self.spread_local_3x3 = bool(spread_local_3x3)

        if class_focus_weights is None:
            # Default: emphasize tiny & rare VisDrone classes (ped, people, bicycle, tricycle, awning-tri, moto)
            class_focus_weights = [2.0, 2.0, 3.0, 0.3, 0.5, 0.5, 3.0, 3.0, 0.3, 2.0]
        assert len(class_focus_weights) == num_classes, (
            f"class_focus_weights len ({len(class_focus_weights)}) != num_classes ({num_classes})"
        )
        self.register_buffer(
            "class_weights",
            torch.tensor(class_focus_weights, dtype=torch.float32).view(1, num_classes, 1, 1),
            persistent=False,
        )

        # Deformable offset (dx, dy) and adaptive scale (sx, sy) predictors — zero-init for clean start
        self.offset_head = nn.Linear(p3_channels, 2)
        self.scale_head = nn.Linear(p3_channels, 2)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

        # Lightweight high-res RGB micro-patch encoder
        self.patch_encoder = MicroPatchEncoder(out_dim=p3_channels, hidden_dim=max(24, p3_channels // 2))
        self.drop = nn.Dropout(patch_dropout)

        # Zero-init projection back to P3 box/cls feature space (preserves pretrained yolo11n.pt at Epoch 0)
        self.box_proj = nn.Linear(p3_channels, p3_channels)
        self.cls_proj = nn.Linear(p3_channels, p3_channels)
        nn.init.zeros_(self.box_proj.weight)
        nn.init.zeros_(self.box_proj.bias)
        nn.init.zeros_(self.cls_proj.weight)
        nn.init.zeros_(self.cls_proj.bias)

        # Optional local 3x3 depthwise spread so immediate neighbor P3 anchors also get high-res context
        if self.spread_local_3x3:
            self.local_spread = nn.Conv2d(p3_channels, p3_channels, kernel_size=3, padding=1, groups=p3_channels, bias=False)
            # Initialize as identity (1 at center, 0 at neighbors)
            nn.init.zeros_(self.local_spread.weight)
            with torch.no_grad():
                self.local_spread.weight[:, 0, 1, 1] = 1.0
        else:
            self.local_spread = None

        # Precompute base P x P sampling grid in [-1, 1]
        lin = torch.linspace(-1.0, 1.0, self.patch_size, dtype=torch.float32)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        base_grid = torch.stack([gx, gy], dim=-1).view(1, 1, self.patch_size, self.patch_size, 2)  # (1, 1, P, P, 2)
        self.register_buffer("base_grid", base_grid, persistent=False)

    def _select_topk_locations(
        self,
        attn_ori: torch.Tensor,
        h3: int,
        w3: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select Top-K P3 spatial indices using Slot Attention foreground & rare-class saliency.

        Returns:
            top_idx: (B, K) flat spatial indices in [0, H3*W3).
            top_scores: (B, K) saliency confidence scores in [0, 1].
            ref_xy: (B, K, 2) normalized reference coordinates in [-1, 1] (x, y).
        """
        bs = attn_ori.shape[0]
        n3 = h3 * w3
        k_eff = min(self.num_patches, n3)

        attn_p3 = attn_ori[:, :, :n3].view(bs, self.num_classes + 1, h3, w3).float()
        fg_sal = (1.0 - attn_p3[:, -1, :, :]).clamp(0.0, 1.0)  # (B, H3, W3)
        weighted_cls = (attn_p3[:, : self.num_classes, :, :] * self.class_weights).sum(dim=1)  # (B, H3, W3)
        priority = fg_sal * (0.25 + weighted_cls)  # (B, H3, W3)

        # Mild 3x3 max-pool boost so sharp isolated tiny objects (even near borders) win Top-K
        priority_flat = (0.7 * priority + 0.3 * F.max_pool2d(priority.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)).flatten(1)
        top_vals, top_idx = torch.topk(priority_flat, k=k_eff, dim=1, sorted=False)  # (B, K)

        # Normalize gating score smoothly to [0, 1]
        top_scores = torch.sigmoid((top_vals - 0.15) * 4.0)

        gy = (top_idx // w3).float()
        gx = (top_idx % w3).float()
        # Map cell centers (gx + 0.5) / w3 in [0, 1] -> [-1, 1] for F.grid_sample(align_corners=False)
        ref_x = ((gx + 0.5) / float(w3)) * 2.0 - 1.0
        ref_y = ((gy + 0.5) / float(h3)) * 2.0 - 1.0
        ref_xy = torch.stack([ref_x, ref_y], dim=-1)  # (B, K, 2)
        return top_idx, top_scores, ref_xy

    def forward(
        self,
        x_hr: torch.Tensor,
        p3_box: torch.Tensor,
        p3_cls: torch.Tensor,
        attn_ori: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Sample deformable high-res micro-patches and inject residuals into P3 (box & cls).

        Args:
            x_hr:   (B, 3, H_hr, W_hr) high-resolution RGB image tensor.
            p3_box: (B, C3, H3, W3) Neck P3 feature map for box regression (`head.cv2[0]`).
            p3_cls: (B, C3, H3, W3) Slot-FiLM P3 feature map for classification (`head.cv3[0]`).
            attn_ori: (B, K_cls + 1, N) multi-scale Slot Attention posterior.

        Returns:
            p3_box_refined: (B, C3, H3, W3)
            p3_cls_refined: (B, C3, H3, W3)
            patch_info: diagnostic dict with 'ref_xy', 'centers_xy', 'radii_xy', 'top_scores'.
        """
        bs, c3, h3, w3 = p3_box.shape
        orig_dtype = p3_box.dtype
        device = p3_box.device

        # Run grid_sample + micro-encoder in float32 for AMP stability (only ~1-2 GFLOPs)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            x_hr_f = x_hr.float()
            p3_box_f = p3_box.float()
            p3_cls_f = p3_cls.float()

            # 1. Select Top-K foreground P3 locations guided by Slot Attention
            top_idx, top_scores, ref_xy = self._select_topk_locations(attn_ori, h3, w3)  # (B, K), (B, K), (B, K, 2)
            k_eff = top_idx.shape[1]

            # Gather P3 features at Top-K locations: (B, C3, H3*W3) -> (B, K, C3)
            p3_flat = p3_box_f.flatten(2)  # (B, C3, N3)
            gather_idx = top_idx.unsqueeze(1).expand(-1, c3, -1)  # (B, C3, K)
            p3_sel = torch.gather(p3_flat, dim=2, index=gather_idx).transpose(1, 2)  # (B, K, C3)

            # 2. Predict learned deformable center offsets & adaptive patch radii
            max_dx = (2.0 * self.max_offset_cells) / float(w3)
            max_dy = (2.0 * self.max_offset_cells) / float(h3)
            raw_off = torch.tanh(self.offset_head(p3_sel))  # (B, K, 2) in [-1, 1]
            delta_xy = torch.stack([raw_off[..., 0] * max_dx, raw_off[..., 1] * max_dy], dim=-1)
            centers_xy = (ref_xy + delta_xy).clamp(-1.0, 1.0)  # (B, K, 2)

            # Adaptive patch half-size in [-1, 1] coordinates (clamped within [0.6x, 1.4x] of base_patch_radius)
            scale_fac = 1.0 + 0.4 * torch.tanh(self.scale_head(p3_sel))  # (B, K, 2)
            radii_xy = self.base_patch_radius * scale_fac  # (B, K, 2)

            # 3. Build batched deformable sampling grid (B, K*P, P, 2) -> single GPU grid_sample!
            #    Reshaping K*P along height lets us sample all K patches per image in ONE F.grid_sample call
            #    WITHOUT duplicating x_hr K times in VRAM!
            grid = (
                centers_xy[:, :, None, None, :]
                + self.base_grid * radii_xy[:, :, None, None, :]
            )  # (B, K, P, P, 2)
            grid_packed = grid.view(bs, k_eff * self.patch_size, self.patch_size, 2)  # (B, K*P, P, 2)
            sampled_packed = F.grid_sample(
                x_hr_f,
                grid_packed,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )  # (B, 3, K*P, P)
            patches = (
                sampled_packed.view(bs, 3, k_eff, self.patch_size, self.patch_size)
                .permute(0, 2, 1, 3, 4)
                .contiguous()
                .view(bs * k_eff, 3, self.patch_size, self.patch_size)
            )  # (B*K, 3, P, P)

            # 4. Encode micro-patches -> (B, K, C3)
            patch_emb = self.drop(self.patch_encoder(patches).view(bs, k_eff, c3))
            gate = top_scores.unsqueeze(-1)  # (B, K, 1)

            delta_box = (self.box_proj(patch_emb) * gate).transpose(1, 2)  # (B, C3, K)
            delta_cls = (self.cls_proj(patch_emb) * gate).transpose(1, 2)  # (B, C3, K)

            # 5. Scatter-add sparse high-res residuals back onto P3 grid
            canvas_box = torch.zeros_like(p3_flat)
            canvas_cls = torch.zeros_like(p3_flat)
            canvas_box.scatter_add_(2, gather_idx, delta_box)
            canvas_cls.scatter_add_(2, gather_idx, delta_cls)

            map_box = canvas_box.view(bs, c3, h3, w3)
            map_cls = canvas_cls.view(bs, c3, h3, w3)
            if self.local_spread is not None:
                map_box = self.local_spread(map_box)
                map_cls = self.local_spread(map_cls)

            p3_box_out = (p3_box_f + map_box).to(dtype=orig_dtype)
            p3_cls_out = (p3_cls_f + map_cls).to(dtype=orig_dtype)

        patch_info = {
            "ref_xy": ref_xy,
            "centers_xy": centers_xy,
            "radii_xy": radii_xy,
            "top_scores": top_scores,
            "num_patches": k_eff,
        }
        return p3_box_out, p3_cls_out, patch_info
