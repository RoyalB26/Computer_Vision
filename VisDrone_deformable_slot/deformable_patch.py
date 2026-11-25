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

    def forward(
        self,
        patches: torch.Tensor,
        return_spatial: bool = False,
    ):
        x = self.stem(patches)
        x = self.block1(x)
        x = self.block2(x)
        gap = x.mean(dim=(2, 3))
        gmp = x.amax(dim=(2, 3))
        emb = self.head(torch.cat([gap, gmp], dim=-1))
        if return_spatial:
            return emb, x
        return emb


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
        use_uncertainty_routing: bool = False,
        uncertainty_floor: float = 0.2,
        use_sparse_subhead: bool = False,
        reg_max: int = 16,
    ):
        super().__init__()
        self.p3_channels = int(p3_channels)
        self.num_classes = int(num_classes)
        self.num_patches = max(1, int(num_patches))
        self.patch_size = max(8, int(patch_size))
        self.base_patch_radius = float(base_patch_radius)
        self.max_offset_cells = float(max_offset_cells)
        self.spread_local_3x3 = bool(spread_local_3x3)
        self.use_uncertainty_routing = bool(use_uncertainty_routing)
        self.uncertainty_floor = float(uncertainty_floor)
        self.use_sparse_subhead = bool(use_sparse_subhead)
        self.reg_max = int(reg_max)
        self.sub_grid_size = max(1, self.patch_size // 8)

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

        # Optional 4x4 Sparse High-Res Sub-Head directly on MicroPatchEncoder's spatial map
        if self.use_sparse_subhead:
            g = self.sub_grid_size
            gn_groups = min(8, p3_channels)
            # Cell centers in patch-local [-1, 1] coordinates (row-major gy, gx matching flatten(2))
            u = ((torch.arange(g, dtype=torch.float32) + 0.5) / float(g)) * 2.0 - 1.0
            sgy, sgx = torch.meshgrid(u, u, indexing="ij")
            sub_grid_uv = torch.stack([sgx, sgy], dim=-1).view(1, 1, g * g, 2)  # (1, 1, G*G, 2)
            self.register_buffer("sub_grid_uv", sub_grid_uv, persistent=False)

            # Global P3 + Slot-FiLM context injection into each patch's 4x4 map
            self.sub_box_ctx = nn.Linear(p3_channels, p3_channels)
            self.sub_cls_ctx = nn.Linear(p3_channels, p3_channels)

            self.sub_box_refine = nn.Sequential(
                nn.Conv2d(p3_channels, p3_channels, kernel_size=3, padding=1, groups=p3_channels, bias=False),
                nn.Conv2d(p3_channels, p3_channels, kernel_size=1, bias=False),
                nn.GroupNorm(gn_groups, p3_channels),
                nn.SiLU(inplace=True),
            )
            self.sub_cls_refine = nn.Sequential(
                nn.Conv2d(p3_channels, p3_channels, kernel_size=3, padding=1, groups=p3_channels, bias=False),
                nn.Conv2d(p3_channels, p3_channels, kernel_size=1, bias=False),
                nn.GroupNorm(gn_groups, p3_channels),
                nn.SiLU(inplace=True),
            )
            self.sub_cv2 = nn.Conv2d(p3_channels, 4 * self.reg_max, kernel_size=1)
            self.sub_cv3 = nn.Conv2d(p3_channels, num_classes, kernel_size=1)
            # YOLOv11-aligned bias init: DFL ~ 1.0, cls prior ~ 0.01 (prevents Epoch-1 focal spike)
            nn.init.constant_(self.sub_cv2.bias, 1.0)
            nn.init.constant_(self.sub_cv3.bias, -4.5)

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
        cls_p3 = attn_p3[:, : self.num_classes, :, :]  # (B, C, H3, W3)
        fg_sal = (1.0 - attn_p3[:, -1, :, :]).clamp(0.0, 1.0)  # (B, H3, W3)
        weighted_cls = (cls_p3 * self.class_weights).sum(dim=1)  # (B, H3, W3)
        priority = fg_sal * (0.25 + weighted_cls)  # (B, H3, W3)

        if self.use_uncertainty_routing:
            # Uncertainty-Guided Routing: de-prioritize already crystal-clear P3 objects
            # and focus high-res patches on FG/BG boundary ambiguity & inter-class confusion.
            u_fg = (4.0 * fg_sal * (1.0 - fg_sal)).clamp(0.0, 1.0)
            q_cls = cls_p3 / (fg_sal.unsqueeze(1) + 1e-6)
            u_cls = (1.0 - q_cls.amax(dim=1)).clamp(0.0, 1.0)
            uncertainty = torch.maximum(u_fg, fg_sal * u_cls)
            fl = self.uncertainty_floor
            priority = priority * (fl + (1.0 - fl) * uncertainty)

        # CenterNet-style 3x3 Local-Peak NMS: prevents Top-K from dumping 6-10 redundant patches
        # onto the same bright object; spreads the K budget cleanly across distinct small/rare objects.
        p_4d = priority.unsqueeze(1)
        is_peak = (p_4d == F.max_pool2d(p_4d, kernel_size=3, stride=1, padding=1)) & (p_4d > 0.0)
        peak_priority = torch.where(is_peak, p_4d, p_4d * 0.05).flatten(1)
        _, top_idx = torch.topk(peak_priority, k=k_eff, dim=1, sorted=False)  # (B, K)

        # Gather un-attenuated priority values for smooth gating score in [0, 1]
        raw_vals = torch.gather(priority.flatten(1), dim=1, index=top_idx)
        top_scores = torch.sigmoid((raw_vals - 0.15) * 4.0)

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

            # 4. Encode micro-patches -> (B, K, C3) (+ optional 4x4 spatial feature map)
            if self.use_sparse_subhead:
                raw_emb, spatial_feat = self.patch_encoder(patches, return_spatial=True)
            else:
                raw_emb = self.patch_encoder(patches, return_spatial=False)
                spatial_feat = None
            patch_emb = self.drop(raw_emb.view(bs, k_eff, c3))
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

            # 6. Optional 4x4 Sparse High-Res Sub-Head directly on each patch's spatial map
            sub_out: Dict[str, Any] = {}
            if self.use_sparse_subhead and spatial_feat is not None:
                g = int(spatial_feat.shape[-1])
                gg = g * g
                p3_cls_flat = p3_cls_f.flatten(2)  # (B, C3, N3)
                p3_cls_sel = torch.gather(p3_cls_flat, dim=2, index=gather_idx).transpose(1, 2)  # (B, K, C3)

                ctx_box = self.sub_box_ctx(p3_sel).reshape(bs * k_eff, c3, 1, 1)
                ctx_cls = self.sub_cls_ctx(p3_cls_sel).reshape(bs * k_eff, c3, 1, 1)

                sub_box_feat = self.sub_box_refine(spatial_feat + ctx_box)
                sub_cls_feat = self.sub_cls_refine(spatial_feat + ctx_cls)

                sub_boxes_raw = (
                    self.sub_cv2(sub_box_feat)
                    .view(bs, k_eff, 4 * self.reg_max, gg)
                    .permute(0, 2, 1, 3)
                    .reshape(bs, 4 * self.reg_max, k_eff * gg)
                )  # (B, 4*reg_max, K*G*G)
                sub_scores_raw = (
                    self.sub_cv3(sub_cls_feat)
                    .view(bs, k_eff, self.num_classes, gg)
                    .permute(0, 2, 1, 3)
                    .reshape(bs, self.num_classes, k_eff * gg)
                )  # (B, num_classes, K*G*G)

                # Sub-anchor centers in [-1, 1] and patch radii in [-1, 1] (aligned with K*G*G order)
                sub_anchors_norm = (
                    centers_xy[:, :, None, :] + self.sub_grid_uv * radii_xy[:, :, None, :]
                ).clamp(-1.0, 1.0).reshape(bs, k_eff * gg, 2)  # (B, K*G*G, 2)
                sub_radii_norm = (
                    radii_xy[:, :, None, :].expand(-1, -1, gg, -1).reshape(bs, k_eff * gg, 2)
                )  # (B, K*G*G, 2)

                sub_out = {
                    "sub_boxes_raw": sub_boxes_raw.to(dtype=orig_dtype),
                    "sub_scores_raw": sub_scores_raw.to(dtype=orig_dtype),
                    "sub_anchors_norm": sub_anchors_norm,
                    "sub_radii_norm": sub_radii_norm,
                    "sub_grid_size": g,
                }

        patch_info = {
            "ref_xy": ref_xy,
            "centers_xy": centers_xy,
            "radii_xy": radii_xy,
            "top_scores": top_scores,
            "top_idx": top_idx,
            "num_patches": k_eff,
            "grid_hw": (h3, w3),
            **sub_out,
        }
        return p3_box_out, p3_cls_out, patch_info

    def compute_patch_loss(
        self,
        patch_info: Dict[str, Any],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        mask_gt: torch.Tensor,
        img_size: Tuple[int, int],
        reach_cells: float = 2.5,
        scale_padding: float = 1.35,
    ) -> Dict[str, torch.Tensor]:
        """Class-weighted deformable patch supervision (offset + adaptive scale).

        For each sampled patch reference `ref_xy`, selects the best GT object within real
        geometric reach (`<= reach_cells` P3 grid cells) by minimizing weighted effective distance:
            d_eff = dist_cells / (sqrt(class_weight) * small_boost)
        so rare/small objects (e.g. bicycle, tricycle) win over adjacent larger cars.
        """
        ref_xy = patch_info["ref_xy"].float()          # (B, K, 2) in [-1, 1]
        centers_xy = patch_info["centers_xy"].float()  # (B, K, 2) in [-1, 1]
        radii_xy = patch_info["radii_xy"].float()      # (B, K, 2) in [0, 1]
        h3, w3 = patch_info["grid_hw"]
        device = ref_xy.device
        zero = torch.tensor(0.0, device=device, dtype=torch.float32)

        bs, max_m, _ = gt_bboxes.shape
        if max_m == 0:
            return {"loss_patch": zero, "loss_patch_offset": zero, "loss_patch_scale": zero}

        with torch.amp.autocast(device_type=device.type, enabled=False):
            img_h, img_w = float(img_size[0]), float(img_size[1])
            valid = mask_gt.squeeze(-1).bool()  # (B, M)
            lbls = gt_labels.squeeze(-1).long().clamp(0, self.num_classes - 1)  # (B, M)

            # GT centers in [-1, 1] and normalized sizes in [0, 1]
            gb = gt_bboxes.float()
            gt_cx_n = (0.5 * (gb[:, :, 0] + gb[:, :, 2]) / max(img_w, 1.0)).clamp(0.0, 1.0)
            gt_cy_n = (0.5 * (gb[:, :, 1] + gb[:, :, 3]) / max(img_h, 1.0)).clamp(0.0, 1.0)
            gt_w_n = ((gb[:, :, 2] - gb[:, :, 0]) / max(img_w, 1.0)).clamp(min=1e-4, max=1.0)
            gt_h_n = ((gb[:, :, 3] - gb[:, :, 1]) / max(img_h, 1.0)).clamp(min=1e-4, max=1.0)

            gt_xy = torch.stack([gt_cx_n * 2.0 - 1.0, gt_cy_n * 2.0 - 1.0], dim=-1)  # (B, M, 2) in [-1, 1]
            gt_wh = torch.stack([gt_w_n, gt_h_n], dim=-1)                           # (B, M, 2) in [0, 1]

            # Real Euclidean distance in P3 cell units: (B, K, M)
            dx_c = (ref_xy[:, :, None, 0] - gt_xy[:, None, :, 0]) * (w3 * 0.5)
            dy_c = (ref_xy[:, :, None, 1] - gt_xy[:, None, :, 1]) * (h3 * 0.5)
            dist_cells = torch.sqrt(dx_c * dx_c + dy_c * dy_c + 1e-8)  # (B, K, M)

            # Priority weights: sqrt(class_focus_weight) * small_boost (1.3x if < 32^2 at 1280)
            cw_flat = self.class_weights.view(self.num_classes)
            gt_cw = cw_flat[lbls].clamp(min=0.1)  # (B, M)
            gt_sqrt_w = torch.sqrt(gt_cw)          # (B, M)
            is_small_gt = (gt_w_n * gt_h_n) < ((32.0 / 1280.0) ** 2)
            small_boost = torch.where(is_small_gt, torch.full_like(gt_sqrt_w, 1.3), torch.ones_like(gt_sqrt_w))
            priority_denom = (gt_sqrt_w * small_boost).clamp(min=0.1)  # (B, M)

            d_eff = dist_cells / priority_denom[:, None, :]  # (B, K, M)
            in_reach = valid[:, None, :] & (dist_cells <= float(reach_cells))  # (B, K, M)
            d_eff_masked = d_eff.masked_fill(~in_reach, 1e9)

            best_d, best_m = d_eff_masked.min(dim=-1)  # (B, K)
            matched = best_d < 1e8                     # (B, K)

            if not matched.any():
                return {"loss_patch": zero, "loss_patch_offset": zero, "loss_patch_scale": zero}

            # Gather matched GT center, size, and class weight
            m_idx2 = best_m.unsqueeze(-1).expand(-1, -1, 2)
            tgt_xy = torch.gather(gt_xy, dim=1, index=m_idx2)      # (B, K, 2)
            tgt_wh = torch.gather(gt_wh, dim=1, index=m_idx2)      # (B, K, 2)
            tgt_w = torch.gather(gt_sqrt_w, dim=1, index=best_m)   # (B, K)

            # Clamp target center within reachable offset range so offset_head isn't penalized for unreachable targets
            max_dx = (2.0 * self.max_offset_cells) / float(w3)
            max_dy = (2.0 * self.max_offset_cells) / float(h3)
            tgt_dx = (tgt_xy[..., 0] - ref_xy[..., 0]).clamp(-max_dx, max_dx)
            tgt_dy = (tgt_xy[..., 1] - ref_xy[..., 1]).clamp(-max_dy, max_dy)
            pred_dx = centers_xy[..., 0] - ref_xy[..., 0]
            pred_dy = centers_xy[..., 1] - ref_xy[..., 1]

            # Express offset error in P3 cell units (well-scaled O(1))
            err_off_x = F.smooth_l1_loss(pred_dx * (w3 * 0.5), tgt_dx * (w3 * 0.5), reduction="none", beta=0.5)
            err_off_y = F.smooth_l1_loss(pred_dy * (h3 * 0.5), tgt_dy * (h3 * 0.5), reduction="none", beta=0.5)
            per_pt_off = 0.5 * (err_off_x + err_off_y)  # (B, K)

            # Adaptive scale target: (scale_padding * gt_wh / base_patch_radius) clamped to [0.6, 1.4]
            pred_sc = radii_xy / max(self.base_patch_radius, 1e-6)  # (B, K, 2)
            tgt_sc = (tgt_wh * float(scale_padding) / max(self.base_patch_radius, 1e-6)).clamp(0.6, 1.4)
            per_pt_sc = F.smooth_l1_loss(pred_sc, tgt_sc, reduction="none", beta=0.25).mean(dim=-1)  # (B, K)

            w_m = tgt_w[matched]
            w_sum = w_m.sum().clamp(min=1e-6)
            loss_off = (per_pt_off[matched] * w_m).sum() / w_sum
            loss_sc = (per_pt_sc[matched] * w_m).sum() / w_sum
            loss_patch = loss_off + 0.5 * loss_sc

        return {
            "loss_patch": loss_patch,
            "loss_patch_offset": loss_off,
            "loss_patch_scale": loss_sc,
        }
