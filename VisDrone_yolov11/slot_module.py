"""Post-Neck Decoupled Slot Attention + FiLM Refinement (Method B+C) for VisDrone YOLOv11."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PositionEmbeddingSine(nn.Module):
    """2D Sine-Cosine Positional Encoding (DETR-style) for spatial feature maps."""

    def __init__(
        self,
        num_pos_feats: int = 16,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        ones = torch.ones((b, h, w), dtype=torch.float32, device=x.device)
        y_embed = ones.cumsum(1)
        x_embed = ones.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos.to(dtype=x.dtype)


class LightweightSlotAttention(nn.Module):
    """VRAM-efficient Slot Attention over multi-scale tokens N = N_p3 + N_p4 + N_p5.

    Supports:
      - Multi-Prototype Sub-Slots (`num_prototypes = M`): M sub-slots per FG class (K*M + 1 total)
        to model multi-modal classes (truck, awning-tricycle, tricycle), cleanly aggregated back to
        K+1 class-level `attn_ori` & `dots` via logsumexp with `-ln(M)` prior correction.
      - ASA Presence Gate: adds detached `log(presence_gate)` bias to FG sub-slots.
      - Sparse `attn_norm`: prunes tiny background tails (< `sparse_thresh`) when aggregating FG slots.
      - NaN-safe logit clamping `[-30, 30]` in float32.
    """

    def __init__(
        self,
        dim: int = 32,
        num_classes: int = 10,
        num_prototypes: int = 1,
        eps: float = 1e-8,
        slot_dropout: float = 0.1,
        sparse_thresh: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.num_prototypes = max(1, int(num_prototypes))
        self.num_fg_slots = num_classes * self.num_prototypes
        self.num_slots = self.num_fg_slots + 1  # K*M foreground sub-slots + 1 background slot
        self.eps = eps
        self.scale = dim ** -0.5
        self.sparse_thresh = float(sparse_thresh)

        self.obj_queries = nn.Parameter(torch.randn(1, self.num_slots, dim) * 0.05)
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.fc1 = nn.Linear(dim, dim * 2)
        self.dropout = nn.Dropout(slot_dropout)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(
        self,
        inputs: torch.Tensor,
        slots: Optional[torch.Tensor] = None,
        presence_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Slot Attention.

        Returns:
            slots_sub: (B, K*M + 1, D) updated sub-slots (for rich FiLM broadcast & ortho).
            attn_sub:  (B, K*M + 1, N) per-pixel sub-slot posterior.
            slots_cls: (B, K + 1, D) class-level mean-pooled slots.
            attn_cls:  (B, K + 1, N) class-level posterior (sum over M sub-slots per class).
            dots_cls:  (B, K + 1, N) class-level logsumexp logits (exact pre-softmax of attn_cls).
        """
        bs, n_tok, _ = inputs.shape
        k_cls = self.num_classes
        m_proto = self.num_prototypes

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        if slots is None:
            slots = self.obj_queries.expand(bs, -1, -1)

        q = self.to_q(self.norm_slots(slots))
        # Compute attention logits in float32 and clamp to [-30, 30] so exp/logsumexp can NEVER overflow to NaN
        dots_sub = (torch.einsum("bsd,bnd->bsn", q.float(), k.float()) * self.scale).clamp(-30.0, 30.0)

        # Prior correction `-ln(M)` so M FG sub-slots don't get an artificial +ln(M) bias over the 1 BG slot
        if m_proto > 1:
            fg_sub = dots_sub[:, : self.num_fg_slots, :] - math.log(float(m_proto))
            bg_sub = dots_sub[:, self.num_fg_slots :, :]
            dots_sub = torch.cat([fg_sub, bg_sub], dim=1)

        # [ASA Presence Gate] Detach gate so background pixels don't crush active class gates
        if presence_gate is not None:
            gate_bias = torch.log(presence_gate.detach().float().clamp(min=1e-4))  # (B, K)
            if m_proto > 1:
                gate_bias = gate_bias.repeat_interleave(m_proto, dim=1)  # (B, K*M)
            dots_sub = torch.cat(
                [dots_sub[:, : self.num_fg_slots, :] + gate_bias.unsqueeze(-1), dots_sub[:, self.num_fg_slots :, :]],
                dim=1,
            ).clamp(-30.0, 30.0)

        attn_sub = dots_sub.softmax(dim=1)  # (B, K*M + 1, N)

        # [Sparse attn_norm] Prune tiny background tails on FG sub-slots before aggregating prototypes
        if self.sparse_thresh > 0.0:
            fg_a = attn_sub[:, : self.num_fg_slots, :]
            bg_a = attn_sub[:, self.num_fg_slots :, :]
            attn_for_agg = torch.cat([fg_a * (fg_a >= self.sparse_thresh).float(), bg_a], dim=1) + self.eps
        else:
            attn_for_agg = attn_sub + self.eps
        attn_norm = attn_for_agg / attn_for_agg.sum(dim=-1, keepdim=True).clamp(min=self.eps)

        slots_sub = torch.einsum("bnd,bsn->bsd", v.float(), attn_norm).to(dtype=v.dtype)
        slots_sub = slots_sub + self.fc2(self.dropout(F.silu(self.fc1(self.norm_pre_ff(slots_sub)))))

        # Aggregate sub-slots -> class-level (K + 1)
        if m_proto > 1:
            dots_fg_cls = torch.logsumexp(dots_sub[:, : self.num_fg_slots, :].view(bs, k_cls, m_proto, n_tok), dim=2)
            dots_cls = torch.cat([dots_fg_cls, dots_sub[:, self.num_fg_slots :, :]], dim=1)
            attn_fg_cls = attn_sub[:, : self.num_fg_slots, :].view(bs, k_cls, m_proto, n_tok).sum(dim=2)
            attn_cls = torch.cat([attn_fg_cls, attn_sub[:, self.num_fg_slots :, :]], dim=1)
            slots_fg_cls = slots_sub[:, : self.num_fg_slots, :].view(bs, k_cls, m_proto, self.dim).mean(dim=2)
            slots_cls = torch.cat([slots_fg_cls, slots_sub[:, self.num_fg_slots :, :]], dim=1)
        else:
            dots_cls = dots_sub
            attn_cls = attn_sub
            slots_cls = slots_sub

        return slots_sub, attn_sub, slots_cls, attn_cls, dots_cls


class PostNeckSlotFiLMRefiner(nn.Module):
    """Post-Neck Decoupled Slot + FiLM Refiner (Method B + C).

    1. Projects (P3, P4, P5) from YOLOv11Neck to `slot_dim` with 2D Sine PosEmb + Level Embed.
    2. Runs LightweightSlotAttention to obtain image-adaptive class slots (B, S, D) and `attn_ori` (B, S, N).
    3. Broadcasts slots back to (P3, P4, P5) via `attn_ori` (NOT `attn_norm`).
    4. Modulates classification features via Zero-Init FiLM: F_cls = F * (1 + d_gamma) + d_beta
       (or Zero-Init Residual), preserving 100% of pretrained yolo11n.pt behavior at epoch 0.
    """

    def __init__(
        self,
        in_channels: Tuple[int, int, int] = (64, 128, 256),
        num_classes: int = 10,
        slot_dim: int = 32,
        num_prototypes: int = 1,
        fusion_mode: str = "film",
        use_bg_slot_write: bool = False,
        use_logit_prior: bool = True,
        use_checkpoint: bool = False,
        slot_dropout: float = 0.1,
        use_asa_gate: bool = False,
        asa_gate_floor: float = 0.05,
        sparse_thresh: float = 0.0,
    ):
        super().__init__()
        assert slot_dim % 2 == 0, f"slot_dim must be even for 2D Sine PosEmb, got {slot_dim}"
        assert fusion_mode in ("film", "residual"), f"Unsupported fusion_mode: {fusion_mode}"

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.slot_dim = slot_dim
        self.num_prototypes = max(1, int(num_prototypes))
        self.fusion_mode = fusion_mode
        self.use_bg_slot_write = use_bg_slot_write
        self.use_logit_prior = use_logit_prior
        self.use_checkpoint = use_checkpoint
        self.use_asa_gate = bool(use_asa_gate)
        self.asa_gate_floor = float(asa_gate_floor)

        # Lateral 1x1 projections using GroupNorm (zero running stats -> immune to AMP inf/NaN poisoning!)
        gn_groups = min(8, slot_dim)
        while slot_dim % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1
        self.proj_in = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, slot_dim, kernel_size=1, bias=False),
                nn.GroupNorm(gn_groups, slot_dim),
                nn.SiLU(inplace=True),
            )
            for ch in in_channels
        ])

        self.pos_emb = PositionEmbeddingSine(num_pos_feats=slot_dim // 2)
        self.level_embed = nn.Parameter(torch.randn(len(in_channels), slot_dim) * 0.02)

        # [ASA Presence Gate] GAP + GMP over P3 + P4 + P5 (preserves peak signal of tiny small objects)
        if self.use_asa_gate:
            asa_in_dim = 2 * sum(in_channels)
            self.presence_head = nn.Sequential(
                nn.Linear(asa_in_dim, slot_dim * 2),
                nn.SiLU(inplace=True),
                nn.Dropout(slot_dropout),
                nn.Linear(slot_dim * 2, num_classes),
            )
        else:
            self.presence_head = None

        self.slot_attention = LightweightSlotAttention(
            dim=slot_dim,
            num_classes=num_classes,
            num_prototypes=self.num_prototypes,
            slot_dropout=slot_dropout,
            sparse_thresh=sparse_thresh,
        )
        self.slot_drop = nn.Dropout2d(slot_dropout)

        # Zero-initialized FiLM or Residual projections back to each scale's channel dim
        if fusion_mode == "film":
            self.modulators = nn.ModuleList([
                nn.Conv2d(slot_dim, 2 * ch, kernel_size=1, bias=True)
                for ch in in_channels
            ])
        else:
            self.modulators = nn.ModuleList([
                nn.Conv2d(slot_dim, ch, kernel_size=1, bias=True)
                for ch in in_channels
            ])

        for m in self.modulators:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        # Optional direct semantic logit prior added to Detect.cv3 outputs (zero-init)
        if use_logit_prior:
            self.logit_prior_scale = nn.Parameter(torch.zeros(1, num_classes, 1))
        else:
            self.register_parameter("logit_prior_scale", None)

    def forward(
        self, feats: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Refine multi-scale neck features for the classification head branch.

        Args:
            feats: [p3, p4, p5] from YOLOv11Neck.

        Returns:
            cls_feats: [p3_cls, p4_cls, p5_cls] modulated features for `head.cv3`.
            slot_info: dict with 'slots', 'attn_ori', 'dots', 'resolutions', 'logit_prior'.
        """
        bs = feats[0].shape[0]
        orig_dtype = feats[0].dtype
        device = feats[0].device

        # Run entire Slot+FiLM refiner in float32 (only ~3% of FLOPs) + sanitize fp16 inf/NaN inputs
        with torch.amp.autocast(device_type=device.type, enabled=False):
            feats_f = [torch.nan_to_num(f.float(), nan=0.0, posinf=1e4, neginf=-1e4) for f in feats]

            # [ASA Presence Gate] GAP + GMP over P3, P4, P5 with soft floor so tiny objects are never hard-killed
            presence_logits = None
            presence_gate = None
            if self.presence_head is not None:
                pooled = torch.cat(
                    [t for f in feats_f for t in (f.mean(dim=(2, 3)), f.amax(dim=(2, 3)))],
                    dim=-1,
                )
                presence_logits = self.presence_head(pooled)
                fl = self.asa_gate_floor
                presence_gate = fl + (1.0 - fl) * torch.sigmoid(presence_logits)

            tokens_list = []
            resolutions: List[Tuple[int, int]] = []
            counts: List[int] = []

            for i, f in enumerate(feats_f):
                h, w = f.shape[2], f.shape[3]
                resolutions.append((h, w))
                counts.append(h * w)

                proj = self.proj_in[i](f)
                proj = proj + self.pos_emb(proj)
                tok = proj.flatten(2).transpose(1, 2) + self.level_embed[i]  # (B, H*W, D)
                tokens_list.append(tok)

            tokens = torch.cat(tokens_list, dim=1)  # (B, N, D)

            if self.use_checkpoint and self.training:
                slots_sub, attn_sub, slots, attn_ori, dots = checkpoint(
                    self.slot_attention, tokens, None, presence_gate, use_reentrant=False
                )
            else:
                slots_sub, attn_sub, slots, attn_ori, dots = self.slot_attention(
                    tokens, presence_gate=presence_gate
                )

            # Write-back (Broadcast) using rich sub-slots (each pixel gets its exact sub-prototype!)
            s_write = self.slot_attention.num_slots if self.use_bg_slot_write else self.slot_attention.num_fg_slots
            attn_write = attn_sub[:, :s_write, :]
            if presence_gate is not None:
                g_sub = presence_gate.repeat_interleave(self.num_prototypes, dim=1)  # (B, K*M)
                if self.use_bg_slot_write:
                    g_sub = torch.cat([g_sub, torch.ones((bs, 1), device=device, dtype=g_sub.dtype)], dim=1)
                attn_write = attn_write * g_sub.unsqueeze(-1)
            slot_broadcast = torch.einsum("bsd,bsn->bdn", slots_sub[:, :s_write, :], attn_write)  # (B, D, N)

            cls_feats: List[torch.Tensor] = []
            offset = 0
            for i, (f, (h, w), n_i) in enumerate(zip(feats_f, resolutions, counts)):
                slot_map_i = slot_broadcast[:, :, offset : offset + n_i].view(bs, self.slot_dim, h, w)
                offset += n_i
                slot_map_i = self.slot_drop(slot_map_i)

                if self.fusion_mode == "film":
                    delta_gamma, delta_beta = self.modulators[i](slot_map_i).chunk(2, dim=1)
                    # Bound both FiLM scale [0.5, 1.5] and shift [-0.5, +0.5] so P3 l2_mean & small_ratio stay clean
                    f_cls = f * (1.0 + 0.5 * torch.tanh(delta_gamma)) + 0.5 * torch.tanh(delta_beta)
                else:
                    f_cls = f + self.modulators[i](slot_map_i)
                cls_feats.append(f_cls.to(dtype=orig_dtype))

            logit_prior = None
            if self.logit_prior_scale is not None:
                logit_prior = self.logit_prior_scale * dots[:, : self.num_classes, :]  # (B, K, N)

        slot_info = {
            "slots": slots,
            "slots_sub": slots_sub,
            "attn_ori": attn_ori,
            "dots": dots,
            "resolutions": tuple(resolutions),
            "logit_prior": logit_prior,
            "presence_logits": presence_logits,
            "presence_gate": presence_gate,
        }
        return cls_feats, slot_info


class SlotAuxiliaryLoss(nn.Module):
    """Auxiliary semantic supervision for Slot Attention inside YOLOv11.

    Uses unified pixel xyxy GT boxes (matching YOLOv11DetectionLoss) so box formats
    never mismatch, and runs strictly in float32 for AMP stability.
    """

    def __init__(
        self,
        num_classes: int = 10,
        bg_attn_weight: float = 0.3,
        fg_attn_weight: float = 0.7,
        suppress_weight: float = 0.05,
        cls_attn_weight: float = 0.8,
        ortho_weight: float = 0.1,
        presence_loss_weight: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bg_attn_weight = float(bg_attn_weight)
        self.fg_attn_weight = float(fg_attn_weight)
        self.suppress_weight = float(suppress_weight)
        self.cls_attn_weight = float(cls_attn_weight)
        self.ortho_weight = float(ortho_weight)
        self.presence_loss_weight = float(presence_loss_weight)

    @staticmethod
    def _rasterize_masks_from_xyxy(
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        mask_gt: torch.Tensor,
        resolutions: Tuple[Tuple[int, int], ...],
        img_size: Tuple[int, int],
        num_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fully vectorized GPU rasterization of bg_mask (B, N), cls_mask (B, K, N), and gt_counts (B, K).

        Zero `.item()` calls and zero Python loops over boxes -> ~10-20x faster on GPU.
        """
        bs, max_m, _ = gt_bboxes.shape
        device = gt_bboxes.device
        img_h, img_w = float(img_size[0]), float(img_size[1])

        valid = mask_gt.squeeze(-1).bool()  # (B, M)
        lbls = gt_labels.squeeze(-1).long()  # (B, M)
        valid_cls = valid & (lbls >= 0) & (lbls < num_classes)
        lbls_safe = lbls.clamp(0, num_classes - 1)

        # (B, M, K) one-hot masked by valid_cls
        one_hot = F.one_hot(lbls_safe, num_classes=num_classes).float() * valid_cls.unsqueeze(-1).float()
        gt_counts = one_hot.sum(dim=1)  # (B, K)

        if max_m == 0:
            total_n = sum(h * w for h, w in resolutions)
            return (
                torch.ones((bs, total_n), device=device, dtype=torch.float32),
                torch.zeros((bs, num_classes, total_n), device=device, dtype=torch.float32),
                gt_counts,
            )

        bg_scales = []
        cls_scales = []
        valid_f = valid.unsqueeze(-1).float()  # (B, M, 1)

        for h, w in resolutions:
            sx = w / max(img_w, 1.0)
            sy = h / max(img_h, 1.0)

            x1 = (gt_bboxes[:, :, 0] * sx).floor().clamp(0, w - 1)             # (B, M)
            y1 = (gt_bboxes[:, :, 1] * sy).floor().clamp(0, h - 1)             # (B, M)
            x2 = torch.maximum(x1 + 1.0, (gt_bboxes[:, :, 2] * sx).ceil().clamp(0, w))  # (B, M)
            y2 = torch.maximum(y1 + 1.0, (gt_bboxes[:, :, 3] * sy).ceil().clamp(0, h))  # (B, M)

            xs = torch.arange(w, device=device, dtype=torch.float32)  # (W,)
            ys = torch.arange(h, device=device, dtype=torch.float32)  # (H,)

            # (B, M, W) and (B, M, H)
            in_x = (xs.view(1, 1, w) >= x1.unsqueeze(-1)) & (xs.view(1, 1, w) < x2.unsqueeze(-1))
            in_y = (ys.view(1, 1, h) >= y1.unsqueeze(-1)) & (ys.view(1, 1, h) < y2.unsqueeze(-1))

            # (B, M, H*W) float mask of pixels inside each valid box
            box_hw = (in_y.unsqueeze(-1) & in_x.unsqueeze(-2)).view(bs, max_m, h * w).float() * valid_f

            # Background mask: 1.0 where no valid box covers the pixel
            bg_flat = (box_hw.sum(dim=1) == 0.0).float()  # (B, H*W)

            # Per-class mask via batched matmul: (B, K, M) @ (B, M, H*W) -> (B, K, H*W)
            cls_flat = torch.bmm(one_hot.transpose(1, 2), box_hw).clamp_(0.0, 1.0)

            bg_scales.append(bg_flat)
            cls_scales.append(cls_flat)

        return torch.cat(bg_scales, dim=1), torch.cat(cls_scales, dim=2), gt_counts

    def forward(
        self,
        slots: torch.Tensor,
        attn_ori: torch.Tensor,
        obj_queries: torch.Tensor,
        resolutions: Tuple[Tuple[int, int], ...],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        mask_gt: torch.Tensor,
        img_size: Tuple[int, int],
        dots: Optional[torch.Tensor] = None,
        slots_sub: Optional[torch.Tensor] = None,
        presence_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = attn_ori.device
        with torch.amp.autocast(device_type=device.type, enabled=False):
            slots = slots.float()
            attn_ori = attn_ori.float()
            obj_queries = obj_queries.float()
            gt_bboxes = gt_bboxes.float()
            dots_f = dots.float() if dots is not None else None

            n3 = resolutions[0][0] * resolutions[0][1]
            n4 = resolutions[1][0] * resolutions[1][1]

            def _balanced_multiscale_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                p = pred.clamp(1e-4, 1.0 - 1e-4)
                if not torch.isfinite(p).all():
                    p = torch.nan_to_num(p, nan=0.5, posinf=1.0 - 1e-4, neginf=1e-4)
                l_p3 = F.binary_cross_entropy(p[..., :n3], target[..., :n3])
                l_p4 = F.binary_cross_entropy(p[..., n3 : n3 + n4], target[..., n3 : n3 + n4])
                l_p5 = F.binary_cross_entropy(p[..., n3 + n4 :], target[..., n3 + n4 :])
                return (l_p3 + l_p4 + l_p5) / 3.0

            def _balanced_multiscale_bce_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                l_p3 = F.binary_cross_entropy_with_logits(logits[..., :n3], target[..., :n3])
                l_p4 = F.binary_cross_entropy_with_logits(logits[..., n3 : n3 + n4], target[..., n3 : n3 + n4])
                l_p5 = F.binary_cross_entropy_with_logits(logits[..., n3 + n4 :], target[..., n3 + n4 :])
                return (l_p3 + l_p4 + l_p5) / 3.0

            bg_mask, cls_mask, gt_counts = self._rasterize_masks_from_xyxy(
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                mask_gt=mask_gt,
                resolutions=resolutions,
                img_size=img_size,
                num_classes=self.num_classes,
            )

            # Exact logit for P(bg) = sigmoid(d_bg - logsumexp(d_fg)): bounded gradient in [-1, 1], zero 1/p explosion
            bg_logit = (
                dots_f[:, -1, :] - torch.logsumexp(dots_f[:, : self.num_classes, :], dim=1)
                if dots_f is not None
                else None
            )

            # 1. Background slot attention loss
            if self.bg_attn_weight > 0:
                if bg_logit is not None:
                    loss_bg = _balanced_multiscale_bce_logits(bg_logit, bg_mask)
                else:
                    loss_bg = _balanced_multiscale_bce(attn_ori[:, -1, :], bg_mask)
            else:
                loss_bg = torch.tensor(0.0, device=device)

            # 2. Foreground slots sum attention loss (1 - P(bg) = sigmoid(-bg_logit))
            if self.fg_attn_weight > 0:
                fg_mask = 1.0 - bg_mask
                if bg_logit is not None:
                    loss_fg = _balanced_multiscale_bce_logits(-bg_logit, fg_mask)
                else:
                    fg_attn_sum = attn_ori[:, : self.num_classes, :].sum(dim=1)
                    loss_fg = _balanced_multiscale_bce(fg_attn_sum, fg_mask)
            else:
                loss_fg = torch.tensor(0.0, device=device)

            # 3. Inactive class suppression
            if self.suppress_weight > 0:
                inactive = gt_counts == 0
                if inactive.any():
                    inactive_attn = attn_ori[:, : self.num_classes, :][inactive]
                    loss_suppress = _balanced_multiscale_bce(inactive_attn, torch.zeros_like(inactive_attn))
                else:
                    loss_suppress = torch.tensor(0.0, device=device)
            else:
                loss_suppress = torch.tensor(0.0, device=device)

            # 4. Per-class attention Dice + Positive CE (using log_softmax when dots is available -> bounded grad)
            active = gt_counts > 0
            if self.cls_attn_weight > 0 and active.any():
                p_cls = attn_ori[:, : self.num_classes, :].clamp(1e-4, 1.0 - 1e-4)
                inter = (p_cls * cls_mask).sum(dim=-1)
                card = p_cls.sum(dim=-1) + cls_mask.sum(dim=-1)
                dice = 1.0 - (2.0 * inter + 1.0) / (card + 1.0)
                loss_dice = dice[active].mean()

                # Class-balanced Positive CE: average per active class first so rare classes
                # (bicycle, tricycle, awning-tricycle) are not drowned out by car/pedestrian pixels
                pos_m = (cls_mask > 0.5).float()  # (B, K, N)
                pos_cnt = pos_m.sum(dim=-1)       # (B, K)
                valid_pos_cls = active & (pos_cnt > 0)
                if valid_pos_cls.any():
                    if dots_f is not None:
                        neg_log_p = -F.log_softmax(dots_f, dim=1)[:, : self.num_classes, :]
                    else:
                        neg_log_p = -torch.log(p_cls)
                    per_cls_pos_ce = (neg_log_p * pos_m).sum(dim=-1) / pos_cnt.clamp(min=1.0)
                    loss_pos = per_cls_pos_ce[valid_pos_cls].mean()
                else:
                    loss_pos = torch.tensor(0.0, device=device)

                loss_cls_attn = loss_dice + 0.5 * loss_pos
            else:
                loss_cls_attn = torch.tensor(0.0, device=device)

            # 5. Slot orthogonality across ALL FG sub-slots (K*M) so multi-prototypes specialize on distinct modes
            slots_for_ortho = slots_sub.float() if slots_sub is not None else slots
            k_fg = slots_for_ortho.shape[1] - 1
            if self.ortho_weight > 0 and k_fg > 1:
                denom = float(k_fg * (k_fg - 1))
                eye_k = torch.eye(k_fg, device=device, dtype=torch.float32)

                norm_slots = F.normalize(slots_for_ortho[:, :k_fg, :], p=2, dim=-1, eps=1e-6)
                sim_s = torch.bmm(norm_slots, norm_slots.transpose(1, 2))
                loss_slot_ortho = ((sim_s * (1.0 - eye_k.unsqueeze(0))) ** 2).sum(dim=(-1, -2)).mean() / denom

                q_fg = obj_queries[:, :k_fg, :]
                norm_q = F.normalize(q_fg, p=2, dim=-1, eps=1e-6)
                sim_q = torch.bmm(norm_q, norm_q.transpose(1, 2))[0] * (1.0 - eye_k)
                loss_q_ortho = (sim_q ** 2).sum() / denom

                loss_ortho = loss_slot_ortho + 0.5 * loss_q_ortho
            else:
                loss_ortho = torch.tensor(0.0, device=device)

            # 6. ASA Presence Gate BCE Loss
            if self.presence_loss_weight > 0 and presence_logits is not None:
                loss_presence = F.binary_cross_entropy_with_logits(presence_logits.float(), active.float())
            else:
                loss_presence = torch.tensor(0.0, device=device)

            slot_total = (
                self.bg_attn_weight * loss_bg
                + self.fg_attn_weight * loss_fg
                + self.suppress_weight * loss_suppress
                + self.cls_attn_weight * loss_cls_attn
                + self.ortho_weight * loss_ortho
                + self.presence_loss_weight * loss_presence
            )

        return {
            "loss_slot": slot_total,
            "loss_bg_attn": loss_bg,
            "loss_fg_attn": loss_fg,
            "loss_suppress": loss_suppress,
            "loss_cls_attn": loss_cls_attn,
            "loss_ortho": loss_ortho,
            "loss_presence": loss_presence,
        }
