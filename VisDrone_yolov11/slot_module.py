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

    Removes the heavy per-token input MLP (FC1/FC2 on 18,900 tokens) to cut ~60% activation VRAM
    while keeping full slot-level FFN and class-supervised attention dynamics.
    """

    def __init__(self, dim: int = 32, num_classes: int = 10, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.num_slots = num_classes + 1  # K foreground classes + 1 background slot
        self.eps = eps
        self.scale = dim ** -0.5

        self.obj_queries = nn.Parameter(torch.randn(1, self.num_slots, dim) * 0.05)
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(
        self,
        inputs: torch.Tensor,
        slots: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Slot Attention.

        Args:
            inputs: (B, N, D) flattened multi-scale feature tokens.
            slots: Optional initial slots (B, S, D).

        Returns:
            slots: (B, S, D) updated image-adaptive class prototypes.
            attn_ori: (B, S, N) per-pixel posterior probability over slots (sum over S == 1).
            dots: (B, S, N) raw scaled dot-product logits before softmax.
        """
        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        if slots is None:
            slots = self.obj_queries.expand(inputs.shape[0], -1, -1)

        q = self.to_q(self.norm_slots(slots))
        # Compute attention logits & softmax in float32 (keep attn_ori/dots in float32 so small
        # probabilities don't underflow to 0.0 in fp16 and lose gradients in SlotAuxiliaryLoss)
        dots = torch.einsum("bsd,bnd->bsn", q.float(), k.float()) * self.scale

        attn_ori = dots.softmax(dim=1)
        attn_eps = attn_ori + self.eps
        attn_norm = attn_eps / attn_eps.sum(dim=-1, keepdim=True).clamp(min=self.eps)

        slots = torch.einsum("bnd,bsn->bsd", v.float(), attn_norm).to(dtype=v.dtype)
        slots = slots + self.fc2(F.silu(self.fc1(self.norm_pre_ff(slots))))
        return slots, attn_ori, dots


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
        fusion_mode: str = "film",
        use_bg_slot_write: bool = False,
        use_logit_prior: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        assert slot_dim % 2 == 0, f"slot_dim must be even for 2D Sine PosEmb, got {slot_dim}"
        assert fusion_mode in ("film", "residual"), f"Unsupported fusion_mode: {fusion_mode}"

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.slot_dim = slot_dim
        self.fusion_mode = fusion_mode
        self.use_bg_slot_write = use_bg_slot_write
        self.use_logit_prior = use_logit_prior
        self.use_checkpoint = use_checkpoint

        # Lateral 1x1 projections from YOLOv11Neck (64, 128, 256) -> slot_dim
        self.proj_in = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, slot_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(slot_dim),
                nn.SiLU(inplace=True),
            )
            for ch in in_channels
        ])

        self.pos_emb = PositionEmbeddingSine(num_pos_feats=slot_dim // 2)
        self.level_embed = nn.Parameter(torch.randn(len(in_channels), slot_dim) * 0.02)

        self.slot_attention = LightweightSlotAttention(dim=slot_dim, num_classes=num_classes)

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
        tokens_list = []
        resolutions: List[Tuple[int, int]] = []
        counts: List[int] = []

        for i, f in enumerate(feats):
            h, w = f.shape[2], f.shape[3]
            resolutions.append((h, w))
            counts.append(h * w)

            proj = self.proj_in[i](f)
            proj = proj + self.pos_emb(proj)
            tok = proj.flatten(2).transpose(1, 2) + self.level_embed[i]  # (B, H*W, D)
            tokens_list.append(tok)

        tokens = torch.cat(tokens_list, dim=1)  # (B, N, D)

        if self.use_checkpoint and self.training:
            slots, attn_ori, dots = checkpoint(self.slot_attention, tokens, use_reentrant=False)
        else:
            slots, attn_ori, dots = self.slot_attention(tokens)

        # Write-back (Broadcast) using attn_ori (posterior over slots)
        s_write = self.slot_attention.num_slots if self.use_bg_slot_write else self.num_classes
        slot_broadcast = torch.einsum(
            "bsd,bsn->bdn",
            slots[:, :s_write, :].float(),
            attn_ori[:, :s_write, :].float(),
        ).to(dtype=feats[0].dtype)  # (B, D, N)

        cls_feats: List[torch.Tensor] = []
        offset = 0
        for i, (f, (h, w), n_i) in enumerate(zip(feats, resolutions, counts)):
            slot_map_i = slot_broadcast[:, :, offset : offset + n_i].view(bs, self.slot_dim, h, w)
            offset += n_i

            if self.fusion_mode == "film":
                delta_gamma, delta_beta = self.modulators[i](slot_map_i).chunk(2, dim=1)
                f_cls = f * (1.0 + delta_gamma) + delta_beta
            else:
                f_cls = f + self.modulators[i](slot_map_i)
            cls_feats.append(f_cls)

        logit_prior = None
        if self.logit_prior_scale is not None:
            logit_prior = self.logit_prior_scale * dots[:, : self.num_classes, :]  # (B, K, N)

        slot_info = {
            "slots": slots,
            "attn_ori": attn_ori,
            "dots": dots,
            "resolutions": tuple(resolutions),
            "logit_prior": logit_prior,
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
        bg_attn_weight: float = 0.5,
        fg_attn_weight: float = 0.5,
        suppress_weight: float = 0.05,
        cls_attn_weight: float = 0.5,
        ortho_weight: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bg_attn_weight = float(bg_attn_weight)
        self.fg_attn_weight = float(fg_attn_weight)
        self.suppress_weight = float(suppress_weight)
        self.cls_attn_weight = float(cls_attn_weight)
        self.ortho_weight = float(ortho_weight)

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
    ) -> Dict[str, torch.Tensor]:
        device = attn_ori.device
        with torch.amp.autocast(device_type=device.type, enabled=False):
            slots = slots.float()
            attn_ori = attn_ori.float()
            obj_queries = obj_queries.float()
            gt_bboxes = gt_bboxes.float()

            n3 = resolutions[0][0] * resolutions[0][1]
            n4 = resolutions[1][0] * resolutions[1][1]

            def _balanced_multiscale_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                p = pred.clamp(1e-6, 1.0 - 1e-6)
                if not torch.isfinite(p).all():
                    p = torch.nan_to_num(p, nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6)
                l_p3 = F.binary_cross_entropy(p[..., :n3], target[..., :n3])
                l_p4 = F.binary_cross_entropy(p[..., n3 : n3 + n4], target[..., n3 : n3 + n4])
                l_p5 = F.binary_cross_entropy(p[..., n3 + n4 :], target[..., n3 + n4 :])
                return (l_p3 + l_p4 + l_p5) / 3.0

            bg_mask, cls_mask, gt_counts = self._rasterize_masks_from_xyxy(
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                mask_gt=mask_gt,
                resolutions=resolutions,
                img_size=img_size,
                num_classes=self.num_classes,
            )

            # 1. Background slot attention loss
            if self.bg_attn_weight > 0:
                loss_bg = _balanced_multiscale_bce(attn_ori[:, -1, :], bg_mask)
            else:
                loss_bg = torch.tensor(0.0, device=device)

            # 2. Foreground slots sum attention loss
            if self.fg_attn_weight > 0:
                fg_mask = 1.0 - bg_mask
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

            # 4. Per-class attention Dice + Positive CE
            active = gt_counts > 0
            if self.cls_attn_weight > 0 and active.any():
                p_cls = attn_ori[:, : self.num_classes, :].clamp(1e-6, 1.0 - 1e-6)
                inter = (p_cls * cls_mask).sum(dim=-1)
                card = p_cls.sum(dim=-1) + cls_mask.sum(dim=-1)
                dice = 1.0 - (2.0 * inter + 1.0) / (card + 1.0)
                loss_dice = dice[active].mean()

                pos_pixels = (cls_mask > 0.5) & active.unsqueeze(-1)
                if pos_pixels.any():
                    loss_pos = -torch.log(p_cls[pos_pixels]).mean()
                else:
                    loss_pos = torch.tensor(0.0, device=device)

                loss_cls_attn = loss_dice + 0.5 * loss_pos
            else:
                loss_cls_attn = torch.tensor(0.0, device=device)

            # 5. Slot orthogonality (anti-collapse)
            if self.ortho_weight > 0 and self.num_classes > 1:
                k = self.num_classes
                denom = float(k * (k - 1))
                eye_k = torch.eye(k, device=device, dtype=torch.float32)

                norm_slots = F.normalize(slots[:, :k, :], p=2, dim=-1)
                sim_s = torch.bmm(norm_slots, norm_slots.transpose(1, 2))
                loss_slot_ortho = ((sim_s * (1.0 - eye_k.unsqueeze(0))) ** 2).sum(dim=(-1, -2)).mean() / denom

                norm_q = F.normalize(obj_queries[:, :k, :], p=2, dim=-1)
                sim_q = torch.bmm(norm_q, norm_q.transpose(1, 2))[0] * (1.0 - eye_k)
                loss_q_ortho = (sim_q ** 2).sum() / denom

                loss_ortho = loss_slot_ortho + 0.5 * loss_q_ortho
            else:
                loss_ortho = torch.tensor(0.0, device=device)

            slot_total = (
                self.bg_attn_weight * loss_bg
                + self.fg_attn_weight * loss_fg
                + self.suppress_weight * loss_suppress
                + self.cls_attn_weight * loss_cls_attn
                + self.ortho_weight * loss_ortho
            )

        return {
            "loss_slot": slot_total,
            "loss_bg_attn": loss_bg,
            "loss_fg_attn": loss_fg,
            "loss_suppress": loss_suppress,
            "loss_cls_attn": loss_cls_attn,
            "loss_ortho": loss_ortho,
        }
