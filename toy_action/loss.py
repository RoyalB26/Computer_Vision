"""Loss functions for Toy Action Slot models."""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def inter_and_union(pred, mask, num_class=1, start_class=0):
    """Compute intersection and union for segmentation evaluation."""
    pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
    mask = mask.data.cpu().numpy().astype(np.uint8)
    pred = np.asarray(pred, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=np.uint8).copy()
    inter = pred * (pred == mask)
    (area_inter, _) = np.histogram(inter, bins=num_class, range=(start_class, num_class))
    (area_pred, _) = np.histogram(pred, bins=num_class, range=(start_class, num_class))
    (area_mask, _) = np.histogram(mask, bins=num_class, range=(start_class, num_class))
    area_union = area_pred + area_mask - area_inter
    return (area_inter, area_union)


class ActionSlotLoss(nn.Module):
    """Action Slot Loss for Toy Action dataset.

    Supervises:
        1. Multi-class slot action predictions (24 slots x 3 states: single, group+, background).
        2. Spatiotemporal Slot Attention mask supervision (using bg_mask).
    """

    def __init__(
        self,
        num_actor_class: int = 48,
        num_slots: int = 6,
        attention_res: Optional[Tuple[int, int]] = (7, 7),
        bce_pos_weight: float = 5.0,
        actor_loss_type: str = "bce",
        use_obj_count_loss: bool = True,
        obj_count_weight: float = 1.0,
        obj_cls_weight: float = 1.0,
        obj_cls_pos_weight: float = 3.0,
        num_obj_classes: int = 12,
        action_attn_weight: float = 1.0,
        bg_attn_weight: float = 0.5,
        feat_recon_weight: float = 0.5,
        dual_stream_aux_weight: float = 0.5,
        bg_mask: bool = True,
        bg_slot: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_actor_class = num_actor_class
        self.num_slots = num_slots
        self.attention_res = attention_res
        self.bce_pos_weight = float(bce_pos_weight)
        self.actor_loss_type = actor_loss_type.lower()
        self.use_obj_count_loss = bool(use_obj_count_loss)
        self.obj_count_weight = float(obj_count_weight)
        self.obj_cls_weight = float(obj_cls_weight)
        self.obj_cls_pos_weight = float(obj_cls_pos_weight)
        self.num_obj_classes = int(num_obj_classes)
        self.action_attn_weight = action_attn_weight
        self.bg_attn_weight = bg_attn_weight
        self.feat_recon_weight = float(feat_recon_weight)
        self.dual_stream_aux_weight = float(dual_stream_aux_weight)
        self.bg_mask = bg_mask
        self.bg_slot = bg_slot
        self.device = device
        class_weights = torch.tensor([bce_pos_weight, bce_pos_weight, 1.0], dtype=torch.float32).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.obj_bce = nn.BCELoss()
        if self.bce_pos_weight != 1.0:
            pos_w = torch.full((num_actor_class,), self.bce_pos_weight, dtype=torch.float32)
        else:
            pos_w = None
        self.register_buffer("bce_pos_weight_tensor", pos_w)
        if self.obj_cls_pos_weight != 1.0:
            obj_pos_w = torch.full((self.num_obj_classes,), self.obj_cls_pos_weight, dtype=torch.float32)
        else:
            obj_pos_w = None
        self.register_buffer("obj_cls_pos_weight_tensor", obj_pos_w)

    def actor_loss_reduce(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """pred: (B, 24, 3) logits, targets: (B, 24) in {0, 1, 2}."""
        b, n, c = pred.shape
        return self.criterion(pred.contiguous().reshape(-1, c), targets.contiguous().reshape(-1))

    def actor_loss_bce(self, pred: torch.Tensor, actor_gt: torch.Tensor) -> torch.Tensor:
        """Multi-label BCEWithLogitsLoss: pred (B, C) logits, actor_gt (B, C) in {0, 1}."""
        return F.binary_cross_entropy_with_logits(
            pred,
            actor_gt.to(device=pred.device, dtype=pred.dtype),
            pos_weight=self.bce_pos_weight_tensor,
            reduction="mean",
        )

    @staticmethod
    def _align_temporal_counts(gt_counts: torch.Tensor, target_len: int) -> torch.Tensor:
        """Align (B, T_gt, S) counts tensor along time to `target_len`."""
        if gt_counts.shape[1] == target_len:
            return gt_counts
        aligned = F.interpolate(
            gt_counts.permute(0, 2, 1).float(),
            size=target_len,
            mode="nearest",
        )
        return aligned.permute(0, 2, 1)

    def obj_count_loss(
        self,
        pred_counts: Optional[torch.Tensor],
        gt_counts: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Compute per-frame, per-class object count regression loss (Smooth L1)."""
        if pred_counts is None or gt_counts is None or self.obj_count_weight <= 0:
            return None
        gt_aligned = self._align_temporal_counts(gt_counts, pred_counts.shape[1])
        gt_aligned = gt_aligned.to(device=pred_counts.device, dtype=pred_counts.dtype)
        loss_val = F.smooth_l1_loss(pred_counts, gt_aligned, beta=0.5, reduction="mean")
        return self.obj_count_weight * loss_val

    def obj_cls_loss(
        self,
        pred_obj_cls: Optional[torch.Tensor],
        actor_gt: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute multi-label BCE loss for object variant classes (e.g. x, x+, ..., O, O+)."""
        if pred_obj_cls is None or self.obj_cls_weight <= 0:
            return None
        b, num_obj_cls = pred_obj_cls.shape
        if actor_gt.shape[1] % num_obj_cls != 0:
            raise ValueError(
                f"actor_gt dim ({actor_gt.shape[1]}) must be divisible by num_obj_cls ({num_obj_cls})."
            )
        obj_cls_gt = (actor_gt.view(b, num_obj_cls, -1) > 0.5).any(dim=-1).to(
            device=pred_obj_cls.device,
            dtype=pred_obj_cls.dtype,
        )
        pos_w = (
            self.obj_cls_pos_weight_tensor
            if self.obj_cls_pos_weight_tensor is not None and self.obj_cls_pos_weight_tensor.numel() == num_obj_cls
            else None
        )
        loss_val = F.binary_cross_entropy_with_logits(
            pred_obj_cls,
            obj_cls_gt,
            pos_weight=pos_w,
            reduction="mean",
        )
        return self.obj_cls_weight * loss_val

    def attn_loss(
        self,
        attn: torch.Tensor,
        bg_seg: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        inactive_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute attention supervision loss using background mask."""
        losses: Dict[str, torch.Tensor] = {}
        if attn is None:
            return losses

        b, l, n, h, w = attn.shape

        # 1. Background slot supervision
        if self.bg_slot and self.bg_attn_weight > 0 and bg_seg is not None:
            # bg_seg shape: (B, 1, L, H_img, W_img) or (B, L, H_img, W_img)
            if bg_seg.dim() == 5:
                bg_seg = bg_seg.squeeze(1)  # (B, L, H_img, W_img)
            # Last slot corresponds to background slot
            bg_attn = attn[:, :, -1, :, :]  # (B, L, H, W)

            # Interpolate bg_attn to bg_seg resolution if needed
            if bg_attn.shape[-2:] != bg_seg.shape[-2:]:
                bg_attn_flat = bg_attn.contiguous().reshape(b * l, 1, h, w)
                bg_attn_up = F.interpolate(bg_attn_flat, size=bg_seg.shape[-2:], mode="bilinear", align_corners=False)
                bg_attn = bg_attn_up.reshape(b, l, bg_seg.shape[-2], bg_seg.shape[-1])

            bg_attn = torch.clamp(bg_attn, 1e-6, 1.0 - 1e-6)
            bg_target = bg_seg.float().to(device=bg_attn.device)
            # In our dataset: 1.0 is background, 0.0 is object
            bg_loss = self.obj_bce(bg_attn, bg_target)
            losses["loss_bg_attn"] = self.bg_attn_weight * bg_loss

        # 2. Inactive slot suppression
        if self.action_attn_weight > 0 and (inactive_mask is not None or targets is not None):
            action_attn = attn[:, :, :self.num_slots, :, :]  # (B, L, num_slots, H, W)
            if inactive_mask is None:
                inactive_mask = (targets == 2)  # (B, num_slots)
            inactive_mask = inactive_mask.to(device=action_attn.device, dtype=torch.bool)
            if inactive_mask.any():
                if inactive_mask.dim() == 2:
                    inactive_expanded = inactive_mask.view(b, 1, self.num_slots, 1, 1).expand_as(action_attn)
                elif inactive_mask.dim() == 3:
                    inactive_expanded = inactive_mask.view(b, l, self.num_slots, 1, 1).expand_as(action_attn)
                else:
                    raise ValueError(f"Unsupported inactive_mask dim: {inactive_mask.dim()}")
                selected_attn = action_attn[inactive_expanded]
                selected_attn = torch.clamp(selected_attn, 1e-6, 1.0 - 1e-6)
                zeros = torch.zeros_like(selected_attn)
                suppression_loss = self.obj_bce(selected_attn, zeros)
                losses["loss_attn_suppress"] = self.action_attn_weight * suppression_loss

        return losses

    def feat_recon_loss(
        self,
        recon_feats: Optional[torch.Tensor],
        target_feats: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Compute MSE feature reconstruction loss (DINOSAUR-style)."""
        if recon_feats is None or target_feats is None or self.feat_recon_weight <= 0:
            return None
        return self.feat_recon_weight * F.mse_loss(
            recon_feats, target_feats.detach().to(device=recon_feats.device, dtype=recon_feats.dtype)
        )

    def dual_stream_aux_loss(
        self,
        pred_var: Optional[torch.Tensor],
        pred_dir: Optional[torch.Tensor],
        actor_gt: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Auxiliary BCE losses supervising Stream A (variants) and Stream B (directions)."""
        if self.dual_stream_aux_weight <= 0 or (pred_var is None and pred_dir is None):
            return None, None
        b = actor_gt.shape[0]
        if actor_gt.shape[1] % (self.num_slots * 2) != 0:
            return None, None

        gt_2x4 = actor_gt.view(b, self.num_slots, 2, -1) > 0.5  # (B, S, 2, 4)
        pos_w_val = min(self.bce_pos_weight, 3.0) if self.bce_pos_weight > 1.0 else None

        loss_var = None
        if pred_var is not None:
            var_gt = gt_2x4.any(dim=-1).to(device=pred_var.device, dtype=pred_var.dtype)
            pos_w = (
                torch.full((pred_var.shape[-1],), pos_w_val, device=pred_var.device, dtype=pred_var.dtype)
                if pos_w_val is not None
                else None
            )
            loss_var = self.dual_stream_aux_weight * F.binary_cross_entropy_with_logits(
                pred_var, var_gt, pos_weight=pos_w, reduction="mean"
            )

        loss_dir = None
        if pred_dir is not None:
            dir_gt = gt_2x4.any(dim=-2).to(device=pred_dir.device, dtype=pred_dir.dtype)
            pos_w = (
                torch.full((pred_dir.shape[-1],), pos_w_val, device=pred_dir.device, dtype=pred_dir.dtype)
                if pos_w_val is not None
                else None
            )
            loss_dir = self.dual_stream_aux_weight * F.binary_cross_entropy_with_logits(
                pred_dir, dir_gt, pos_weight=pos_w, reduction="mean"
            )

        return loss_var, loss_dir

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        label: Union[Dict[str, torch.Tensor], torch.Tensor],
        validate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or raw tensor label
        if isinstance(label, dict):
            actor_gt = label.get("actor", label.get("target", label.get("targets", None)))
            bg_seg = label.get("bg_seg", label.get("bg_mask", None))
            gt_counts = label.get("obj_counts", None)
        else:
            actor_gt = label
            bg_seg = None
            gt_counts = None

        if actor_gt is None:
            raise ValueError("Ground-truth actor labels must be provided.")

        B = actor_gt.shape[0]
        pred_actor = pred["actor"]
        attn = pred.get("attn", None)

        if self.actor_loss_type == "bce" or pred_actor.dim() == 2:
            loss_actor = self.actor_loss_bce(pred_actor, actor_gt)
            ce_targets = None
            if self.use_obj_count_loss and gt_counts is not None and attn is not None:
                counts_aligned = self._align_temporal_counts(gt_counts, attn.shape[1])
                inactive_mask = counts_aligned < 0.5  # (B, L, num_slots)
            elif actor_gt.shape[1] % self.num_slots == 0:
                inactive_mask = (actor_gt.view(B, self.num_slots, -1) <= 0.5).all(dim=-1)  # (B, num_slots)
            else:
                inactive_mask = None
        else:
            # Legacy (B, 24, 3) CE mode: 6 symbols x 2 variants [single, group+] x 4 actions
            reshaped = actor_gt.view(B, 6, 2, 4)
            fg = reshaped.permute(0, 1, 3, 2).reshape(B, 24, 2)
            ce_targets = torch.where(
                fg[:, :, 0] > 0.5,
                torch.zeros(B, 24, dtype=torch.long, device=actor_gt.device),
                torch.where(
                    fg[:, :, 1] > 0.5,
                    torch.ones(B, 24, dtype=torch.long, device=actor_gt.device),
                    torch.full((B, 24), 2, dtype=torch.long, device=actor_gt.device),
                ),
            )
            loss_actor = self.actor_loss_reduce(pred_actor, ce_targets)
            inactive_mask = None

        attn_losses = self.attn_loss(
            attn,
            bg_seg=bg_seg,
            targets=ce_targets,
            inactive_mask=inactive_mask,
        )
        if self.use_obj_count_loss:
            loss_count = self.obj_count_loss(pred.get("obj_counts", None), gt_counts)
            loss_obj_cls = None
        else:
            loss_count = None
            loss_obj_cls = self.obj_cls_loss(pred.get("obj_cls", None), actor_gt)

        loss_recon = self.feat_recon_loss(
            pred.get("recon_feats", None),
            pred.get("target_feats", None),
        )
        loss_stream_var, loss_stream_dir = self.dual_stream_aux_loss(
            pred.get("var_logits", None),
            pred.get("dir_logits", None),
            actor_gt,
        )

        total_loss = loss_actor
        for v in attn_losses.values():
            total_loss = total_loss + v
        if loss_count is not None:
            total_loss = total_loss + loss_count
        if loss_obj_cls is not None:
            total_loss = total_loss + loss_obj_cls
        if loss_recon is not None:
            total_loss = total_loss + loss_recon
        if loss_stream_var is not None:
            total_loss = total_loss + loss_stream_var
        if loss_stream_dir is not None:
            total_loss = total_loss + loss_stream_dir

        loss_dict: Dict[str, torch.Tensor] = {
            "loss": total_loss,
            "loss_actor": loss_actor,
        }
        if loss_count is not None:
            loss_dict["loss_obj_count"] = loss_count
        if loss_obj_cls is not None:
            loss_dict["loss_obj_cls"] = loss_obj_cls
        if loss_recon is not None:
            loss_dict["loss_feat_recon"] = loss_recon
        if loss_stream_var is not None:
            loss_dict["loss_stream_var"] = loss_stream_var
        if loss_stream_dir is not None:
            loss_dict["loss_stream_dir"] = loss_stream_dir
        loss_dict.update(attn_losses)
        return loss_dict
