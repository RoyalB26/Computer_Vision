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
        num_slots: int = 24,
        attention_res: Optional[Tuple[int, int]] = (7, 7),
        bce_pos_weight: float = 5.0,
        action_attn_weight: float = 1.0,
        bg_attn_weight: float = 0.5,
        bg_mask: bool = True,
        bg_slot: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_actor_class = num_actor_class
        self.num_slots = num_slots
        self.attention_res = attention_res
        self.bce_pos_weight = bce_pos_weight
        self.action_attn_weight = action_attn_weight
        self.bg_attn_weight = bg_attn_weight
        self.bg_mask = bg_mask
        self.bg_slot = bg_slot
        self.device = device
        class_weights = torch.tensor([bce_pos_weight, bce_pos_weight, 1.0], dtype=torch.float32).to(device)
        self.criterion = nn.CrossEntropyLoss(weight= class_weights, reduction="mean")
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.obj_bce = nn.BCELoss()

    def actor_loss_reduce(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """pred: (B, 24, 3) logits, targets: (B, 24) in {0, 1, 2}."""
        b, n, c = pred.shape
        return self.criterion(pred.contiguous().reshape(-1, c), targets.contiguous().reshape(-1))

    def attn_loss(
        self,
        attn: torch.Tensor,
        bg_seg: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
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

        # 2. Inactive action slot suppression
        if self.action_attn_weight > 0 and targets is not None:
            # targets: (B, 24) where 2 means inactive
            action_attn = attn[:, :, :self.num_slots, :, :]  # (B, L, num_slots, H, W)
            inactive_mask = (targets == 2)  # (B, num_slots)
            if inactive_mask.any():
                inactive_expanded = inactive_mask.view(b, 1, self.num_slots, 1, 1).expand_as(action_attn)
                selected_attn = action_attn[inactive_expanded]
                selected_attn = torch.clamp(selected_attn, 1e-6, 1.0 - 1e-6)
                zeros = torch.zeros_like(selected_attn)
                suppression_loss = self.obj_bce(selected_attn, zeros)
                losses["loss_attn_suppress"] = self.action_attn_weight * suppression_loss

        return losses

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
        else:
            actor_gt = label
            bg_seg = None

        if actor_gt is None:
            raise ValueError("Ground-truth actor labels must be provided.")

        B = actor_gt.shape[0]

        # Convert (B, 48) multi-hot vector into (B, 24) slot targets in {0, 1, 2}
        # 6 symbols x 2 variants [single, group+] x 4 actions
        reshaped = actor_gt.view(B, 6, 2, 4)
        fg = reshaped.permute(0, 1, 3, 2).reshape(B, 24, 2)

        targets = torch.where(
            fg[:, :, 0] > 0.5,
            torch.zeros(B, 24, dtype=torch.long, device=actor_gt.device),
            torch.where(
                fg[:, :, 1] > 0.5,
                torch.ones(B, 24, dtype=torch.long, device=actor_gt.device),
                torch.full((B, 24), 2, dtype=torch.long, device=actor_gt.device),
            ),
        )

        loss_actor = self.actor_loss_reduce(pred["actor"], targets)
        attn_losses = self.attn_loss(pred.get("attn", None), bg_seg=bg_seg, targets=targets)

        total_loss = loss_actor
        for v in attn_losses.values():
            total_loss = total_loss + v

        loss_dict: Dict[str, torch.Tensor] = {
            "loss": total_loss,
            "loss_actor": loss_actor,
        }
        loss_dict.update(attn_losses)
        return loss_dict
