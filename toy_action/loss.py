"""Loss functions for Toy Action Deformable-Mamba models."""

from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableActionLoss(nn.Module):
    """Multi-label action classification loss for Deformable-Mamba Toy Action models."""

    def __init__(
        self,
        num_actor_class: int = 48,
        bce_pos_weight: float = 5.0,
        ego_loss_weight: float = 0.2,
    ):
        super().__init__()
        self.num_actor_class = int(num_actor_class)
        self.bce_pos_weight = float(bce_pos_weight)
        self.ego_loss_weight = float(ego_loss_weight)

        pos_w = (
            torch.full((self.num_actor_class,), self.bce_pos_weight, dtype=torch.float32)
            if self.bce_pos_weight != 1.0
            else None
        )
        self.register_buffer("bce_pos_weight_tensor", pos_w)
        self.ego_ce = nn.CrossEntropyLoss(reduction="mean")

    def actor_loss(self, pred: torch.Tensor, actor_gt: torch.Tensor) -> torch.Tensor:
        """Multi-label BCEWithLogitsLoss over (B, C) action logits."""
        return F.binary_cross_entropy_with_logits(
            pred,
            actor_gt.to(device=pred.device, dtype=pred.dtype),
            pos_weight=self.bce_pos_weight_tensor,
            reduction="mean",
        )

    def forward(
        self,
        pred: Dict[str, Optional[torch.Tensor]],
        label: Union[Dict[str, torch.Tensor], torch.Tensor],
        validate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(label, dict):
            actor_gt = label.get("actor", label.get("target", label.get("targets", None)))
            ego_gt = label.get("ego", None)
        else:
            actor_gt = label
            ego_gt = None

        if actor_gt is None:
            raise ValueError("Ground-truth actor labels must be provided.")

        loss_actor = self.actor_loss(pred["actor"], actor_gt)
        total_loss = loss_actor

        loss_dict: Dict[str, torch.Tensor] = {
            "loss": total_loss,
            "loss_actor": loss_actor,
        }

        pred_ego = pred.get("ego", None)
        if pred_ego is not None and ego_gt is not None and self.ego_loss_weight > 0:
            loss_ego = self.ego_loss_weight * self.ego_ce(
                pred_ego, ego_gt.to(device=pred_ego.device, dtype=torch.long)
            )
            loss_dict["loss_ego"] = loss_ego
            loss_dict["loss"] = loss_dict["loss"] + loss_ego

        return loss_dict
