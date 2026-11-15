"""Action and Ego Classification Heads for Toy Action models."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Allocated_Head(nn.Module):
    """Direct classification head predicting 1 binary logit per class."""

    def __init__(
        self,
        in_channel: int,
        num_actor_classes: int = 48,
        num_ego_classes: int = 0,
        ego_channel: int = 0,
        **kwargs
    ):
        super().__init__()
        self.num_ego_classes = num_ego_classes
        self.num_actor_classes = num_actor_classes

        if self.num_ego_classes > 0:
            c = ego_channel if ego_channel > 0 else in_channel
            self.fc_ego = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(c, num_ego_classes),
            )
        else:
            self.fc_ego = None

        self.fc_actor = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(in_channel, 1),
            )
            for _ in range(num_actor_classes)
        ])

    def forward(self, x: torch.Tensor, ego_x: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        b, n, _ = x.shape
        y_actor = [self.fc_actor[i](x[:, i, :]) for i in range(min(n, self.num_actor_classes))]
        y_actor = torch.stack(y_actor, dim=0).permute(1, 0, 2).reshape(b, -1)

        y_ego = self.fc_ego(ego_x) if self.fc_ego is not None and ego_x is not None else None
        return y_ego, y_actor


class ReduceAllocatedHead(nn.Module):
    """Slot-allocated head predicting 3 logits per action-symbol pair:

    [single, group+, background].
    For 24 action slots -> 24 * 2 = 48 foreground action classes!
    """

    def __init__(
        self,
        in_channel: int,
        num_actor_classes: int = 48,
        num_slots: int = 24,
        num_ego_classes: int = 0,
        ego_channel: int = 0,
        **kwargs
    ):
        super().__init__()
        self.num_ego_classes = num_ego_classes
        self.num_actor_classes = num_actor_classes
        self.num_slots = num_slots

        if self.num_ego_classes > 0:
            c = ego_channel if ego_channel > 0 else in_channel
            self.fc_ego = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(c, num_ego_classes),
            )
        else:
            self.fc_ego = None

        # 3 classes per slot: [0: single, 1: group+, 2: background]
        self.fc_actor = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(in_channel, 3),
            )
            for _ in range(num_slots)
        ])

    def forward(self, x: torch.Tensor, ego_x: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        b, n, _ = x.shape
        # Use first self.num_slots slots (excluding optional background slot)
        slots_to_use = min(n, self.num_slots)
        y_actor = [self.fc_actor[i](x[:, i, :]) for i in range(slots_to_use)]
        y_actor = torch.stack(y_actor, dim=0).permute(1, 0, 2).reshape(b, slots_to_use, 3)

        y_ego = self.fc_ego(ego_x) if self.fc_ego is not None and ego_x is not None else None
        return y_ego, y_actor
