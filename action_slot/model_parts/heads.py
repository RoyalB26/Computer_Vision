"""Action and Ego Classification Heads."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Allocated_Head(nn.Module):
    """Slot-allocated classification head."""

    def __init__(self, in_channel: int, num_ego_classes: int, num_actor_classes: int, ego_channel: int = 0, **kwargs):
        super().__init__()
        self.num_ego_classes = num_ego_classes
        if self.num_ego_classes != 0:
            feat_c = ego_channel if ego_channel != 0 else in_channel
            self.fc_ego = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(feat_c, num_ego_classes)
            )
        else:
            self.fc_ego = None

        self.fc_actor = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(in_channel, 1)
        )

    def forward(self, x: torch.Tensor, feat_map: torch.Tensor = None, ego_x: torch.Tensor = None, **kwargs):
        # x is (B, num_slots, dim)
        y_actor = self.fc_actor(x).squeeze(-1)  # (B, num_slots)
        y_ego = None
        if self.fc_ego is not None:
            target_ego = ego_x if ego_x is not None else x
            y_ego = self.fc_ego(target_ego)
        return y_ego, y_actor
