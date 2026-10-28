"""Classification Head for MobileNetV3."""
from typing import Optional
import torch
import torch.nn as nn


class MobileNetV3Classifier(nn.Module):
    """Classification Head for MobileNetV3.

    Structure:
      avgpool: AdaptiveAvgPool2d(1)
      0: Linear(in_features, mid_features) [576 -> 1024 for small]
      1: Hardswish()
      2: Dropout(p=dropout)
      3: Linear(mid_features, num_classes) [1024 -> 1000]
    """

    def __init__(
        self,
        in_features: int = 576,
        mid_features: int = 1024,
        num_classes: int = 1000,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, mid_features, bias=True),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(mid_features, num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
