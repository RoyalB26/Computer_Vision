"""Simple Convolutional Network for image classification inheriting from BaseModel."""
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from BaseModel.core import MODELS
from BaseModel.model import BaseModel


class AdaptiveMinPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x):
        return -self.adaptive_max_pool(-x)

@MODELS.register_module()
class SimpleConvNet(BaseModel):
    """Simple 3-layer CNN classifier.

    Args:
        in_channels: Number of input image channels.
        num_classes: Number of target categories.
        hidden_dim: Number of hidden channels in conv layers.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.num_classes = num_classes

        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=1, padding=0),
            nn.BatchNorm2d(6),
            nn.Hardswish(),
            nn.Conv2d(6, 9, kernel_size=3, padding=1),
            nn.Conv2d(9, hidden_dim, kernel_size=3, padding=1),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features1(x)
        return feat

    def forward_train(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        feat = self.forward_features(inputs)
        logits = self.classifier(feat)

        loss = self.criterion(logits, targets)
        return {"loss": loss, "loss_cls": loss}

    def predict(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        feat = self.forward_features(inputs)
        logits = self.classifier(feat)
        return logits
