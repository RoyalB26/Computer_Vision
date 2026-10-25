"""Simple Convolutional Network for image classification inheriting from BaseModel."""
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from BaseModel.core import MODELS
from BaseModel.model import BaseModel


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

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return torch.flatten(feat, 1)

    def forward_train(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        feat = self.forward_features(inputs)
        logits = self.classifier(feat)

        loss = self.criterion(logits, targets)
        return {"loss": loss, "loss_cls": loss}

    def predict(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        feat = self.forward_features(inputs)
        logits = self.classifier(feat)
        return logits
