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

class Cross_Attention(nn.Module):
    def __init__(self, dim, kernel= 1, stride= 1, padding= 0):
        super().__init__()
        self.to_q= nn.Linear(dim, dim)
        self.to_v= nn.Linear(dim, dim)
        self.to_k= nn.Conv2d(dim, dim, kernel_size= kernel, stride= stride, padding= padding)
        self.scale= dim ** 0.5

    def forward(self, x, slots):
        """
        x: B, d, H, W
        slots: B, S, d
        """
        q= self.to_q(slots)
        v= self.to_v(slots)
        k= self.to_k(x)
        B,d,H,W= k.shape
        k=k.view(B, d, -1)
        dots= torch.einsum('bdn, bsd -> bns', k, q)
        dots= dots * self.scale
        attn= torch.softmax(dots, dim= 1)
        x= torch.einsum('bns, bsd -> bdn', attn, v).view(B, d, H, W)

        return x

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
        self.dim= hidden_dim
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
        )
        self.feature= nn.ModuleList([
            Cross_Attention(hidden_dim, 3, 2, 1), # 8
            Cross_Attention(hidden_dim, 3, 2, 1), # 4
            Cross_Attention(hidden_dim, 3, 2, 1), # 2
            Cross_Attention(hidden_dim, 3, 2, 1), # 1
        ]
        )
        self.features2= nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        self.mu = nn.Parameter(torch.randn(self.num_classes, hidden_dim))
        self.logsigma = nn.Parameter(torch.zeros(self.num_classes, hidden_dim))

        nn.init.xavier_uniform_(self.mu)
        nn.init.zeros_(self.logsigma)
        torch.manual_seed(42)
        
    def initalize_slots(self, batch_size=1, device=None):
        # Handle the case where device is passed as the first positional argument
        if isinstance(batch_size, (torch.device, str)):
            device = batch_size
            batch_size = 1

        device = device if device is not None else self.mu.device

        mu = self.mu.to(device)
        sigma = torch.exp(self.logsigma).to(device)

        noise = torch.randn(batch_size, self.num_classes, self.dim, device=device)

        slots = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise

        return slots

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B, _, _,_ = x.shape
        self.slots= self.initalize_slots(B, device= x.device)
        feat = self.features1(x)
        for f in self.feature:
            feat= f(feat, self.slots)
        # feat = self.features2(feat)
        feat = feat.squeeze(-1).squeeze(-1)
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
