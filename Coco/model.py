"""Simple Convolutional Network for multi-label image classification inheriting from BaseModel."""
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from BaseModel.core import MODELS
from BaseModel.model import BaseModel
import numpy as np
from torch import Tensor
import math

class Attention_Slot(nn.Module):
    def __init__(self, dim, num_slot):
        super().__init__()
        self.slot_dim = dim
        self.num_slot = num_slot
        self.total_slots = num_slot  

        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_q = nn.Linear(dim, dim)

        # Shape: (total_slots, dim) -> each slot gets unique learned mu and logsigma
        self.mu = nn.Parameter(torch.randn(self.total_slots, dim))
        self.logsigma = nn.Parameter(torch.zeros(self.total_slots, dim))

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

        # Initialization
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

        noise = torch.randn(batch_size, self.total_slots, self.slot_dim, device=device)

        slots = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise

        return slots

    def positional_encoding(self, H, W, D):
        
        grid_w = np.arange(W, dtype= np.float32) # (W, )
        grid_h = np.arange(H, dtype= np.float32) # (H, )

        X,Y = np.meshgrid(grid_w, grid_h)
        x_flat = X.reshape(-1) # (H * W, )
        y_flat = Y.reshape(-1) # (H * W, )

        emb_x = self.sincos_1D_embedding(D // 2, x_flat)  # [H * W, D // 2]
        emb_y = self.sincos_1D_embedding(D // 2, y_flat)  # [H * W, D // 2]

        # Ghép lại thành vector vị trí 2D hoàn chỉnh: [H * W, D]
        return np.concatenate([emb_x, emb_y], axis=1)

    def sincos_1D_embedding(self, D, grid):
        omega = np.arange(D // 2, dtype= np.float32) # (D // 2, )
        omega/= (D / 2.0)
        omega = 1.0 / (10000 ** omega)

        out = np.einsum('m,d->md', grid, omega) # [N, D // 2]
        return np.concatenate([np.sin(out), np.cos(out)], axis=1) # [N, D]
    
    def forward(self, x: Tensor):
        """
        x: [B, 5, C, D]
        """
        B,K,N,D = x.shape
        k = self.to_k(x) # [B, N, dim]

        v = self.to_v(x) # [B, N, dim]
        slots = self.initalize_slots(batch_size= B,device= x.device) # [B, num_slot, dim]
        q = self.to_q(slots) # [B, num_slot, dim]

        B, N, D = q.shape
        scale = 1.0 / math.sqrt(D)

        # Đưa N vào batch: (B*N, ...)
        # q: (B*N, 1, D)
        q_reshaped = q.view(B * N, 1, D)

        # k ban đầu: (B, 5, N, D) -> đổi chỗ thành (B, N, 5, D) -> view (B*N, 5, D)
        k_reshaped = k.permute(0, 2, 1, 3).reshape(B * N, 5, D)

        # Tính attention scores: (B*N, 1, D) @ (B*N, D, 5) -> (B*N, 1, 5)
        scores = torch.bmm(q_reshaped, k_reshaped.transpose(1, 2)) * scale
        weights = F.softmax(scores, dim=-1)  # shape: (B*N, 1, 5)

        # Nếu nhân tiếp với v: (B, 5, N, D) -> (B*N, 5, D)
        v_reshaped = v.permute(0, 2, 1, 3).reshape(B * N, 5, D)

        # out: (B*N, 1, 5) @ (B*N, 5, D) -> (B*N, 1, D) -> (B, N, D)
        x = torch.bmm(weights, v_reshaped).view(B, N, D)
        x = self.fc2(F.relu(self.fc1(self.norm_out(x))))
        slots = slots + x
        return slots




@MODELS.register_module()
class ConvNet(BaseModel):
    """Simple 3-layer CNN classifier for multi-label classification.

    Args:
        in_channels: Number of input image channels.
        num_classes: Number of target categories (C).
        hidden_dim: Number of hidden channels in conv layers.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 80, hidden_dim: int = 80):
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
            nn.Conv2d(16, hidden_dim, kernel_size=3, padding=1, stride=2),
            nn.Hardswish(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=2),
            nn.Hardswish(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=2),
            nn.AdaptiveMaxPool2d(4),
        )

        self.features_14 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
            nn.Conv2d(16, hidden_dim, kernel_size=3, padding=1),
            nn.Hardswish(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Hardswish(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.AdaptiveMaxPool2d(4),
        )
        
        self.attn_slot = Attention_Slot(16, num_classes)

        self.classifier = nn.Linear(16, 1)
        # Đối với multi-label classification, dùng BCEWithLogitsLoss
        self.criterion = nn.BCEWithLogitsLoss()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[1], x.shape[2]
        feats= []
        for i in range(0, 5):
            if i < 4:
                feat = self.features(x)
            else:
                feat = self.features_14(x)
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
            B,C,H,W = feat.shape
            feat = feat.view(B, C, H*W)
            feats.append(feat)
        feat = torch.stack(feats, dim=1)
        return feat

    def forward_train(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        feat = self.forward_features(inputs)
        feat = self.attn_slot(feat)
        logits = self.classifier(feat).squeeze(2)  # (B, C)
        loss = self.criterion(logits, targets)
        return {"loss": loss, "loss_cls": loss}

    def predict(self, inputs: torch.Tensor, return_probs: bool = False, **kwargs) -> torch.Tensor:
        """Dự đoán cho batch ảnh inputs (B, C, H, W).
        
        Trả về:
            Tensor shape (B, C) là logits hoặc probabilities (qua sigmoid) cho từng class.
        """
        feat = self.forward_features(inputs)
        feat = self.attn_slot(feat)
        logits = self.classifier(feat).squeeze(2)  # (B, C)
        # print(logits.shape)
        # exit(0)
        if return_probs:
            return torch.sigmoid(logits)
        return logits
