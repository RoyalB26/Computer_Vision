"""Micro-Benchmark so sánh 16 kiến trúc Backbone Block kinh điển & hiện đại trên tập con VisDrone.

Danh sách 16 kiến trúc (từ 2014 -> 2025):
  1. vgg         (2014): Plain 3x3 Conv + BN + ReLU (không skip)
  2. inception   (2014): Multi-branch song song (1x1, 3x3, 5x5 factorized, MaxPool)
  3. resnet      (2015): Residual BasicBlock (3x3 -> 3x3 + Skip)
  4. resnext     (2016): Grouped Convolution Bottleneck + Skip (Split-Transform-Merge)
  5. densenet    (2017): Dense Feature Reuse (Concat nhiều tầng -> 1x1 Transition)
  6. senet       (2017): Residual + Squeeze-and-Excitation (Channel Attention)
  7. cbam        (2018): Residual + Channel Attention + Spatial 7x7 Attention
  8. inverted    (2018): MobileNetV2/V3 Inverted Residual (Expand -> DWConv 3x3 -> SE -> Project)
  9. csp         (2020): YOLOv5/v8 C2f / Cross-Stage Partial Bottleneck
 10. repvgg      (2021): Multi-branch Structural Reparameterization (3x3 + 1x1 + Identity)
 11. convnext    (2022): Large-Kernel DWConv 7x7 + LayerNorm + GELU MLP + LayerScale
 12. window_attn (2021): Local Window Multi-Head Self-Attention (Swin / Hybrid ViT style)
 13. vmamba      (2024): Visual State Space Block (SS2D 4-way Selective Scan + SiLU Gate)
 14. mobilevit   (2022): Apple MobileViT (Local Conv -> Unfold Global Transformer -> Fold -> Fusion)
 15. fastvit     (2023): Apple FastViT (CPE 7x7 + RepMixer Token Mixer + ConvFFN + LayerScale)
 16. area_attn   (2025): YOLOv12 Area Attention (A2: Strip-Area SDPA + DWConv 7x7 Value-PE + MLP)

Chỉ số đánh giá tự động:
  - P3 Small In/Out  : Tỉ lệ L2-Norm Energy trong box vật nhỏ vs ngoài nền tại P3
  - P3 All In/Out    : Tỉ lệ L2-Norm Energy trong mọi box vs ngoài nền tại P3
  - PCA FG/BG Fisher : Độ phân tách tuyến tính (Fisher Ratio) giữa FG và BG trên không gian chiếu PCA-3D
  - Class Sep (Cos)  : Độ tách cụm ngữ nghĩa giữa các class khác nhau (car vs motorcycle vs people...) tại P3
  - P3 FG Mass%      : % năng lượng Foreground Attention rơi trúng GT Box tại P3
  - Val MAE          : Sai số đếm trung bình trên tập validation

Ví dụ chạy:
  # Chạy nhanh 4 kiến trúc tiêu biểu (mất ~1-2 phút)
  python Visdrone/benchmark_blocks.py --blocks vgg resnet inverted csp --epochs 12

  # Chạy toàn bộ cả 12 kiến trúc
  python Visdrone/benchmark_blocks.py --blocks all --epochs 15 --n-train 160 --n-val 40
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import pickle
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from BaseModel.dataset import open_lmdb
from Visdrone.dataset import VISDRONE_CLASSES
from Visdrone.inspect_scales import build_box_mask_at_res, compute_pca_rgb, draw_boxes_on_ax
from Visdrone.model import PANetNeck, PositionEmbeddingSine, SlotAttention


# =============================================================================
# 1. TỔNG HỢP 12 KIẾN TRÚC BACKBONE BLOCK (2014 -> 2024)
# =============================================================================

class SELayer(nn.Module):
    """Squeeze-and-Excitation (Channel Attention)."""
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        mid = max(8, ch // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)

class CBAMV2(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        mid = max(8, ch // 4)
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(ch, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1),
        )
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.bn_att = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        # Channel attention (Avg + Max)
        ca = torch.sigmoid(self.ca_mlp(F.adaptive_avg_pool2d(out, 1)) + self.ca_mlp(F.adaptive_max_pool2d(out, 1)))
        out = out * ca
        # Spatial attention (Avg + Max across channels)
        avg_s = out.mean(dim=1, keepdim=True)
        max_s, _ = out.max(dim=1, keepdim=True)
        sa = torch.sigmoid(self.sa_conv(torch.cat([avg_s, max_s], dim=1)))
        out = self.bn_att(out + out * sa)
        return F.relu(out + x, inplace=True)

class VGGBlock(nn.Module):
    """2014 — Classic Plain Conv 3x3 (không có residual skip)."""
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InceptionBlock(nn.Module):
    """2014 — Multi-branch Inception (1x1, 3x3, 5x5 factorized, MaxPool)."""
    def __init__(self, ch: int):
        super().__init__()
        c_b = max(8, ch // 4)
        c_last = ch - 3 * c_b
        self.b1 = nn.Sequential(
            nn.Conv2d(ch, c_b, 1, bias=False),
            nn.BatchNorm2d(c_b),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(ch, c_b, 1, bias=False),
            nn.BatchNorm2d(c_b),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_b, c_b, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_b),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(ch, c_b, 1, bias=False),
            nn.BatchNorm2d(c_b),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_b, c_b, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_b),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_b, c_b, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_b),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(ch, c_last, 1, bias=False),
            nn.BatchNorm2d(c_last),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class ResNetBasicBlock(nn.Module):
    """2015 — ResNet BasicBlock (Conv 3x3 -> Conv 3x3 + Identity Skip)."""
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x, inplace=True)


class ResNeXtBlock(nn.Module):
    """2016 — ResNeXt Grouped Convolution Bottleneck (Split-Transform-Merge)."""
    def __init__(self, ch: int, groups: int = 8):
        super().__init__()
        g = groups if ch % groups == 0 else max(1, math.gcd(ch, groups))
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, groups=g, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x), inplace=True)


class DenseNetBlock(nn.Module):
    """2017 — DenseNet Feature Reuse (Concat nhiều tầng liên tiếp + 1x1 compression)."""
    def __init__(self, ch: int, growth: Optional[int] = None):
        super().__init__()
        g = growth if growth is not None else max(16, ch // 2)
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, g, 3, padding=1, bias=False),
        )
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(ch + g),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch + g, g, 3, padding=1, bias=False),
        )
        self.compress = nn.Sequential(
            nn.BatchNorm2d(ch + 2 * g),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch + 2 * g, ch, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.layer1(x)
        c1 = torch.cat([x, h1], dim=1)
        h2 = self.layer2(c1)
        c2 = torch.cat([c1, h2], dim=1)
        return self.compress(c2)


class SENetBlock(nn.Module):
    """2017 — ResNet BasicBlock + Squeeze-and-Excitation."""
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.se = SELayer(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.se(self.bn2(self.conv2(out)))
        return F.relu(out + x, inplace=True)


class CBAMBlock(nn.Module):
    """2018 — Residual + Channel Attention + Spatial 7x7 Attention."""
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        mid = max(8, ch // 4)
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(ch, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1),
        )
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        # Channel attention (Avg + Max)
        ca = torch.sigmoid(self.ca_mlp(F.adaptive_avg_pool2d(out, 1)) + self.ca_mlp(F.adaptive_max_pool2d(out, 1)))
        out = out * ca
        # Spatial attention (Avg + Max across channels)
        avg_s = out.mean(dim=1, keepdim=True)
        max_s, _ = out.max(dim=1, keepdim=True)
        sa = torch.sigmoid(self.sa_conv(torch.cat([avg_s, max_s], dim=1)))
        out = out * sa
        return F.relu(out + x, inplace=True)


class InvertedResidualBlock(nn.Module):
    """2018-2019 — MobileNetV2/V3 Inverted Residual + SE."""
    def __init__(self, ch: int, expand_ratio: int = 3):
        super().__init__()
        mid = ch * expand_ratio
        self.net = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.Hardswish(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.Hardswish(inplace=True),
            SELayer(mid),
            nn.Conv2d(mid, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CSPC2fBlock(nn.Module):
    """2020-2023 — YOLOv8 C2f / Cross-Stage Partial Bottleneck."""
    def __init__(self, ch: int, n: int = 2):
        super().__init__()
        c_half = ch // 2
        self.cv1 = nn.Sequential(
            nn.Conv2d(ch, 2 * c_half, 1, bias=False),
            nn.BatchNorm2d(2 * c_half),
            nn.SiLU(inplace=True),
        )
        self.m = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_half, c_half, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_half),
                nn.SiLU(inplace=True),
                nn.Conv2d(c_half, c_half, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_half),
                nn.SiLU(inplace=True),
            )
            for _ in range(n)
        ])
        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + n) * c_half, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        for m in self.m:
            y.append(y[-1] + m(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class RepVGGBlock(nn.Module):
    """2021 — RepVGG Training-time Multi-branch (3x3 + 1x1 + Identity)."""
    def __init__(self, ch: int):
        super().__init__()
        self.rbr_dense = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.rbr_identity = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.rbr_dense(x) + self.rbr_1x1(x) + self.rbr_identity(x), inplace=True)


class ConvNeXtBlock(nn.Module):
    """2022 — Modern Large-Kernel ConvNeXt Block (DWConv 7x7 + LN + GELU MLP + LayerScale)."""
    def __init__(self, ch: int, mlp_ratio: int = 3):
        super().__init__()
        self.dwconv = nn.Conv2d(ch, ch, kernel_size=7, padding=3, groups=ch)
        self.norm = nn.LayerNorm(ch, eps=1e-6)
        self.pwconv1 = nn.Linear(ch, ch * mlp_ratio)
        self.pwconv2 = nn.Linear(ch * mlp_ratio, ch)
        self.gamma = nn.Parameter(1e-2 * torch.ones(ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv2(F.gelu(self.pwconv1(x)))
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return shortcut + x


class WindowSelfAttentionBlock(nn.Module):
    """2021-2023 — Local Window Multi-Head Self-Attention (Swin / Hybrid ViT style)."""
    def __init__(self, ch: int, num_heads: int = 4, window_size: int = 6):
        super().__init__()
        self.ws = window_size
        self.local_pos = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.norm1 = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(ch)
        self.mlp = nn.Sequential(
            nn.Linear(ch, ch * 2),
            nn.GELU(),
            nn.Linear(ch * 2, ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.local_pos(x)
        B, C, H, W = x.shape
        ws = self.ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_pad = x
        Hp, Wp = x_pad.shape[2], x_pad.shape[3]
        # Chia thành các cửa sổ không chồng lấp (B * nW, ws*ws, C)
        xw = x_pad.view(B, C, Hp // ws, ws, Wp // ws, ws)
        xw = xw.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, ws * ws, C)
        xn = self.norm1(xw)
        attn_out, _ = self.attn(xn, xn, xn, need_weights=False)
        xw = xw + attn_out
        xw = xw + self.mlp(self.norm2(xw))
        # Ghép ngược lại về (B, C, Hp, Wp)
        out = xw.view(B, Hp // ws, Wp // ws, ws, ws, C).permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, Hp, Wp)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        return out


class VMambaBlock(nn.Module):
    """2024 — Visual State Space Block (VMamba / SS2D 4-way Selective Scan)."""
    def __init__(self, ch: int, expand: float = 1.5):
        super().__init__()
        d_inner = max(16, int(ch * expand))
        self.d_inner = d_inner
        self.norm_in = nn.LayerNorm(ch)
        self.in_proj = nn.Linear(ch, d_inner * 2, bias=False)
        self.dwconv = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=True)
        # Input-dependent selective decay gates cho 4 hướng quét (L->R, R->L, T->B, B->T)
        self.dt_h = nn.Conv2d(d_inner, d_inner * 2, kernel_size=1, bias=True)
        self.dt_v = nn.Conv2d(d_inner, d_inner * 2, kernel_size=1, bias=True)
        self.norm_out = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, ch, bias=False)
        self.gamma = nn.Parameter(1e-1 * torch.ones(ch))

    @staticmethod
    def _scan_1d_bidir(u: torch.Tensor, gate_fwd: torch.Tensor, gate_bwd: torch.Tensor) -> torch.Tensor:
        """Quét chọn lọc 2 chiều dọc trục cuối L: h_t = (1 - g_t) * h_{t-1} + g_t * u_t."""
        # u, gate_fwd, gate_bwd: (N, C, L)
        L = u.shape[-1]
        a_f = 1.0 - gate_fwd
        b_f = gate_fwd * u
        h_f = b_f[..., 0]
        outs_f = [h_f]
        for t in range(1, L):
            h_f = a_f[..., t] * h_f + b_f[..., t]
            outs_f.append(h_f)
        y_f = torch.stack(outs_f, dim=-1)

        a_b = 1.0 - gate_bwd
        b_b = gate_bwd * u
        h_b = b_b[..., L - 1]
        outs_b = [h_b]
        for t in range(L - 2, -1, -1):
            h_b = a_b[..., t] * h_b + b_b[..., t]
            outs_b.append(h_b)
        outs_b.reverse()
        y_b = torch.stack(outs_b, dim=-1)
        return y_f + y_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        B, C, H, W = x.shape
        x_ln = self.norm_in(x.permute(0, 2, 3, 1))
        xz = self.in_proj(x_ln).permute(0, 3, 1, 2)  # (B, 2*D, H, W)
        u, z = xz.chunk(2, dim=1)
        u = F.silu(self.dwconv(u))

        # Quét ngang (L->R và R->L) song song trên tất cả H hàng
        gh_f, gh_b = torch.sigmoid(self.dt_h(u)).chunk(2, dim=1)
        u_h = u.permute(0, 2, 1, 3).contiguous().view(B * H, self.d_inner, W)
        gh_f = gh_f.permute(0, 2, 1, 3).contiguous().view(B * H, self.d_inner, W)
        gh_b = gh_b.permute(0, 2, 1, 3).contiguous().view(B * H, self.d_inner, W)
        y_h = self._scan_1d_bidir(u_h, gh_f, gh_b).view(B, H, self.d_inner, W).permute(0, 2, 1, 3)

        # Quét dọc (T->B và B->T) song song trên tất cả W cột
        gv_f, gv_b = torch.sigmoid(self.dt_v(u)).chunk(2, dim=1)
        u_v = u.permute(0, 3, 1, 2).contiguous().view(B * W, self.d_inner, H)
        gv_f = gv_f.permute(0, 3, 1, 2).contiguous().view(B * W, self.d_inner, H)
        gv_b = gv_b.permute(0, 3, 1, 2).contiguous().view(B * W, self.d_inner, H)
        y_v = self._scan_1d_bidir(u_v, gv_f, gv_b).view(B, W, self.d_inner, H).permute(0, 2, 3, 1)

        y = (y_h + y_v) * 0.25
        y = self.norm_out(y.permute(0, 2, 3, 1)) * F.silu(z.permute(0, 2, 3, 1))
        out = (self.gamma * self.out_proj(y)).permute(0, 3, 1, 2)
        return shortcut + out


class MobileViTBlock(nn.Module):
    """2022 — Apple MobileViT Block (Local Conv -> Unfold Global Transformer -> Fold -> Fusion)."""
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = ch // num_heads
        # Local representation (Depthwise Separable Conv chuẩn MobileViT-v2/v3, cân bằng tham số)
        self.local_rep = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 1, bias=False),
        )
        # Global Transformer across patches
        self.norm1 = nn.LayerNorm(ch)
        self.qkv = nn.Linear(ch, ch * 3, bias=False)
        self.proj = nn.Linear(ch, ch, bias=False)
        self.norm2 = nn.LayerNorm(ch)
        self.ffn = nn.Sequential(
            nn.Linear(ch, ch * 2),
            nn.SiLU(inplace=True),
            nn.Linear(ch * 2, ch),
        )
        self.norm_out = nn.LayerNorm(ch)
        # Fusion with original input (1x1 + DWConv 3x3 để giữ tổng params ~1.3M công bằng với các block khác)
        self.conv_proj = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.local_rep(x)
        B, C, H, W = y.shape
        # Patch size p=4 cho P3 (48x48 -> 12x12=144 patches), p=2 cho P4/P5
        p = 4 if H >= 32 else 2
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h > 0 or pad_w > 0:
            y = F.pad(y, (0, pad_w, 0, pad_h))
        Hp, Wp = y.shape[2], y.shape[3]
        nh, nw = Hp // p, Wp // p

        # Unfold: (B * p*p, nh*nw, C) — attention TOÀN CỤC giữa các patch ở cùng vị trí cục bộ
        tokens = y.view(B, C, nh, p, nw, p).permute(0, 3, 5, 2, 4, 1).contiguous().view(B * p * p, nh * nw, C)

        # Transformer (SDPA)
        tn = self.norm1(tokens)
        qkv = self.qkv(tn).view(B * p * p, nh * nw, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B * p * p, nh * nw, C)
        tokens = tokens + self.proj(attn)
        tokens = tokens + self.ffn(self.norm2(tokens))
        tokens = self.norm_out(tokens)

        # Fold ngược lại về (B, C, Hp, Wp)
        y = tokens.view(B, p, p, nh, nw, C).permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, Hp, Wp)
        if pad_h > 0 or pad_w > 0:
            y = y[:, :, :H, :W]

        y = self.conv_proj(y)
        return self.fusion(torch.cat([x, y], dim=1))


class FastViTBlock(nn.Module):
    """2023 — Apple FastViT Block (CPE 7x7 + RepMixer Token Mixer + ConvFFN + LayerScale)."""
    def __init__(self, ch: int, mlp_ratio: int = 3):
        super().__init__()
        # Conditional Positional Encoding
        self.cpe = nn.Conv2d(ch, ch, kernel_size=7, padding=3, groups=ch, bias=True)
        # RepMixer: x + gamma1 * (DWConv3x3(x) - BN(x))
        self.norm_mix = nn.BatchNorm2d(ch)
        self.mixer = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False)
        self.gamma1 = nn.Parameter(1e-2 * torch.ones(1, ch, 1, 1))
        # Channel ConvFFN
        mid = ch * mlp_ratio
        self.ffn = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.Conv2d(ch, mid, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, ch, kernel_size=1, bias=False),
        )
        self.gamma2 = nn.Parameter(1e-2 * torch.ones(1, ch, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.cpe(x)
        x = x + self.gamma1 * (self.mixer(x) - self.norm_mix(x))
        x = x + self.gamma2 * self.ffn(x)
        return x


class AreaAttentionBlock(nn.Module):
    """2025 — YOLOv12 Area Attention Block (A2: Strip-Area SDPA + DWConv 7x7 Value-PE + MLP)."""
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = ch // num_heads
        self.qkv = nn.Conv2d(ch, ch * 3, kernel_size=1, bias=False)
        self.pe = nn.Conv2d(ch, ch, kernel_size=7, padding=3, groups=ch, bias=True)
        self.proj = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        # Chia không gian thành các dải (areas) ngang không cần padding/window-shift (chuẩn YOLOv12)
        area = 4 if H >= 24 else 2
        if N % area != 0:
            area = 1

        qkv = self.qkv(x)  # (B, 3*C, H, W)
        q, k, v = qkv.chunk(3, dim=1)
        # Signature YOLOv12: DWConv 7x7 positional encoding cộng trực tiếp từ v
        pe = self.pe(v)

        # Reshape theo area strips: (B * area, heads, N // area, head_dim)
        seq_len = N // area
        q = q.flatten(2).transpose(1, 2).contiguous().view(B * area, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2).contiguous().view(B * area, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2).contiguous().view(B * area, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, N, C).transpose(1, 2).view(B, C, H, W)

        x = x + self.proj(attn + pe)
        x = x + self.mlp(x)
        return x


BLOCK_REGISTRY = {
    "vgg": VGGBlock,
    "inception": InceptionBlock,
    "resnet": ResNetBasicBlock,
    "resnext": ResNeXtBlock,
    "densenet": DenseNetBlock,
    "senet": SENetBlock,
    "cbam": CBAMBlock,
    "cbamv2": CBAMV2,
    "inverted": InvertedResidualBlock,
    "csp": CSPC2fBlock,
    "repvgg": RepVGGBlock,
    "convnext": ConvNeXtBlock,
    "window_attn": WindowSelfAttentionBlock,
    "vmamba": VMambaBlock,
    "mobilevit": MobileViTBlock,
    "fastvit": FastViTBlock,
    "area_attn": AreaAttentionBlock,
}


# =============================================================================
# 2. MÔ HÌNH MINI DÙNG CHUNG (CONTROLLED ABLATION)
# =============================================================================

class MiniModularBackbone(nn.Module):
    """Backbone mini 3 tầng (P3 //8, P4 //16, P5 //32) thay đổi linh hoạt lõi Block."""
    def __init__(self, block_type: str, stage_channels: Tuple[int, int, int] = (48, 96, 160), blocks_per_stage: int = 2):
        super().__init__()
        if block_type not in BLOCK_REGISTRY:
            raise ValueError(f"Unknown block_type '{block_type}'. Choose from: {list(BLOCK_REGISTRY.keys())}")
        block_cls = BLOCK_REGISTRY[block_type]
        c3, c4, c5 = stage_channels

        # Stem chung (stride 4)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage P3 (stride 8)
        self.down3 = nn.Sequential(
            nn.Conv2d(32, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(*[block_cls(c3) for _ in range(blocks_per_stage)])

        # Stage P4 (stride 16)
        self.down4 = nn.Sequential(
            nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(*[block_cls(c4) for _ in range(blocks_per_stage)])

        # Stage P5 (stride 32)
        self.down5 = nn.Sequential(
            nn.Conv2d(c4, c5, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )
        self.stage5 = nn.Sequential(*[block_cls(c5) for _ in range(blocks_per_stage)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        p3 = self.stage3(self.down3(x))
        p4 = self.stage4(self.down4(p3))
        p5 = self.stage5(self.down5(p4))
        return p3, p4, p5


class MiniSlotCounter(nn.Module):
    """Mô hình đếm mini dùng chung Neck + Positional Encoding + SlotAttention + Multi-scale Mass."""
    def __init__(self, block_type: str, num_classes: int = 10, dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        stage_ch = (48, 96, 160)
        self.backbone = MiniModularBackbone(block_type=block_type, stage_channels=stage_ch, blocks_per_stage=2)
        self.neck = PANetNeck(in_channels=stage_ch, out_dim=dim)
        self.pos_emb = PositionEmbeddingSine(num_pos_feats=dim // 2)
        self.level_embed = nn.Parameter(torch.randn(3, dim) * 0.02)
        self.slot_attention = SlotAttention(dim=dim, num_class=num_classes)
        self.fc_actor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim + 3, dim),
                nn.ReLU(inplace=False),
                nn.Linear(dim, 1),
                nn.ReLU(inplace=False),
            )
            for _ in range(num_classes)
        ])

    def forward_all(self, x: torch.Tensor):
        c3, c4, c5 = self.backbone(x)
        feat_p3, feat_p4, feat_p5 = self.neck(c3, c4, c5)

        p3_pos = feat_p3 + self.pos_emb(feat_p3)
        p4_pos = feat_p4 + self.pos_emb(feat_p4)
        p5_pos = feat_p5 + self.pos_emb(feat_p5)

        resolutions = (
            (feat_p3.shape[2], feat_p3.shape[3]),
            (feat_p4.shape[2], feat_p4.shape[3]),
            (feat_p5.shape[2], feat_p5.shape[3]),
        )
        t3 = p3_pos.flatten(2).transpose(1, 2) + self.level_embed[0]
        t4 = p4_pos.flatten(2).transpose(1, 2) + self.level_embed[1]
        t5 = p5_pos.flatten(2).transpose(1, 2) + self.level_embed[2]
        tokens = torch.cat([t3, t4, t5], dim=1)

        slots, attn_ori, _ = self.slot_attention(tokens)

        (H3, W3), (H4, W4), _ = resolutions
        n3, n4 = H3 * W3, H4 * W4
        cls_attn = attn_ori[:, : self.num_classes, :]
        mass_p3 = cls_attn[:, :, :n3].mean(dim=-1, keepdim=True) * 10.0
        mass_p4 = cls_attn[:, :, n3 : n3 + n4].mean(dim=-1, keepdim=True) * 10.0
        mass_p5 = cls_attn[:, :, n3 + n4 :].mean(dim=-1, keepdim=True) * 10.0
        attn_mass_3 = torch.cat([mass_p3, mass_p4, mass_p5], dim=-1)

        preds = torch.stack(
            [self.fc_actor[i](torch.cat([slots[:, i, :], attn_mass_3[:, i, :]], dim=-1)).squeeze(-1) for i in range(self.num_classes)],
            dim=-1,
        )
        return preds, attn_ori, (feat_p3, feat_p4, feat_p5), resolutions, slots


# =============================================================================
# 3. MICRO-DATASET TRONG RAM TỪ VISDRONE LMDB THẬT
# =============================================================================

def rasterize_boxes(boxes: np.ndarray, resolutions: Tuple[Tuple[int, int], ...]) -> np.ndarray:
    """Tạo vector multi-scale background mask (1.0 = BG, 0.0 = Box)."""
    has_b = boxes is not None and len(boxes) > 0
    if has_b:
        bxs = np.asarray(boxes, dtype=np.float32)
        cx, cy, w, h = bxs[:, 0], bxs[:, 1], bxs[:, 2], bxs[:, 3]
    parts = []
    for H, W in resolutions:
        m = np.ones((H, W), dtype=np.float32)
        if has_b:
            x1 = np.clip(np.trunc((cx - w / 2.0) * W).astype(np.int64), 0, W - 1)
            y1 = np.clip(np.trunc((cy - h / 2.0) * H).astype(np.int64), 0, H - 1)
            x2 = np.maximum(x1 + 1, np.clip(np.ceil((cx + w / 2.0) * W).astype(np.int64), 0, W))
            y2 = np.maximum(y1 + 1, np.clip(np.ceil((cy + h / 2.0) * H).astype(np.int64), 0, H))
            for i in range(len(x1)):
                m[y1[i] : y2[i], x1[i] : x2[i]] = 0.0
        parts.append(m.reshape(-1))
    return np.concatenate(parts, axis=0)


class InMemoryVisDroneMicroDataset(Dataset):
    """Nạp sẵn tập con VisDrone thật (resize img_size x img_size) vào RAM để benchmark siêu nhanh."""
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        return {
            "image": s["image"],
            "counts": s["counts"],
            "bg_mask": s["bg_mask"],
        }


def _select_stratified_indices(
    all_counts: np.ndarray,
    valid_indices: np.ndarray,
    n_select: int,
    target_box_dist: np.ndarray,
    target_pres_dist: np.ndarray,
    rng: np.random.Generator,
    pool_size: int = 256,
) -> Tuple[List[int], np.ndarray]:
    """Chọn n_select ảnh sao cho phân phối số lượng box và tỷ lệ hiện diện của 10 class
    khớp sát nhất với phân phối gốc của toàn bộ dataset (Iterative Greedy Stratified Matching).
    """
    avail_mask = np.ones(len(valid_indices), dtype=bool)
    selected_pos: List[int] = []
    cur_box_sum = np.zeros(10, dtype=np.float64)
    cur_pres_sum = np.zeros(10, dtype=np.float64)

    counts_valid = all_counts[valid_indices]  # (M, 10)
    pres_valid = (counts_valid > 0).astype(np.float64)  # (M, 10)
    mean_boxes_per_img = float(counts_valid.sum(axis=1).mean())

    # Trọng số chuẩn hóa (chi-square style) để các class hiếm (awning-tricycle, tricycle, bus...) không bị lấn át bởi car/pedestrian
    box_weights = 1.0 / (target_box_dist + 0.005)
    pres_weights = 1.0 / (target_pres_dist + 0.02)

    for step in range(1, n_select + 1):
        avail_pos = np.flatnonzero(avail_mask)
        if len(avail_pos) == 0:
            break
        # Lấy ngẫu nhiên một tập ứng viên đủ lớn để vừa đảm bảo tính ngẫu nhiên theo seed vừa tối ưu cực nhanh
        if len(avail_pos) > pool_size:
            cand_pos = rng.choice(avail_pos, size=pool_size, replace=False)
        else:
            cand_pos = avail_pos

        cand_counts = counts_valid[cand_pos]  # (P, 10)
        cand_pres = pres_valid[cand_pos]      # (P, 10)

        new_box_sums = cur_box_sum[None, :] + cand_counts
        new_box_totals = new_box_sums.sum(axis=1, keepdims=True) + 1e-8
        new_box_dists = new_box_sums / new_box_totals

        new_pres_dists = (cur_pres_sum[None, :] + cand_pres) / float(step)

        # Sai số phân phối tỷ lệ box + sai số tỷ lệ ảnh có mặt class + giữ mật độ trung bình ổn định
        err_box = ((new_box_dists - target_box_dist[None, :]) ** 2 * box_weights[None, :]).sum(axis=1)
        err_pres = ((new_pres_dists - target_pres_dist[None, :]) ** 2 * pres_weights[None, :]).sum(axis=1)
        err_density = ((new_box_totals[:, 0] / float(step) - mean_boxes_per_img) / (mean_boxes_per_img + 1e-6)) ** 2

        total_err = err_box + 0.35 * err_pres + 0.05 * err_density
        best_pos = int(cand_pos[int(np.argmin(total_err))])

        avail_mask[best_pos] = False
        selected_pos.append(best_pos)
        cur_box_sum += counts_valid[best_pos]
        cur_pres_sum += pres_valid[best_pos]

    chosen_real_indices = [int(valid_indices[p]) for p in selected_pos]
    remaining_mask = avail_mask
    return chosen_real_indices, remaining_mask


def load_micro_splits(
    lmdb_path: str,
    n_train: int = 160,
    n_val: int = 40,
    img_size: int = 384,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Tải tập con VisDrone thật vào RAM sao cho cả tập train và val đều cân bằng
    và bám sát phân phối của 10 class trong toàn bộ dataset gốc.
    """
    env, txn = open_lmdb(lmdb_path, readonly=True)
    total_bytes = txn.get(b"__len__")
    total = int(total_bytes.decode("ascii")) if total_bytes else env.stat()["entries"]

    # Cache bảng đếm class toàn cục để các lần chạy sau không cần quét lại LMDB
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lmdb_class_counts_cache.npz")
    all_counts: Optional[np.ndarray] = None
    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path)
            if cached["counts"].shape == (total, 10):
                all_counts = cached["counts"]
        except Exception:
            all_counts = None

    if all_counts is None:
        print(f"[*] Đang quét phân phối 10 class trên toàn bộ {total} ảnh của {lmdb_path}...")
        all_counts = np.zeros((total, 10), dtype=np.float64)
        for i in range(total):
            bf = txn.get(f"{i:08d}".encode("ascii"))
            if bf is None:
                continue
            d = pickle.loads(bf)
            raw_labels = d.get("labels", [])
            if len(raw_labels) == 0:
                continue
            lbl_arr = np.asarray(raw_labels, dtype=np.int64)
            lbl_arr = lbl_arr[(lbl_arr >= 0) & (lbl_arr < 10)]
            if len(lbl_arr) > 0:
                all_counts[i] = np.bincount(lbl_arr, minlength=10)
        try:
            np.savez_compressed(cache_path, counts=all_counts)
        except Exception:
            pass

    # Lọc các ảnh hợp lệ có ít nhất 5 vật thể
    row_totals = all_counts.sum(axis=1)
    valid_indices = np.flatnonzero(row_totals >= 5)

    # Phân phối mục tiêu trên toàn bộ dataset (tỷ lệ số lượng box & tỷ lệ hiện diện từng class)
    global_box_sum = all_counts[valid_indices].sum(axis=0)
    target_box_dist = global_box_sum / (global_box_sum.sum() + 1e-8)
    target_pres_dist = (all_counts[valid_indices] > 0).mean(axis=0)

    rng = np.random.default_rng(seed)

    # Bước 1: Chọn tập Val khớp phân phối toàn cục trước (vì n_val nhỏ hơn nên ưu tiên chọn trước để đạt độ chính xác cao nhất)
    val_indices, rem_mask = _select_stratified_indices(
        all_counts=all_counts,
        valid_indices=valid_indices,
        n_select=n_val,
        target_box_dist=target_box_dist,
        target_pres_dist=target_pres_dist,
        rng=rng,
    )
    # Bước 2: Chọn tập Train từ phần còn lại cũng khớp phân phối toàn cục
    rem_valid_indices = valid_indices[rem_mask]
    train_indices, _ = _select_stratified_indices(
        all_counts=all_counts,
        valid_indices=rem_valid_indices,
        n_select=n_train,
        target_box_dist=target_box_dist,
        target_pres_dist=target_pres_dist,
        rng=rng,
    )

    # Đảo thứ tự ngẫu nhiên trong nội bộ train và val
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    # In bảng đối chiếu phân phối class giữa Toàn bộ Data vs Train vs Val
    train_box_sum = all_counts[train_indices].sum(axis=0)
    val_box_sum = all_counts[val_indices].sum(axis=0)
    train_dist = train_box_sum / (train_box_sum.sum() + 1e-8)
    val_dist = val_box_sum / (val_box_sum.sum() + 1e-8)

    print("\n" + "=" * 105)
    print(f"  PHÂN PHỐI CLASS (STRATIFIED BALANCED SAMPLING) — Toàn bộ Data ({len(valid_indices)} ảnh) vs Train ({len(train_indices)}) vs Val ({len(val_indices)})")
    print("=" * 105)
    print(f"{'Class Name':<18} | {'Global %':<10} | {'Train %':<10} | {'Val %':<10} | {'Train Boxes':<12} | {'Val Boxes':<10} | {'Val Images':<10}")
    print("-" * 105)
    val_img_pres = (all_counts[val_indices] > 0).sum(axis=0)
    for c in range(10):
        print(
            f"{VISDRONE_CLASSES[c]:<18} | "
            f"{target_box_dist[c] * 100:>8.2f}% | "
            f"{train_dist[c] * 100:>8.2f}% | "
            f"{val_dist[c] * 100:>8.2f}% | "
            f"{int(train_box_sum[c]):>12d} | "
            f"{int(val_box_sum[c]):>10d} | "
            f"{int(val_img_pres[c]):>10d}"
        )
    print("-" * 105)
    tv_train = 0.5 * float(np.abs(train_dist - target_box_dist).sum()) * 100.0
    tv_val = 0.5 * float(np.abs(val_dist - target_box_dist).sum()) * 100.0
    print(f"  Total Variation Distance so với Global: Train TVD = {tv_train:.2f}% | Val TVD = {tv_val:.2f}%")
    print("=" * 105)

    resolutions = ((img_size // 8, img_size // 8), (img_size // 16, img_size // 16), (img_size // 32, img_size // 32))

    def _load_indices(idx_list: List[int]) -> List[Dict]:
        loaded: List[Dict] = []
        for real_idx in idx_list:
            byteflow = txn.get(f"{real_idx:08d}".encode("ascii"))
            if byteflow is None:
                continue
            data = pickle.loads(byteflow)
            raw_boxes = data.get("boxes", [])
            raw_labels = data.get("labels", [])

            img = Image.open(io.BytesIO(data["image"])).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            img_np = np.array(img, dtype=np.uint8)
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

            counts = torch.from_numpy(all_counts[real_idx].astype(np.float32))
            bg_mask = torch.from_numpy(rasterize_boxes(raw_boxes, resolutions))
            boxes_norm = [
                (int(l), float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                for b, l in zip(raw_boxes, raw_labels)
                if 0 <= int(l) < 10
            ]
            loaded.append({
                "image": img_t,
                "img_np": img_np,
                "counts": counts,
                "bg_mask": bg_mask,
                "boxes_norm": boxes_norm,
            })
        return loaded

    print(f"[*] Đang nạp {len(train_indices) + len(val_indices)} ảnh đã chọn ({img_size}x{img_size}) vào RAM...")
    train_samples = _load_indices(train_indices)
    val_samples = _load_indices(val_indices)
    print(f"[+] Đã nạp xong {len(train_samples)} ảnh train và {len(val_samples)} ảnh val cân bằng theo phân phối toàn cục.")
    return train_samples, val_samples


# =============================================================================
# 4. ĐÁNH GIÁ ĐA TẦNG (P3, P4, P5) & TRÊN SLOTS CUỐI CÙNG
# =============================================================================

def _compute_fisher_full_and_pca3d(fg_vecs: np.ndarray, bg_vecs: np.ndarray) -> Tuple[float, float]:
    """Tính Fisher Discriminant Ratio giữa tập vector FG (N_fg, D) và BG (N_bg, D) trên không gian gốc D-chiều và PCA-3D."""
    if len(fg_vecs) == 0 or len(bg_vecs) == 0:
        return 0.0, 0.0
    mu_fg_f, mu_bg_f = fg_vecs.mean(axis=0), bg_vecs.mean(axis=0)
    var_fg_f = float(np.sum(np.var(fg_vecs, axis=0)))
    var_bg_f = float(np.sum(np.var(bg_vecs, axis=0)))
    full_fisher = float(np.sum((mu_fg_f - mu_bg_f) ** 2)) / (var_fg_f + var_bg_f + 1e-6)

    # Chiếu PCA-3D trên tập hợp chung [FG; BG]
    all_vecs = np.concatenate([fg_vecs, bg_vecs], axis=0).astype(np.float32)
    centered = all_vecs - all_vecs.mean(axis=0, keepdims=True)
    if all_vecs.shape[0] >= 3 and all_vecs.shape[1] >= 3:
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ vt[:3].T
        except np.linalg.LinAlgError:
            proj = centered[:, :3]
    else:
        proj = centered[:, : min(3, centered.shape[1])]
    # Chuẩn hóa min-max từng kênh [0, 1] tương đương compute_pca_rgb
    p_min = proj.min(axis=0, keepdims=True)
    p_max = proj.max(axis=0, keepdims=True)
    proj_norm = (proj - p_min) / (p_max - p_min + 1e-6)

    n_fg = len(fg_vecs)
    fg_pca = proj_norm[:n_fg]
    bg_pca = proj_norm[n_fg:]
    mu_fg_p, mu_bg_p = fg_pca.mean(axis=0), bg_pca.mean(axis=0)
    var_fg_p = float(np.sum(np.var(fg_pca, axis=0)))
    var_bg_p = float(np.sum(np.var(bg_pca, axis=0)))
    pca_fisher = float(np.sum((mu_fg_p - mu_bg_p) ** 2)) / (var_fg_p + var_bg_p + 1e-6)
    return full_fisher, pca_fisher


def _build_pairwise_matrix(proto_sum: np.ndarray, proto_cnt: np.ndarray, dim: int) -> Tuple[np.ndarray, float]:
    """Tính ma trận khoảng cách Cosine 10x10 (45 cặp class) và trung bình cộng 45 cặp."""
    pairwise_dist = np.full((10, 10), np.nan, dtype=np.float32)
    unit_protos = np.zeros((10, dim), dtype=np.float64)
    valid_cls = proto_cnt > 0
    for c in range(10):
        if valid_cls[c]:
            v = proto_sum[c] / float(proto_cnt[c])
            unit_protos[c] = v / (np.linalg.norm(v) + 1e-8)
    for i in range(10):
        for j in range(10):
            if valid_cls[i] and valid_cls[j]:
                pairwise_dist[i, j] = 0.0 if i == j else float(1.0 - np.dot(unit_protos[i], unit_protos[j]))
    off_vals = pairwise_dist[~np.eye(10, dtype=bool)]
    off_vals = off_vals[np.isfinite(off_vals)]
    mean_45 = float(off_vals.mean()) if off_vals.size > 0 else 0.0
    return pairwise_dist, mean_45


@torch.no_grad()
def evaluate_architecture(
    model: MiniSlotCounter,
    val_samples: List[Dict],
    device: torch.device,
    small_area_thresh: float = 0.0025,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Đo đầy đủ tất cả chỉ số trên từng tầng P3, P4, P5 VÀ trên Slots cuối cùng sau khi qua P3+P4+P5."""
    model.eval()

    scale_names = ("p3", "p4", "p5")
    sc_acc: Dict[str, Dict[str, List[float]]] = {
        sn: {
            "small_ratios": [],
            "all_ratios": [],
            "full_fishers": [],
            "pca_fishers": [],
            "class_seps": [],
            "fg_mass_pcts": [],
        }
        for sn in scale_names
    }
    sc_proto_sum = {sn: np.zeros((10, model.dim), dtype=np.float64) for sn in scale_names}
    sc_proto_cnt = {sn: np.zeros(10, dtype=np.int64) for sn in scale_names}

    # Bộ tích lũy cho SLOTS cuối cùng (sau khi đi qua toàn bộ P3, P4, P5)
    slot_small_ratios: List[float] = []
    slot_all_ratios: List[float] = []
    slot_act_inact_ratios: List[float] = []
    slot_class_seps: List[float] = []
    slot_all10_seps: List[float] = []
    slot_fg_mass_pcts: List[float] = []
    slot_share_p3_pcts: List[float] = []
    slot_share_p4_pcts: List[float] = []
    slot_share_p5_pcts: List[float] = []

    slot_fg_vecs_all: List[np.ndarray] = []
    slot_bg_vecs_all: List[np.ndarray] = []
    slot_proto_sum = np.zeros((10, model.dim), dtype=np.float64)
    slot_proto_cnt = np.zeros(10, dtype=np.int64)

    abs_errors: List[float] = []
    vis_pack: Dict[str, np.ndarray] = {}

    for idx, s in enumerate(val_samples):
        img_t = s["image"].unsqueeze(0).to(device)
        counts_gt = s["counts"].numpy()
        boxes_norm = s["boxes_norm"]
        small_boxes = [b for b in boxes_norm if (b[3] * b[4]) < small_area_thresh]
        small_cls_set = {int(b[0]) for b in small_boxes}

        preds, attn_ori, feats, resolutions, slots = model.forward_all(img_t)
        pred_np = preds[0].cpu().numpy()
        abs_errors.append(float(np.abs(pred_np - counts_gt).mean()))

        # Chia khoảng token cho P3, P4, P5
        (H3, W3), (H4, W4), (H5, W5) = resolutions
        n3, n4, n5 = H3 * W3, H4 * W4, H5 * W5
        slices = [(0, n3), (n3, n3 + n4), (n3 + n4, n3 + n4 + n5)]

        all_masks_flat = []
        fg_attn_all_tokens = attn_ori[0, :10, :].sum(dim=0).cpu().numpy()  # (n3+n4+n5,)

        # ---------------------------------------------------------------------
        # A. ĐO TRÊN TỪNG TẦNG P3, P4, P5
        # ---------------------------------------------------------------------
        for s_idx, sn in enumerate(scale_names):
            Hs, Ws = resolutions[s_idx]
            t_start, t_end = slices[s_idx]
            fs = feats[s_idx][0]  # (C, Hs, Ws)
            l2_map = torch.linalg.vector_norm(fs.float(), dim=0).cpu().numpy()  # (Hs, Ws)

            all_box_mask = build_box_mask_at_res(boxes_norm, Hs, Ws)
            small_box_mask = build_box_mask_at_res(small_boxes, Hs, Ws)
            bg_mask_bool = ~all_box_mask
            all_masks_flat.append(all_box_mask.reshape(-1))

            # 1. Energy Contrast Ratio (In-Box / Out-Box)
            if all_box_mask.any() and bg_mask_bool.any():
                bg_energy = float(l2_map[bg_mask_bool].mean()) + 1e-6
                sc_acc[sn]["all_ratios"].append(float(l2_map[all_box_mask].mean()) / bg_energy)
                if small_box_mask.any():
                    sc_acc[sn]["small_ratios"].append(float(l2_map[small_box_mask].mean()) / bg_energy)

            # 2a. Full-Dim FG/BG Fisher
            fs_hwc = fs.detach().cpu().float().permute(1, 2, 0).numpy()  # (Hs, Ws, C)
            if all_box_mask.any() and bg_mask_bool.any():
                fg_full = fs_hwc[all_box_mask]
                bg_full = fs_hwc[bg_mask_bool]
                mu_fg_f, mu_bg_f = fg_full.mean(axis=0), bg_full.mean(axis=0)
                var_fg_f = float(np.sum(np.var(fg_full, axis=0)))
                var_bg_f = float(np.sum(np.var(bg_full, axis=0)))
                sc_acc[sn]["full_fishers"].append(float(np.sum((mu_fg_f - mu_bg_f) ** 2)) / (var_fg_f + var_bg_f + 1e-6))

            # 2b. PCA-3D FG/BG Fisher
            pca_rgb = compute_pca_rgb(fs)  # (Hs, Ws, 3)
            if all_box_mask.any() and bg_mask_bool.any():
                fg_pts = pca_rgb[all_box_mask]
                bg_pts = pca_rgb[bg_mask_bool]
                mu_fg, mu_bg = fg_pts.mean(axis=0), bg_pts.mean(axis=0)
                var_fg = float(np.sum(np.var(fg_pts, axis=0)))
                var_bg = float(np.sum(np.var(bg_pts, axis=0)))
                sc_acc[sn]["pca_fishers"].append(float(np.sum((mu_fg - mu_bg) ** 2)) / (var_fg + var_bg + 1e-6))

            # 3. Inter-Class Cosine Separation tại tầng sn
            fs_norm = F.normalize(fs.float(), dim=0).cpu().numpy()  # (C, Hs, Ws)
            class_protos = []
            for c in range(10):
                c_mask = build_box_mask_at_res(boxes_norm, Hs, Ws, class_filter=c)
                cnt_c = int(c_mask.sum())
                if cnt_c > 0:
                    sc_proto_sum[sn][c] += fs_norm[:, c_mask].sum(axis=1)
                    sc_proto_cnt[sn][c] += cnt_c
                if cnt_c >= 2:
                    proto = fs_norm[:, c_mask].mean(axis=1)
                    proto = proto / (np.linalg.norm(proto) + 1e-8)
                    class_protos.append(proto)
            if len(class_protos) >= 2:
                P = np.stack(class_protos, axis=0)
                sim = P @ P.T
                K = len(class_protos)
                off_diag = sim[~np.eye(K, dtype=bool)]
                sc_acc[sn]["class_seps"].append(float(1.0 - off_diag.mean()))

            # 4. Foreground Attention Mass rơi trúng GT Box tại tầng sn
            fg_attn_s = fg_attn_all_tokens[t_start:t_end].reshape(Hs, Ws)
            if all_box_mask.any():
                sc_acc[sn]["fg_mass_pcts"].append(
                    float(fg_attn_s[all_box_mask].sum()) / (float(fg_attn_s.sum()) + 1e-8) * 100.0
                )

            if idx == 0:
                if sn == "p3":
                    vis_pack = {
                        "img_np": s["img_np"],
                        "boxes_norm": boxes_norm,
                        "l2_map": l2_map,
                        "pca_rgb": pca_rgb,
                        "fg_attn_p3": fg_attn_s,
                    }
                vis_pack[f"l2_map_{sn}"] = l2_map
                vis_pack[f"pca_rgb_{sn}"] = pca_rgb
                vis_pack[f"fg_attn_{sn}"] = fg_attn_s

        # ---------------------------------------------------------------------
        # B. ĐO TRÊN SLOTS CUỐI CÙNG (SAU KHI QUA TOÀN BỘ P3 + P4 + P5)
        # ---------------------------------------------------------------------
        slots_np = slots[0].detach().cpu().float().numpy()  # (11, D): 0..9 là 10 class, 10 là BG slot
        fg_slots_np = slots_np[:10]  # (10, D)
        bg_slot_np = slots_np[10]    # (D,)

        slot_norms = np.linalg.norm(fg_slots_np, axis=1)  # (10,)
        bg_slot_norm = float(np.linalg.norm(bg_slot_np)) + 1e-6

        active_cls = [c for c in range(10) if counts_gt[c] > 0]
        inactive_cls = [c for c in range(10) if counts_gt[c] == 0]
        small_active_cls = [c for c in active_cls if c in small_cls_set]

        # 1. Slot Energy Ratios (Active All / BG Slot, Active Small / BG Slot, Active / Inactive)
        if active_cls:
            slot_all_ratios.append(float(slot_norms[active_cls].mean()) / bg_slot_norm)
            if small_active_cls:
                slot_small_ratios.append(float(slot_norms[small_active_cls].mean()) / bg_slot_norm)
            if inactive_cls:
                inact_norm = float(slot_norms[inactive_cls].mean()) + 1e-6
                slot_act_inact_ratios.append(float(slot_norms[active_cls].mean()) / inact_norm)

        # 2. Tích lũy vector FG Slots (active) và BG Slot để tính Slot FG/BG Fisher (Full-Dim & PCA-3D)
        for c in active_cls:
            slot_fg_vecs_all.append(fg_slots_np[c])
        slot_bg_vecs_all.append(bg_slot_np)

        # 3. Slot Inter-Class Cosine Separation (Local trong 1 ảnh & Global 45-pair tích lũy)
        fg_slots_unit = fg_slots_np / (np.linalg.norm(fg_slots_np, axis=1, keepdims=True) + 1e-8)  # (10, D)
        # 3a. Trên tất cả 10 class slots
        sim10 = fg_slots_unit @ fg_slots_unit.T
        slot_all10_seps.append(float(1.0 - sim10[~np.eye(10, dtype=bool)].mean()))

        # 3b. Trên các class active có mặt trong ảnh + tích lũy Global Slot Prototype cho mỗi class
        for c in active_cls:
            slot_proto_sum[c] += fg_slots_unit[c]
            slot_proto_cnt[c] += 1
        if len(active_cls) >= 2:
            P_slot = fg_slots_unit[active_cls]  # (K, D)
            sim_act = P_slot @ P_slot.T
            K_act = len(active_cls)
            slot_class_seps.append(float(1.0 - sim_act[~np.eye(K_act, dtype=bool)].mean()))

        # 4. Tổng Foreground Attention Mass của Slots trên toàn bộ chuỗi [P3, P4, P5] rơi trúng GT Box
        combined_box_mask = np.concatenate(all_masks_flat, axis=0)  # (n3+n4+n5,)
        total_fg_attn_sum = float(fg_attn_all_tokens.sum()) + 1e-8
        if combined_box_mask.any():
            slot_fg_mass_pcts.append(float(fg_attn_all_tokens[combined_box_mask].sum()) / total_fg_attn_sum * 100.0)

        # Tỷ trọng phân bổ Attention của 10 Foreground Slots cho P3 vs P4 vs P5
        slot_share_p3_pcts.append(float(fg_attn_all_tokens[:n3].sum()) / total_fg_attn_sum * 100.0)
        slot_share_p4_pcts.append(float(fg_attn_all_tokens[n3 : n3 + n4].sum()) / total_fg_attn_sum * 100.0)
        slot_share_p5_pcts.append(float(fg_attn_all_tokens[n3 + n4 :].sum()) / total_fg_attn_sum * 100.0)

    # Tính ma trận khoảng cách Cosine 10x10 (45 cặp class) cho P3, P4, P5 và cho SLOTS cuối cùng
    pw_p3, mean45_p3 = _build_pairwise_matrix(sc_proto_sum["p3"], sc_proto_cnt["p3"], model.dim)
    pw_p4, mean45_p4 = _build_pairwise_matrix(sc_proto_sum["p4"], sc_proto_cnt["p4"], model.dim)
    pw_p5, mean45_p5 = _build_pairwise_matrix(sc_proto_sum["p5"], sc_proto_cnt["p5"], model.dim)
    pw_slots, mean45_slots = _build_pairwise_matrix(slot_proto_sum, slot_proto_cnt, model.dim)

    vis_pack["pairwise_cos_sep"] = pw_p3
    vis_pack["pairwise_cos_sep_p4"] = pw_p4
    vis_pack["pairwise_cos_sep_p5"] = pw_p5
    vis_pack["pairwise_cos_sep_slots"] = pw_slots

    # Tính Fisher FG/BG trên không gian Slots cuối cùng (Full-Dim D chiều & PCA-3D)
    if slot_fg_vecs_all and slot_bg_vecs_all:
        slot_full_fisher, slot_pca_fisher = _compute_fisher_full_and_pca3d(
            np.stack(slot_fg_vecs_all, axis=0),
            np.stack(slot_bg_vecs_all, axis=0),
        )
    else:
        slot_full_fisher, slot_pca_fisher = 0.0, 0.0

    def _m(lst: List[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    metrics = {
        # --- Tầng P3 (giữ tương thích ngược với các tên cột cũ) ---
        "p3_small_ratio": _m(sc_acc["p3"]["small_ratios"]),
        "p3_all_ratio": _m(sc_acc["p3"]["all_ratios"]),
        "full_fg_bg_fisher": _m(sc_acc["p3"]["full_fishers"]),
        "pca_fg_bg_fisher": _m(sc_acc["p3"]["pca_fishers"]),
        "p3_full_fisher": _m(sc_acc["p3"]["full_fishers"]),
        "p3_pca_fisher": _m(sc_acc["p3"]["pca_fishers"]),
        "class_cos_sep": _m(sc_acc["p3"]["class_seps"]),
        "p3_class_cos_sep": _m(sc_acc["p3"]["class_seps"]),
        "p3_global_45pair_sep": mean45_p3,
        "p3_fg_mass_pct": _m(sc_acc["p3"]["fg_mass_pcts"]),
        # --- Tầng P4 ---
        "p4_small_ratio": _m(sc_acc["p4"]["small_ratios"]),
        "p4_all_ratio": _m(sc_acc["p4"]["all_ratios"]),
        "p4_full_fisher": _m(sc_acc["p4"]["full_fishers"]),
        "p4_pca_fisher": _m(sc_acc["p4"]["pca_fishers"]),
        "p4_class_cos_sep": _m(sc_acc["p4"]["class_seps"]),
        "p4_global_45pair_sep": mean45_p4,
        "p4_fg_mass_pct": _m(sc_acc["p4"]["fg_mass_pcts"]),
        # --- Tầng P5 ---
        "p5_small_ratio": _m(sc_acc["p5"]["small_ratios"]),
        "p5_all_ratio": _m(sc_acc["p5"]["all_ratios"]),
        "p5_full_fisher": _m(sc_acc["p5"]["full_fishers"]),
        "p5_pca_fisher": _m(sc_acc["p5"]["pca_fishers"]),
        "p5_class_cos_sep": _m(sc_acc["p5"]["class_seps"]),
        "p5_global_45pair_sep": mean45_p5,
        "p5_fg_mass_pct": _m(sc_acc["p5"]["fg_mass_pcts"]),
        # --- SLOTS cuối cùng (sau khi đi qua toàn bộ P3 + P4 + P5) ---
        "slot_small_ratio": _m(slot_small_ratios),
        "slot_all_ratio": _m(slot_all_ratios),
        "slot_act_inact_ratio": _m(slot_act_inact_ratios),
        "slot_full_fisher": slot_full_fisher,
        "slot_pca_fisher": slot_pca_fisher,
        "slot_class_cos_sep": _m(slot_class_seps),
        "slot_all10_cos_sep": _m(slot_all10_seps),
        "slot_global_45pair_sep": mean45_slots,
        "slot_fg_mass_pct": _m(slot_fg_mass_pcts),
        "slot_share_p3_pct": _m(slot_share_p3_pcts),
        "slot_share_p4_pct": _m(slot_share_p4_pcts),
        "slot_share_p5_pct": _m(slot_share_p5_pcts),
        # --- Sai số đếm ---
        "val_mae": _m(abs_errors),
    }
    return metrics, vis_pack


def train_and_eval_block(
    block_type: str,
    train_loader: DataLoader,
    val_samples: List[Dict],
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Huấn luyện và đánh giá 1 kiến trúc block với seed cố định."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MiniSlotCounter(block_type=block_type, num_classes=10, dim=64).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            imgs = batch["image"].to(device, non_blocking=True)
            gt_counts = batch["counts"].to(device, non_blocking=True)
            bg_mask = batch["bg_mask"].to(device, non_blocking=True)
            fg_mask = 1.0 - bg_mask

            optimizer.zero_grad()
            preds, attn_ori, _, _, _ = model.forward_all(imgs)

            loss_count = F.smooth_l1_loss(preds, gt_counts, beta=1.0)
            bg_attn = attn_ori[:, -1, :].clamp(1e-6, 1.0 - 1e-6)
            fg_attn = attn_ori[:, :10, :].sum(dim=1).clamp(1e-6, 1.0 - 1e-6)
            loss_bg = F.binary_cross_entropy(bg_attn, bg_mask)
            loss_fg = F.binary_cross_entropy(fg_attn, fg_mask)

            inactive = gt_counts == 0
            if inactive.any():
                inact_attn = attn_ori[:, :10, :][inactive].clamp(1e-6, 1.0 - 1e-6)
                loss_sup = F.binary_cross_entropy(inact_attn, torch.zeros_like(inact_attn))
            else:
                loss_sup = torch.tensor(0.0, device=device)

            loss = loss_count + 1.0 * loss_bg + 1.0 * loss_fg + 0.05 * loss_sup
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

    train_sec = time.perf_counter() - t0
    metrics, vis_pack = evaluate_architecture(model, val_samples, device)
    metrics["block"] = block_type
    metrics["params_m"] = params_m
    metrics["train_sec"] = train_sec
    return metrics, vis_pack


def _save_pairwise_csv_and_heatmap(
    names: List[str],
    vis_packs: Dict[str, Dict[str, np.ndarray]],
    mat_key: str,
    csv_path: str,
    hm_path: Optional[str],
    title_prefix: str,
) -> None:
    """Lưu file CSV 45 cặp class và lưới Heatmap 10x10 cho một tầng hoặc cho Slots cuối cùng."""
    pair_fields = ["class_i", "class_j", "pair"] + names + ["mean_blocks"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pair_fields)
        writer.writeheader()
        for i in range(10):
            for j in range(i + 1, 10):
                row_dict: Dict[str, object] = {
                    "class_i": VISDRONE_CLASSES[i],
                    "class_j": VISDRONE_CLASSES[j],
                    "pair": f"{VISDRONE_CLASSES[i]} vs {VISDRONE_CLASSES[j]}",
                }
                valid_vals = []
                for bname in names:
                    val = float(vis_packs[bname][mat_key][i, j])
                    if np.isnan(val):
                        row_dict[bname] = ""
                    else:
                        row_dict[bname] = round(val, 4)
                        valid_vals.append(val)
                row_dict["mean_blocks"] = round(float(np.mean(valid_vals)), 4) if valid_vals else ""
                writer.writerow(row_dict)

    if hm_path is None:
        return

    short_labels = ["ped", "ppl", "bic", "car", "van", "trk", "tri", "awn", "bus", "mot"]
    n_blocks = len(names)
    ncols = min(4, n_blocks)
    nrows = math.ceil(n_blocks / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.9 * nrows), constrained_layout=True)
    axes_flat = np.array(axes).reshape(-1) if n_blocks > 1 else [axes]

    all_mats = [vis_packs[b][mat_key] for b in names]
    finite_vals = np.concatenate([m[np.isfinite(m)] for m in all_mats]) if all_mats else np.array([0.5])
    vmax_hm = float(max(np.percentile(finite_vals, 98), 0.1)) if finite_vals.size > 0 else 1.0

    for idx_b, bname in enumerate(names):
        ax = axes_flat[idx_b]
        mat = vis_packs[bname][mat_key]
        off_vals = mat[~np.eye(10, dtype=bool)]
        off_vals = off_vals[np.isfinite(off_vals)]
        mean_sep = float(off_vals.mean()) if off_vals.size > 0 else 0.0

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0.0, vmax=vmax_hm)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_title(f"[{bname.upper()}] {title_prefix}\nMean(45 pairs): {mean_sep:.4f}", fontsize=10.5, fontweight="bold")

        for r_i in range(10):
            for c_j in range(10):
                val = mat[r_i, c_j]
                if r_i == c_j:
                    txt = "-"
                    t_color = "#888888"
                elif np.isnan(val):
                    txt = "na"
                    t_color = "#888888"
                else:
                    txt = f"{val:.2f}"
                    t_color = "white" if val > 0.6 * vmax_hm else "black"
                ax.text(c_j, r_i, txt, ha="center", va="center", fontsize=6.5, color=t_color)

    for idx_empty in range(n_blocks, len(axes_flat)):
        axes_flat[idx_empty].axis("off")

    fig.colorbar(im, ax=list(axes_flat[:n_blocks]), fraction=0.015, pad=0.02, label="Cosine Distance (1 - cos_sim)")
    plt.suptitle(f"Pairwise Inter-Class Cosine Separation — {title_prefix} (All 45 Class Pairs)", fontsize=14, fontweight="bold")
    plt.savefig(hm_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_visual_comparison(
    results: List[Dict[str, float]],
    vis_packs: Dict[str, Dict[str, np.ndarray]],
    out_dir: str,
) -> None:
    """Lưu đầy đủ CSV đa tầng (P3, P4, P5) + CSV trên Slots cuối cùng + biểu đồ & heatmap."""
    os.makedirs(out_dir, exist_ok=True)

    # 1a. Lưu CSV tổng hợp đầy đủ (giữ tương thích các cột cũ + toàn bộ P3, P4, P5 và Slots)
    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    fieldnames = [
        "block", "params_m", "train_sec",
        "p3_small_ratio", "p3_all_ratio", "full_fg_bg_fisher", "pca_fg_bg_fisher",
        "p3_full_fisher", "p3_pca_fisher",
        "class_cos_sep", "p3_class_cos_sep", "p3_fg_mass_pct", "p3_global_45pair_sep",
        "p4_small_ratio", "p4_all_ratio", "p4_full_fisher", "p4_pca_fisher",
        "p4_class_cos_sep", "p4_fg_mass_pct", "p4_global_45pair_sep",
        "p5_small_ratio", "p5_all_ratio", "p5_full_fisher", "p5_pca_fisher",
        "p5_class_cos_sep", "p5_fg_mass_pct", "p5_global_45pair_sep",
        "slot_small_ratio", "slot_all_ratio", "slot_act_inact_ratio",
        "slot_full_fisher", "slot_pca_fisher",
        "slot_class_cos_sep", "slot_all10_cos_sep", "slot_global_45pair_sep",
        "slot_fg_mass_pct", "slot_share_p3_pct", "slot_share_p4_pct", "slot_share_p5_pct",
        "val_mae",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: round(r[k], 4) if isinstance(r[k], float) else r[k] for k in fieldnames})

    # 1b. Lưu bảng CSV chuyên biệt cho SLOTS cuối cùng (sau khi đi qua P3+P4+P5) & đối chiếu P3/P4/P5
    slots_csv_path = os.path.join(out_dir, "benchmark_slots_results.csv")
    slots_fields = [
        "block", "params_m", "val_mae",
        "slot_small_ratio", "slot_all_ratio", "slot_act_inact_ratio",
        "slot_full_fisher", "slot_pca_fisher",
        "slot_class_cos_sep", "slot_all10_cos_sep", "slot_global_45pair_sep",
        "slot_fg_mass_pct", "slot_share_p3_pct", "slot_share_p4_pct", "slot_share_p5_pct",
        "p3_small_ratio", "p4_small_ratio", "p5_small_ratio",
        "p3_all_ratio", "p4_all_ratio", "p5_all_ratio",
        "p3_full_fisher", "p4_full_fisher", "p5_full_fisher",
        "p3_pca_fisher", "p4_pca_fisher", "p5_pca_fisher",
        "p3_fg_mass_pct", "p4_fg_mass_pct", "p5_fg_mass_pct",
        "p3_class_cos_sep", "p4_class_cos_sep", "p5_class_cos_sep",
        "p3_global_45pair_sep", "p4_global_45pair_sep", "p5_global_45pair_sep",
    ]
    with open(slots_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=slots_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: round(r[k], 4) if isinstance(r[k], float) else r[k] for k in slots_fields})

    # 2. Lưới ảnh so sánh P3, P4, P5 của tất cả các kiến trúc trên cùng 1 ảnh val
    n = len(results)
    grid_paths: Dict[str, str] = {}
    for sn_tag, sn_lbl in [("p3", "P3"), ("p4", "P4"), ("p5", "P5")]:
        fig, axes = plt.subplots(n, 3, figsize=(16, 4.2 * n))
        if n == 1:
            axes = axes[None, :]

        for row, r in enumerate(results):
            bname = r["block"]
            vp = vis_packs[bname]
            img_np = vp["img_np"]
            H, W = img_np.shape[:2]
            boxes_px = [
                (
                    c,
                    int(max(0, round((cx - w / 2) * W))),
                    int(max(0, round((cy - h / 2) * H))),
                    int(min(W, round((cx + w / 2) * W))),
                    int(min(H, round((cy + h / 2) * H))),
                )
                for c, cx, cy, w, h in vp["boxes_norm"]
            ]
            gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            gray_rgb = np.stack([gray] * 3, axis=-1)

            l2_map_s = vp.get(f"l2_map_{sn_tag}", vp["l2_map"])
            pca_rgb_s = vp.get(f"pca_rgb_{sn_tag}", vp["pca_rgb"])
            fg_attn_s = vp.get(f"fg_attn_{sn_tag}", vp["fg_attn_p3"])

            l2_up = cv2.resize(l2_map_s, (W, H), interpolation=cv2.INTER_NEAREST)
            pca_up = cv2.resize(pca_rgb_s, (W, H), interpolation=cv2.INTER_NEAREST)
            fg_up = cv2.resize(fg_attn_s, (W, H), interpolation=cv2.INTER_NEAREST)

            axes[row, 0].imshow(l2_up, cmap="magma")
            draw_boxes_on_ax(axes[row, 0], boxes_px, color="#00FF66", linewidth=0.7)
            axes[row, 0].set_title(
                f"[{bname.upper()}] {sn_lbl} L2-Norm\n{sn_lbl} Small: {r[f'{sn_tag}_small_ratio']:.2f}x | {sn_lbl} All: {r[f'{sn_tag}_all_ratio']:.2f}x",
                fontsize=10.5, fontweight="bold",
            )
            axes[row, 0].axis("off")

            axes[row, 1].imshow(pca_up)
            draw_boxes_on_ax(axes[row, 1], boxes_px, color="#FFFFFF", linewidth=0.7)
            axes[row, 1].set_title(
                f"[{bname.upper()}] {sn_lbl} PCA-RGB\n{sn_lbl}Fisher: {r[f'{sn_tag}_full_fisher']:.2f} (PCA: {r[f'{sn_tag}_pca_fisher']:.2f}) | {sn_lbl}Sep: {r[f'{sn_tag}_class_cos_sep']:.3f}",
                fontsize=10.0, fontweight="bold",
            )
            axes[row, 1].axis("off")

            axes[row, 2].imshow(gray_rgb)
            axes[row, 2].imshow(fg_up, cmap="jet", alpha=0.6, vmin=0.0, vmax=1.0)
            draw_boxes_on_ax(axes[row, 2], boxes_px, color="#00FF66", linewidth=0.7)
            axes[row, 2].set_title(
                f"[{bname.upper()}] {sn_lbl} FG Attn\n{sn_lbl}Mass: {r[f'{sn_tag}_fg_mass_pct']:.1f}% | SlotAllMass: {r['slot_fg_mass_pct']:.1f}% | MAE: {r['val_mae']:.2f}",
                fontsize=10.5, fontweight="bold",
            )
            axes[row, 2].axis("off")

        plt.tight_layout()
        g_name = "benchmark_comparison_grid.png" if sn_tag == "p3" else f"benchmark_comparison_grid_{sn_tag}.png"
        g_path = os.path.join(out_dir, g_name)
        plt.savefig(g_path, dpi=150, bbox_inches="tight")
        plt.close()
        grid_paths[sn_tag] = g_path
    grid_path = grid_paths["p3"]

    # 3. Biểu đồ cột tổng hợp 4 chỉ số chính cho từng tầng P3, P4, P5 và Final Slots
    names = [r["block"] for r in results]
    x = np.arange(len(names))
    summary_chart_paths: Dict[str, str] = {}

    scale_chart_configs = [
        ("p3", "P3", "p3_small_ratio", "p3_full_fisher", "p3_pca_fisher", "p3_class_cos_sep", "benchmark_summary_chart.png"),
        ("p4", "P4", "p4_small_ratio", "p4_full_fisher", "p4_pca_fisher", "p4_class_cos_sep", "benchmark_summary_chart_p4.png"),
        ("p5", "P5", "p5_small_ratio", "p5_full_fisher", "p5_pca_fisher", "p5_class_cos_sep", "benchmark_summary_chart_p5.png"),
        ("slots", "Final Slots", "slot_small_ratio", "slot_full_fisher", "slot_pca_fisher", "slot_class_cos_sep", "benchmark_summary_chart_slots.png"),
    ]

    for sc_tag, sc_lbl, k_small, k_ffish, k_pfish, k_csep, c_fname in scale_chart_configs:
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        panels = [
            (k_small, f"{sc_lbl} Small Object Energy (In/Out Ratio, cao hơn = tốt hơn)", "#e74c3c"),
            ("fisher_both", f"{sc_lbl} FG vs BG Fisher Separation: Full-Dim (gốc) vs PCA-3D", "#3498db"),
            (k_csep, f"Inter-Class Cosine Separation tại {sc_lbl} (cao hơn = tách class tốt hơn)", "#2ecc71"),
            ("val_mae", "Validation Counting MAE (thấp hơn = đếm chính xác hơn)", "#9b59b6"),
        ]
        for ax, (key, title, color) in zip(axes.flatten(), panels):
            if key == "fisher_both":
                v_full = [r[k_ffish] for r in results]
                v_pca = [r[k_pfish] for r in results]
                w_bar = 0.38
                b1 = ax.bar(x - w_bar / 2, v_full, width=w_bar, color="#2980b9", alpha=0.9, label="Full-Dim (Gốc)")
                b2 = ax.bar(x + w_bar / 2, v_pca, width=w_bar, color="#7fb3d5", alpha=0.9, label="PCA-3D (Chiếu)")
                ax.legend(fontsize=8.5)
                for bar, v in zip(b1, v_full):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
                for bar, v in zip(b2, v_pca):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
            else:
                vals = [r[key] for r in results]
                bars = ax.bar(x, vals, color=color, alpha=0.85)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9.5, fontweight="bold")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        c_path = os.path.join(out_dir, c_fname)
        plt.savefig(c_path, dpi=160, bbox_inches="tight")
        plt.close()
        summary_chart_paths[sc_tag] = c_path
    chart_path = summary_chart_paths["p3"]

    # 3b. Biểu đồ đa tầng so sánh P3 vs P4 vs P5 vs Final Slots (2x4 panels: bao gồm cả PCA-3D Fisher & Val MAE)
    fig_ms, axes_ms = plt.subplots(2, 4, figsize=(26, 10))
    ms_panels = [
        (
            [("p3_small_ratio", "P3", "#e74c3c"), ("p4_small_ratio", "P4", "#f39c12"), ("p5_small_ratio", "P5", "#95a5a6"), ("slot_small_ratio", "Slots", "#8e44ad")],
            "Small Object Energy Ratio (P3 / P4 / P5 / Final Slots)",
        ),
        (
            [("p3_all_ratio", "P3", "#e74c3c"), ("p4_all_ratio", "P4", "#f39c12"), ("p5_all_ratio", "P5", "#95a5a6"), ("slot_all_ratio", "Slots", "#8e44ad")],
            "All Object Energy Ratio (P3 / P4 / P5 / Final Slots)",
        ),
        (
            [("p3_full_fisher", "P3", "#2980b9"), ("p4_full_fisher", "P4", "#27ae60"), ("p5_full_fisher", "P5", "#7f8c8d"), ("slot_full_fisher", "Slots", "#8e44ad")],
            "Full-Dim FG vs BG Fisher Separation (P3 / P4 / P5 / Final Slots)",
        ),
        (
            [("p3_pca_fisher", "P3", "#3498db"), ("p4_pca_fisher", "P4", "#2ecc71"), ("p5_pca_fisher", "P5", "#95a5a6"), ("slot_pca_fisher", "Slots", "#9b59b6")],
            "PCA-3D FG vs BG Fisher Separation (P3 / P4 / P5 / Final Slots)",
        ),
        (
            [("p3_fg_mass_pct", "P3", "#e67e22"), ("p4_fg_mass_pct", "P4", "#2ecc71"), ("p5_fg_mass_pct", "P5", "#3498db"), ("slot_fg_mass_pct", "Slots(All)", "#8e44ad")],
            "FG Attention Mass in Box % (P3 / P4 / P5 / Combined Slots)",
        ),
        (
            [("p3_class_cos_sep", "P3", "#16a085"), ("p4_class_cos_sep", "P4", "#2980b9"), ("p5_class_cos_sep", "P5", "#7f8c8d"), ("slot_class_cos_sep", "Slots", "#8e44ad")],
            "Per-Image Class Cosine Separation (P3 / P4 / P5 / Final Slots)",
        ),
        (
            [("p3_global_45pair_sep", "P3", "#c0392b"), ("p4_global_45pair_sep", "P4", "#d35400"), ("p5_global_45pair_sep", "P5", "#7f8c8d"), ("slot_global_45pair_sep", "Slots", "#8e44ad")],
            "Global 45-Pair Cosine Separation (P3 / P4 / P5 / Final Slots)",
        ),
        (
            [("slot_share_p3_pct", "Share P3%", "#e74c3c"), ("slot_share_p4_pct", "Share P4%", "#f39c12"), ("slot_share_p5_pct", "Share P5%", "#3498db"), ("slot_act_inact_ratio", "Act/Inact(x)", "#8e44ad")],
            "Slot Attention Share (P3/P4/P5 %) & Active/Inactive Ratio",
        ),
    ]
    w_m = 0.20
    offsets = [-1.5 * w_m, -0.5 * w_m, 0.5 * w_m, 1.5 * w_m]
    for ax, (series_list, title) in zip(axes_ms.flatten(), ms_panels):
        for off, (k_col, lbl, clr) in zip(offsets, series_list):
            vals = [r[k_col] for r in results]
            ax.bar(x + off, vals, width=w_m, color=clr, alpha=0.88, label=lbl)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    ms_chart_path = os.path.join(out_dir, "benchmark_multiscale_slots_chart.png")
    plt.savefig(ms_chart_path, dpi=160, bbox_inches="tight")
    plt.close()

    # 4 & 5. Lưu CSV 45 cặp class & Heatmap 10x10 cho P3, P4, P5 và cho SLOTS cuối cùng
    pair_csv_path = os.path.join(out_dir, "pairwise_cosine_sep.csv")
    pair_hm_path = os.path.join(out_dir, "pairwise_cosine_sep_heatmaps.png")
    _save_pairwise_csv_and_heatmap(names, vis_packs, "pairwise_cos_sep", pair_csv_path, pair_hm_path, "At P3")

    pair_p4_csv_path = os.path.join(out_dir, "pairwise_cosine_sep_p4.csv")
    pair_p4_hm_path = os.path.join(out_dir, "pairwise_cosine_sep_p4_heatmaps.png")
    _save_pairwise_csv_and_heatmap(names, vis_packs, "pairwise_cos_sep_p4", pair_p4_csv_path, pair_p4_hm_path, "At P4")

    pair_p5_csv_path = os.path.join(out_dir, "pairwise_cosine_sep_p5.csv")
    pair_p5_hm_path = os.path.join(out_dir, "pairwise_cosine_sep_p5_heatmaps.png")
    _save_pairwise_csv_and_heatmap(names, vis_packs, "pairwise_cos_sep_p5", pair_p5_csv_path, pair_p5_hm_path, "At P5")

    pair_slots_csv_path = os.path.join(out_dir, "pairwise_cosine_sep_slots.csv")
    pair_slots_hm_path = os.path.join(out_dir, "pairwise_cosine_sep_slots_heatmaps.png")
    _save_pairwise_csv_and_heatmap(
        names, vis_packs, "pairwise_cos_sep_slots", pair_slots_csv_path, pair_slots_hm_path, "Final Slots (P3+P4+P5)"
    )

    print("\n[+] Đã lưu kết quả benchmark tại:")
    print(f"    1. Bảng CSV tổng hợp (P3,P4,P5,Slots) : {os.path.abspath(csv_path)}")
    print(f"    2. Bảng CSV chuyên biệt Final Slots    : {os.path.abspath(slots_csv_path)}")
    print(f"    3. Lưới ảnh so sánh P3 / P4 / P5       : {os.path.abspath(grid_path)} | {os.path.abspath(grid_paths['p4'])} | {os.path.abspath(grid_paths['p5'])}")
    print(f"    4. Biểu đồ tổng hợp 4 chỉ số (P3..Slot): {os.path.abspath(chart_path)} (+ p4, p5, slots)")
    print(f"    5. Biểu đồ Đa tầng (P3/P4/P5 vs Slots) : {os.path.abspath(ms_chart_path)}")
    print(f"    6. CSV Cosine Sep 45 cặp (P3)          : {os.path.abspath(pair_csv_path)} (+ heatmap: {os.path.abspath(pair_hm_path)})")
    print(f"    7. CSV Cosine Sep 45 cặp (P4)          : {os.path.abspath(pair_p4_csv_path)} (+ heatmap: {os.path.abspath(pair_p4_hm_path)})")
    print(f"    8. CSV Cosine Sep 45 cặp (P5)          : {os.path.abspath(pair_p5_csv_path)} (+ heatmap: {os.path.abspath(pair_p5_hm_path)})")
    print(f"    9. CSV Cosine Sep 45 cặp (Final Slots) : {os.path.abspath(pair_slots_csv_path)} (+ heatmap: {os.path.abspath(pair_slots_hm_path)})")


def main():
    parser = argparse.ArgumentParser(description="Benchmark 16 Backbone Block Architectures on VisDrone Micro-Dataset")
    parser.add_argument(
        "--blocks",
        type=str,
        nargs="+",
        default=["all"],
        help=f"Danh sách block cần test hoặc 'all'. Hỗ trợ: {', '.join(BLOCK_REGISTRY.keys())}",
    )
    parser.add_argument("--lmdb", type=str, default="D:/apps/Datasets/VisDrone/train_data.lmdb")
    parser.add_argument("--n-train", type=int, default=160, help="Số ảnh train trong RAM")
    parser.add_argument("--n-val", type=int, default=40, help="Số ảnh val trong RAM")
    parser.add_argument("--size", type=int, default=384, help="Kích thước ảnh (khuyên dùng 384, chia hết 32 & 6)")
    parser.add_argument("--epochs", type=int, default=15, help="Số epoch train mỗi kiến trúc")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu", action="store_true", help="Chạy trên CPU")
    parser.add_argument("--out-dir", type=str, default="Visdrone/output_benchmark")
    parser.add_argument("--run", "--runs", dest="runs", type=int, default=1, help="Số lần chạy lặp lại (mỗi lần lưu vào out-dir/run_i)")
    args = parser.parse_args()

    device = torch.device("cpu" if (args.cpu or not torch.cuda.is_available()) else args.device)
    print(f"[*] Thiết bị benchmark: {device}")

    if "all" in args.blocks:
        block_list = list(BLOCK_REGISTRY.keys())
    else:
        block_list = args.blocks

    train_samples, val_samples = load_micro_splits(
        lmdb_path=args.lmdb,
        n_train=args.n_train,
        n_val=args.n_val,
        img_size=args.size,
        seed=args.seed,
    )
    train_loader = DataLoader(InMemoryVisDroneMicroDataset(train_samples), batch_size=args.batch_size, shuffle=True)

    for run_idx in range(1, args.runs + 1):
        if args.runs > 1:
            run_out_dir = os.path.join(args.out_dir, f"run_{run_idx}")
            run_seed = args.seed + (run_idx - 1) * 1000
        else:
            run_out_dir = os.path.join(args.out_dir, "run_1")
            run_seed = args.seed

        print("\n" + "=" * 150)
        print(f"  BẮT ĐẦU BENCHMARK {len(block_list)} KIẾN TRÚC [RUN {run_idx}/{args.runs}] ({args.epochs} epochs | Seed: {run_seed} | {args.size}x{args.size} | {len(train_samples)} train / {len(val_samples)} val)")
        print(f"  Thư mục lưu kết quả: {run_out_dir}")
        print("=" * 150)
        print(
            f"{'Block':<12} | {'Params':<7} | {'P3 Small':<8} | {'P3 All':<7} | {'P4 Small':<8} | {'P4 All':<7} | "
            f"{'P5 Small':<8} | {'P5 All':<7} | {'P3 Fisher':<9} | {'P4 Fisher':<9} | {'P5 Fisher':<9} | {'Val MAE':<7}"
        )
        print("-" * 150)

        results: List[Dict[str, float]] = []
        vis_packs: Dict[str, Dict[str, np.ndarray]] = {}

        for bname in block_list:
            metrics, vpack = train_and_eval_block(
                block_type=bname,
                train_loader=train_loader,
                val_samples=val_samples,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                seed=run_seed,
            )
            results.append(metrics)
            vis_packs[bname] = vpack
            print(
                f"{bname:<12} | {metrics['params_m']:<6.2f}M | "
                f"{metrics['p3_small_ratio']:<7.2f}x | {metrics['p3_all_ratio']:<6.2f}x | "
                f"{metrics['p4_small_ratio']:<7.2f}x | {metrics['p4_all_ratio']:<6.2f}x | "
                f"{metrics['p5_small_ratio']:<7.2f}x | {metrics['p5_all_ratio']:<6.2f}x | "
                f"{metrics['p3_full_fisher']:<9.3f} | {metrics['p4_full_fisher']:<9.3f} | {metrics['p5_full_fisher']:<9.3f} | "
                f"{metrics['val_mae']:<7.3f}",
                flush=True,
            )

        print("=" * 150)
        print("\n" + "=" * 150)
        print(f"  BẢNG ĐO CHI TIẾT ĐA TẦNG P3 / P4 / P5 (PCA Fisher, Class Cosine Sep, 45-Pair Sep, FG Mass%) [RUN {run_idx}/{args.runs}]")
        print("=" * 150)
        print(
            f"{'Block':<12} | {'P3/P4/P5 PCAFish':<20} | {'P3/P4/P5 ClsSep':<22} | "
            f"{'P3/P4/P5 45PairSep':<22} | {'P3/P4/P5 FGMass(%)':<20} | {'Val MAE':<7}"
        )
        print("-" * 150)
        for m in results:
            pca_str = f"{m['p3_pca_fisher']:5.2f}/{m['p4_pca_fisher']:5.2f}/{m['p5_pca_fisher']:5.2f}"
            csep_str = f"{m['p3_class_cos_sep']:6.4f}/{m['p4_class_cos_sep']:6.4f}/{m['p5_class_cos_sep']:6.4f}"
            p45_str = f"{m['p3_global_45pair_sep']:6.4f}/{m['p4_global_45pair_sep']:6.4f}/{m['p5_global_45pair_sep']:6.4f}"
            mass_str = f"{m['p3_fg_mass_pct']:5.1f}/{m['p4_fg_mass_pct']:5.1f}/{m['p5_fg_mass_pct']:5.1f}"
            print(
                f"{m['block']:<12} | {pca_str:<20} | {csep_str:<22} | {p45_str:<22} | {mass_str:<20} | {m['val_mae']:<7.3f}"
            )
        print("=" * 150)
        print("\n" + "=" * 150)
        print(f"  BẢNG ĐO TRÊN SLOTS CUỐI CÙNG [RUN {run_idx}/{args.runs}] (SAU KHI ĐI QUA TOÀN BỘ P3 + P4 + P5)")
        print("=" * 150)
        print(
            f"{'Block':<12} | {'SlotSmall':<9} | {'SlotAll':<8} | {'Act/Inact':<9} | "
            f"{'SlotFullFish':<12} | {'SlotPCAFish':<11} | {'SlotClsSep':<10} | {'Slot45Pair':<10} | "
            f"{'SlotFGMass%':<11} | {'Share P3/P4/P5 (%)':<20} | {'Val MAE':<7}"
        )
        print("-" * 150)
        for m in results:
            share_str = f"{m['slot_share_p3_pct']:4.1f}/{m['slot_share_p4_pct']:4.1f}/{m['slot_share_p5_pct']:4.1f}"
            print(
                f"{m['block']:<12} | {m['slot_small_ratio']:<8.2f}x | {m['slot_all_ratio']:<7.2f}x | {m['slot_act_inact_ratio']:<8.2f}x | "
                f"{m['slot_full_fisher']:<12.3f} | {m['slot_pca_fisher']:<11.3f} | "
                f"{m['slot_class_cos_sep']:<10.4f} | {m['slot_global_45pair_sep']:<10.4f} | "
                f"{m['slot_fg_mass_pct']:<10.1f}% | {share_str:<20} | {m['val_mae']:<7.3f}"
            )
        print("=" * 150)

        save_visual_comparison(results, vis_packs, run_out_dir)


if __name__ == "__main__":
    main()
