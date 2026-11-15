"""
model.py
--------
Kiến trúc mô hình Mamba hoàn chỉnh và MambaLMHeadModel phục vụ mô hình ngôn ngữ (Language Modeling).

Kiến trúc tổng thể:
    Input Token IDs (B, L)
            │
      Token Embedding
            │
    ┌───────────────────────┐
    │  ResidualBlock 1      │
    │  ResidualBlock 2      │  x n_layer
    │  ...                  │
    │  ResidualBlock N      │
    └───────────────────────┘
            │
        RMSNorm (norm_f)
            │
      LM Head (Linear)
            │
      Logits (B, L, vocab_size)

Đặc biệt:
    Hỗ trợ phương thức generate() với cơ chế Recurrent State Caching O(1) per token:
    Tốc độ sinh token không phụ thuộc vào độ dài chuỗi ngữ cảnh quá khứ!
"""

import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import MambaConfig
from .norm import RMSNorm
from .mamba_layer import ResidualBlock


class Mamba(nn.Module):
    """
    Backbone của mô hình Mamba.

    Args:
        config (MambaConfig): Cấu hình siêu tham số.
    """
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        # 2. Xếp chồng n_layer tầng Mamba ResidualBlock
        self.layers = nn.ModuleList([
            ResidualBlock(config) for _ in range(config.n_layer)
        ])

        # 3. Lớp chuẩn hóa RMSNorm cuối cùng
        self.norm_f = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Xử lý toàn bộ chuỗi token: (B, L, S, d_model).
        """
        x = inputs

        # Đi qua từng tầng Mamba
        for layer in self.layers:
            x, _ = layer(x)

        # Chuẩn hóa đầu ra cuối cùng
        x = self.norm_f(x)
        return x




if __name__ == "__main__":
    print("=" * 60)
    print("TEST MÔ HÌNH TOÀN DIỆN MAMBA LM HEAD (model.py)")
    print("=" * 60)

    # Cố định seed ngẫu nhiên để kết quả tái lập được
    torch.manual_seed(42)

    # 1. Khởi tạo cấu hình nhỏ phục vụ test nhanh
    cfg = MambaConfig(
        d_model=64,
        n_layer=3,
        vocab_size=256,
        d_state=16,
        expand=2,
        d_conv=4
    )
    model = Mamba(cfg)

    # Đếm số lượng tham số
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Đã khởi tạo MambaLMHeadModel thành công:")
    print(f"  - Số tầng (n_layer): {cfg.n_layer}")
    print(f"  - d_model          : {cfg.d_model}")
    print(f"  - d_inner          : {cfg.d_inner}")
    print(f"  - Tổng số tham số  : {n_params:,} parameters")

    batch_size = 2
    seq_len = 5
    u = torch.randn(batch_size, seq_len, cfg.num_slot, cfg.d_model, requires_grad=True)
    out = model(u)
    print("out = ", out[:, -1, :, :].shape)