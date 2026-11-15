"""
mamba_layer.py
--------------
Lớp Mamba hoàn chỉnh kèm Residual Connection và Chuẩn hóa Pre-Norm (RMSNorm).

Kiến trúc:
        x ───────────────( + )───> Output
        │                  ▲
     RMSNorm               │
        │                  │
    MambaBlock ────────────┘
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from .configs import MambaConfig
from .norm import RMSNorm
from .mamba_block import MambaBlock


class ResidualBlock(nn.Module):
    """
    Một tầng (layer) hoàn chỉnh trong kiến trúc Mamba:
    Gồm Chuẩn hóa RMSNorm + Khối Mixer (MambaBlock) + Kết nối tắt Residual.

    Args:
        config (MambaConfig): Cấu hình siêu tham số của mô hình.
    """
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        # Pre-normalization sử dụng RMSNorm
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        # Khối biến đổi cốt lõi Mamba
        self.mixer = MambaBlock(config)

    def forward(
        self,
        x: torch.Tensor,
        initial_ssm_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Xử lý toàn bộ chuỗi: y = x + MambaBlock(RMSNorm(x))

        Args:
            x (torch.Tensor): Tensor đầu vào (B, L, d_model)
            initial_ssm_state: Trạng thái ban đầu của SSM

        Returns:
            out (torch.Tensor): Tensor đầu ra (B, L, d_model)
            final_ssm_state: Trạng thái SSM tại bước cuối
        """
        # Áp dụng chuẩn hóa trước (Pre-norm)
        normed_x = self.norm(x)
        # Đi qua khối Mamba mixer
        mixer_out, final_ssm_state = self.mixer(normed_x, initial_ssm_state=initial_ssm_state)
        # Cộng residual connection
        out = x + mixer_out
        return out, final_ssm_state


if __name__ == "__main__":
    print("=" * 60)
    print("TEST TẦNG RESIDUAL BLOCK (mamba_layer.py)")
    print("=" * 60)

    # 1. Khởi tạo cấu hình và tầng ResidualBlock
    cfg = MambaConfig(d_model=64, d_state=16, expand=2, d_conv=4, num_slot= 24)
    layer = ResidualBlock(cfg)
    print(f"[OK] Đã khởi tạo ResidualBlock:")
    print(f"  - Pre-norm: RMSNorm(d_model={cfg.d_model})")
    print(f"  - Mixer   : MambaBlock(d_model={cfg.d_model}, d_inner={cfg.d_inner})")

    # 2. Test forward pass toàn bộ chuỗi
    batch_size = 2
    seq_len = 6
    x = torch.randn(batch_size, seq_len, cfg.num_slot, cfg.d_model, requires_grad=True)
    out_seq, final_ssm = layer(x)

    print(f"\n[OK] Chạy forward chuỗi thành công:")
    print(f"  - Input shape    : {tuple(x.shape)} (B, L, d_model)")
    print(f"  - Output shape   : {tuple(out_seq.shape)} (B, L, d_model)")
    print(f"  - Final SSM state: {tuple(final_ssm.shape)}")
    assert out_seq.shape == x.shape, "Output shape phải khớp với input shape!"

    # 4. Kiểm tra Backward pass
    loss = out_seq.sum()
    loss.backward()
    assert x.grad is not None, "Gradient của x bị None!"
    assert layer.norm.weight.grad is not None, "Gradient của norm.weight bị None!"
    print(f"\n[OK] Kiểm tra Backward pass thành công:")
    print(f"  - Grad norm norm.weight: {layer.norm.weight.grad.norm().item():.5f}")
    print(f"  - Grad norm input x    : {x.grad.norm().item():.5f}")

    print("\n=> [PASS] mamba_layer.py kiểm tra hoàn tất thành công!")
