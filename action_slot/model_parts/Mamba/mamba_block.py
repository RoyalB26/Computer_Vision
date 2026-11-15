"""
mamba_block.py
--------------
Khối Mamba hoàn chỉnh (Mamba Block / Mixer Block):
Kết hợp cơ chế mở rộng chiều ẩn (Expansion), tích chập 1D nhân quả (Causal Conv1d),
hàm kích hoạt phi tuyến tính (SiLU), khối Selective SSM (S6), và cổng nhân (Gating).

Kiến trúc bên trong 1 khối Mamba:
        
             x (B, L, d_model)
                   │
                in_proj
               /       \\
              u         z
              │         │
           Conv1d       │
              │         │
            SiLU        │
              │         │
             SSM      SiLU
              \\       /
               \\     /
               Multiply (Gating)
                   │
                out_proj
                   │
             y (B, L, d_model)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import MambaConfig
from .ssm import SelectiveSSM


class MambaBlock(nn.Module):
    """
    Một khối Mamba Block hoàn chỉnh.

    Args:
        config (MambaConfig): Cấu hình siêu tham số của mô hình.
    """
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.d_conv = config.d_conv
        self.num_slots = config.num_slot
        # 1. Phép chiếu mở rộng đầu vào: d_model -> 2 * d_inner (chia đôi thành nhánh u và z)
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=config.bias)

        # 2. Tích chập 1D cục bộ (Depthwise Causal Conv1d)
        # padding = d_conv - 1 để đảm bảo tính nhân quả (chỉ nhìn về quá khứ)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            bias=config.conv_bias,
            padding=self.d_conv - 1
        )

        # 3. Hàm kích hoạt phi tuyến tính
        self.act = nn.SiLU()

        # 4. Khối lõi Selective State Space Model (S6)
        self.ssm = SelectiveSSM(config)

        # 5. Phép chiếu đầu ra: d_inner -> d_model
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        initial_ssm_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Xử lý toàn bộ chuỗi đầu vào x: (B, L, S, d_model).

        Returns:
            out: (B, L, d_model)
            final_ssm_state: (B, d_inner, d_state)
        """
        batch_size, seq_len, S, _ = x.shape

        # 1. Chiếu tuyến tính và tách thành 2 nhánh: u (nhánh tính toán) và z (nhánh gating)
        xz = self.in_proj(x)  # (B, L, S, 2 * d_inner)
        u, z = torch.chunk(xz, 2, dim=-1)  # u: (B, L, S, d_inner), z: (B, L, S, d_inner)
        B,L,S,D= u.shape
        # 2. Causal 1D Convolution trên nhánh u
        # Chuyển (B, L, S, D) -> (B, D, S * L) cho Conv1d
        u_conv_in = u.view(B,L * S, D).transpose(1, 2)
        # Áp dụng Conv1d và cắt bớt phần padding bên phải để đảm bảo tính nhân quả và giữ nguyên độ dài L
        u_conv_out = self.conv1d(u_conv_in)[:, :, :(S * seq_len)]
        # Chuyển ngược lại về (B, L, S, D)
        u_conv = u_conv_out.transpose(1, 2).view(B,L,S, D)

        # 3. Kích hoạt SiLU trên nhánh conv
        u_act = self.act(u_conv)

        # 4. Đi qua khối Selective SSM
        y_ssm, final_ssm_state = self.ssm(u_act, initial_state=initial_ssm_state)

        # 5. Nhân gating với nhánh z: y = y_ssm * SiLU(z)
        y_gate = y_ssm * self.act(z)

        # 6. Chiếu tuyến tính về số chiều d_model ban đầu
        out = self.out_proj(y_gate)
        return out, final_ssm_state


if __name__ == "__main__":
    print("=" * 60)
    print("TEST KHỐI MAMBA BLOCK (mamba_block.py)")
    print("=" * 60)

    # 1. Khởi tạo cấu hình và khối Mamba
    num_slot= 24
    cfg = MambaConfig(d_model=64, d_state=16, expand=2, d_conv=4, num_slot= num_slot)
    block = MambaBlock(cfg)
    print(f"[OK] Đã khởi tạo MambaBlock:")
    print(f"  - d_model: {block.d_model}")
    print(f"  - d_inner: {block.d_inner}")
    print(f"  - d_conv : {block.d_conv}")

    # 2. Test forward pass toàn bộ chuỗi
    batch_size = 2
    seq_len = 5
    
    x = torch.randn(batch_size, seq_len, num_slot, cfg.d_model, requires_grad=True)
    out_seq, final_ssm = block(x)

    print(f"\n[OK] Chạy forward chuỗi thành công:")
    print(f"  - Input shape    : {tuple(x.shape)} (B, L, d_model)")
    print(f"  - Output shape   : {tuple(out_seq.shape)} (B, L, d_model)")
    print(f"  - Final SSM state: {tuple(final_ssm.shape)}")
    assert out_seq.shape == x.shape, "Kích thước output phải bằng kích thước input!"

    # 4. Test Backward pass
    loss = out_seq.sum()
    loss.backward()
    assert x.grad is not None, "Gradient của input x bị None!"
    assert block.in_proj.weight.grad is not None, "Gradient của in_proj bị None!"
    assert block.conv1d.weight.grad is not None, "Gradient của conv1d bị None!"
    assert block.out_proj.weight.grad is not None, "Gradient của out_proj bị None!"
    print(f"\n[OK] Kiểm tra Backward pass thành công:")
    print(f"  - Grad norm in_proj : {block.in_proj.weight.grad.norm().item():.5f}")
    print(f"  - Grad norm conv1d  : {block.conv1d.weight.grad.norm().item():.5f}")
    print(f"  - Grad norm out_proj: {block.out_proj.weight.grad.norm().item():.5f}")
    print(f"  - Grad norm input x : {x.grad.norm().item():.5f}")

    print("\n=> [PASS] mamba_block.py kiểm tra hoàn tất thành công!")
