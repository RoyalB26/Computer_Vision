"""
ssm.py
------
Module SelectiveSSM (S6):
Bao bọc lớp toán học Selective Scan thành một PyTorch Module hoàn chỉnh.

Nhiệm vụ:
    1. Quản lý các tham số học được:
       - A_log: Ma trận A dạng tham số hóa log để đảm bảo A luôn âm (hệ thống ổn định).
       - D: Vector skip-connection.
       - x_proj: Chiếu tuyến tính u -> (delta_proj, B, C) (Cơ chế Selection).
       - dt_proj: Chiếu delta_proj -> delta với khởi tạo đặc biệt (Inverse Softplus).
    2. Cung cấp 2 chế độ tính toán:
       - forward(): Xử lý song song toàn bộ chuỗi (Training / Prompt processing).
       - step(): Xử lý từng token với độ phức tạp O(1) (Inference / Generation).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import MambaConfig
from .selective_scan import selective_scan_seq, selective_scan_step


class SelectiveSSM(nn.Module):
    """
    Khối Selective State Space Model (S6).

    Args:
        config (MambaConfig): Cấu hình siêu tham số của mô hình.
    """
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.d_inner = config.d_inner
        self.d_state = config.d_state
        self.dt_rank = config.dt_rank_val
        self.num_slot = config.num_slot
        # 1. Khởi tạo ma trận A theo HiPPO / S4D (diagonal real)
        # A_i,j = -(j + 1) -> Khởi tạo log để tham số hóa A = -exp(A_log)
        # Giúp đảm bảo A luôn âm trong suốt quá trình tối ưu hóa
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.num_slot, self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # 2. Skip-connection parameter D (khởi tạo 0 để gradient chảy qua trạng thái ẩn SSM)
        self.D = nn.Parameter(torch.zeros(self.d_inner))
        self.D._no_weight_decay = True

        # 3. Chiếu x thành (delta_proj, B, C)
        # delta_proj có số chiều dt_rank
        # B có số chiều d_state
        # C có số chiều d_state
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * self.d_state,
            bias=False
        )

        # 4. Chiếu dt_proj từ dt_rank lên d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Khởi tạo trọng số đặc biệt cho dt_proj theo bài báo Mamba
        dt_init_std = (self.dt_rank ** -0.5) * config.dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Khởi tạo dt_bias sao cho softplus(dt_bias) phân bố đều trong [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        # Hàm ngược của softplus: inv_softplus(x) = log(exp(x) - 1)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

    def forward(
        self,
        u: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Xử lý toàn bộ chuỗi tensor u: (B, L, S, d_inner).

        Returns:
            ys: (B, L, S, d_inner) - Chuỗi đầu ra
            final_state: (B, S, d_inner, d_state) - Trạng thái ẩn tại bước cuối
        """
        # 1. Chiếu u để lấy tham số động delta, B, C phụ thuộc vào ngữ cảnh
        x_dbl = self.x_proj(u)  # (B, L, S, dt_rank + 2 * d_state)
        delta_proj, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # 2. Chiếu delta_proj lên d_inner và đưa qua hàm kích hoạt softplus (riêng cho từng vị trí/slot S)
        delta = F.softplus(self.dt_proj(delta_proj))  # (B, L, S, d_inner)

        # 3. Tính ma trận A từ A_log
        A = -torch.exp(self.A_log.float())  # (S, d_inner, d_state)

        # 4. Thực hiện thuật toán Selective Scan
        ys, final_state = selective_scan_seq(
            u=u,
            delta=delta,
            A=A,
            B=B,
            C=C,
            D=self.D,
            initial_state=initial_state
        )

        return ys, final_state

    def step(
        self,
        u_t: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Xử lý 1 token tại bước t: u_t: (B, d_inner).
        Độ phức tạp tính toán là O(1) không đổi theo chiều dài ngữ cảnh.

        Returns:
            y_t: (B, d_inner) - Token đầu ra
            next_state: (B, d_inner, d_state) - Trạng thái ẩn mới
        """
        # 1. Chiếu u_t thành (delta_proj, B_t, C_t)
        x_dbl = self.x_proj(u_t)  # (B, dt_rank + 2 * d_state)
        delta_proj, B_t, C_t = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # 2. Chiếu delta_proj và tính softplus
        delta_t = F.softplus(self.dt_proj(delta_proj))  # (B, d_inner)

        # 3. Lấy ma trận A
        A = -torch.exp(self.A_log.float())

        # 4. Bước quét đệ quy O(1)
        y_t, next_state = selective_scan_step(
            u_t=u_t,
            delta_t=delta_t,
            A=A,
            B_t=B_t,
            C_t=C_t,
            D=self.D,
            prev_state=prev_state
        )
        return y_t, next_state


if __name__ == "__main__":
    print("=" * 60)
    print("TEST MODULE SELECTIVE SSM (ssm.py)")
    print("=" * 60)

    # 1. Khởi tạo cấu hình và module SSM
    num_slot = 24
    cfg = MambaConfig(d_model=64, d_state=16, expand=2, num_slot= num_slot)
    ssm = SelectiveSSM(cfg)
    print(f"[OK] Đã khởi tạo SelectiveSSM:")
    print(f"  - d_inner: {ssm.d_inner}")
    print(f"  - d_state: {ssm.d_state}")
    print(f"  - dt_rank: {ssm.dt_rank}")
    print(f"  - A_log shape  : {tuple(ssm.A_log.shape)}")
    print(f"  - x_proj weight: {tuple(ssm.x_proj.weight.shape)}")
    print(f"  - dt_proj weight: {tuple(ssm.dt_proj.weight.shape)}")

    # 2. Test forward pass toàn bộ chuỗi
    batch_size = 2
    seq_len = 5
    u = torch.randn(batch_size, seq_len, num_slot, ssm.d_inner, requires_grad=True)
    ys_seq, final_state = ssm(u)
    print(f"\n[OK] Chạy forward chuỗi thành công:")
    print(f"  - Input u shape    : {tuple(u.shape)}")
    print(f"  - Output ys shape  : {tuple(ys_seq.shape)}")
    print(f"  - Final state shape: {tuple(final_state.shape)}")
    assert ys_seq.shape == u.shape, "Output shape phải khớp với input shape!"

    # 4. Test tính toán Gradient (Backward pass)
    loss = ys_seq.sum()
    loss.backward()
    assert ssm.A_log.grad is not None, "Gradient của A_log bị None!"
    assert ssm.x_proj.weight.grad is not None, "Gradient của x_proj bị None!"
    assert ssm.dt_proj.weight.grad is not None, "Gradient của dt_proj bị None!"
    assert ssm.D.grad is not None, "Gradient của D bị None!"
    assert u.grad is not None, "Gradient của u bị None!"
    print(f"\n[OK] Kiểm tra Backward pass:")
    print(f"  - Grad norm A_log  : {ssm.A_log.grad.norm().item():.5f}")
    print(f"  - Grad norm x_proj : {ssm.x_proj.weight.grad.norm().item():.5f}")
    print(f"  - Grad norm dt_proj: {ssm.dt_proj.weight.grad.norm().item():.5f}")
    print(f"  - Grad norm input u: {u.grad.norm().item():.5f}")

    print("\n=> [PASS] ssm.py kiểm tra hoàn tất thành công!")
