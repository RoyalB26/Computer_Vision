"""
configs.py
----------
Cấu hình siêu tham số cho kiến trúc Mamba (Selective State Space Model).
"""

from dataclasses import dataclass, field
import math
from typing import Union


@dataclass
class MambaConfig:
    """
    Bảng cấu hình các siêu tham số cho Mamba.

    Attributes:
        d_model: Chiều không gian ẩn (hidden dimension) của mô hình.
        n_layer: Số lượng lớp Mamba (ResidualBlock).
        vocab_size: Kích thước từ vựng (vocabulary size).
        d_state: Kích thước trạng thái ẩn SSM (State Space Model state dimension, N). Mặc định là 16.
        expand: Hệ số mở rộng kích thước chiều ẩn bên trong khối SSM (E). Mặc định là 2.
        dt_rank: Hạng của phép chiếu delta (rank of delta projection). 
                 Nếu là 'auto', sẽ lấy ceil(d_model / 16).
        d_conv: Kích thước kernel của 1D Convolution cục bộ (local 1D convolution kernel size). Mặc định là 4.
        conv_bias: Có sử dụng bias cho Conv1d hay không.
        bias: Có sử dụng bias cho các Linear projection hay không.
        dt_min: Giá trị nhỏ nhất dùng khởi tạo delta.
        dt_max: Giá trị lớn nhất dùng khởi tạo delta.
        dt_init_floor: Ngưỡng dưới khi khởi tạo delta.
        dt_scale: Hệ số tỉ lệ khi khởi tạo weight cho dt_proj.
        rms_norm_eps: Epsilon chống chia cho 0 trong RMSNorm.
    """
    d_model: int = 128
    n_layer: int = 1
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_scale: float = 1.0
    rms_norm_eps: float = 1e-5
    num_slot: int = 24
    
    @property
    def d_inner(self) -> int:
        """Kích thước chiều không gian ẩn bên trong khối SSM (D = expand * d_model)."""
        return int(self.expand * self.d_model)

    @property
    def dt_rank_val(self) -> int:
        """Tính toán giá trị nguyên cụ thể của dt_rank."""
        if self.dt_rank == "auto":
            return math.ceil(self.d_model / 16)
        return int(self.dt_rank)


if __name__ == "__main__":
    print("=" * 60)
    print("TEST CẤU HÌNH MAMBA (configs.py)")
    print("=" * 60)

    # 1. Khởi tạo cấu hình mặc định (phù hợp test cục bộ)
    cfg_default = MambaConfig()
    print(f"[OK] Cấu hình mặc định:")
    print(f"  - d_model     : {cfg_default.d_model}")
    print(f"  - n_layer     : {cfg_default.n_layer}")
    print(f"  - vocab_size  : {cfg_default.vocab_size}")
    print(f"  - d_state (N) : {cfg_default.d_state}")
    print(f"  - expand (E)  : {cfg_default.expand}")
    print(f"  - d_inner (D) : {cfg_default.d_inner} (tính từ expand * d_model = {cfg_default.expand} * {cfg_default.d_model})")
    print(f"  - dt_rank_val : {cfg_default.dt_rank_val} (tính từ ceil(d_model / 16) = {cfg_default.dt_rank_val})")
    print(f"  - d_conv      : {cfg_default.d_conv}")

    # 2. Khởi tạo cấu hình tương đương phiên bản tiêu chuẩn (Mamba-130M)
    cfg_130m = MambaConfig(
        d_model=768,
        n_layer=24,
        vocab_size=50277,
        d_state=16,
        expand=2,
        dt_rank="auto",
        d_conv=4
    )
    print(f"\n[OK] Cấu hình Mamba-130M:")
    print(f"  - d_model     : {cfg_130m.d_model}")
    print(f"  - n_layer     : {cfg_130m.n_layer}")
    print(f"  - d_inner     : {cfg_130m.d_inner}")
    print(f"  - dt_rank_val : {cfg_130m.dt_rank_val}")

    assert cfg_default.d_inner == 256, f"Sai d_inner mặc định: {cfg_default.d_inner}"
    assert cfg_default.dt_rank_val == 8, f"Sai dt_rank mặc định: {cfg_default.dt_rank_val}"
    assert cfg_130m.dt_rank_val == 48, f"Sai dt_rank 130m: {cfg_130m.dt_rank_val}"
    print("\n=> [PASS] configs.py kiểm tra hoàn tất thành công!")
