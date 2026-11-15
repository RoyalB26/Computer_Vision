"""
norm.py
-------
Cài đặt RMSNorm (Root Mean Square Layer Normalization).
Đây là kỹ thuật chuẩn hóa được sử dụng xuyên suốt trong kiến trúc Mamba 
thay cho LayerNorm truyền thống (loại bỏ bước trừ giá trị trung bình để tăng tốc độ tính toán).

Công thức:
    RMS(x) = sqrt( mean(x^2, dim=-1, keepdim=True) + eps )
    y = (x / RMS(x)) * weight
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.

    Args:
        d_model (int): Số chiều của tensor đầu vào tại chiều cuối cùng.
        eps (float): Hằng số nhỏ để tránh chia cho 0 (mặc định 1e-5).
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Trọng số scale học được (learnable parameter), khởi tạo bằng 1
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện chuẩn hóa RMSNorm trên tensor đầu vào.

        Args:
            x (torch.Tensor): Tensor đầu vào với kích thước (*, d_model), 
                              thường là (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Tensor sau chuẩn hóa có cùng kích thước và dtype với x.
        """
        # Tính toán ở float32 để đảm bảo độ chính xác số học
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        # Tính mean(x^2) theo chiều đặc trưng cuối cùng
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        # Tính nghịch đảo của RMS: 1 / sqrt(mean(x^2) + eps)
        rsqrt = torch.rsqrt(variance + self.eps)

        normed = x_fp32 * rsqrt
        # Nhân với tham số học được weight và đưa về kiểu dữ liệu ban đầu
        output = normed.to(input_dtype) * self.weight
        return output


if __name__ == "__main__":
    print("=" * 60)
    print("TEST CHUẨN HÓA RMSNORM (norm.py)")
    print("=" * 60)

    # 1. Khởi tạo kích thước test
    batch_size = 2
    seq_len = 5
    d_model = 128
    eps = 1e-5

    norm_layer = RMSNorm(d_model=d_model, eps=eps)
    print(f"[OK] Đã khởi tạo RMSNorm(d_model={d_model}, eps={eps})")

    # 2. Tạo input giả lập ngẫu nhiên
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    print(f"  - Input shape : {tuple(x.shape)}")

    # 3. Chạy forward pass
    y = norm_layer(x)
    print(f"  - Output shape: {tuple(y.shape)}")
    assert x.shape == y.shape, "Kích thước đầu ra không khớp với đầu vào!"

    # 4. Kiểm tra tính chất toán học của RMSNorm:
    # Sau khi chuẩn hóa (khi weight = 1), RMS(y) phải xấp xỉ bằng 1.0
    rms_y = torch.sqrt(torch.mean(y.pow(2), dim=-1))
    print(f"  - RMS trung bình của output: {rms_y.mean().item():.5f} (kỳ vọng ~ 1.0)")
    assert torch.allclose(rms_y, torch.ones_like(rms_y), atol=1e-3), "RMS của output không xấp xỉ 1!"

    # 5. Kiểm tra backward pass và lan truyền gradient
    loss = y.sum()
    loss.backward()
    assert norm_layer.weight.grad is not None, "Gradient của weight bị None!"
    assert x.grad is not None, "Gradient của input x bị None!"
    print(f"  - Gradient norm của weight: {norm_layer.weight.grad.norm().item():.5f}")
    print(f"  - Gradient norm của input x: {x.grad.norm().item():.5f}")

    # 6. Kiểm tra độ ổn định số học khi input toàn số 0
    zeros_in = torch.zeros(1, 1, d_model)
    zeros_out = norm_layer(zeros_in)
    assert not torch.isnan(zeros_out).any(), "Xảy ra NaN khi input là 0!"
    print(f"  - Xử lý tensor 0: An toàn (Không bị lỗi chia cho 0)")

    print("\n=> [PASS] norm.py kiểm tra hoàn tất thành công!")
