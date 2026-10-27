"""Attention Kernel module performing dynamic convolution of class kernels over feature map."""
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class attn_kernel(nn.Module):
    """Dynamic Kernel Convolution Module.

    Nhận đầu vào:
      - x: Feature maps ảnh có shape (B, C, H, W)
      - kr: Kernel đại diện cho các lớp có shape (B, num_class, D) hoặc (num_class, D)

    Thực hiện:
      1. Tính m = sqrt(D), reshape kr thành (B, num_class, m, m)
      2. Dùng nn.Linear đưa x từ (B, C, H, W) về (B, 1, H, W)
      3. Quét từng kernel m x m của mỗi class qua x để sinh ra feature map có cùng shape (H, W)
      4. Tổng hợp toàn bộ class cho ra output shape (B, num_class, H, W)

    Args:
        in_channels: Số channel C của ảnh đầu vào x (mặc định: 64, có thể tự động suy luận nếu None).
        bias: Có sử dụng bias trong nn.Linear hay không (mặc định: True).
    """

    def __init__(self, in_channels: Optional[int] = None, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        if in_channels is not None:
            self.linear = nn.Linear(in_channels, 1, bias=bias)
        else:
            self.linear = None
        self.bias = bias

    def _init_linear_if_needed(self, c: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.linear is None:
            self.in_channels = c
            self.linear = nn.Linear(c, 1, bias=self.bias).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, kr: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor ảnh shape (B, C, H, W)
            kr: Tensor kernel shape (B, num_class, D) hoặc (num_class, D)

        Returns:
            Tensor shape (B, num_class, H, W)
        """
        B, C, H, W = x.shape

        # Hỗ trợ trường hợp kr chỉ có shape (num_class, D) -> mở rộng thành (B, num_class, D)
        if kr.ndim == 2:
            kr = kr.unsqueeze(0).expand(B, -1, -1)

        b_kr, num_class, D = kr.shape
        assert b_kr == B, f"Batch size mismatch: x has {B}, but kr has {b_kr}"

        # 1. Tính m = sqrt(D) và view kr thành (B, num_class, m, m)
        m = int(math.isqrt(D))
        assert m * m == D, f"D phải là số chính phương (ví dụ: 16 -> m=4), nhận được D={D}"
        kr_2d = kr.view(B, num_class, m, m)

        # 2. Thực hiện nn.Linear để đưa x từ (B, C, H, W) về (B, 1, H, W)
        self._init_linear_if_needed(C, device=x.device, dtype=x.dtype)
        # Permute (B, C, H, W) -> (B, H, W, C) để đưa qua nn.Linear
        x_perm = x.permute(0, 2, 3, 1)
        x_1ch = self.linear(x_perm).permute(0, 3, 1, 2)  # (B, 1, H, W)

        # 3. Tính toán padding để đảm bảo output convolution giữ nguyên kích thước (H, W)
        # Với kernel kích thước m x m: tổng padding cần thêm là m - 1
        pad_left = (m - 1) // 2
        pad_right = (m - 1) - pad_left
        pad_top = (m - 1) // 2
        pad_bottom = (m - 1) - pad_top

        x_padded = F.pad(x_1ch, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

        # 4. Quét từng kernel m x m của mỗi class qua x bằng Grouped Convolution tối ưu hoá
        # Gom nhóm toàn bộ batch và class để thực thi trong 1 phép tính conv2d duy nhất:
        # - x_grouped: (1, B, H_pad, W_pad)
        # - weights: (B * num_class, 1, m, m)
        # - groups = B: mỗi kênh b của x chỉ tích chập với num_class kernels tương ứng của mẫu b đó
        x_grouped = x_padded.view(1, B, x_padded.shape[2], x_padded.shape[3])
        weights = kr_2d.reshape(B * num_class, 1, m, m)

        out = F.conv2d(x_grouped, weights, groups=B)  # shape: (1, B * num_class, H, W)

        # 5. Reshape về kết quả cuối cùng (B, num_class, H, W)
        out = out.view(B, num_class, H, W)
        return out


# Alias hỗ trợ quy ước CamelCase
AttnKernel = attn_kernel

__all__ = ["attn_kernel", "AttnKernel"]
