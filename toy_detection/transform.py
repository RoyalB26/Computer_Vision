
import os
import sys

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from BaseModel.core.registrar import TRANSFORMS
# from core.registry import build_from_cfg

@TRANSFORMS.register_module("LocalToTensor")
class LocalToTensor:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "image" in data:
            img = data["image"]  # Mảng numpy 2D (H, W) chứa 0 và 255
            h, w = img.shape

            # 1. Tạo mặt nạ boolean: True là nét vẽ, False là nền
            fg_mask = torch.from_numpy(img > 0)
            bg_mask = ~fg_mask

            # 2. Sinh màu ngẫu nhiên trong khoảng [0.0, 1.0] cho nền và nét vẽ
            # shape: (3, 1, 1) để broadcast trực tiếp trên không gian (H, W)
            bg_color = torch.rand(3, 1, 1, dtype=torch.float32)
            fg_color = torch.rand(3, 1, 1, dtype=torch.float32)

            # (Tùy chọn) Đảm bảo độ tương phản: tránh trường hợp 2 màu quá giống nhau
            while torch.norm(fg_color - bg_color) < 0.3:
                fg_color = torch.rand(3, 1, 1, dtype=torch.float32)

            # 3. Khởi tạo canvas 3 kênh: (3, H, W)
            rgb_img = torch.empty((3, h, w), dtype=torch.float32)

            # 4. Gán màu nền và màu nét vẽ
            rgb_img[:, bg_mask] = bg_color.squeeze()[:, None]
            rgb_img[:, fg_mask] = fg_color.squeeze()[:, None]

            data["image"] = rgb_img

        return data