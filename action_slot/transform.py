
import os
import sys

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from typing import Any, Callable, Dict, List, Optional, Union
from torch import Tensor
import numpy as np
import torch
import torchvision.transforms as transforms
from BaseModel.core.registrar import TRANSFORMS
# from core.registry import build_from_cfg


@TRANSFORMS.register_module("Action_Slot_Transform")
class Action_Slot_Transform:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if data['inputs']:
            data['inputs'] = self.process_videos(data['inputs'])

        if "bg_seg" in data['targets']:
            data['targets']['bg_seg'] = self.process_masks(data['targets']['bg_seg'])
        
        return data

    def process_videos(self, data: List[Any]) -> torch.Tensor:
        """Vectorized transform: chuyển đổi danh sách ảnh thành 1 Batch Tensor và resize đồng loạt."""
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode
        from PIL import Image

        # Đảm bảo chuyển tất cả các frame thành numpy array (16, H, W, 3)
        np_frames = []
        for img in data:
            if isinstance(img, Image.Image):
                np_frames.append(np.asarray(img))
            elif isinstance(img, np.ndarray):
                np_frames.append(img)
            elif isinstance(img, torch.Tensor):
                np_frames.append(img.cpu().numpy())
            else:
                np_frames.append(np.asarray(img))

        # Gom 16 frames thành Tensor 4D: (T, C, H, W) kiểu uint8
        arr = np.stack(np_frames, axis=0) # (16, H, W, C)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2) # (16, C, H, W)

        # Resize toàn bộ 16 frame cùng lúc bằng C++ Tensor backend
        tensor = TF.resize(tensor, [256, 768], interpolation=InterpolationMode.BILINEAR, antialias=True)
        # Chuyển float32 và chuẩn hóa nhanh
        tensor = tensor.float().div_(255.0)
        tensor = TF.normalize(tensor, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        return tensor

    def process_masks(self, data: List[Any]) -> torch.Tensor:
        """Vectorized transform cho background mask (T_mask, 1, 32, 96)."""
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode
        from PIL import Image

        np_masks = []
        for m in data:
            if isinstance(m, Image.Image):
                np_masks.append(np.asarray(m))
            elif isinstance(m, np.ndarray):
                np_masks.append(m)
            elif isinstance(m, torch.Tensor):
                np_masks.append(m.cpu().numpy())
            else:
                np_masks.append(np.asarray(m))

        arr = np.stack(np_masks, axis=0) # (T_mask, H, W)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=1) # (T_mask, 1, H, W)
        elif arr.ndim == 4 and arr.shape[-1] == 1:
            arr = np.transpose(arr, (0, 3, 1, 2))

        tensor = torch.from_numpy(arr) # uint8
        tensor = TF.resize(tensor, [32, 96], interpolation=InterpolationMode.NEAREST)
        tensor = tensor.float().div_(255.0)
        return tensor


    