"""Custom Data Transforms for VisDrone Object Detection."""
from typing import Any, Dict, List, Tuple, Union
import random
import cv2
import numpy as np
from PIL import Image
import torch

from BaseModel.core.registrar import TRANSFORMS


@TRANSFORMS.register_module(name=["CLAHE", "CLAHE_LAB"], force=True)
class CLAHE:
    """Cân bằng Histogram cục bộ thích ứng độ tương phản (Contrast Limited Adaptive
    Histogram Equalization - CLAHE) trên kênh L (Luminance) trong không gian màu LAB.

    Giúp tăng cường độ tương phản cục bộ, làm rõ nét biên cạnh và cấu trúc của các vật thể
    nhỏ (drone imagery / VisDrone) mà vẫn bảo toàn màu sắc gốc trên các kênh A và B.
    Hỗ trợ hoàn toàn PyTorch DataLoader đa tiến trình (multiprocessing spawn) trên Windows.

    Args:
        clip_limit (float): Ngưỡng cắt tỉa tương phản để chống khuếch đại nhiễu (mặc định: 2.0).
        tile_grid_size (Tuple[int, int] hoặc List[int]): Kích thước ô lưới cục bộ (mặc định: (8, 8)).
        p (float): Xác suất áp dụng transform (0.0 đến 1.0, mặc định: 1.0).
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Union[Tuple[int, int], List[int]] = (8, 8),
        p: float = 1.0,
    ):
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(tile_grid_size)
        self.p = float(p)
        self._clahe = None

    @property
    def clahe(self):
        """Khởi tạo lười (lazy initialization) để tránh lưu C++ object khi pickle qua subprocess."""
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size,
            )
        return self._clahe

    def __getstate__(self):
        """Bỏ C++ binding cv2.CLAHE khi pickle qua multiprocessing worker của DataLoader."""
        state = self.__dict__.copy()
        state["_clahe"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._clahe = None

    def _apply_clahe_lab(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        """Chuyển RGB sang LAB -> áp CLAHE trên L -> chuyển lại RGB."""
        lab = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_clahe = self.clahe.apply(l_channel)
        lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "image" not in data or self.p <= 0.0:
            return data

        if self.p < 1.0 and random.random() > self.p:
            return data

        img = data["image"]

        if isinstance(img, Image.Image):
            # PIL Image RGB
            img_np = np.array(img.convert("RGB"))
            out_np = self._apply_clahe_lab(img_np)
            data["image"] = Image.fromarray(out_np)

        elif isinstance(img, np.ndarray):
            # Numpy array
            if img.dtype != np.uint8:
                img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)
                out_u8 = self._apply_clahe_lab(img_u8)
                data["image"] = out_u8.astype(img.dtype) / 255.0
            else:
                data["image"] = self._apply_clahe_lab(img)

        elif isinstance(img, torch.Tensor):
            # Tensor [C, H, W]
            device = img.device
            dtype = img.dtype
            img_cpu = img.detach().cpu().numpy()
            if img_cpu.ndim == 3 and img_cpu.shape[0] == 3:
                img_np = np.transpose(img_cpu, (1, 2, 0))
                is_float = img.dtype in [torch.float32, torch.float64, torch.float16]
                if is_float:
                    img_u8 = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img_u8 = img_np.astype(np.uint8)

                out_u8 = self._apply_clahe_lab(img_u8)

                if is_float:
                    out_tensor = torch.from_numpy(out_u8.astype(np.float32) / 255.0).to(device=device, dtype=dtype)
                else:
                    out_tensor = torch.from_numpy(out_u8).to(device=device, dtype=dtype)
                data["image"] = out_tensor.permute(2, 0, 1)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(clip_limit={self.clip_limit}, tile_grid_size={self.tile_grid_size}, p={self.p})"


@TRANSFORMS.register_module(name="NormalizeBoxes")
class NormalizeBoxes:
    def __call__(self, data: Dict[str, Any], image_size: int = 320) -> Dict[str, Any]:
        if "target" in data and "boxes" in data['target']:
            data["target"]['boxes'] /= image_size
        return data
