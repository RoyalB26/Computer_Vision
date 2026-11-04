"""Transforms and Data Augmentation Pipeline."""
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from ...core.registrar import TRANSFORMS
from ...core.registry import build_from_cfg


@TRANSFORMS.register_module()
class Compose:
    """Compose multiple transforms together into a single pipeline.

    Args:
        transforms: List of transforms or transform configs.
    """

    def __init__(self, transforms: Optional[List[Union[Callable, Dict[str, Any]]]] = None):
        self.transforms: List[Callable] = []
        if transforms:
            for t in transforms:
                if isinstance(t, dict):
                    self.transforms.append(build_from_cfg(t, TRANSFORMS))
                elif callable(t):
                    self.transforms.append(t)
                else:
                    raise TypeError(f"Transform must be callable or dict, got {type(t)}")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transforms sequentially to the data dict."""
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


@TRANSFORMS.register_module()
class ToTensor:
    """Convert image in data dict (numpy array or PIL) to torch.Tensor with shape (C, H, W).
    
    If 'mask' is present, converts it to torch.LongTensor with shape (H, W).
    """

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "image" in data:
            img = data["image"]
            if isinstance(img, np.ndarray):
                if img.ndim == 2:
                    img = img[:, :, None]
                img = img.transpose((2, 0, 1))
                tensor = torch.from_numpy(img).float()
                if tensor.max() > 1.0 and tensor.dtype == torch.float32:
                    tensor = tensor / 255.0
                data["image"] = tensor
            elif hasattr(img, "convert"):  # PIL Image
                import torchvision.transforms.functional as TF
                data["image"] = TF.to_tensor(img)

        # Handle segmentation mask if present
        if "mask" in data:
            mask = data["mask"]
            if hasattr(mask, "convert") or isinstance(mask, np.ndarray):
                data["mask"] = torch.from_numpy(np.array(mask)).long()

        return data


@TRANSFORMS.register_module()
class Normalize:
    """Normalize image tensor with mean and standard deviation."""

    def __init__(self, mean: List[float] = (0.485, 0.456, 0.406), std: List[float] = (0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "image" in data and isinstance(data["image"], torch.Tensor):
            data["image"] = (data["image"] - self.mean) / self.std
        return data


@TRANSFORMS.register_module()
class Resize:
    """Resize image tensor or array to target size (height, width).
    
    If 'mask' is present, resizes it using Nearest Neighbor interpolation to preserve discrete class labels.
    """

    def __init__(self, size: Union[int, tuple]):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode

        if "image" in data:
            img = data["image"]
            if isinstance(img, torch.Tensor) or hasattr(img, "resize"):
                data["image"] = TF.resize(img, self.size, interpolation=InterpolationMode.BILINEAR)

        if "mask" in data:
            mask = data["mask"]
            if isinstance(mask, torch.Tensor) or hasattr(mask, "resize"):
                data["mask"] = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        return data



@TRANSFORMS.register_module(name=["CLAHE", "CLAHE_LAB"], force=True)
class CLAHE:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel in LAB color space."""

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Union[tuple, list] = (8, 8),
        p: float = 1.0,
    ):
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(tile_grid_size)
        self.p = float(p)
        self._clahe = None

    @property
    def clahe(self):
        if self._clahe is None:
            import cv2
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size,
            )
        return self._clahe

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_clahe"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._clahe = None

    def _apply_clahe_lab(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        import cv2
        lab = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_clahe = self.clahe.apply(l_channel)
        lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        import random
        from PIL import Image

        if "image" not in data or self.p <= 0.0:
            return data

        if self.p < 1.0 and random.random() > self.p:
            return data

        img = data["image"]

        if isinstance(img, Image.Image):
            img_np = np.array(img.convert("RGB"))
            out_np = self._apply_clahe_lab(img_np)
            data["image"] = Image.fromarray(out_np)
        elif isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)
                out_u8 = self._apply_clahe_lab(img_u8)
                data["image"] = out_u8.astype(img.dtype) / 255.0
            else:
                data["image"] = self._apply_clahe_lab(img)
        elif isinstance(img, torch.Tensor):
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
