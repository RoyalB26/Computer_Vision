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
