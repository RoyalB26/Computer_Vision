
import os
import sys

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from BaseModel.core.registrar import TRANSFORMS


@TRANSFORMS.register_module("Action_Slot_Transform")
class Action_Slot_Transform:
    """Chuẩn hóa video clip và background mask; hỗ trợ truyền uint8 lên GPU để giảm 4x băng thông."""

    def __init__(
        self,
        video_size: Tuple[int, int] = (256, 768),
        mask_size: Tuple[int, int] = (32, 96),
        normalize_on_gpu: bool = True,
    ):
        self.video_size = tuple(video_size)
        self.mask_size = tuple(mask_size)
        self.normalize_on_gpu = normalize_on_gpu

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        inputs = data.get("inputs")
        if isinstance(inputs, torch.Tensor):
            if inputs.shape[-2:] != self.video_size:
                import torchvision.transforms.functional as TF
                from torchvision.transforms import InterpolationMode
                inputs = TF.resize(inputs, list(self.video_size), interpolation=InterpolationMode.BILINEAR, antialias=False)
            if not self.normalize_on_gpu and inputs.dtype == torch.uint8:
                inputs = inputs.float().mul_(1.0 / (255.0 * 0.225)).add_(-0.45 / 0.225)
            data["inputs"] = inputs
        elif inputs:
            data["inputs"] = self.process_videos(inputs)

        targets = data.get("targets", {})
        bg_seg = targets.get("bg_seg")
        if isinstance(bg_seg, torch.Tensor):
            if bg_seg.shape[-2:] != self.mask_size:
                import torchvision.transforms.functional as TF
                from torchvision.transforms import InterpolationMode
                bg_seg = TF.resize(bg_seg, list(self.mask_size), interpolation=InterpolationMode.NEAREST)
            if bg_seg.dtype == torch.uint8:
                bg_seg = bg_seg.float().div_(255.0)
            targets["bg_seg"] = bg_seg
        elif bg_seg:
            targets["bg_seg"] = self.process_masks(bg_seg)

        return data

    def process_videos(self, data: List[Any]) -> torch.Tensor:
        """Fallback path cho danh sách PIL.Image/ndarray: resize ở uint8 trước khi chuẩn hóa."""
        from PIL import Image
        import torchvision.transforms.functional as TF

        target_h, target_w = self.video_size
        arr = np.empty((len(data), target_h, target_w, 3), dtype=np.uint8)
        for i, img in enumerate(data):
            if isinstance(img, Image.Image):
                if img.size != (target_w, target_h):
                    img = img.resize((target_w, target_h), Image.BILINEAR)
                arr[i] = np.asarray(img, dtype=np.uint8)
            else:
                np_img = img.cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
                if np_img.shape[:2] != (target_h, target_w):
                    im_pil = Image.fromarray(np_img.astype(np.uint8, copy=False)).resize((target_w, target_h), Image.BILINEAR)
                    arr[i] = np.asarray(im_pil, dtype=np.uint8)
                else:
                    arr[i] = np_img

        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        if not self.normalize_on_gpu:
            tensor = tensor.float().mul_(1.0 / (255.0 * 0.225)).add_(-0.45 / 0.225)
        return tensor

    def process_masks(self, data: List[Any]) -> torch.Tensor:
        """Fallback path cho danh sách background mask (T_mask, 1, 32, 96)."""
        from PIL import Image

        target_h, target_w = self.mask_size
        arr = np.empty((len(data), 1, target_h, target_w), dtype=np.uint8)
        for i, m in enumerate(data):
            if isinstance(m, Image.Image):
                if m.size != (target_w, target_h):
                    m = m.resize((target_w, target_h), Image.NEAREST)
                arr[i, 0] = np.asarray(m, dtype=np.uint8)
            else:
                np_m = np.squeeze(m.cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m))
                if np_m.shape[:2] != (target_h, target_w):
                    m_pil = Image.fromarray(np_m.astype(np.uint8, copy=False)).resize((target_w, target_h), Image.NEAREST)
                    arr[i, 0] = np.asarray(m_pil, dtype=np.uint8)
                else:
                    arr[i, 0] = np_m

        return torch.from_numpy(arr).float().div_(255.0)


    