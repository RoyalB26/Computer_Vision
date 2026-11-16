
import os
import sys

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import random
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from BaseModel.core.registrar import TRANSFORMS

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False


@TRANSFORMS.register_module("Action_Slot_Transform")
class Action_Slot_Transform:
    """Chuẩn hóa video clip, background mask và photometric augmentation đồng nhất theo clip."""

    def __init__(
        self,
        video_size: Tuple[int, int] = (256, 768),
        mask_size: Tuple[int, int] = (32, 96),
        normalize_on_gpu: bool = True,
        photometric_aug: bool = False,
        aug_prob: float = 0.8,
        brightness: float = 0.25,
        contrast: float = 0.25,
        saturation: float = 0.25,
        gamma: float = 0.15,
        channel_jitter: float = 0.08,
        grayscale_prob: float = 0.1,
        blur_prob: float = 0.15,
    ):
        self.video_size = tuple(video_size)
        self.mask_size = tuple(mask_size)
        self.normalize_on_gpu = normalize_on_gpu
        self.photometric_aug = photometric_aug
        self.aug_prob = float(aug_prob)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.gamma = float(gamma)
        self.channel_jitter = float(channel_jitter)
        self.grayscale_prob = float(grayscale_prob)
        self.blur_prob = float(blur_prob)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        inputs = data.get("inputs")
        if isinstance(inputs, torch.Tensor):
            if inputs.shape[-2:] != self.video_size:
                import torchvision.transforms.functional as TF
                from torchvision.transforms import InterpolationMode
                inputs = TF.resize(inputs, list(self.video_size), interpolation=InterpolationMode.BILINEAR, antialias=False)
        elif inputs:
            inputs = self.process_videos(inputs)

        if isinstance(inputs, torch.Tensor):
            if self.photometric_aug and inputs.dtype == torch.uint8:
                inputs = self._apply_photometric_aug(inputs)
            if not self.normalize_on_gpu and inputs.dtype == torch.uint8:
                inputs = inputs.float().mul_(1.0 / (255.0 * 0.225)).add_(-0.45 / 0.225)
            data["inputs"] = inputs

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

    def _apply_photometric_aug(self, video: torch.Tensor) -> torch.Tensor:
        """Áp dụng biến đổi màu sắc/ánh sáng đồng nhất trên toàn bộ T frame (bảo toàn hướng trái/phải)."""
        if video.ndim != 4 or video.shape[1] != 3:
            return video

        arr = video.numpy().copy()
        t_len = arr.shape[0]

        # 1. Gamma + Contrast + Brightness + RGB channel gain gộp vào 1 LUT uint8/kênh (~12ms)
        if random.random() < self.aug_prob:
            gamma_val = random.uniform(max(0.5, 1.0 - self.gamma), 1.0 + self.gamma) if self.gamma > 0 else 1.0
            contrast_val = random.uniform(max(0.3, 1.0 - self.contrast), 1.0 + self.contrast) if self.contrast > 0 else 1.0
            brightness_shift = random.uniform(-self.brightness, self.brightness) * 255.0 if self.brightness > 0 else 0.0

            base = np.arange(256, dtype=np.float32) / 255.0
            if gamma_val != 1.0:
                base = np.power(base, gamma_val)
            base = base * 255.0
            if contrast_val != 1.0:
                base = (base - 127.5) * contrast_val + 127.5
            base = base + brightness_shift

            for c in range(3):
                gain = random.uniform(max(0.5, 1.0 - self.channel_jitter), 1.0 + self.channel_jitter) if self.channel_jitter > 0 else 1.0
                lut = np.clip(base * gain, 0.0, 255.0).astype(np.uint8)
                if _HAS_CV2:
                    for t in range(t_len):
                        cv2.LUT(arr[t, c], lut, dst=arr[t, c])
                else:
                    arr[:, c] = lut[arr[:, c]]

        # 2. Random Grayscale hoặc Saturation jitter dùng trọng số luma cố định (không cần chuyển HSV)
        is_gray = random.random() < self.grayscale_prob
        do_sat = is_gray or (self.saturation > 0 and random.random() < 0.5)
        if do_sat:
            sat = 0.0 if is_gray else random.uniform(max(0.2, 1.0 - self.saturation), 1.0 + self.saturation)
            luma = (
                0.299 * arr[:, 0:1].astype(np.float32)
                + 0.587 * arr[:, 1:2].astype(np.float32)
                + 0.114 * arr[:, 2:3].astype(np.float32)
            )
            blended = luma + sat * (arr.astype(np.float32) - luma)
            arr = np.clip(blended, 0.0, 255.0).astype(np.uint8)

        # 3. Gaussian Blur nhẹ đồng nhất theo clip để giảm học thuộc chi tiết sắc cạnh
        if self.blur_prob > 0 and random.random() < self.blur_prob:
            sigma = random.uniform(0.2, 1.0)
            if _HAS_CV2:
                for t in range(t_len):
                    for c in range(3):
                        cv2.GaussianBlur(arr[t, c], (3, 3), sigma, dst=arr[t, c])
            else:
                import torchvision.transforms.functional as TF
                return TF.gaussian_blur(torch.from_numpy(arr), [3, 3], [sigma, sigma]).contiguous()

        return torch.from_numpy(arr).contiguous()

    def process_videos(self, data: List[Any]) -> torch.Tensor:
        """Fallback path cho danh sách PIL.Image/ndarray: trả về uint8 Tensor (T, 3, H, W)."""
        from PIL import Image

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

        return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()

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


    