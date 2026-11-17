
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
import torch.nn as nn
import torch.nn.functional as F
from BaseModel.core.registrar import TRANSFORMS

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False


class GPUBatchPhotometricAug(nn.Module):
    """Vectorized clip-consistent photometric augmentation running on GPU batches `(B, T, 3, H, W)` in `[0, 1]`."""

    def __init__(
        self,
        aug_prob: float = 0.8,
        brightness: float = 0.25,
        contrast: float = 0.25,
        saturation: float = 0.25,
        gamma: float = 0.15,
        channel_jitter: float = 0.08,
        grayscale_prob: float = 0.1,
        blur_prob: float = 0.15,
    ):
        super().__init__()
        self.aug_prob = float(aug_prob)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.gamma = float(gamma)
        self.channel_jitter = float(channel_jitter)
        self.grayscale_prob = float(grayscale_prob)
        self.blur_prob = float(blur_prob)
        # Precompute 3x3 grid offsets for vectorized Gaussian blur kernels
        coords = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("_blur_dist2", (gx * gx + gy * gy).unsqueeze(0), persistent=False)

    @torch.no_grad()
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """Apply clip-consistent photometric augmentations to `(B, T, 3, H, W)` float tensor in `[0, 1]`."""
        if not self.training or videos.ndim != 5 or videos.shape[2] != 3:
            return videos

        b, t_len, _, h, w = videos.shape
        device = videos.device
        dtype = videos.dtype
        out = videos

        # 1. Gamma + Contrast + Brightness + RGB channel gain (shared across all T frames of each clip)
        if self.aug_prob > 0.0:
            apply_main = torch.rand((b, 1, 1, 1, 1), device=device) < self.aug_prob
            if apply_main.any():
                if self.gamma > 0.0:
                    g_lo = max(0.5, 1.0 - self.gamma)
                    g_hi = 1.0 + self.gamma
                    gamma_val = torch.empty((b, 1, 1, 1, 1), device=device, dtype=dtype).uniform_(g_lo, g_hi)
                    gamma_val = torch.where(apply_main, gamma_val, torch.ones_like(gamma_val))
                    out = out.clamp(0.0, 1.0).pow(gamma_val)

                if self.contrast > 0.0:
                    c_lo = max(0.3, 1.0 - self.contrast)
                    c_hi = 1.0 + self.contrast
                    contrast_val = torch.empty((b, 1, 1, 1, 1), device=device, dtype=dtype).uniform_(c_lo, c_hi)
                    contrast_val = torch.where(apply_main, contrast_val, torch.ones_like(contrast_val))
                    out = (out - 0.5) * contrast_val + 0.5

                if self.brightness > 0.0:
                    shift = torch.empty((b, 1, 1, 1, 1), device=device, dtype=dtype).uniform_(
                        -self.brightness, self.brightness
                    )
                    shift = torch.where(apply_main, shift, torch.zeros_like(shift))
                    out = out + shift

                if self.channel_jitter > 0.0:
                    j_lo = max(0.5, 1.0 - self.channel_jitter)
                    j_hi = 1.0 + self.channel_jitter
                    gain = torch.empty((b, 1, 3, 1, 1), device=device, dtype=dtype).uniform_(j_lo, j_hi)
                    gain = torch.where(apply_main, gain, torch.ones_like(gain))
                    out = out * gain

                out = out.clamp_(0.0, 1.0)

        # 2. Random Grayscale or Saturation Jitter
        if self.grayscale_prob > 0.0 or self.saturation > 0.0:
            is_gray = torch.rand((b, 1, 1, 1, 1), device=device) < self.grayscale_prob
            do_sat = is_gray | (
                (torch.rand((b, 1, 1, 1, 1), device=device) < 0.5)
                if self.saturation > 0.0
                else torch.zeros_like(is_gray)
            )
            if do_sat.any():
                s_lo = max(0.2, 1.0 - self.saturation)
                s_hi = 1.0 + self.saturation
                sat = torch.empty((b, 1, 1, 1, 1), device=device, dtype=dtype).uniform_(s_lo, s_hi)
                sat = torch.where(is_gray, torch.zeros_like(sat), sat)
                sat = torch.where(do_sat, sat, torch.ones_like(sat))
                luma = (
                    0.299 * out[:, :, 0:1]
                    + 0.587 * out[:, :, 1:2]
                    + 0.114 * out[:, :, 2:3]
                )
                out = (luma + sat * (out - luma)).clamp_(0.0, 1.0)

        # 3. Clip-consistent 3x3 Gaussian Blur on the selected subset of clips
        if self.blur_prob > 0.0:
            blur_mask = torch.rand((b,), device=device) < self.blur_prob
            if blur_mask.any():
                sub = out[blur_mask]  # (K, T, 3, H, W)
                k_cnt = sub.shape[0]
                sigmas = torch.empty((k_cnt, 1, 1), device=device, dtype=torch.float32).uniform_(0.2, 1.0)
                dist2 = self._blur_dist2.to(device=device, dtype=torch.float32)
                kernels = torch.exp(-dist2 / (2.0 * sigmas * sigmas))
                kernels = (kernels / kernels.sum(dim=(-2, -1), keepdim=True)).to(dtype=dtype)
                # Expand kernel identically across RGB channels: (K*3, 1, 3, 3)
                kernels = kernels.unsqueeze(1).expand(k_cnt, 3, 3, 3).reshape(k_cnt * 3, 1, 3, 3)

                # Reshape to (T, K*3, H, W) so depthwise conv blurs each frame independently
                sub_2d = sub.permute(1, 0, 2, 3, 4).reshape(t_len, k_cnt * 3, h, w)
                sub_pad = F.pad(sub_2d, (1, 1, 1, 1), mode="reflect")
                blurred = F.conv2d(sub_pad, kernels, groups=k_cnt * 3)
                blurred = blurred.reshape(t_len, k_cnt, 3, h, w).permute(1, 0, 2, 3, 4)
                out = out.clone()
                out[blur_mask] = blurred

        return out


@TRANSFORMS.register_module("Action_Slot_Transform")
class Action_Slot_Transform:
    """Chuẩn hóa video clip, background mask và photometric augmentation đồng nhất theo clip."""

    def __init__(
        self,
        video_size: Tuple[int, int] = (256, 768),
        mask_size: Tuple[int, int] = (32, 96),
        normalize_on_gpu: bool = True,
        photometric_aug: bool = False,
        aug_on_gpu: bool = True,
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
        self.aug_on_gpu = bool(aug_on_gpu)
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
            if self.photometric_aug and not self.aug_on_gpu and inputs.dtype == torch.uint8:
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


    