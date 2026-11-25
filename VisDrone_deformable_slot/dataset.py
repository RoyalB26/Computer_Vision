"""High-Speed Dual-Resolution LMDB Dataset for VisDroneYOLO11DeformableSlot.

Optimized specifically for CPU-bound environments (e.g. Kaggle 2xT4 with 4 vCPUs):
  1. Uses OpenCV C++ SIMD (`cv2.imdecode` + `cv2.resize`) instead of PIL (3-4x faster).
  2. Performs Mosaic4 quadrant assembly and RandomScale/Crop entirely in `uint8` HWC
     before a single `uint8 -> float32` conversion (5x less CPU memory bandwidth).
  3. Skips redundant CPU `F.interpolate` RandomScale on already-Mosaic'd samples and
     uses fast in-place photometric jitter.
  4. Zero RAM duplication under Python 3.13 `forkserver` DDP when `cache_images: null`.
"""
from __future__ import annotations

import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from BaseModel.core import DATASETS
from VisDrone_yolov11.dataset import (
    VISDRONE_CLASSES,
    CloseMosaicHook,
    VisDroneLMDBDataset,
    WarmupLRHook,
)


@DATASETS.register_module(name="VisDroneDualResLMDBDataset", force=True)
class VisDroneDualResLMDBDataset(VisDroneLMDBDataset):
    """Fast OpenCV-accelerated Dual-Resolution LMDB Dataset with Drone Photometric & Blur Augment."""

    def __init__(
        self,
        *args,
        drone_aug: bool = True,
        hsv_prob: float = 0.7,
        blur_prob: float = 0.15,
        haze_gamma_prob: float = 0.25,
        noise_prob: float = 0.10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.drone_aug = bool(drone_aug)
        self.hsv_prob = float(hsv_prob)
        self.blur_prob = float(blur_prob)
        self.haze_gamma_prob = float(haze_gamma_prob)
        self.noise_prob = float(noise_prob)
        self._lut_x = np.arange(256, dtype=np.float32)

    def _apply_drone_photometric_blur(self, rgb: np.ndarray) -> np.ndarray:
        """Fast uint8 OpenCV-SIMD Drone Photometric & Blur Augmentation (~2-3ms at 1280x1280).

        Simulates aerial drone conditions without harming tiny objects:
          1. HSV color/white-balance/exposure jitter via 256-entry cv2.LUT.
          2. Mild directional drone motion blur (3x3 / 5x5) or optical defocus blur (3x3).
          3. Dusk/glare gamma curve (cv2.LUT) or mild high-altitude atmospheric haze.
          4. Mild high-ISO night sensor noise.
        """
        # 1. HSV Jitter (LUT-accelerated, zero float32 full-image allocation)
        if self.hsv_prob > 0.0 and random.random() < self.hsv_prob:
            r = np.random.uniform(-1.0, 1.0, 3) * np.array([0.015, 0.50, 0.35], dtype=np.float32) + 1.0
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            x = self._lut_x
            lut_h = ((x * r[0]) % 180.0).astype(np.uint8)
            lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
            lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
            hsv[..., 0] = cv2.LUT(hsv[..., 0], lut_h)
            hsv[..., 1] = cv2.LUT(hsv[..., 1], lut_s)
            hsv[..., 2] = cv2.LUT(hsv[..., 2], lut_v)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # 2. Dusk/Glare Gamma OR Mild High-Altitude Haze (0.3ms via LUT / addWeighted)
        if self.haze_gamma_prob > 0.0 and random.random() < self.haze_gamma_prob:
            if random.random() < 0.6:
                gamma = random.uniform(0.75, 1.35)
                lut_g = np.clip(((self._lut_x / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
                rgb = cv2.LUT(rgb, lut_g)
            else:
                alpha = random.uniform(0.05, 0.18)
                fog_val = random.randint(205, 238)
                # Blend toward bright atmospheric haze: (1 - alpha) * rgb + alpha * fog_val
                rgb = cv2.convertScaleAbs(rgb, alpha=1.0 - alpha, beta=alpha * fog_val)

        # 3. Mild Drone Motion Blur (gimbal pan/vibration) OR Optical Defocus Blur
        if self.blur_prob > 0.0 and random.random() < self.blur_prob:
            if random.random() < 0.6:
                k = random.choice((3, 5))
                kernel = np.zeros((k, k), dtype=np.float32)
                mode = random.randint(0, 3)
                if mode == 0:  # horizontal motion
                    kernel[k // 2, :] = 1.0 / k
                elif mode == 1:  # vertical motion
                    kernel[:, k // 2] = 1.0 / k
                elif mode == 2:  # main diagonal
                    np.fill_diagonal(kernel, 1.0 / k)
                else:  # anti-diagonal
                    np.fill_diagonal(np.fliplr(kernel), 1.0 / k)
                rgb = cv2.filter2D(rgb, -1, kernel)
            else:
                sigma = random.uniform(0.4, 1.0)
                rgb = cv2.GaussianBlur(rgb, (3, 3), sigmaX=sigma)

        # 4. Mild High-ISO Sensor Noise
        if self.noise_prob > 0.0 and random.random() < self.noise_prob:
            sigma = random.uniform(2.0, 6.0)
            noise = np.empty_like(rgb, dtype=np.int8)
            cv2.randn(noise, 0, sigma)
            rgb = np.clip(rgb.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)

        return rgb

    def _read_uint8_rgb_and_target(self, idx: int) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Read resized (H, W, 3) uint8 RGB numpy array + target dict using fast OpenCV SIMD."""
        h, w = self.cache_size

        if self._ram_images is not None and self._ram_targets is not None:
            rgb = self._ram_images[idx].permute(1, 2, 0).numpy().copy()
            t_np = self._ram_targets[idx]
            target: Dict[str, Any] = {"_is_cxcywh": t_np.get("_is_cxcywh", True)}
            for k in ("counts", "boxes", "labels"):
                if k in t_np:
                    v = t_np[k]
                    target[k] = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            return rgb, target

        if self._jpeg_cache is not None:
            img_bytes, t_np = self._jpeg_cache[idx]
            target = {"_is_cxcywh": t_np.get("_is_cxcywh", True)}
            for k in ("counts", "boxes", "labels"):
                if k in t_np:
                    v = t_np[k]
                    target[k] = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        else:
            self._init_db()
            real_idx = self.data_infos[idx]["lmdb_idx"]
            key = f"{real_idx:08d}".encode("ascii")
            byteflow = self.txn.get(key)
            if byteflow is None:
                return None, {}
            data = pickle.loads(byteflow)
            img_bytes = data["image"]
            target = self._build_target(data, key)

        buf = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return None, {}
        if bgr.shape[0] != h or bgr.shape[1] != w:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, target

    def _apply_mosaic4_uint8(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """4-image Class-Aware Object-Centered Mosaic assembled directly in uint8 HWC."""
        n_total = len(self.data_infos)
        quad_rare_cls: List[Optional[int]] = [None, None, None, None]
        if self.mosaic_rare_prob > 0.0 and self._rare_class_ids and random.random() < self.mosaic_rare_prob:
            num_rare_inject = 2 if (random.random() < 0.5) else 1
            chosen_rc = random.choices(self._rare_class_ids, weights=self._rare_class_weights, k=num_rare_inject)
            extra_pairs: List[Tuple[int, Optional[int]]] = [
                (random.choice(self._rare_class_pools[rc]), rc) for rc in chosen_rc
            ]
            while len(extra_pairs) < 3:
                extra_pairs.append((random.randint(0, n_total - 1), None))
            random.shuffle(extra_pairs)
            indices = [idx] + [p[0] for p in extra_pairs]
            quad_rare_cls = [None] + [p[1] for p in extra_pairs]
        else:
            indices = [idx] + [random.randint(0, n_total - 1) for _ in range(3)]

        h, w = self.cache_size
        yc = random.randint(int(0.25 * h), int(0.75 * h))
        xc = random.randint(int(0.25 * w), int(0.75 * w))

        canvas = np.full((h, w, 3), 114, dtype=np.uint8)
        all_boxes: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        quads = [
            (h - yc, h, w - xc, w, 0, yc, 0, xc),
            (h - yc, h, 0, w - xc, 0, yc, xc, w),
            (0, h - yc, w - xc, w, yc, h, 0, xc),
            (0, h - yc, 0, w - xc, yc, h, xc, w),
        ]

        for q_idx, (sy1, sy2, sx1, sx2, dy1, dy2, dx1, dx2), target_rc in zip(indices, quads, quad_rare_cls):
            rgb_q, t_q = self._read_uint8_rgb_and_target(q_idx)
            if rgb_q is None:
                continue

            boxes_q = t_q.get("boxes", torch.empty((0, 4), dtype=torch.float32))
            labels_q = t_q.get("labels", torch.empty((0,), dtype=torch.int64))

            if target_rc is not None and len(boxes_q) > 0:
                rc_mask = labels_q == target_rc
                if rc_mask.any():
                    rc_indices = torch.nonzero(rc_mask, as_tuple=False).squeeze(-1)
                    pick = int(rc_indices[random.randint(0, len(rc_indices) - 1)].item())
                    rc_cx = float(boxes_q[pick, 0].item()) * w
                    rc_cy = float(boxes_q[pick, 1].item()) * h
                    qh, qw = dy2 - dy1, dx2 - dx1
                    jx = random.uniform(-0.25, 0.25) * qw
                    jy = random.uniform(-0.25, 0.25) * qh
                    sx1 = max(0, min(w - qw, int(round(rc_cx - 0.5 * qw + jx))))
                    sy1 = max(0, min(h - qh, int(round(rc_cy - 0.5 * qh + jy))))
                    sx2 = sx1 + qw
                    sy2 = sy1 + qh

            canvas[dy1:dy2, dx1:dx2] = rgb_q[sy1:sy2, sx1:sx2]

            if len(boxes_q) == 0:
                continue

            dx = dx1 - sx1
            dy = dy1 - sy1
            cx = boxes_q[:, 0] * w
            cy = boxes_q[:, 1] * h
            bw = boxes_q[:, 2] * w
            bh = boxes_q[:, 3] * h
            orig_area = (bw * bh).clamp(min=1.0)

            bx1 = (cx - 0.5 * bw + dx).clamp(float(dx1), float(dx2))
            by1 = (cy - 0.5 * bh + dy).clamp(float(dy1), float(dy2))
            bx2 = (cx + 0.5 * bw + dx).clamp(float(dx1), float(dx2))
            by2 = (cy + 0.5 * bh + dy).clamp(float(dy1), float(dy2))

            nbw = bx2 - bx1
            nbh = by2 - by1
            keep = (nbw >= 2.5) & (nbh >= 2.5) & (((nbw * nbh) / orig_area) >= 0.15)
            if keep.any():
                ncx = ((bx1 + bx2) * 0.5) / float(w)
                ncy = ((by1 + by2) * 0.5) / float(h)
                all_boxes.append(torch.stack([ncx, ncy, nbw / float(w), nbh / float(h)], dim=-1)[keep])
                all_labels.append(labels_q[keep])

        if all_boxes:
            boxes_cat = torch.cat(all_boxes, dim=0)
            labels_cat = torch.cat(all_labels, dim=0)
        else:
            boxes_cat = torch.empty((0, 4), dtype=torch.float32)
            labels_cat = torch.empty((0,), dtype=torch.int64)

        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        if len(labels_cat) > 0:
            valid = labels_cat[(labels_cat >= 0) & (labels_cat < self.num_classes)]
            if len(valid) > 0:
                counts = torch.bincount(valid, minlength=self.num_classes).float()

        return canvas, {
            "_is_cxcywh": True,
            "counts": counts,
            "boxes": boxes_cat,
            "labels": labels_cat,
        }

    def _fast_augment_uint8(
        self,
        rgb: np.ndarray,
        target: Dict[str, Any],
        do_scale_crop: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Fast uint8 OpenCV geometric augment + single float32 conversion + in-place photometric jitter."""
        h, w = rgb.shape[:2]
        boxes = target.get("boxes", torch.empty((0, 4), dtype=torch.float32)).clone()
        labels = target.get("labels", torch.empty((0,), dtype=torch.int64)).clone()

        # 1. Random Horizontal Flip (p=0.5) in uint8
        if random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            if len(boxes) > 0:
                boxes[:, 0] = (1.0 - boxes[:, 0]).clamp(0.0, 1.0)

        # 2. Random Scale + Crop/Pad (p=0.5) in uint8 via OpenCV (5x faster than CPU F.interpolate!)
        if do_scale_crop and random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            nh, nw = max(32, int(round(h * scale))), max(32, int(round(w * scale)))
            rgb_s = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

            if scale >= 1.0:
                y0 = random.randint(0, nh - h)
                x0 = random.randint(0, nw - w)
                rgb = np.ascontiguousarray(rgb_s[y0 : y0 + h, x0 : x0 + w])
                if len(boxes) > 0:
                    cx = (boxes[:, 0] * nw - x0) / float(w)
                    cy = (boxes[:, 1] * nh - y0) / float(h)
                    bw = boxes[:, 2] * (nw / float(w))
                    bh = boxes[:, 3] * (nh / float(h))
                    x1 = (cx - 0.5 * bw).clamp(0.0, 1.0)
                    y1 = (cy - 0.5 * bh).clamp(0.0, 1.0)
                    x2 = (cx + 0.5 * bw).clamp(0.0, 1.0)
                    y2 = (cy + 0.5 * bh).clamp(0.0, 1.0)
                    nbw, nbh = x2 - x1, y2 - y1
                    keep = (nbw > (2.0 / w)) & (nbh > (2.0 / h))
                    boxes = torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, nbw, nbh], dim=-1)[keep]
                    labels = labels[keep]
            else:
                canvas = np.full((h, w, 3), 114, dtype=np.uint8)
                y0 = random.randint(0, h - nh)
                x0 = random.randint(0, w - nw)
                canvas[y0 : y0 + nh, x0 : x0 + nw] = rgb_s
                rgb = canvas
                if len(boxes) > 0:
                    cx = (boxes[:, 0] * nw + x0) / float(w)
                    cy = (boxes[:, 1] * nh + y0) / float(h)
                    bw = boxes[:, 2] * (nw / float(w))
                    bh = boxes[:, 3] * (nh / float(h))
                    boxes = torch.stack([cx, cy, bw, bh], dim=-1)

        # 3. Drone Photometric & Blur Augmentation in uint8 (OpenCV SIMD) + single float32 conversion
        if self.drone_aug:
            rgb = self._apply_drone_photometric_blur(rgb)

        img = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)
        if not self.drone_aug and random.random() < 0.8:
            alpha = random.uniform(0.8, 1.2)  # fallback contrast/brightness scale
            beta = random.uniform(-0.08, 0.08)
            img.mul_(alpha).add_(beta).clamp_(0.0, 1.0)

        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        if len(labels) > 0:
            valid = labels[(labels >= 0) & (labels < self.num_classes)]
            if len(valid) > 0:
                counts = torch.bincount(valid, minlength=self.num_classes).float()

        return img, {
            "_is_cxcywh": True,
            "counts": counts,
            "boxes": boxes,
            "labels": labels,
        }

    def prepare_data(self, idx: int) -> Optional[Dict[str, Any]]:
        """Fast OpenCV + uint8 pipeline (3-6x faster on CPU-bound Kaggle workers)."""
        if self.augment and self.mosaic_prob > 0.0 and len(self.data_infos) >= 4 and random.random() < self.mosaic_prob:
            rgb, target = self._apply_mosaic4_uint8(idx)
            # Skip redundant CPU RandomScale on already-Mosaic'd image (cuts ~35ms/sample!)
            img, target = self._fast_augment_uint8(rgb, target, do_scale_crop=False)
            return {"image": img, "target": target}

        rgb, target = self._read_uint8_rgb_and_target(idx)
        if rgb is None:
            return None
        if self.augment:
            img, target = self._fast_augment_uint8(rgb, target, do_scale_crop=True)
        else:
            img = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)
        return {"image": img, "target": target}


__all__ = [
    "VISDRONE_CLASSES",
    "VisDroneDualResLMDBDataset",
    "WarmupLRHook",
    "CloseMosaicHook",
]
