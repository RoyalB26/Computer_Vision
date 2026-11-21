"""Ablation Study: So sánh Neck PANet Có ReLU cuối (gốc) vs BỎ ReLU ở tầng cuối của Neck.
Đều sử dụng chung Backbone SENet (chuẩn).

Vấn đề khảo sát:
- Ở đầu ra của PANet Neck:
    out_p3 = td_conv_p3(...)   (ở FPN)
    out_p4 = bu_conv_p4(...)   (ở PAN)
    out_p5 = bu_conv_p5(...)   (ở PAN)
  Trong bản gốc, td_conv_p3, bu_conv_p4, bu_conv_p5 đều kết thúc bằng:
      nn.Conv2d -> nn.BatchNorm2d -> nn.ReLU(inplace=True)
  Điều này triệt tiêu toàn bộ giá trị âm (trục < 0) của feature map ngay trước khi:
      feat + PositionalEncoding -> Token -> SlotAttention (Dot-Product Q, K, V)

- Giả thuyết thực nghiệm:
  Bỏ ReLU ở 3 khối đầu ra cuối cùng của Neck (chỉ để Conv + BN tuyến tính):
      nn.Conv2d -> nn.BatchNorm2d
  1. Cho phép tokens giữ trọn vẹn cả pha âm và pha dương trên không gian C-chiều (Zero-centered embedding),
     rất có lợi cho dot-product attention và cosine similarity trong Slot Attention.
  2. Đo đạc trực tiếp tác động lên val_mae, slot_act_inact_ratio, class_cos_sep và 45 cặp lớp trên Final Slots.

Lệnh chạy:
    python Visdrone/ablation_neck_relu.py --epochs 40 --n-train 500 --n-val 100 --out-dir Visdrone/output_ablation_neck_relu
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import pickle
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from BaseModel.dataset import open_lmdb
from Visdrone.benchmark_blocks import BLOCK_REGISTRY, MiniModularBackbone
from Visdrone.dataset import VISDRONE_CLASSES
from Visdrone.inspect_scales import build_box_mask_at_res, compute_pca_rgb, draw_boxes_on_ax
from Visdrone.model import PositionEmbeddingSine, SlotAttention


# =============================================================================
# 1. BACKBONE SENET CHUẨN
# =============================================================================

class SELayer(nn.Module):
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        mid = max(8, ch // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class SENetBlock(nn.Module):
    """SENet Block chuẩn (He et al. / Hu et al.)."""
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.se = SELayer(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.se(self.bn2(self.conv2(out)))
        return F.relu(out + x, inplace=True)


class SENetBackbone(nn.Module):
    """Backbone SENet 3 tầng chuẩn (P3 //8, P4 //16, P5 //32)."""
    def __init__(self, stage_channels: Tuple[int, int, int] = (48, 96, 160), blocks_per_stage: int = 2):
        super().__init__()
        c3, c4, c5 = stage_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(32, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(*[SENetBlock(c3) for _ in range(blocks_per_stage)])

        self.down4 = nn.Sequential(
            nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(*[SENetBlock(c4) for _ in range(blocks_per_stage)])

        self.down5 = nn.Sequential(
            nn.Conv2d(c4, c5, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )
        self.stage5 = nn.Sequential(*[SENetBlock(c5) for _ in range(blocks_per_stage)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        p3 = self.stage3(self.down3(x))
        p4 = self.stage4(self.down4(p3))
        p5 = self.stage5(self.down5(p4))
        return p3, p4, p5


# =============================================================================
# 2. PANET NECK: 5 BIẾN THỂ ĐỐI ĐẦU (BAO GỒM HÀM KÍCH HOẠT 1 CHIỀU ÂM/DƯƠNG)
# =============================================================================

class OneDimSignedReLU(nn.Module):
    """Hàm kích hoạt cho phép ĐÚNG 1 chiều (kênh) bất kỳ mang giá trị âm hoặc dương (-inf, +inf),
    còn toàn bộ C - 1 chiều còn lại bị ép >= 0 qua ReLU.

    Không gian biểu diễn: R^1 x R_+^(C-1).
    """
    def __init__(self, num_channels: int, free_dim: Optional[int] = None):
        super().__init__()
        self.num_channels = num_channels
        if free_dim is None:
            # Chọn ngẫu nhiên 1 chiều bất kỳ trong [0, num_channels - 1]
            chosen_dim = int(torch.randint(0, num_channels, (1,)).item())
        else:
            chosen_dim = int(free_dim) % num_channels
        self.free_dim = chosen_dim

        # Mask (1, C, 1, 1): True tại đúng 1 chiều được tự do âm/dương, False tại C-1 chiều >= 0
        free_mask = torch.zeros(1, num_channels, 1, 1, dtype=torch.bool)
        free_mask[0, self.free_dim, 0, 0] = True
        self.register_buffer("free_mask", free_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(self.free_mask, x, F.relu(x))

    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}, free_signed_dim={self.free_dim}"


class PANetNeckAblation(nn.Module):
    """PANet Neck hỗ trợ 5 chế độ thí nghiệm:
    1. 'with_relu'         : Gốc (td_p3, bu_p4, bu_p5 đều qua Conv -> BN -> ReLU)
    2. 'no_last_relu'      : Bỏ ReLU ở cả 3 khối đầu ra (chỉ Conv -> BN)
    3. 'preserve_bu_relu'  : Nhánh xuất ra tokens thì BỎ ReLU (Conv->BN), nhưng nhánh truyền nội bộ
                             Bottom-Up (p3_down, p4_down) vẫn giữ ReLU(out) để tránh suy biến tích chập liên tiếp.
    4. 'silu'              : Thay ReLU cuối bằng SiLU (Smooth non-linear floor, ép giá trị âm sâu về 0 nhưng
                             cho phép đáy âm nhẹ [-2, 0] để mở rộng góc Cosine).
    5. 'one_dim_signed'    : Dùng OneDimSignedReLU — chỉ cho phép đúng 1 chiều bất kỳ mang giá trị âm hoặc dương,
                             còn C - 1 chiều còn lại đều >= 0 (qua ReLU).
    """
    def __init__(
        self,
        in_channels: Tuple[int, int, int] = (48, 96, 160),
        out_dim: int = 64,
        mode: str = "with_relu",
    ):
        super().__init__()
        c3_in, c4_in, c5_in = in_channels
        self.mode = mode

        # 1. Lateral projections
        self.lat_p3 = nn.Sequential(
            nn.Conv2d(c3_in, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.lat_p4 = nn.Sequential(
            nn.Conv2d(c4_in, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.lat_p5 = nn.Sequential(
            nn.Conv2d(c5_in, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        # 2. Top-Down smoothing convs (FPN)
        self.td_conv_p4 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        def make_head_block(m: str):
            if m in ("no_last_relu", "preserve_bu_relu"):
                return nn.Sequential(
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_dim),
                )
            elif m == "silu":
                return nn.Sequential(
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.SiLU(inplace=True),
                )
            elif m == "one_dim_signed":
                return nn.Sequential(
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    OneDimSignedReLU(num_channels=out_dim),
                )
            else:  # 'with_relu'
                return nn.Sequential(
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                )

        self.td_conv_p3 = make_head_block(mode)

        # 3. Bottom-Up downsampling & fusion convs (PAN)
        self.down_p3_to_p4 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.bu_conv_p4 = make_head_block(mode)

        self.down_p4_to_p5 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.bu_conv_p5 = make_head_block(mode)

    def forward(
        self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3_lat = self.lat_p3(c3)
        p4_lat = self.lat_p4(c4)
        p5_lat = self.lat_p5(c5)

        td_p5 = p5_lat
        td_p4 = self.td_conv_p4(p4_lat + F.interpolate(td_p5, size=p4_lat.shape[-2:], mode="nearest"))
        td_p3 = self.td_conv_p3(p3_lat + F.interpolate(td_p4, size=p3_lat.shape[-2:], mode="nearest"))

        out_p3 = td_p3

        # Nhánh truyền Bottom-Up:
        # Nếu preserve_bu_relu: áp ReLU cho feature map đi lên P4 để giữ phi tuyến giữa 2 lớp conv
        feed_p3 = F.relu(out_p3) if self.mode == "preserve_bu_relu" else out_p3
        p3_down = self.down_p3_to_p4(feed_p3)
        if p3_down.shape[-2:] != td_p4.shape[-2:]:
            p3_down = F.interpolate(p3_down, size=td_p4.shape[-2:], mode="bilinear", align_corners=False)
        out_p4 = self.bu_conv_p4(td_p4 + p3_down)

        feed_p4 = F.relu(out_p4) if self.mode == "preserve_bu_relu" else out_p4
        p4_down = self.down_p4_to_p5(feed_p4)
        if p4_down.shape[-2:] != td_p5.shape[-2:]:
            p4_down = F.interpolate(p4_down, size=td_p5.shape[-2:], mode="bilinear", align_corners=False)
        out_p5 = self.bu_conv_p5(td_p5 + p4_down)

        return out_p3, out_p4, out_p5


class MiniSlotCounterNeckAblation(nn.Module):
    """Model đếm dùng Backbone tùy chọn (mặc định SENet) + Neck PANet (chọn 1 trong 4 mode)."""
    def __init__(self, mode: str = "with_relu", backbone_type: str = "senet", num_classes: int = 10, dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.backbone_type = backbone_type
        stage_ch = (48, 96, 160)
        self.backbone = MiniModularBackbone(block_type=backbone_type, stage_channels=stage_ch, blocks_per_stage=2)
        self.neck = PANetNeckAblation(in_channels=stage_ch, out_dim=dim, mode=mode)
        self.pos_emb = PositionEmbeddingSine(num_pos_feats=dim // 2)
        self.level_embed = nn.Parameter(torch.randn(3, dim) * 0.02)
        self.slot_attention = SlotAttention(dim=dim, num_class=num_classes)
        self.fc_actor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim + 3, dim),
                nn.ReLU(inplace=False),
                nn.Linear(dim, 1),
                nn.ReLU(),
            )
            for _ in range(num_classes)
        ])

    def forward_all(self, x: torch.Tensor):
        c3, c4, c5 = self.backbone(x)
        feat_p3, feat_p4, feat_p5 = self.neck(c3, c4, c5)

        p3_pos = feat_p3 + self.pos_emb(feat_p3)
        p4_pos = feat_p4 + self.pos_emb(feat_p4)
        p5_pos = feat_p5 + self.pos_emb(feat_p5)

        resolutions = (
            (feat_p3.shape[2], feat_p3.shape[3]),
            (feat_p4.shape[2], feat_p4.shape[3]),
            (feat_p5.shape[2], feat_p5.shape[3]),
        )
        t3 = p3_pos.flatten(2).transpose(1, 2) + self.level_embed[0]
        t4 = p4_pos.flatten(2).transpose(1, 2) + self.level_embed[1]
        t5 = p5_pos.flatten(2).transpose(1, 2) + self.level_embed[2]
        tokens = torch.cat([t3, t4, t5], dim=1)

        slots, attn_ori, _ = self.slot_attention(tokens)

        (H3, W3), (H4, W4), _ = resolutions
        n3, n4 = H3 * W3, H4 * W4
        cls_attn = attn_ori[:, : self.num_classes, :]
        mass_p3 = cls_attn[:, :, :n3].mean(dim=-1, keepdim=True) * 10.0
        mass_p4 = cls_attn[:, :, n3 : n3 + n4].mean(dim=-1, keepdim=True) * 10.0
        mass_p5 = cls_attn[:, :, n3 + n4 :].mean(dim=-1, keepdim=True) * 10.0
        attn_mass_3 = torch.cat([mass_p3, mass_p4, mass_p5], dim=-1)

        preds = torch.stack(
            [self.fc_actor[i](torch.cat([slots[:, i, :], attn_mass_3[:, i, :]], dim=-1)).squeeze(-1) for i in range(self.num_classes)],
            dim=-1,
        )
        return preds, attn_ori, (feat_p3, feat_p4, feat_p5), resolutions, slots


# =============================================================================
# 3. MICRO-DATASET TRONG RAM
# =============================================================================

def rasterize_boxes(boxes: np.ndarray, resolutions: Tuple[Tuple[int, int], ...]) -> np.ndarray:
    has_b = boxes is not None and len(boxes) > 0
    if has_b:
        bxs = np.asarray(boxes, dtype=np.float32)
        cx, cy, w, h = bxs[:, 0], bxs[:, 1], bxs[:, 2], bxs[:, 3]
    parts = []
    for H, W in resolutions:
        m = np.ones((H, W), dtype=np.float32)
        if has_b:
            x1 = np.clip(np.trunc((cx - w / 2.0) * W).astype(np.int64), 0, W - 1)
            y1 = np.clip(np.trunc((cy - h / 2.0) * H).astype(np.int64), 0, H - 1)
            x2 = np.maximum(x1 + 1, np.clip(np.ceil((cx + w / 2.0) * W).astype(np.int64), 0, W))
            y2 = np.maximum(y1 + 1, np.clip(np.ceil((cy + h / 2.0) * H).astype(np.int64), 0, H))
            for i in range(len(x1)):
                m[y1[i] : y2[i], x1[i] : x2[i]] = 0.0
        parts.append(m.reshape(-1))
    return np.concatenate(parts, axis=0)


class InMemoryVisDroneMicroDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image": self.samples[idx]["image"],
            "counts": self.samples[idx]["counts"],
            "bg_mask": self.samples[idx]["bg_mask"],
        }


def load_micro_splits(
    lmdb_path: str,
    n_train: int = 500,
    n_val: int = 100,
    img_size: int = 384,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    resolutions = ((img_size // 8, img_size // 8), (img_size // 16, img_size // 16), (img_size // 32, img_size // 32))
    env, txn = open_lmdb(lmdb_path, readonly=True)
    total_bytes = txn.get(b"__len__")
    total = int(total_bytes.decode("ascii")) if total_bytes else env.stat()["entries"]

    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)

    needed = n_train + n_val
    samples: List[Dict] = []
    print(f"[*] Đang nạp {needed} ảnh VisDrone thật ({img_size}x{img_size}) vào RAM từ {lmdb_path}...")
    for real_idx in indices:
        if len(samples) >= needed:
            break
        byteflow = txn.get(f"{real_idx:08d}".encode("ascii"))
        if byteflow is None:
            continue
        data = pickle.loads(byteflow)
        raw_labels = data.get("labels", [])
        raw_boxes = data.get("boxes", [])

        img = Image.open(io.BytesIO(data["image"])).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.uint8)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        counts = torch.zeros(10, dtype=torch.float32)
        lbl_arr = np.asarray(raw_labels, dtype=np.int64)
        for c in range(10):
            counts[c] = float((lbl_arr == c).sum())

        bg_mask = torch.from_numpy(rasterize_boxes(raw_boxes, resolutions))
        boxes_norm = [
            (int(l), float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            for b, l in zip(raw_boxes, raw_labels)
            if 0 <= int(l) < 10
        ]
        samples.append({
            "image": img_t,
            "img_np": img_np,
            "counts": counts,
            "bg_mask": bg_mask,
            "boxes_norm": boxes_norm,
        })
    print(f"[+] Đã nạp xong {len(samples)} ảnh vào RAM ({n_train} train, {len(samples) - n_train} val).")
    return samples[:n_train], samples[n_train:]


# =============================================================================
# 4. ĐÁNH GIÁ ĐA TẦNG & TRÊN SLOTS CUỐI CÙNG
# =============================================================================

def _compute_fisher_full_and_pca3d(fg_vecs: np.ndarray, bg_vecs: np.ndarray) -> Tuple[float, float]:
    if len(fg_vecs) == 0 or len(bg_vecs) == 0:
        return 0.0, 0.0
    mu_fg_f, mu_bg_f = fg_vecs.mean(axis=0), bg_vecs.mean(axis=0)
    var_fg_f = float(np.sum(np.var(fg_vecs, axis=0)))
    var_bg_f = float(np.sum(np.var(bg_vecs, axis=0)))
    full_fisher = float(np.sum((mu_fg_f - mu_bg_f) ** 2)) / (var_fg_f + var_bg_f + 1e-6)

    all_vecs = np.concatenate([fg_vecs, bg_vecs], axis=0).astype(np.float32)
    centered = all_vecs - all_vecs.mean(axis=0, keepdims=True)
    if all_vecs.shape[0] >= 3 and all_vecs.shape[1] >= 3:
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ vt[:3].T
        except np.linalg.LinAlgError:
            proj = centered[:, :3]
    else:
        proj = centered[:, : min(3, centered.shape[1])]
    p_min = proj.min(axis=0, keepdims=True)
    p_max = proj.max(axis=0, keepdims=True)
    proj_norm = (proj - p_min) / (p_max - p_min + 1e-6)

    n_fg = len(fg_vecs)
    fg_pca = proj_norm[:n_fg]
    bg_pca = proj_norm[n_fg:]
    mu_fg_p, mu_bg_p = fg_pca.mean(axis=0), bg_pca.mean(axis=0)
    var_fg_p = float(np.sum(np.var(fg_pca, axis=0)))
    var_bg_p = float(np.sum(np.var(bg_pca, axis=0)))
    pca_fisher = float(np.sum((mu_fg_p - mu_bg_p) ** 2)) / (var_fg_p + var_bg_p + 1e-6)
    return full_fisher, pca_fisher


def _build_pairwise_matrix(proto_sum: np.ndarray, proto_cnt: np.ndarray, dim: int) -> Tuple[np.ndarray, float]:
    pairwise_dist = np.full((10, 10), np.nan, dtype=np.float32)
    unit_protos = np.zeros((10, dim), dtype=np.float64)
    valid_cls = proto_cnt > 0
    for c in range(10):
        if valid_cls[c]:
            v = proto_sum[c] / float(proto_cnt[c])
            unit_protos[c] = v / (np.linalg.norm(v) + 1e-8)
    for i in range(10):
        for j in range(10):
            if valid_cls[i] and valid_cls[j]:
                pairwise_dist[i, j] = 0.0 if i == j else float(1.0 - np.dot(unit_protos[i], unit_protos[j]))
    off_vals = pairwise_dist[~np.eye(10, dtype=bool)]
    off_vals = off_vals[np.isfinite(off_vals)]
    mean_45 = float(off_vals.mean()) if off_vals.size > 0 else 0.0
    return pairwise_dist, mean_45


@torch.no_grad()
def evaluate_architecture(
    model: MiniSlotCounterNeckAblation,
    val_samples: List[Dict],
    device: torch.device,
    small_area_thresh: float = 0.0025,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    model.eval()

    scale_names = ("p3", "p4", "p5")
    sc_acc: Dict[str, Dict[str, List[float]]] = {
        sn: {
            "small_ratios": [],
            "all_ratios": [],
            "full_fishers": [],
            "pca_fishers": [],
            "class_seps": [],
            "fg_mass_pcts": [],
        }
        for sn in scale_names
    }
    sc_proto_sum = {sn: np.zeros((10, model.dim), dtype=np.float64) for sn in scale_names}
    sc_proto_cnt = {sn: np.zeros(10, dtype=np.int64) for sn in scale_names}

    slot_small_ratios: List[float] = []
    slot_all_ratios: List[float] = []
    slot_act_inact_ratios: List[float] = []
    slot_class_seps: List[float] = []
    slot_all10_seps: List[float] = []
    slot_fg_mass_pcts: List[float] = []
    slot_share_p3_pcts: List[float] = []
    slot_share_p4_pcts: List[float] = []
    slot_share_p5_pcts: List[float] = []

    slot_fg_vecs_all: List[np.ndarray] = []
    slot_bg_vecs_all: List[np.ndarray] = []
    slot_proto_sum = np.zeros((10, model.dim), dtype=np.float64)
    slot_proto_cnt = np.zeros(10, dtype=np.int64)

    abs_errors: List[float] = []
    vis_pack: Dict[str, np.ndarray] = {}

    for idx, s in enumerate(val_samples):
        img_t = s["image"].unsqueeze(0).to(device)
        counts_gt = s["counts"].numpy()
        boxes_norm = s["boxes_norm"]
        small_boxes = [b for b in boxes_norm if (b[3] * b[4]) < small_area_thresh]
        small_cls_set = {int(b[0]) for b in small_boxes}

        preds, attn_ori, feats, resolutions, slots = model.forward_all(img_t)
        pred_np = preds[0].cpu().numpy()
        abs_errors.append(float(np.abs(pred_np - counts_gt).mean()))

        (H3, W3), (H4, W4), (H5, W5) = resolutions
        n3, n4, n5 = H3 * W3, H4 * W4, H5 * W5
        slices = [(0, n3), (n3, n3 + n4), (n3 + n4, n3 + n4 + n5)]

        all_masks_flat = []
        fg_attn_all_tokens = attn_ori[0, :10, :].sum(dim=0).cpu().numpy()

        for s_idx, sn in enumerate(scale_names):
            Hs, Ws = resolutions[s_idx]
            t_start, t_end = slices[s_idx]
            fs = feats[s_idx][0]
            l2_map = torch.linalg.vector_norm(fs.float(), dim=0).cpu().numpy()

            all_box_mask = build_box_mask_at_res(boxes_norm, Hs, Ws)
            small_box_mask = build_box_mask_at_res(small_boxes, Hs, Ws)
            bg_mask_bool = ~all_box_mask
            all_masks_flat.append(all_box_mask.reshape(-1))

            if all_box_mask.any() and bg_mask_bool.any():
                bg_energy = float(l2_map[bg_mask_bool].mean()) + 1e-6
                sc_acc[sn]["all_ratios"].append(float(l2_map[all_box_mask].mean()) / bg_energy)
                if small_box_mask.any():
                    sc_acc[sn]["small_ratios"].append(float(l2_map[small_box_mask].mean()) / bg_energy)

            fs_hwc = fs.detach().cpu().float().permute(1, 2, 0).numpy()
            if all_box_mask.any() and bg_mask_bool.any():
                fg_full = fs_hwc[all_box_mask]
                bg_full = fs_hwc[bg_mask_bool]
                mu_fg_f, mu_bg_f = fg_full.mean(axis=0), bg_full.mean(axis=0)
                var_fg_f = float(np.sum(np.var(fg_full, axis=0)))
                var_bg_f = float(np.sum(np.var(bg_full, axis=0)))
                sc_acc[sn]["full_fishers"].append(float(np.sum((mu_fg_f - mu_bg_f) ** 2)) / (var_fg_f + var_bg_f + 1e-6))

            pca_rgb = compute_pca_rgb(fs)
            if all_box_mask.any() and bg_mask_bool.any():
                fg_pts = pca_rgb[all_box_mask]
                bg_pts = pca_rgb[bg_mask_bool]
                mu_fg, mu_bg = fg_pts.mean(axis=0), bg_pts.mean(axis=0)
                var_fg = float(np.sum(np.var(fg_pts, axis=0)))
                var_bg = float(np.sum(np.var(bg_pts, axis=0)))
                sc_acc[sn]["pca_fishers"].append(float(np.sum((mu_fg - mu_bg) ** 2)) / (var_fg + var_bg + 1e-6))

            fs_norm = F.normalize(fs.float(), dim=0).cpu().numpy()
            class_protos = []
            for c in range(10):
                c_mask = build_box_mask_at_res(boxes_norm, Hs, Ws, class_filter=c)
                cnt_c = int(c_mask.sum())
                if cnt_c > 0:
                    sc_proto_sum[sn][c] += fs_norm[:, c_mask].sum(axis=1)
                    sc_proto_cnt[sn][c] += cnt_c
                if cnt_c >= 2:
                    proto = fs_norm[:, c_mask].mean(axis=1)
                    proto = proto / (np.linalg.norm(proto) + 1e-8)
                    class_protos.append(proto)
            if len(class_protos) >= 2:
                P = np.stack(class_protos, axis=0)
                sim = P @ P.T
                K = len(class_protos)
                off_diag = sim[~np.eye(K, dtype=bool)]
                sc_acc[sn]["class_seps"].append(float(1.0 - off_diag.mean()))

            fg_attn_s = fg_attn_all_tokens[t_start:t_end].reshape(Hs, Ws)
            if all_box_mask.any():
                sc_acc[sn]["fg_mass_pcts"].append(
                    float(fg_attn_s[all_box_mask].sum()) / (float(fg_attn_s.sum()) + 1e-8) * 100.0
                )

            if idx == 0 and sn == "p3":
                vis_pack = {
                    "img_np": s["img_np"],
                    "boxes_norm": boxes_norm,
                    "l2_map": l2_map,
                    "pca_rgb": pca_rgb,
                    "fg_attn_p3": fg_attn_s,
                }

        slots_np = slots[0].detach().cpu().float().numpy()
        fg_slots_np = slots_np[:10]
        bg_slot_np = slots_np[10]

        slot_norms = np.linalg.norm(fg_slots_np, axis=1)
        bg_slot_norm = float(np.linalg.norm(bg_slot_np)) + 1e-6

        active_cls = [c for c in range(10) if counts_gt[c] > 0]
        inactive_cls = [c for c in range(10) if counts_gt[c] == 0]
        small_active_cls = [c for c in active_cls if c in small_cls_set]

        if active_cls:
            slot_all_ratios.append(float(slot_norms[active_cls].mean()) / bg_slot_norm)
            if small_active_cls:
                slot_small_ratios.append(float(slot_norms[small_active_cls].mean()) / bg_slot_norm)
            if inactive_cls:
                inact_norm = float(slot_norms[inactive_cls].mean()) + 1e-6
                slot_act_inact_ratios.append(float(slot_norms[active_cls].mean()) / inact_norm)

        for c in active_cls:
            slot_fg_vecs_all.append(fg_slots_np[c])
        slot_bg_vecs_all.append(bg_slot_np)

        fg_slots_unit = fg_slots_np / (np.linalg.norm(fg_slots_np, axis=1, keepdims=True) + 1e-8)
        sim10 = fg_slots_unit @ fg_slots_unit.T
        slot_all10_seps.append(float(1.0 - sim10[~np.eye(10, dtype=bool)].mean()))

        for c in active_cls:
            slot_proto_sum[c] += fg_slots_unit[c]
            slot_proto_cnt[c] += 1
        if len(active_cls) >= 2:
            P_slot = fg_slots_unit[active_cls]
            sim_act = P_slot @ P_slot.T
            K_act = len(active_cls)
            slot_class_seps.append(float(1.0 - sim_act[~np.eye(K_act, dtype=bool)].mean()))

        combined_box_mask = np.concatenate(all_masks_flat, axis=0)
        total_fg_attn_sum = float(fg_attn_all_tokens.sum()) + 1e-8
        if combined_box_mask.any():
            slot_fg_mass_pcts.append(float(fg_attn_all_tokens[combined_box_mask].sum()) / total_fg_attn_sum * 100.0)

        slot_share_p3_pcts.append(float(fg_attn_all_tokens[:n3].sum()) / total_fg_attn_sum * 100.0)
        slot_share_p4_pcts.append(float(fg_attn_all_tokens[n3 : n3 + n4].sum()) / total_fg_attn_sum * 100.0)
        slot_share_p5_pcts.append(float(fg_attn_all_tokens[n3 + n4 :].sum()) / total_fg_attn_sum * 100.0)

    pw_p3, mean45_p3 = _build_pairwise_matrix(sc_proto_sum["p3"], sc_proto_cnt["p3"], model.dim)
    pw_p4, mean45_p4 = _build_pairwise_matrix(sc_proto_sum["p4"], sc_proto_cnt["p4"], model.dim)
    pw_p5, mean45_p5 = _build_pairwise_matrix(sc_proto_sum["p5"], sc_proto_cnt["p5"], model.dim)
    pw_slots, mean45_slots = _build_pairwise_matrix(slot_proto_sum, slot_proto_cnt, model.dim)

    vis_pack["pairwise_cos_sep"] = pw_p3
    vis_pack["pairwise_cos_sep_p4"] = pw_p4
    vis_pack["pairwise_cos_sep_p5"] = pw_p5
    vis_pack["pairwise_cos_sep_slots"] = pw_slots

    if slot_fg_vecs_all and slot_bg_vecs_all:
        slot_full_fisher, slot_pca_fisher = _compute_fisher_full_and_pca3d(
            np.stack(slot_fg_vecs_all, axis=0),
            np.stack(slot_bg_vecs_all, axis=0),
        )
    else:
        slot_full_fisher, slot_pca_fisher = 0.0, 0.0

    def _m(lst: List[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    metrics = {
        "p3_small_ratio": _m(sc_acc["p3"]["small_ratios"]),
        "p3_all_ratio": _m(sc_acc["p3"]["all_ratios"]),
        "p3_full_fisher": _m(sc_acc["p3"]["full_fishers"]),
        "p3_pca_fisher": _m(sc_acc["p3"]["pca_fishers"]),
        "p3_class_cos_sep": _m(sc_acc["p3"]["class_seps"]),
        "p3_global_45pair_sep": mean45_p3,
        "p3_fg_mass_pct": _m(sc_acc["p3"]["fg_mass_pcts"]),
        "p4_small_ratio": _m(sc_acc["p4"]["small_ratios"]),
        "p4_all_ratio": _m(sc_acc["p4"]["all_ratios"]),
        "p4_full_fisher": _m(sc_acc["p4"]["full_fishers"]),
        "p4_pca_fisher": _m(sc_acc["p4"]["pca_fishers"]),
        "p4_class_cos_sep": _m(sc_acc["p4"]["class_seps"]),
        "p4_global_45pair_sep": mean45_p4,
        "p4_fg_mass_pct": _m(sc_acc["p4"]["fg_mass_pcts"]),
        "p5_small_ratio": _m(sc_acc["p5"]["small_ratios"]),
        "p5_all_ratio": _m(sc_acc["p5"]["all_ratios"]),
        "p5_full_fisher": _m(sc_acc["p5"]["full_fishers"]),
        "p5_pca_fisher": _m(sc_acc["p5"]["pca_fishers"]),
        "p5_class_cos_sep": _m(sc_acc["p5"]["class_seps"]),
        "p5_global_45pair_sep": mean45_p5,
        "p5_fg_mass_pct": _m(sc_acc["p5"]["fg_mass_pcts"]),
        "slot_small_ratio": _m(slot_small_ratios),
        "slot_all_ratio": _m(slot_all_ratios),
        "slot_act_inact_ratio": _m(slot_act_inact_ratios),
        "slot_full_fisher": slot_full_fisher,
        "slot_pca_fisher": slot_pca_fisher,
        "slot_class_cos_sep": _m(slot_class_seps),
        "slot_all10_cos_sep": _m(slot_all10_seps),
        "slot_global_45pair_sep": mean45_slots,
        "slot_fg_mass_pct": _m(slot_fg_mass_pcts),
        "slot_share_p3_pct": _m(slot_share_p3_pcts),
        "slot_share_p4_pct": _m(slot_share_p4_pcts),
        "slot_share_p5_pct": _m(slot_share_p5_pcts),
        "val_mae": _m(abs_errors),
    }
    return metrics, vis_pack


# =============================================================================
# 5. HUẤN LUYỆN & XUẤT BẢNG / BIỂU ĐỒ SO SÁNH
# =============================================================================

def train_and_eval_variant(
    mode: str,
    label_name: str,
    train_loader: DataLoader,
    val_samples: List[Dict],
    device: torch.device,
    backbone_type: str = "senet",
    epochs: int = 40,
    lr: float = 1e-3,
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MiniSlotCounterNeckAblation(mode=mode, backbone_type=backbone_type, num_classes=10, dim=64).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    t0 = time.perf_counter()
    model.train()
    print(f"\n[+] Bắt đầu train '{label_name}' (Backbone: {backbone_type.upper()} | Neck Mode: '{mode}') ({epochs} epochs)...")
    for ep in range(1, epochs + 1):
        loss_ep = 0.0
        n_b = 0
        for batch in train_loader:
            imgs = batch["image"].to(device, non_blocking=True)
            gt_counts = batch["counts"].to(device, non_blocking=True)
            bg_mask = batch["bg_mask"].to(device, non_blocking=True)
            fg_mask = 1.0 - bg_mask

            optimizer.zero_grad()
            preds, attn_ori, _, _, _ = model.forward_all(imgs)

            loss_count = F.smooth_l1_loss(preds, gt_counts, beta=1.0)
            bg_attn = attn_ori[:, -1, :].clamp(1e-6, 1.0 - 1e-6)
            fg_attn = attn_ori[:, :10, :].sum(dim=1).clamp(1e-6, 1.0 - 1e-6)
            loss_bg = F.binary_cross_entropy(bg_attn, bg_mask)
            loss_fg = F.binary_cross_entropy(fg_attn, fg_mask)

            inactive = gt_counts == 0
            if inactive.any():
                inact_attn = attn_ori[:, :10, :][inactive].clamp(1e-6, 1.0 - 1e-6)
                loss_sup = F.binary_cross_entropy(inact_attn, torch.zeros_like(inact_attn))
            else:
                loss_sup = torch.tensor(0.0, device=device)

            loss = loss_count + 1.0 * loss_bg + 1.0 * loss_fg + 0.05 * loss_sup
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_ep += float(loss.item())
            n_b += 1
        scheduler.step()
        if ep % 10 == 0 or ep == epochs:
            print(f"    Epoch {ep:02d}/{epochs:02d} | Loss: {loss_ep / max(1, n_b):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    train_sec = time.perf_counter() - t0
    metrics, vis_pack = evaluate_architecture(model, val_samples, device)
    metrics["variant"] = label_name
    metrics["params_m"] = params_m
    metrics["train_sec"] = train_sec
    return metrics, vis_pack


def save_ablation_results(
    results: List[Dict[str, float]],
    vis_packs: Dict[str, Dict[str, np.ndarray]],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    names = [r["variant"] for r in results]

    # 1. CSV tổng hợp chi tiết
    csv_path = os.path.join(out_dir, "ablation_neck_relu_results.csv")
    cols = [
        "variant", "val_mae", "params_m", "train_sec",
        "slot_act_inact_ratio", "slot_class_cos_sep", "slot_global_45pair_sep",
        "slot_small_ratio", "slot_all_ratio", "slot_full_fisher", "slot_pca_fisher", "slot_fg_mass_pct",
        "slot_share_p3_pct", "slot_share_p4_pct", "slot_share_p5_pct",
        "p3_small_ratio", "p4_small_ratio", "p5_small_ratio",
        "p3_full_fisher", "p4_full_fisher", "p5_full_fisher",
        "p3_class_cos_sep", "p4_class_cos_sep", "p5_class_cos_sep",
        "p3_global_45pair_sep", "p4_global_45pair_sep", "p5_global_45pair_sep",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in results:
            writer.writerow({k: round(r[k], 4) if isinstance(r[k], float) else r[k] for k in cols})

    # 2. CSV so sánh 45 cặp lớp trên Final Slots cho tất cả các biến thể
    pair_csv = os.path.join(out_dir, "pairwise_cosine_sep_slots.csv")
    pair_fields = ["class_i", "class_j", "pair"] + names
    with open(pair_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pair_fields)
        writer.writeheader()
        for i in range(10):
            for j in range(i + 1, 10):
                row_dict = {
                    "class_i": VISDRONE_CLASSES[i],
                    "class_j": VISDRONE_CLASSES[j],
                    "pair": f"{VISDRONE_CLASSES[i]} vs {VISDRONE_CLASSES[j]}",
                }
                for vname in names:
                    v = float(vis_packs[vname]["pairwise_cos_sep_slots"][i, j])
                    row_dict[vname] = round(v, 4) if not np.isnan(v) else ""
                writer.writerow(row_dict)

    # 3. Biểu đồ so sánh trực quan thanh đối đầu
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    metrics_to_plot = [
        ("val_mae", "Validation Counting MAE ↓ (Thấp hơn = Tốt hơn)"),
        ("slot_act_inact_ratio", "Slot Active vs Inactive Ratio ↑ (Cao hơn = Tắt slot rỗng tốt hơn)"),
        ("slot_class_cos_sep", "Slot Class Cosine Separation ↑ (Cao hơn = Tách class trong ảnh tốt hơn)"),
        ("slot_global_45pair_sep", "Slot Global 45-Pair Separation ↑ (Cao hơn = Tách 45 cặp toàn tập tốt hơn)"),
        ("slot_full_fisher", "Slot FG vs BG Fisher Ratio ↑ (Cao hơn = Tách nền FG/BG trên Slots tốt hơn)"),
        ("p3_small_ratio", "P3 Small Object Energy Ratio ↑ (In/Out Box tại P3)"),
    ]
    x = np.arange(len(names))
    color_map = {
        "neck_with_relu": "#2980b9",
        "neck_no_last_relu": "#e67e22",
        "neck_preserve_bu_relu": "#27ae60",
        "neck_silu": "#8e44ad",
        "neck_one_dim_signed": "#e74c3c",
    }
    label_map = {
        "neck_with_relu": "With ReLU\n(Gốc: >=0 cả 64 chiều)",
        "neck_no_last_relu": "No Last ReLU\n(Tự do cả 64 chiều)",
        "neck_preserve_bu_relu": "Preserve BU\n(Giữ ReLU BottomUp)",
        "neck_silu": "SiLU Neck\n(Smooth Floor)",
        "neck_one_dim_signed": "1-Dim Signed\n(1 chiều Âm/Dương, 63 chiều >=0)",
    }
    bar_colors = [color_map.get(n, "#34495e") for n in names]
    bar_labels = [label_map.get(n, n) for n in names]

    for ax, (m_key, title) in zip(axes.flatten(), metrics_to_plot):
        vals = [r[m_key] for r in results]
        bars = ax.bar(x, vals, color=bar_colors, width=0.55, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle(f"Ablation Study: {len(names)} PANet Neck Variants (Epoch 40 | 500 Train / 100 Val)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = os.path.join(out_dir, "ablation_neck_relu_chart.png")
    plt.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close()

    # 4. Lưới Heatmap 10x10 so sánh các biến thể
    ncols = min(5, len(names))
    fig, axes = plt.subplots(1, ncols, figsize=(5.4 * ncols, 5.2), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    short_labels = ["ped", "ppl", "bic", "car", "van", "trk", "tri", "awn", "bus", "mot"]
    for idx_b, bname in enumerate(names):
        ax = axes[idx_b]
        mat = vis_packs[bname]["pairwise_cos_sep_slots"]
        off_vals = mat[~np.eye(10, dtype=bool)]
        off_vals = off_vals[np.isfinite(off_vals)]
        mean_sep = float(off_vals.mean()) if off_vals.size > 0 else 0.0
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0.0, vmax=0.15)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_title(f"{bname}\nMean(45 pairs): {mean_sep:.4f}", fontsize=10.5, fontweight="bold")
        for r_i in range(10):
            for c_j in range(10):
                val = mat[r_i, c_j]
                txt = "-" if r_i == c_j else (f"{val:.2f}" if not np.isnan(val) else "na")
                t_color = "white" if (not np.isnan(val) and val > 0.09) else "black"
                ax.text(c_j, r_i, txt, ha="center", va="center", fontsize=6.5, color=t_color)

    fig.colorbar(im, ax=list(axes), fraction=0.015, pad=0.02, label="Cosine Distance (1 - cos_sim)")
    hm_path = os.path.join(out_dir, "ablation_pairwise_slots_heatmap.png")
    plt.savefig(hm_path, dpi=160, bbox_inches="tight")
    plt.close()

    print("\n[+] Đã lưu toàn bộ kết quả ablation tại:")
    print(f"    1. Bảng số liệu CSV so sánh       : {os.path.abspath(csv_path)}")
    print(f"    2. Bảng phân tách 45 cặp trên Slots: {os.path.abspath(pair_csv)}")
    print(f"    3. Biểu đồ đối đầu các chỉ số chính: {os.path.abspath(chart_path)}")
    print(f"    4. Heatmap 10x10 Slots so sánh     : {os.path.abspath(hm_path)}")


def main():
    parser = argparse.ArgumentParser(description="Ablation: 4 PANet Neck Variants on configurable Backbone")
    parser.add_argument("--lmdb", type=str, default="D:/apps/Datasets/VisDrone/train_data.lmdb")
    parser.add_argument("--n-train", type=int, default=500, help="Số ảnh train trong RAM (mặc định 500)")
    parser.add_argument("--n-val", type=int, default=100, help="Số ảnh val trong RAM (mặc định 100)")
    parser.add_argument("--size", type=int, default=384, help="Kích thước ảnh")
    parser.add_argument("--epochs", type=int, default=40, help="Số epoch train mỗi kiến trúc (mặc định 40)")
    parser.add_argument("--runs", "--run", type=int, default=1, dest="runs", help="Số lần chạy lặp lại (mỗi run lưu vào thư mục con run_X)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu", action="store_true", help="Chạy trên CPU")
    parser.add_argument(
        "--backbone",
        type=str,
        default="senet",
        choices=list(BLOCK_REGISTRY.keys()),
        help=f"Tên backbone block (từ benchmark_blocks.py). Hỗ trợ: {', '.join(BLOCK_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["all"],
        help="Biến thể chạy: all hoặc danh sách từ: neck_with_relu, neck_no_last_relu, neck_preserve_bu_relu, neck_silu, neck_one_dim_signed",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Thư mục xuất kết quả (mặc định: Visdrone/output_ablation_neck_relu_{backbone})")
    args = parser.parse_args()

    backbone_name = args.backbone.lower()
    if args.out_dir is None:
        out_dir = f"Visdrone/output_ablation_neck_relu_{backbone_name}"
    else:
        out_dir = args.out_dir

    device = torch.device("cpu" if (args.cpu or not torch.cuda.is_available()) else args.device)
    print(f"[*] Thiết bị chạy ablation: {device}")
    print(f"[*] Backbone được chọn: {backbone_name.upper()} | Output dir: {out_dir}")

    train_samples, val_samples = load_micro_splits(
        lmdb_path=args.lmdb,
        n_train=args.n_train,
        n_val=args.n_val,
        img_size=args.size,
        seed=args.seed,
    )
    train_loader = DataLoader(InMemoryVisDroneMicroDataset(train_samples), batch_size=args.batch_size, shuffle=True)

    all_experiments = [
        ("with_relu", "neck_with_relu"),
        ("no_last_relu", "neck_no_last_relu"),
        ("preserve_bu_relu", "neck_preserve_bu_relu"),
        ("silu", "neck_silu"),
        ("one_dim_signed", "neck_one_dim_signed"),
    ]
    if "all" in args.variants:
        experiments = all_experiments
    else:
        experiments = [(m, l) for m, l in all_experiments if l in args.variants or m in args.variants]

    for run_idx in range(1, args.runs + 1):
        if args.runs > 1:
            run_out_dir = os.path.join(out_dir, f"run_{run_idx}")
            run_seed = args.seed + (run_idx - 1) * 1000
        else:
            run_out_dir = os.path.join(out_dir, "run_1")
            run_seed = args.seed

        print("\n" + "=" * 135)
        print(f"  BẮT ĐẦU ABLATION PANET NECK [RUN {run_idx}/{args.runs}] ({args.epochs} epochs | Backbone: {backbone_name.upper()} | Seed: {run_seed} | {args.size}x{args.size} | {len(train_samples)} train / {len(val_samples)} val)")
        print(f"  Thư mục lưu kết quả: {run_out_dir}")
        print("=" * 135)
        print(f"{'Biến thể Neck':<23} | {'Params':<7} | {'Time':<6} | {'Val MAE':<8} | {'Act/Inact':<9} | {'Slot ClsSep':<11} | {'Slot 45Pair':<11} | {'P3 Small':<8}")
        print("-" * 135)

        results: List[Dict[str, float]] = []
        vis_packs: Dict[str, Dict[str, np.ndarray]] = {}

        for mode, lbl in experiments:
            metrics, vpack = train_and_eval_variant(
                mode=mode,
                label_name=lbl,
                train_loader=train_loader,
                val_samples=val_samples,
                device=device,
                backbone_type=backbone_name,
                epochs=args.epochs,
                lr=args.lr,
                seed=run_seed,
            )
            results.append(metrics)
            vis_packs[lbl] = vpack
            print(
                f"{lbl:<23} | {metrics['params_m']:<6.2f}M | {metrics['train_sec']:<5.1f}s | "
                f"{metrics['val_mae']:<8.4f} | {metrics['slot_act_inact_ratio']:<8.2f}x | "
                f"{metrics['slot_class_cos_sep']:<11.4f} | {metrics['slot_global_45pair_sep']:<11.4f} | {metrics['p3_small_ratio']:<7.2f}x",
                flush=True,
            )

        print("=" * 135)
        save_ablation_results(results, vis_packs, run_out_dir)


if __name__ == "__main__":
    main()
