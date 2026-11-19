"""VisDrone Class Counting Model using MobileNetV3 Backbone, 2D Positional Encoding, and Attention Mass."""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from BaseModel.core import MODELS
from BaseModel.model import BaseModel

class PositionEmbeddingSine(nn.Module):
    """2D Sine-Cosine Positional Encoding (chuẩn DETR) mã hóa tọa độ không gian (x, y)."""
    def __init__(self, num_pos_feats: int = 16, temperature: int = 10000, normalize: bool = True, scale: Optional[float] = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # dim // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos  # (B, 2 * num_pos_feats, H, W)
    
def filter_color_bg(
    images: torch.Tensor | np.ndarray,
    target_res: tuple[int, int] = (8, 24),
    exp_factor: float = 5.0,
) -> torch.Tensor:
    """Filter flat background vs foreground on a batch of images using color distance, LayerNorm, and ReLU.
    Formula:
        score = mean(exp(exp_factor * |x - global_mean|), dim=C)
        relu_out = ReLU(LayerNorm_spatial(score))
    """
    if isinstance(images, np.ndarray):
        img = torch.from_numpy(images)
    elif isinstance(images, torch.Tensor):
        img = images
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(images)}")
    if img.dim() == 3:
        img = img.unsqueeze(0)  # (1, C, H, W)
    elif img.dim() != 4:
        raise ValueError(f"Expected 4D batch (B, C, H, W) or 3D image (C, H, W), got {img.shape}")
    H_feat, W_feat = target_res
    # 1. Normalize image về [0, 1]
    if img.dtype == torch.uint8 or img.max() > 1.0:
        v_float = img.float().div(255.0)
    else:
        v_float = img.float()
        if v_float.min() < -0.5:
            v_float = (v_float * 0.225 + 0.45).clamp(0.0, 1.0)
    # 2. Downsample ảnh về kích thước (B, C, H_feat, W_feat)
    x = F.interpolate(v_float, size=(H_feat, W_feat), mode="area")
    # 3. Tính màu trung bình độc lập cho từng ảnh trong batch theo không gian (H, W)
    global_mean = x.mean(dim=(2, 3), keepdim=True)
    # 4. Tính khoảng cách màu theo kênh C và khuếch đại bằng hàm mũ
    color_diff = torch.abs(x - global_mean)
    exp_diff = torch.exp(float(exp_factor) * color_diff)
    score_map = exp_diff.mean(dim=1)
    # 5. LayerNorm độc lập trên từng ảnh theo không gian (H_feat, W_feat), sau đó qua ReLU
    mean_v = score_map.mean(dim=(1, 2), keepdim=True)
    std_v = score_map.std(dim=(1, 2), keepdim=True)
    norm_v = (score_map - mean_v) / (std_v + 1e-6)
    relu_v = torch.relu(norm_v)
    # Vùng có relu_v == 0 (tức norm_v <= 0) là vùng nền đồng nhất
    flat_mask = (relu_v == 0).float()
    return flat_mask

class PANetNeck(nn.Module):
    """Path Aggregation Network (PANet) Neck cho 3 tầng P3 (//8), P4 (//16), P5 (//32).

    Kết hợp 2 luồng thông tin:
      1. Top-Down (FPN): truyền ngữ nghĩa cấp cao từ P5 -> P4 -> P3 (giúp P3 nhận diện vật thể nhỏ tốt hơn).
      2. Bottom-Up (PAN): truyền chi tiết vị trí/biên cạnh từ P3 -> P4 -> P5 (giúp P4, P5 định vị chính xác hơn).
    """

    def __init__(self, in_channels: Tuple[int, int, int] = (24, 48, 576), out_dim: int = 32):
        super().__init__()
        c3_in, c4_in, c5_in = in_channels

        # 1. Lateral 1x1 projections đưa C3, C4, C5 về cùng số kênh out_dim
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
        self.td_conv_p3 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        # 3. Bottom-Up downsampling & fusion convs (PAN)
        self.down_p3_to_p4 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.bu_conv_p4 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        self.down_p4_to_p5 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.bu_conv_p5 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Lateral projections
        p3_lat = self.lat_p3(c3)
        p4_lat = self.lat_p4(c4)
        p5_lat = self.lat_p5(c5)

        # Top-Down Pathway (P5 -> P4 -> P3)
        td_p5 = p5_lat
        td_p4 = self.td_conv_p4(p4_lat + F.interpolate(td_p5, size=p4_lat.shape[-2:], mode="nearest"))
        td_p3 = self.td_conv_p3(p3_lat + F.interpolate(td_p4, size=p3_lat.shape[-2:], mode="nearest"))

        # Bottom-Up Pathway (P3 -> P4 -> P5)
        out_p3 = td_p3
        p3_down = self.down_p3_to_p4(out_p3)
        if p3_down.shape[-2:] != td_p4.shape[-2:]:
            p3_down = F.interpolate(p3_down, size=td_p4.shape[-2:], mode="bilinear", align_corners=False)
        out_p4 = self.bu_conv_p4(td_p4 + p3_down)

        p4_down = self.down_p4_to_p5(out_p4)
        if p4_down.shape[-2:] != td_p5.shape[-2:]:
            p4_down = F.interpolate(p4_down, size=td_p5.shape[-2:], mode="bilinear", align_corners=False)
        out_p5 = self.bu_conv_p5(td_p5 + p4_down)

        return out_p3, out_p4, out_p5


class SlotAttention(nn.Module):
    """Slot Attention module nhận chuỗi tokens đa quy mô N = N_p3 + N_p4 + N_p5.
    num_slots: num_class + 1 (10 class slots + 1 background slot).
    """
    def __init__(self, dim: int = 32, num_class: int = 10, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.num_class = num_class
        num_slots: int = num_class + 1
        self.num_slots = num_slots
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.obj_queries = nn.Parameter(torch.randn(1, num_slots, dim) * 0.05)
        self.FC1 = nn.Linear(dim, dim)
        self.FC2 = nn.Linear(dim, dim)
        self.LN = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.eps = eps
        self.scale = dim**-0.5
    def forward(self, inputs: Tensor, slots: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """inputs shape: (B, N, dim). N = 14400 (P3) + 3600 (P4) + 900 (P5) = 18900 tokens."""
        inputs = inputs + self.FC2(F.relu(self.FC1(self.LN(inputs))))
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        if slots is None:
            slots = self.obj_queries.expand(inputs.shape[0], -1, -1)
        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
        attn_ori = dots.softmax(dim=1) + self.eps
        attn_norm = attn_ori / attn_ori.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        slots = torch.einsum("bjd,bij->bid", v, attn_norm)
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots, attn_ori, attn_norm
@MODELS.register_module(name="VisDroneDetector", force=True)
@MODELS.register_module(name="VisDroneSlotCounter", force=True)
class VisDroneDetector(BaseModel):
    """Mô hình đếm số lượng class VisDrone sử dụng MobileNetV3 Backbone, 2D Positional Encoding và Attention Mass.
    1. Backbone: MobileNetV3 Small (Nano) trích xuất P3 (//8), P4 (//16), P5 (//32).
    2. 2D Positional Encoding: Cung cấp tọa độ (x, y) chuẩn xác cho từng điểm token.
    3. Level Embeddings: Phân biệt quy mô //8, //16, //32.
    4. Slot Attention: Cập nhật đặc trưng ngữ nghĩa cho từng class slot.
    5. Attention Mass: Cung cấp trực tiếp tổng diện tích attention vào fc_actor [slots, mass] -> count.
    """
    def __init__(
        self,
        num_classes: int = 10,
        dim: int = 32,
        pretrained: bool = True,
        exp_factor: float = 5.0,
        count_loss_weight: float = 1.0,
        bg_attn_weight: float = 0.5,
        fg_attn_weight: float = 0.5,
        suppress_weight: float = 0.05,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.exp_factor = exp_factor
        self.count_loss_weight = float(count_loss_weight)
        self.bg_attn_weight = float(bg_attn_weight)
        self.fg_attn_weight = float(fg_attn_weight)
        self.suppress_weight = float(suppress_weight)
        # 1. Backbone MobileNetV3 Small (Nano)
        try:
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            m = mobilenet_v3_small(weights=weights)
        except Exception:
            m = mobilenet_v3_small(weights=None)
        return_nodes = {
            "features.3": "p3",   # stride 8  (24 channels)
            "features.8": "p4",   # stride 16 (48 channels)
            "features.12": "p5",  # stride 32 (576 channels)
        }
        self.backbone = create_feature_extractor(m, return_nodes=return_nodes)
        # 2. PANet Neck tổng hợp đặc trưng 2 chiều (Top-Down + Bottom-Up) cho P3, P4, P5
        self.neck = PANetNeck(in_channels=(24, 48, 576), out_dim=dim)
        # 3. 2D Positional Encoding (Sine-Cosine)
        assert dim % 2 == 0, f"dim must be even for 2D Positional Encoding, got {dim}"
        self.pos_emb = PositionEmbeddingSine(num_pos_feats=dim // 2)
        # Level Embeddings: giúp các slot nhận biết token đến từ tầng //8, //16 hay //32
        self.level_embed = nn.Parameter(torch.randn(3, dim) * 0.02)
        # 4. Slot Attention Module
        self.slot_attention = SlotAttention(dim=dim, num_class=num_classes)
        # 5. fc_actor: nhận [slot_vector (dim), mass_p3, mass_p4, mass_p5] -> dự đoán số lượng (không âm)
        self.fc_actor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim + 3, dim),
                nn.ReLU(inplace=False),
                nn.Linear(dim, 1),
                nn.ReLU(inplace=False),  # Số lượng đối tượng không âm
            )
            for _ in range(num_classes)
        ])
    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        """Trích xuất P3, P4, P5 qua PANet Neck, cộng 2D Positional Encoding và Level Embeddings."""
        feats = self.backbone(x)
        feat_p3, feat_p4, feat_p5 = self.neck(feats["p3"], feats["p4"], feats["p5"])
        # Cộng 2D Positional Encoding (x, y) để mạng biết vị trí địa lý của từng token
        feat_p3 = feat_p3 + self.pos_emb(feat_p3)
        feat_p4 = feat_p4 + self.pos_emb(feat_p4)
        feat_p5 = feat_p5 + self.pos_emb(feat_p5)
        res_p3 = (feat_p3.shape[2], feat_p3.shape[3])
        res_p4 = (feat_p4.shape[2], feat_p4.shape[3])
        res_p5 = (feat_p5.shape[2], feat_p5.shape[3])
        # Trải phẳng và cộng Level Embedding theo quy mô
        p3 = feat_p3.flatten(2).transpose(1, 2) + self.level_embed[0]  # (B, 14400, dim)
        p4 = feat_p4.flatten(2).transpose(1, 2) + self.level_embed[1]  # (B, 3600, dim)
        p5 = feat_p5.flatten(2).transpose(1, 2) + self.level_embed[2]  # (B, 900, dim)
        # Ghép nối thành chuỗi 18,900 tokens
        inputs = torch.cat([p3, p4, p5], dim=1)  # (B, 18900, dim)
        return inputs, (res_p3, res_p4, res_p5)
    def extract_slots(self, inputs_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        """Chạy qua Backbone, Positional Encoding và Slot Attention để lấy slots và attention matrix."""
        tokens, resolutions = self.extract_features(inputs_img)
        slots, attn_ori, _ = self.slot_attention(tokens)
        return slots, attn_ori, resolutions
    def get_spatial_attention_maps(
        self,
        attn_ori: torch.Tensor,
        resolutions: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        target_size: Tuple[int, int] = (960, 960),
        fuse_mode: str = "max",
    ) -> torch.Tensor:
        """Tái cấu trúc attention matrix (B, 11, 18900) thành bản đồ không gian 2D (B, 11, H, W).

        fuse_mode:
          - 'max' (mặc định): lấy cực đại theo 3 tầng P3/P4/P5 giúp vật nhỏ (P3) và vật lớn (P5) không bị làm mờ lẫn nhau.
          - 'mean': trung bình cộng 3 tầng sau khi upsample.
        """
        B, num_slots, _ = attn_ori.shape
        (H3, W3), (H4, W4), (H5, W5) = resolutions
        n3 = H3 * W3  # 14400
        n4 = H4 * W4  # 3600
        attn_p3 = attn_ori[:, :, :n3].view(B, num_slots, H3, W3)
        attn_p4 = attn_ori[:, :, n3:n3 + n4].view(B, num_slots, H4, W4)
        attn_p5 = attn_ori[:, :, n3 + n4:].view(B, num_slots, H5, W5)
        p3_up = F.interpolate(attn_p3, size=target_size, mode="bilinear", align_corners=False)
        p4_up = F.interpolate(attn_p4, size=target_size, mode="bilinear", align_corners=False)
        p5_up = F.interpolate(attn_p5, size=target_size, mode="bilinear", align_corners=False)
        if fuse_mode == "mean":
            return (p3_up + p4_up + p5_up) / 3.0
        return torch.maximum(torch.maximum(p3_up, p4_up), p5_up)

    def _predict_counts(
        self,
        slots: torch.Tensor,
        attn_ori: torch.Tensor,
        resolutions: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    ) -> torch.Tensor:
        """Dự đoán số lượng 10 class bằng cách ghép [slots, mass_p3, mass_p4, mass_p5].
        Tách riêng tỉ lệ diện tích attention của từng tầng (mean theo số token của tầng đó)
        để P5 (vật lớn) không bị P3 (nhiều token gấp 16x) áp đảo.
        """
        (H3, W3), (H4, W4), _ = resolutions
        n3 = H3 * W3
        n4 = H4 * W4
        cls_attn = attn_ori[:, :self.num_classes, :]  # (B, 10, N)
        # Tỉ lệ phủ không gian theo từng tầng (nhân 10.0 để đưa về biên độ phù hợp cho MLP)
        mass_p3 = cls_attn[:, :, :n3].mean(dim=-1, keepdim=True) * 10.0
        mass_p4 = cls_attn[:, :, n3:n3 + n4].mean(dim=-1, keepdim=True) * 10.0
        mass_p5 = cls_attn[:, :, n3 + n4:].mean(dim=-1, keepdim=True) * 10.0
        attn_mass_3 = torch.cat([mass_p3, mass_p4, mass_p5], dim=-1)  # (B, 10, 3)

        pred_list = []
        for i in range(self.num_classes):
            feat_i = torch.cat([slots[:, i, :], attn_mass_3[:, i, :]], dim=-1)  # (B, dim + 3)
            pred_list.append(self.fc_actor[i](feat_i).squeeze(-1))
        pred_counts = torch.stack(pred_list, dim=-1)  # (B, 10)
        return pred_counts
    def _normalize_targets(
        self,
        targets: Optional[Union[torch.Tensor, List[Any], Dict[str, Any]]],
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        """Chuẩn hóa targets về dạng Tensor (B, 10) float32."""
        if targets is None:
            return torch.zeros((batch_size, self.num_classes), device=device, dtype=torch.float32)
        if isinstance(targets, torch.Tensor):
            t = targets.to(device=device, dtype=torch.float32)
            if t.dim() == 1 and t.numel() == self.num_classes:
                return t.unsqueeze(0)
            elif t.dim() == 2 and t.shape[-1] == self.num_classes:
                return t
            return t.view(batch_size, self.num_classes)
        if isinstance(targets, dict):
            if "counts" in targets:
                return self._normalize_targets(targets["counts"], device, batch_size)
            if "labels" in targets:
                targets = [targets]
        if isinstance(targets, (list, tuple)):
            gt_counts = torch.zeros((batch_size, self.num_classes), device=device, dtype=torch.float32)
            for b, item in enumerate(targets):
                if b >= batch_size:
                    break
                if isinstance(item, torch.Tensor):
                    gt_counts[b] = item.to(device=device, dtype=torch.float32)
                elif isinstance(item, dict):
                    if "counts" in item:
                        gt_counts[b] = item["counts"].to(device=device, dtype=torch.float32)
                    elif "labels" in item and len(item["labels"]) > 0:
                        lbls = item["labels"].to(device)
                        for c in range(self.num_classes):
                            gt_counts[b, c] = (lbls == c).sum().float()
            return gt_counts

        return torch.zeros((batch_size, self.num_classes), device=device, dtype=torch.float32)

    def _create_bg_masks_from_boxes(
        self,
        targets: Optional[Union[torch.Tensor, List[Any], Dict[str, Any]]],
        resolutions: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        batch_size: int,
        device: torch.device,
        fallback_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Lấy hoặc tạo multi-scale background segmentation mask chuẩn xác từ Bounding Box.
        - Giá trị 1.0: Vùng Nền (Background) bên ngoài mọi bounding box.
        - Giá trị 0.0: Vùng Vật thể (Foreground) bên trong các bounding box.

        Ưu tiên theo thứ tự tốc độ:
          1. Nạp trực tiếp `bg_mask` đã pre-generate từ sidecar LMDB (`bg_seg.lmdb`) -> O(1), không tốn CPU.
          2. Rasterize trực tiếp từ `boxes` trong `targets` (nếu chưa có `bg_mask` hoặc khác độ phân giải).
          3. Fallback về `filter_color_bg` nếu không có cả `bg_mask` và `boxes`.
        """
        total_tokens = sum(h * w for h, w in resolutions)

        # --- Nhánh 1 (Nhanh nhất): Đọc pre-generated bg_mask từ sidecar LMDB ---
        if isinstance(targets, (list, tuple)) and len(targets) >= batch_size:
            if all(isinstance(item, dict) and "bg_mask" in item and item["bg_mask"] is not None for item in targets[:batch_size]):
                first_m = targets[0]["bg_mask"]
                if hasattr(first_m, "numel") and first_m.numel() == total_tokens:
                    stacked = torch.stack(
                        [item["bg_mask"].reshape(-1) for item in targets[:batch_size]], dim=0
                    )
                    return stacked.to(device=device, dtype=torch.float32)
        elif isinstance(targets, dict) and "bg_mask" in targets and targets["bg_mask"] is not None:
            bm = targets["bg_mask"]
            if isinstance(bm, torch.Tensor):
                if bm.dim() == 1 and bm.numel() == total_tokens and batch_size == 1:
                    return bm.unsqueeze(0).to(device=device, dtype=torch.float32)
                elif bm.dim() == 2 and bm.shape == (batch_size, total_tokens):
                    return bm.to(device=device, dtype=torch.float32)
            elif isinstance(bm, (list, tuple)) and len(bm) >= batch_size:
                if bm[0].numel() == total_tokens:
                    return torch.stack([m.reshape(-1) for m in bm[:batch_size]], dim=0).to(
                        device=device, dtype=torch.float32
                    )

        # --- Nhánh 2: Tạo on-the-fly từ bounding boxes ---
        has_boxes = False
        boxes_list = []
        if isinstance(targets, (list, tuple)):
            for item in targets:
                if isinstance(item, dict) and "boxes" in item:
                    has_boxes = True
                    boxes_list.append(item["boxes"])
                else:
                    boxes_list.append(None)
        elif isinstance(targets, dict) and "boxes" in targets:
            has_boxes = True
            bxs = targets["boxes"]
            if isinstance(bxs, (list, tuple)):
                boxes_list = list(bxs)
            elif isinstance(bxs, torch.Tensor) and bxs.dim() == 3:
                boxes_list = [bxs[b] for b in range(bxs.shape[0])]
            else:
                boxes_list = [bxs]

        if not has_boxes:
            # --- Nhánh 3: Fallback nếu targets không cung cấp bg_mask hay bounding box ---
            if fallback_inputs is not None:
                (H3, W3), (H4, W4), (H5, W5) = resolutions
                m3 = filter_color_bg(fallback_inputs, target_res=(H3, W3), exp_factor=self.exp_factor)
                m4 = filter_color_bg(fallback_inputs, target_res=(H4, W4), exp_factor=self.exp_factor)
                m5 = filter_color_bg(fallback_inputs, target_res=(H5, W5), exp_factor=self.exp_factor)
                return torch.cat([m3.flatten(1), m4.flatten(1), m5.flatten(1)], dim=1).to(device)
            else:
                return torch.ones((batch_size, total_tokens), device=device, dtype=torch.float32)

        multiscale_masks = []
        for H, W in resolutions:
            mask = torch.ones((batch_size, H, W), device=device, dtype=torch.float32)
            for b in range(min(batch_size, len(boxes_list))):
                bxs = boxes_list[b]
                if bxs is None or len(bxs) == 0:
                    continue
                if isinstance(bxs, np.ndarray):
                    bxs = torch.from_numpy(bxs)
                bxs = bxs.to(device=device, dtype=torch.float32)

                # VisDrone bounding box format: [cx, cy, w, h] chuẩn hóa trong [0, 1]
                cx, cy, w, h = bxs[:, 0], bxs[:, 1], bxs[:, 2], bxs[:, 3]
                x1 = ((cx - w / 2.0) * W).long().clamp(0, W - 1)
                y1 = ((cy - h / 2.0) * H).long().clamp(0, H - 1)
                x2 = torch.maximum(x1 + 1, ((cx + w / 2.0) * W).ceil().long().clamp(0, W))
                y2 = torch.maximum(y1 + 1, ((cy + h / 2.0) * H).ceil().long().clamp(0, H))

                for i in range(len(bxs)):
                    mask[b, y1[i]:y2[i], x1[i]:x2[i]] = 0.0

            multiscale_masks.append(mask.flatten(1))

        return torch.cat(multiscale_masks, dim=1)  # (B, 18900)

    def _compute_losses(
        self,
        inputs: torch.Tensor,
        pred_counts: torch.Tensor,
        attn_ori: torch.Tensor,
        resolutions: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        gt_counts: torch.Tensor,
        targets: Optional[Union[torch.Tensor, List[Any], Dict[str, Any]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tính toán tổng hợp: Count Loss + Multi-scale Attention Mask Loss (từ Bounding Box)."""
        B, num_slots, total_tokens = attn_ori.shape
        (H3, W3), (H4, W4), (H5, W5) = resolutions
        n3 = H3 * W3
        n4 = H4 * W4

        def _balanced_multiscale_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Tính BCE riêng cho P3, P4, P5 rồi lấy trung bình 3 tầng"""
            p = pred.clamp(1e-6, 1.0 - 1e-6)
            l_p3 = F.binary_cross_entropy(p[..., :n3], target[..., :n3])
            l_p4 = F.binary_cross_entropy(p[..., n3:n3 + n4], target[..., n3:n3 + n4])
            l_p5 = F.binary_cross_entropy(p[..., n3 + n4:], target[..., n3 + n4:])
            return (l_p3 + l_p4 + l_p5) / 3.0

        # 1. Count regression loss (Smooth L1)
        loss_count = F.smooth_l1_loss(pred_counts, gt_counts, beta=1.0)

        # 2. Lấy hoặc tạo background mask chuẩn xác từ Bounding Box (ưu tiên đọc từ sidecar bg_seg.lmdb)
        if self.bg_attn_weight > 0 or self.fg_attn_weight > 0:
            mask_multiscale = self._create_bg_masks_from_boxes(
                targets=targets,
                resolutions=resolutions,
                batch_size=B,
                device=inputs.device,
                fallback_inputs=inputs,
            )
        else:
            mask_multiscale = None

        # 3. Background Slot Loss: Slot cuối (index -1) tập trung vào mask = 1 (vùng nền ngoài box)
        if self.bg_attn_weight > 0:
            bg_attn = attn_ori[:, -1, :]  # (B, 18900)
            loss_bg_attn = _balanced_multiscale_bce(bg_attn, mask_multiscale)
        else:
            loss_bg_attn = torch.tensor(0.0, device=inputs.device)
        # 4. Foreground Slots Loss: 10 slot foreground tập trung vào mask = 0 (1.0 - mask, vùng trong box)
        if self.fg_attn_weight > 0:
            fg_mask = 1.0 - mask_multiscale
            fg_attn_sum = attn_ori[:, :self.num_classes, :].sum(dim=1)  # (B, 18900)
            loss_fg_attn = _balanced_multiscale_bce(fg_attn_sum, fg_mask)
        else:
            loss_fg_attn = torch.tensor(0.0, device=inputs.device)
        # 5. Inactive Slot Suppression (cân bằng đều giữa P3, P4, P5)
        if self.suppress_weight > 0:
            inactive_classes = (gt_counts == 0)  # (B, 10)
            class_attn = attn_ori[:, :self.num_classes, :]  # (B, 10, 18900)
            if inactive_classes.any():
                inactive_selected = class_attn[inactive_classes]
                zeros = torch.zeros_like(inactive_selected)
                loss_suppress = _balanced_multiscale_bce(inactive_selected, zeros)
            else:
                loss_suppress = torch.tensor(0.0, device=inputs.device)
        else:
            loss_suppress = torch.tensor(0.0, device=inputs.device)
        # Tổng hợp loss
        total_loss = (
            self.count_loss_weight * loss_count
            + self.bg_attn_weight * loss_bg_attn
            + self.fg_attn_weight * loss_fg_attn
            + self.suppress_weight * loss_suppress
        )
        return {
            "loss": total_loss,
            "loss_count": loss_count,
            "loss_bg_attn": loss_bg_attn,
            "loss_fg_attn": loss_fg_attn,
            "loss_suppress": loss_suppress,
        }
    def forward_train(
        self,
        inputs: torch.Tensor,
        targets: Optional[Union[torch.Tensor, List[Any], Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Huấn luyện mô hình với Backbone MobileNetV3, PANet Neck, Positional Encoding, Multi-scale Attention Mass và BG Mask."""
        B = inputs.shape[0]
        slots, attn_ori, resolutions = self.extract_slots(inputs)
        pred_counts = self._predict_counts(slots, attn_ori, resolutions)
        gt_counts = self._normalize_targets(targets, device=inputs.device, batch_size=B)
        return self._compute_losses(
            inputs=inputs,
            pred_counts=pred_counts,
            attn_ori=attn_ori,
            resolutions=resolutions,
            gt_counts=gt_counts,
            targets=targets,
        )
    def forward_val(
        self,
        inputs: torch.Tensor,
        targets: Optional[Union[torch.Tensor, List[Any], Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Validation: tính loss và xuất preds (B, 10) để trainer cập nhật metrics."""
        B = inputs.shape[0]
        slots, attn_ori, resolutions = self.extract_slots(inputs)
        pred_counts = self._predict_counts(slots, attn_ori, resolutions)
        gt_counts = self._normalize_targets(targets, device=inputs.device, batch_size=B)
        loss_dict = self._compute_losses(
            inputs=inputs,
            pred_counts=pred_counts,
            attn_ori=attn_ori,
            resolutions=resolutions,
            gt_counts=gt_counts,
            targets=targets,
        )
        output: Dict[str, Any] = {}
        output.update(loss_dict)
        output["preds"] = pred_counts
        output["targets"] = gt_counts
        return output
    def predict(
        self,
        inputs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Inference: dự đoán số lượng 10 class từ batch ảnh đầu vào."""
        slots, attn_ori, resolutions = self.extract_slots(inputs)
        return self._predict_counts(slots, attn_ori, resolutions)
    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[Union[torch.Tensor, List[Any], Dict[str, Any]]] = None,
        **kwargs,
    ):
        """Standard nn.Module forward dispatching to forward_train or forward_val."""
        if self.training:
            return self.forward_train(inputs, targets=targets, **kwargs)
        else:
            return self.forward_val(inputs, targets=targets, **kwargs)