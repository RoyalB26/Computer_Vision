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
        # 2. Chiếu 3 mức đặc trưng về cùng số chiều dim
        self.proj_p3 = nn.Conv2d(24, dim, kernel_size=1)
        self.proj_p4 = nn.Conv2d(48, dim, kernel_size=1)
        self.proj_p5 = nn.Conv2d(576, dim, kernel_size=1)
        # 3. 2D Positional Encoding (Sine-Cosine)
        assert dim % 2 == 0, f"dim must be even for 2D Positional Encoding, got {dim}"
        self.pos_emb = PositionEmbeddingSine(num_pos_feats=dim // 2)
        # Level Embeddings: giúp các slot nhận biết token đến từ tầng //8, //16 hay //32
        self.level_embed = nn.Parameter(torch.randn(3, dim) * 0.02)
        # 4. Slot Attention Module
        self.slot_attention = SlotAttention(dim=dim, num_class=num_classes)
        # 5. fc_actor: nhận [slot_vector (dim), attn_mass (1)] -> dự đoán số lượng (không âm)
        self.fc_actor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim + 1, dim),
                nn.ReLU(inplace=False),
                nn.Linear(dim, 1),
                nn.ReLU(inplace=False),  # Số lượng đối tượng không âm
            )
            for _ in range(num_classes)
        ])
    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        """Trích xuất P3, P4, P5, cộng 2D Positional Encoding và Level Embeddings."""
        feats = self.backbone(x)
        feat_p3 = self.proj_p3(feats["p3"])  # (B, dim, 120, 120)
        feat_p4 = self.proj_p4(feats["p4"])  # (B, dim, 60, 60)
        feat_p5 = self.proj_p5(feats["p5"])  # (B, dim, 30, 30)
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
    ) -> torch.Tensor:
        """Tái cấu trúc attention matrix (B, 11, 18900) thành bản đồ không gian 2D (B, 11, H, W)."""
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
        attn_full = (p3_up + p4_up + p5_up) / 3.0
        return attn_full
    def _predict_counts(self, slots: torch.Tensor, attn_ori: torch.Tensor) -> torch.Tensor:
        """Dự đoán số lượng 10 class thông qua fc_actor bằng cách ghép [slots, attn_mass].
        attn_mass: tổng diện tích attention trước khi chia chuẩn hóa (phản ánh trực tiếp số lượng đối tượng).
        """
        # attn_mass shape: (B, 10, 1) - chia 100 để đưa về biên độ số học phù hợp cho MLP
        attn_mass = attn_ori[:, :self.num_classes, :].sum(dim=-1, keepdim=True) / 100.0
        pred_list = []
        for i in range(self.num_classes):
            feat_i = torch.cat([slots[:, i, :], attn_mass[:, i, :]], dim=-1)  # (B, dim + 1)
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

    def _compute_losses(
        self,
        inputs: torch.Tensor,
        pred_counts: torch.Tensor,
        attn_ori: torch.Tensor,
        resolutions: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        gt_counts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Tính toán tổng hợp: Count Loss + Multi-scale Attention Mask Loss."""
        B, num_slots, total_tokens = attn_ori.shape
        (H3, W3), (H4, W4), (H5, W5) = resolutions
        # 1. Count regression loss (Smooth L1)
        loss_count = F.smooth_l1_loss(pred_counts, gt_counts, beta=1.0)
        # 2. Tạo flat background mask tương ứng cho từng quy mô
        mask_p3 = filter_color_bg(inputs, target_res=(H3, W3), exp_factor=self.exp_factor)
        mask_p4 = filter_color_bg(inputs, target_res=(H4, W4), exp_factor=self.exp_factor)
        mask_p5 = filter_color_bg(inputs, target_res=(H5, W5), exp_factor=self.exp_factor)
        # Ghép nối mask thành vector 18,900 khớp với tokens
        mask_multiscale = torch.cat(
            [mask_p3.flatten(1), mask_p4.flatten(1), mask_p5.flatten(1)], dim=1
        ).to(device=inputs.device, dtype=torch.float32)  # (B, 18900)
        # 3. Background Slot Loss: Slot cuối (index -1) tập trung vào mask = 1
        bg_attn = attn_ori[:, -1, :]  # (B, 18900)
        loss_bg_attn = F.binary_cross_entropy(bg_attn.clamp(1e-6, 1.0 - 1e-6), mask_multiscale)
        # 4. Foreground Slots Loss: 10 slot foreground tập trung vào mask = 0 (1.0 - mask)
        fg_mask = 1.0 - mask_multiscale
        fg_attn_sum = attn_ori[:, :self.num_classes, :].sum(dim=1)  # (B, 18900)
        loss_fg_attn = F.binary_cross_entropy(fg_attn_sum.clamp(1e-6, 1.0 - 1e-6), fg_mask)
        # 5. Inactive Slot Suppression
        inactive_classes = (gt_counts == 0)  # (B, 10)
        class_attn = attn_ori[:, :self.num_classes, :]  # (B, 10, 18900)
        if inactive_classes.any():
            inactive_selected = class_attn[inactive_classes].clamp(1e-6, 1.0 - 1e-6)
            zeros = torch.zeros_like(inactive_selected)
            loss_suppress = F.binary_cross_entropy(inactive_selected, zeros)
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
        """Huấn luyện mô hình với Backbone MobileNetV3, Positional Encoding và Attention Mass."""
        B = inputs.shape[0]
        slots, attn_ori, resolutions = self.extract_slots(inputs)
        pred_counts = self._predict_counts(slots, attn_ori)
        gt_counts = self._normalize_targets(targets, device=inputs.device, batch_size=B)
        return self._compute_losses(
            inputs=inputs,
            pred_counts=pred_counts,
            attn_ori=attn_ori,
            resolutions=resolutions,
            gt_counts=gt_counts,
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
        pred_counts = self._predict_counts(slots, attn_ori)
        gt_counts = self._normalize_targets(targets, device=inputs.device, batch_size=B)
        loss_dict = self._compute_losses(
            inputs=inputs,
            pred_counts=pred_counts,
            attn_ori=attn_ori,
            resolutions=resolutions,
            gt_counts=gt_counts,
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
        slots, attn_ori, _ = self.extract_slots(inputs)
        return self._predict_counts(slots, attn_ori)
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