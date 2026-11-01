"""COCO / Hand Digit Object Detection Model predicting bounding boxes and class labels with FPN."""
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
from torch import Tensor
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models.feature_extraction import create_feature_extractor

from BaseModel.core import MODELS
from BaseModel.model import BaseModel


def build_2d_sincos_position_embedding(H: int, W: int, embed_dim: int, device=None, dtype=torch.float32) -> Tensor:
    """
    Tạo 2D Sin-Cos Positional Encoding hoàn toàn bằng PyTorch trên device thích hợp.
    Output shape: [1, embed_dim, H, W]
    """
    assert embed_dim % 4 == 0, f"embed_dim ({embed_dim}) phải chia hết cho 4."
    dim_per_axis = embed_dim // 2
    # Tọa độ lưới
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    grid_x = grid_x.reshape(-1)  # [H * W]
    grid_y = grid_y.reshape(-1)  # [H * W]

    def get_1d_sincos(grid: Tensor, dim: int) -> Tensor:
        omega = torch.arange(dim // 2, device=device, dtype=dtype)
        omega /= (dim / 2.0)
        omega = 1.0 / (10000.0 ** omega)  # [dim // 2]
        out = torch.einsum('n, d -> nd', grid, omega)  # [H * W, dim // 2]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)  # [H * W, dim]

    emb_x = get_1d_sincos(grid_x, dim_per_axis)  # [H * W, dim_per_axis]
    emb_y = get_1d_sincos(grid_y, dim_per_axis)  # [H * W, dim_per_axis]
    pos_emb = torch.cat([emb_x, emb_y], dim=-1)  # [H * W, embed_dim]
    # Đổi về format chuẩn ảnh: [1, embed_dim, H, W]
    pos_emb = pos_emb.view(H, W, embed_dim).permute(2, 0, 1).unsqueeze(0)
    return pos_emb


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN) đa tầng cho MobileNetV3-Small.
    Đầu vào các stage features từ MobileNetV3:
      - c0:  [B, 16, H//2, W//2]   (stride 2, ví dụ 160x160 với ảnh 320x320)
      - c1:  [B, 16, H//4, W//4]   (stride 4, ví dụ 80x80)
      - c3:  [B, 24, H//8, W//8]   (stride 8, ví dụ 40x40)
      - c5:  [B, 40, H//16, W//16] (stride 16, ví dụ 20x20)
      - c12: [B, 576, H//32, W//32] (stride 32, ví dụ 10x10)
    
    Đầu ra:
      - p5: [B, dim, H//32, W//32] (10x10) - trích xuất ngữ nghĩa toàn cục cho Slot Attention
      - map_pred (p2): [B, dim, H//4, W//4] (80x80) - feature map cho slot dự đoán
      - feat_deform (p1): [B, dim, H//2, W//2] (160x160) - feature map cho deformable attention sampling
    """
    def __init__(self, in_channels: Dict[str, int], out_dim: int):
        super().__init__()
        self.lat_c12 = nn.Conv2d(in_channels['c12'], out_dim, kernel_size=1)
        self.lat_c5 = nn.Conv2d(in_channels['c5'], out_dim, kernel_size=1)
        self.lat_c3 = nn.Conv2d(in_channels['c3'], out_dim, kernel_size=1)
        self.lat_c1 = nn.Conv2d(in_channels['c1'], out_dim, kernel_size=1)
        self.lat_c0 = nn.Conv2d(in_channels['c0'], out_dim, kernel_size=1)

        # Depthwise Separable Convolutions: Giảm ~86% FLOPs so với Standard Conv 3x3
        def build_dw_separable_conv(dim: int) -> nn.Sequential:
            return nn.Sequential(
                # Depthwise Conv 3x3
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                # Pointwise Conv 1x1
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            )

        self.smooth_p5 = build_dw_separable_conv(out_dim)
        self.smooth_p2 = build_dw_separable_conv(out_dim)
        self.smooth_p1 = build_dw_separable_conv(out_dim)

    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c0 = features['c0']
        c1 = features['c1']
        c3 = features['c3']
        c5 = features['c5']
        c12 = features['c12']

        # Top-down pathway
        p5 = self.lat_c12(c12)
        p4 = self.lat_c5(c5) + F.interpolate(p5, size=c5.shape[-2:], mode="nearest")
        p3 = self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat_c1(c1) + F.interpolate(p3, size=c1.shape[-2:], mode="nearest")
        p1 = self.lat_c0(c0) + F.interpolate(p2, size=c0.shape[-2:], mode="nearest")

        out_p5 = self.smooth_p5(p5)       # [B, dim, H//32, W//32] (10x10)
        map_pred = self.smooth_p2(p2)     # [B, dim, H//4, W//4]   (80x80)
        feat_deform = self.smooth_p1(p1)  # [B, dim, H//2, W//2]   (160x160)

        return out_p5, map_pred, feat_deform


class DeformableSlotDetectionHead(nn.Module):
    """
    Deformable Slot Detection Head:
    - map_pred (H // 4, W // 4): không gian dự đoán bounding box và class cho các slot
    - feat_deform (H // 2, W // 2): bản đồ đặc trưng chi tiết cao để lấy mẫu deformable sampling
    """
    def __init__(
        self,
        num_slots: int,
        dim: int,
        num_points: int = 4,   # 4 điểm lấy mẫu xung quanh mỗi query
        num_heads: int = 4,    # 4 attention heads
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim phải chia hết cho num_heads"
        self.num_classes = num_slots
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads

        # Module chiếu map dự đoán (H // 4, W // 4)
        self.pred_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        # ========================================================
        # 1. HỆ SỐ HỌC ĐƯỢC CHO TỪNG SLOT (SLOT-SPECIFIC EMBEDDING)
        # ========================================================
        self.slot_embed = nn.Embedding(num_slots, dim)
        self.slot_modulator = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim * 2)  # gamma, beta
        )
        # Hệ số scale kích thước hộp riêng cho từng slot (ngăn collapse w, h về 0)
        self.slot_box_scale = nn.Parameter(torch.zeros(num_slots, 2))

        # ========================================================
        # 2. DEFORMABLE ATTENTION ENGINE
        # ========================================================
        self.value_proj = nn.Linear(dim, dim)
        self.sampling_offsets = nn.Linear(dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(dim, num_heads * num_points)
        self.output_proj = nn.Linear(dim, dim)

        # ========================================================
        # 3. DECOUPLED PREDICTION HEADS (CHO TỪNG SLOT)
        # ========================================================
        self.fc_box_slots = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Linear(dim, 4)  # [delta_x, delta_y, delta_w, delta_h]
            )
            for _ in range(num_slots)
        ])

        self.fc_obj_slots = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Linear(dim, 1)  # logit
            )
            for _ in range(num_slots)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        # 1. Slot Embeddings
        nn.init.normal_(self.slot_embed.weight, std=0.02)

        # 2. Modulator
        nn.init.constant_(self.slot_modulator[-1].weight, 0.0)
        nn.init.constant_(self.slot_modulator[-1].bias, 0.0)

        # 3. Sampling Offsets: Khởi tạo các điểm lấy mẫu tỏa đều các hướng
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, 1, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= (i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))

        # 4. Attention Weights & Linear Projections
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

        # 5. Khởi tạo Box Heads (Zero-init lớp cuối để box xuất phát đúng prior)
        for box_mlp in self.fc_box_slots:
            nn.init.kaiming_uniform_(box_mlp[0].weight, a=1)
            nn.init.constant_(box_mlp[0].bias, 0.0)
            nn.init.normal_(box_mlp[-1].weight, std=1e-3)
            nn.init.constant_(box_mlp[-1].bias, 0.0)

        # 6. Khởi tạo Objectness Heads (Prior bias = -4.595 chống background flooding)
        prior_prob = 0.01
        obj_bias_val = -math.log((1.0 - prior_prob) / prior_prob)
        for obj_mlp in self.fc_obj_slots:
            nn.init.kaiming_uniform_(obj_mlp[0].weight, a=1)
            nn.init.constant_(obj_mlp[0].bias, 0.0)
            nn.init.constant_(obj_mlp[-1].weight, 0.0)
            nn.init.constant_(obj_mlp[-1].bias, obj_bias_val)

    def _get_reference_grid(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tạo hoặc lấy từ cache tọa độ reference grid chuẩn hóa [0, 1] và logits tương ứng."""
        cache_key = (H, W, device, dtype)
        if not hasattr(self, "_grid_cache_key") or self._grid_cache_key != cache_key:
            y_grid, x_grid = torch.meshgrid(
                torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device, dtype=dtype),
                torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device, dtype=dtype),
                indexing="ij"
            )
            grid = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=-1)  # [H*W, 2]
            eps = 1e-5
            grid_clamped = grid.clamp(min=eps, max=1.0 - eps)
            grid_logits = torch.log(grid_clamped / (1.0 - grid_clamped))  # [H*W, 2]
            self._cached_grid = grid
            self._cached_grid_logits = grid_logits
            self._grid_cache_key = cache_key
        return self._cached_grid, self._cached_grid_logits

    def forward(
        self,
        slot_feat: torch.Tensor,    # [B, S, dim] (đặc trưng slot sau khi gom cụm)
        map_pred: torch.Tensor,     # [B, dim, H_pred, W_pred] (map slot dự đoán, H // 4, W // 4)
        feat_deform: torch.Tensor,  # [B, dim, H_def, W_def] (feat map thực hiện deformable sampling, H // 2, W // 2)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = slot_feat.shape
        _, _, H_pred, W_pred = map_pred.shape
        _, _, H_def, W_def = feat_deform.shape
        device = slot_feat.device
        BS = B * S
        Q = H_pred * W_pred

        # 1. Modulation theo từng Slot
        slot_ids = torch.arange(S, device=device)
        slot_id_embed = self.slot_embed(slot_ids)
        combined_slot = slot_feat + slot_id_embed.unsqueeze(0)
        
        gamma, beta = self.slot_modulator(combined_slot).chunk(2, dim=-1)
        modulated_slot = (1.0 + gamma) * combined_slot + beta  # [B, S, dim]

        # 2. Xây dựng Spatial Features từ map_pred (H // 4, W // 4)
        spatial_map = self.pred_proj(map_pred)  # [B, dim, H_pred, W_pred]
        pos_emb = build_2d_sincos_position_embedding(H_pred, W_pred, self.dim, device=device, dtype=map_pred.dtype)
        spatial_tokens = (spatial_map + pos_emb).flatten(2).permute(0, 2, 1)  # [B, Q, dim]

        # 3. Tạo Query Feat: [B, S, Q, dim]
        query_feat = modulated_slot.unsqueeze(2) + spatial_tokens.unsqueeze(1)
        query_flat = query_feat.view(BS, Q, self.dim)

        # 4. Tọa độ Reference Points dạng [0, 1]: [BS, Q, 2]
        ref_grid, grid_logits = self._get_reference_grid(H_pred, W_pred, device=device, dtype=query_feat.dtype)
        ref_flat = ref_grid.view(1, 1, Q, 2).expand(B, S, -1, -1).reshape(BS, Q, 2)
        anchor_logits = grid_logits.view(1, 1, Q, 2).expand(B, S, -1, -1)  # [B, S, Q, 2]

        # 5. Biến đổi Value từ feat_deform (H // 2, W // 2): [BS * num_heads, head_dim, H_def, W_def]
        feat_deform_spatial = feat_deform.permute(0, 2, 3, 1)  # [B, H_def, W_def, dim]
        value = self.value_proj(feat_deform_spatial)
        value = value.view(B, H_def, W_def, self.num_heads, self.head_dim)
        value = value.unsqueeze(1).expand(-1, S, -1, -1, -1, -1).contiguous().view(BS, H_def, W_def, self.num_heads, self.head_dim)
        value_permuted = value.permute(0, 3, 4, 1, 2).contiguous().view(BS * self.num_heads, self.head_dim, H_def, W_def)

        # 6. Lấy mẫu Deformable Offsets & Attention Weights
        offsets = self.sampling_offsets(query_flat).view(BS, Q, self.num_heads, self.num_points, 2)
        spatial_scale = torch.tensor([1.0 / W_def, 1.0 / H_def], device=device, dtype=offsets.dtype).view(1, 1, 1, 1, 2)
        sampling_locations = ref_flat.unsqueeze(2).unsqueeze(3) + offsets * spatial_scale
        
        # Grid sample yêu cầu đoạn [-1, 1]
        sampling_grid = 2.0 * sampling_locations - 1.0
        grid_permuted = sampling_grid.permute(0, 2, 1, 3, 4).contiguous().view(
            BS * self.num_heads, Q, self.num_points, 2
        )

        attn_weights = self.attention_weights(query_flat).view(BS, Q, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Bilinear interpolation trên feat_deform
        sampled_feat = F.grid_sample(
            value_permuted,
            grid_permuted,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # [BS * num_heads, head_dim, Q, num_points]

        sampled_feat = sampled_feat.view(BS, self.num_heads, self.head_dim, Q, self.num_points)
        sampled_feat = sampled_feat.permute(0, 3, 1, 4, 2)  # [BS, Q, num_heads, num_points, head_dim]

        # Tổng hợp đặc trưng qua các điểm lấy mẫu
        out = (sampled_feat * attn_weights.unsqueeze(-1)).sum(dim=3)
        out = self.output_proj(out.view(BS, Q, self.dim)).view(B, S, Q, self.dim)

        # Đặc trưng tinh chỉnh kết hợp ngữ nghĩa slot và chi tiết điểm ảnh độ phân giải cao
        enhanced_feat = query_feat + out  # [B, S, Q, dim]

        # 7. Dự đoán song song: Box và Objectness cho từng Slot
        box_deltas_list = []
        obj_logits_list = []

        for s in range(self.num_classes):
            f_s = enhanced_feat[:, s]  # [B, Q, dim]
            box_deltas_list.append(self.fc_box_slots[s](f_s))
            obj_logits_list.append(self.fc_obj_slots[s](f_s))

        box_deltas = torch.stack(box_deltas_list, dim=1)  # [B, S, Q, 4]
        obj_logits = torch.stack(obj_logits_list, dim=1)  # [B, S, Q, 1]

        # 8. Tính tọa độ hoàn chỉnh [cx, cy, w, h] chuẩn hóa [0, 1]
        cxcy = (anchor_logits + box_deltas[..., :2]).sigmoid()

        base_scale = self.slot_box_scale.view(1, S, 1, 2)
        scale_factor = torch.exp((base_scale + box_deltas[..., 2:]).clamp(min=-4.0, max=4.0))
        wh = (0.10 * scale_factor).clamp(min=1e-3, max=1.0)
        pred_boxes = torch.cat([cxcy, wh], dim=-1)
        
        pred_boxes = pred_boxes.view(B, S, H_pred, W_pred, 4)
        obj_logits = obj_logits.squeeze(-1).view(B, S, H_pred, W_pred)

        sampling_info = {
            "sampling_locations": sampling_locations.view(B, S, H_pred, W_pred, self.num_heads, self.num_points, 2),
            "attn_weights": attn_weights.view(B, S, H_pred, W_pred, self.num_heads, self.num_points),
            "offsets": offsets.view(B, S, H_pred, W_pred, self.num_heads, self.num_points, 2),
        }
        return obj_logits, pred_boxes, sampling_info


@MODELS.register_module(name="VisDroneDetector")
class VisDroneDetector(BaseModel):
    """Detector for Hand Digit Object Detection predicting bounding boxes and class labels with FPN."""
    def __init__(
        self,         
        num_classes: int = 10,
        dim: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.slot_dim = dim
        self.num_classes = num_classes
        self.total_slots = num_classes + 1  # num_classes foreground slots + 1 background slot

        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_q = nn.Linear(dim, dim)

        # Phân phối khởi tạo học được cho từng slot
        self.mu = nn.Parameter(torch.randn(self.total_slots, dim))
        self.logsigma = nn.Parameter(torch.zeros(self.total_slots, dim))

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        self.FC1 = nn.Linear(dim, dim)
        self.FC2 = nn.Linear(dim, dim)
        self.LN = nn.LayerNorm(dim)

        # Backbone MobileNetV3-small với intermediate feature extraction
        base_backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        return_nodes = {
            'features.0': 'c0',   # stride 2  (16 channels, H//2 x W//2)
            'features.1': 'c1',   # stride 4  (16 channels, H//4 x W//4)
            'features.3': 'c3',   # stride 8  (24 channels, H//8 x W//8)
            'features.5': 'c5',   # stride 16 (40 channels, H//16 x W//16)
            'features.12': 'c12', # stride 32 (576 channels, H//32 x W//32)
        }
        self.backbone = create_feature_extractor(base_backbone, return_nodes=return_nodes)

        # FPN Module
        in_channels = {'c0': 16, 'c1': 16, 'c3': 24, 'c5': 40, 'c12': 576}
        self.fpn = FeaturePyramidNetwork(in_channels=in_channels, out_dim=dim)

        self.norm_out = nn.LayerNorm(dim)
        self.head = DeformableSlotDetectionHead(self.num_classes, self.slot_dim)

        # Initialization
        nn.init.xavier_uniform_(self.mu)
        nn.init.zeros_(self.logsigma)
        torch.manual_seed(42)

    def initialize_slots(self, batch_size=1, device=None) -> Tensor:
        if isinstance(batch_size, (torch.device, str)):
            device = batch_size
            batch_size = 1

        device = device if device is not None else self.mu.device
        mu = self.mu.to(device)
        sigma = torch.exp(self.logsigma).to(device)

        noise = torch.randn(batch_size, self.total_slots, self.slot_dim, device=device, dtype=mu.dtype)
        slots = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise
        return slots

    def positional_encoding(self, H: int, W: int, D: int) -> np.ndarray:
        grid_w = np.arange(W, dtype=np.float32)
        grid_h = np.arange(H, dtype=np.float32)

        X, Y = np.meshgrid(grid_w, grid_h)
        x_flat = X.reshape(-1)
        y_flat = Y.reshape(-1)

        emb_x = self.sincos_1D_embedding(D // 2, x_flat)
        emb_y = self.sincos_1D_embedding(D // 2, y_flat)

        return np.concatenate([emb_x, emb_y], axis=1)

    def sincos_1D_embedding(self, D: int, grid: np.ndarray) -> np.ndarray:
        omega = np.arange(D // 2, dtype=np.float32)
        omega /= (D / 2.0)
        omega = 1.0 / (10000.0 ** omega)

        out = np.einsum('m,d->md', grid, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    def extract_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Trích xuất đặc trưng đa tầng qua Backbone + FPN và cập nhật Slot Attention:
        - slots: [B, num_classes, dim] (foreground slot representations)
        - map_pred: [B, dim, H//4, W//4] (bản đồ dự đoán các slot)
        - feat_deform: [B, dim, H//2, W//2] (bản đồ cho deformable attention sampling)
        - p5: [B, dim, H//32, W//32] (bản đồ ngữ nghĩa toàn cục)
        - attn: [B, total_slots, N] (slot attention weights trên p5)
        """
        features = self.backbone(x)
        p5, map_pred, feat_deform = self.fpn(features)

        B, D, H5, W5 = p5.shape
        x_sem = p5.permute(0, 2, 3, 1).reshape(B, H5 * W5, D)  # [B, N, D]

        # Positional Encoding với caching để tránh CPU-GPU transfer liên tục
        cache_key = (H5, W5, D, x_sem.device, x_sem.dtype)
        if not hasattr(self, "_pos_cache_key") or self._pos_cache_key != cache_key:
            pos_np = self.positional_encoding(H5, W5, D)
            self._pos_cache = torch.from_numpy(pos_np).to(dtype=x_sem.dtype, device=x_sem.device)
            self._pos_cache_key = cache_key
        position = self._pos_cache

        x_sem = x_sem + position  # [B, N, D]
        x_sem = self.LN(x_sem)
        x_sem = self.FC1(x_sem)
        x_sem = F.relu(x_sem)
        x_sem = self.FC2(x_sem)

        k = self.to_k(x_sem)  # [B, N, dim]
        v = self.to_v(x_sem)  # [B, N, dim]
        slots = self.initialize_slots(batch_size=B, device=x_sem.device)  # [B, total_slots, dim]
        q = self.to_q(slots)  # [B, total_slots, dim]

        dot = torch.einsum("b s d, b p d -> b s p", q, k)  # [B, total_slots, N]
        dot = dot / (self.slot_dim ** 0.5)

        attn = torch.softmax(dot, dim=1)  # [B, total_slots, N]
        attn = attn / attn.sum(dim=-1, keepdim=True)
        x_update = torch.einsum("b s p, b p d -> b s d", attn, v)  # [B, total_slots, dim]
        x_update = self.fc2(F.relu(self.fc1(self.norm_out(x_update))))
        slots = slots + x_update
        slots_fg = slots[:, :self.num_classes, :]  # Chỉ lấy class foreground slots [B, num_classes, dim]

        return slots_fg, map_pred, feat_deform, p5, attn

    def slot_orthogonality_loss(self, slots: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Loss ép cosine similarity giữa các cặp slot khác nhau về 0."""
        B, K, D = slots.shape
        slots_norm = F.normalize(slots, p=2, dim=-1, eps=eps)  # [B, K, D]
        cos_sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # [B, K, K]
        eye = torch.eye(K, device=slots.device).unsqueeze(0)  # [1, K, K]
        off_diagonal = cos_sim - eye
        if K > 1:
            loss = (off_diagonal ** 2).sum(dim=(-2, -1)) / (K * (K - 1))
            return loss.mean()
        return torch.tensor(0.0, device=slots.device)

    def _get_crop_grid(self, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache tọa độ lưới cho binary crop mask để tránh tạo meshgrid liên tục."""
        if not hasattr(self, "_cached_crop_grids"):
            self._cached_crop_grids = {}
        key = (H, W, device)
        if key not in self._cached_crop_grids:
            y_coords = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
            x_coords = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            self._cached_crop_grids[key] = (grid_y.unsqueeze(0), grid_x.unsqueeze(0))
        return self._cached_crop_grids[key]

    def create_crop_mask_single_image(
        self,
        image: torch.Tensor, 
        boxes_norm: torch.Tensor, 
        combine: bool = False
    ) -> torch.Tensor:
        """Tạo binary mask cho 1 ảnh với N bounding boxes dạng [x1, y1, w, h] chuẩn hóa [0, 1]."""
        C, H, W = image.shape
        N = boxes_norm.shape[0]
        device = image.device

        if N == 0:
            if combine:
                return torch.zeros((H, W), dtype=torch.bool, device=device)
            return torch.zeros((0, H, W), dtype=torch.bool, device=device)

        x1, y1, bw, bh = boxes_norm.unbind(dim=-1)
        x2 = x1 + bw
        y2 = y1 + bh

        x1 = x1.view(N, 1, 1)
        y1 = y1.view(N, 1, 1)
        x2 = x2.view(N, 1, 1)
        y2 = y2.view(N, 1, 1)

        grid_y, grid_x = self._get_crop_grid(H, W, device)

        mask = (grid_x >= x1) & (grid_x < x2) & (grid_y >= y1) & (grid_y < y2)
        if combine:
            mask = mask.any(dim=0)
        return mask

    def group_boxes_by_class(
        self,
        gt_boxes: torch.Tensor, 
        gt_labels: torch.Tensor, 
        num_classes: int
    ) -> List[torch.Tensor]:
        device = gt_boxes.device
        if gt_boxes.numel() == 0 or gt_labels.numel() == 0:
            return [torch.empty((0, 4), device=device, dtype=gt_boxes.dtype) for _ in range(num_classes)]

        grouped_boxes = []
        for c in range(num_classes):
            mask = (gt_labels == c)
            grouped_boxes.append(gt_boxes[mask])
        return grouped_boxes

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Chuyển đổi boxes từ [cx, cy, w, h] sang [x1, y1, x2, y2]."""
        if boxes.numel() == 0:
            return torch.empty((0, 4), dtype=boxes.dtype, device=boxes.device)
        cx = boxes[..., 0]
        cy = boxes[..., 1]
        w = boxes[..., 2]
        h = boxes[..., 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Chuyển đổi boxes từ [x1, y1, w, h] sang [x1, y1, x2, y2]."""
        if boxes.numel() == 0:
            return torch.empty((0, 4), dtype=boxes.dtype, device=boxes.device)
        x1 = boxes[..., 0]
        y1 = boxes[..., 1]
        w = boxes[..., 2]
        h = boxes[..., 3]
        return torch.stack([x1, y1, x1 + w, y1 + h], dim=-1)

    @staticmethod
    def _xywh_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
        """Chuyển đổi boxes từ [x1, y1, w, h] sang [cx, cy, w, h]."""
        if boxes.numel() == 0:
            return torch.empty((0, 4), dtype=boxes.dtype, device=boxes.device)
        x1 = boxes[..., 0]
        y1 = boxes[..., 1]
        w = boxes[..., 2]
        h = boxes[..., 3]
        return torch.stack([x1 + w / 2.0, y1 + h / 2.0, w, h], dim=-1)

    def _compute_losses(
        self, 
        cls_logits: torch.Tensor, 
        pred_boxes: torch.Tensor, 
        targets: Optional[List[Dict[str, torch.Tensor]]], 
        device: torch.device,
        slots: Optional[torch.Tensor] = None,
        feat_map: Optional[torch.Tensor] = None,
        attn_map: Optional[torch.Tensor] = None,
        sampling_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tính toán Gaussian Focal Loss (cls), DIoU Loss (box), Deformable Sampling Loss và Orthogonality Loss (Vectorized)."""
        batch_size, num_classes, grid_h, grid_w = cls_logits.shape

        total_box_loss = torch.tensor(0.0, device=device)
        total_deform_in_box = torch.tensor(0.0, device=device)
        total_map_loss = torch.tensor(0.0, device=device)
        total_slot_correlation_loss = torch.tensor(0.0, device=device)
        total_slot_attn_loss = torch.tensor(0.0, device=device)
        total_bg_loss = torch.tensor(0.0, device=device)
        num_pos_total = 0

        # Khởi tạo center mask cho toàn bộ batch [B, num_classes, H, W]
        cls_center = torch.zeros((batch_size, num_classes, grid_h, grid_w), device=device, dtype=cls_logits.dtype)

        for b in range(batch_size):
            img_box = pred_boxes[b]      # [num_classes, grid_h, grid_w, 4]
            target_dict = targets[b] if targets is not None and b < len(targets) else None
            gt_boxes = target_dict.get("boxes", torch.empty((0, 4), device=device)) if target_dict else torch.empty((0, 4), device=device)
            gt_labels = target_dict.get("labels", torch.empty((0,), device=device, dtype=torch.long)) if target_dict else torch.empty((0,), device=device, dtype=torch.long)
            num_gt = len(gt_boxes)

            if attn_map is not None:
                boxes_per_class = self.group_boxes_by_class(gt_boxes, gt_labels, num_classes=self.num_classes)
                for s in range(self.num_classes):
                    N = len(boxes_per_class[s])
                    if N > 0:
                        attn_s_mask = self.create_crop_mask_single_image(attn_map[b, s].unsqueeze(0), boxes_per_class[s].to(device))
                        attn_s_map = attn_map[b, s].unsqueeze(0).expand(N, -1, -1) * attn_s_mask.to(attn_map.dtype)
                        norm = torch.sqrt(torch.sum(attn_s_map ** 2, dim=(-2, -1)) + 1e-8)
                        total_slot_attn_loss += torch.sqrt((N - norm.sum())**2) / N
                        total_slot_correlation_loss += norm.numel()**2 * torch.var(norm, unbiased=False)

            if num_gt > 0:
                gt_boxes = gt_boxes.to(device)
                gt_labels = gt_labels.to(device)

                is_cxcywh = target_dict.get("_is_cxcywh", False) if target_dict else False
                if is_cxcywh:
                    gt_cx = gt_boxes[:, 0].clamp(0.0, 0.999)
                    gt_cy = gt_boxes[:, 1].clamp(0.0, 0.999)
                    gt_xyxy = self._cxcywh_to_xyxy(gt_boxes)
                else:
                    # Dataset cung cấp định dạng [x1, y1, w, h] chuẩn hóa [0, 1]
                    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2] / 2.0).clamp(0.0, 0.999)
                    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3] / 2.0).clamp(0.0, 0.999)
                    gt_xyxy = self._xywh_to_xyxy(gt_boxes)

                grid_x = (gt_cx * grid_w).long().clamp(0, grid_w - 1)
                grid_y = (gt_cy * grid_h).long().clamp(0, grid_h - 1)

                valid_mask = (gt_labels >= 0) & (gt_labels < num_classes)
                valid_labels = gt_labels[valid_mask]
                grid_x = grid_x[valid_mask]
                grid_y = grid_y[valid_mask]
                gt_xyxy = gt_xyxy[valid_mask]

                if feat_map is not None and attn_map is not None:
                    _, d, _, _ = feat_map.shape
                    mask = self.create_crop_mask_single_image(feat_map[b], gt_boxes, combine=True).unsqueeze(0)
                    feat_obj = feat_map[b][mask.expand(d, -1, -1)]
                    attn_bg_map = attn_map[b, self.num_classes].unsqueeze(0)
                    attn_bg_map = attn_bg_map[~mask]

                    total_bg_loss += (torch.sqrt(torch.sum(attn_map[b, self.num_classes] ** 2, dim=(-2, -1)) + 1e-8) - torch.sqrt(torch.sum(attn_bg_map ** 2, dim=(-1)) + 1e-8))
                    total_map_loss += (torch.sqrt(torch.sum(feat_map[b] ** 2, dim=(-3, -2, -1)) + 1e-8) - torch.sqrt(torch.sum(feat_obj ** 2, dim=(-1)) + 1e-8)) * 0.01

                num_pos = len(valid_labels)
                num_pos_total += num_pos
                if num_pos > 0:
                    cls_center[b, valid_labels, grid_y, grid_x] = 1.0

                    p_box = img_box[valid_labels, grid_y, grid_x]  # [num_pos, 4] dạng [cx, cy, w, h]
                    p_xyxy = self._cxcywh_to_xyxy(p_box)

                    box_loss = torchvision.ops.distance_box_iou_loss(
                        p_xyxy, gt_xyxy, reduction="sum"
                    )
                    total_box_loss += box_loss

                    # ========================================================
                    # LOSS LẤY MẪU DEFORMABLE: RÀNG BUỘC IN-BOX ALIGNMENT
                    # ========================================================
                    if sampling_info is not None:
                        # Điểm lấy mẫu tại các vị trí GT positive của slot tương ứng
                        b_locs = sampling_info["sampling_locations"][b]  # [S, H, W, nh, np, 2]
                        b_attn = sampling_info["attn_weights"][b]        # [S, H, W, nh, np]
                        pos_pts = b_locs[valid_labels, grid_y, grid_x]   # [num_pos, nh, np, 2]
                        pos_w = b_attn[valid_labels, grid_y, grid_x]     # [num_pos, nh, np]

                        px = pos_pts[..., 0]  # [num_pos, nh, np]
                        py = pos_pts[..., 1]  # [num_pos, nh, np]

                        x1 = gt_xyxy[:, 0].view(num_pos, 1, 1)
                        y1 = gt_xyxy[:, 1].view(num_pos, 1, 1)
                        x2 = gt_xyxy[:, 2].view(num_pos, 1, 1)
                        y2 = gt_xyxy[:, 3].view(num_pos, 1, 1)

                        # Khoảng cách vi phạm ra ngoài GT Box
                        dx = F.relu(x1 - px) + F.relu(px - x2)
                        dy = F.relu(y1 - py) + F.relu(py - y2)
                        in_box_dist = dx + dy  # [num_pos, nh, np]

                        # Nhân trọng số attention của từng điểm lấy mẫu
                        weighted_dist = (in_box_dist * pos_w).sum(dim=(-2, -1))  # [num_pos]
                        total_deform_in_box += weighted_dist.sum()

        # ========================================================
        # 1. VECTORIZED GAUSSIAN HEATMAP TARGET GENERATION
        # ========================================================
        gaussian_k = torch.tensor([
            [0.3679, 0.6065, 0.3679],
            [0.6065, 1.0000, 0.6065],
            [0.3679, 0.6065, 0.3679]
        ], device=device, dtype=cls_logits.dtype)
        padded = F.pad(cls_center, (1, 1, 1, 1), mode="constant", value=0.0)
        shifts = [
            padded[:, :, dy:dy+grid_h, dx:dx+grid_w] * gaussian_k[dy, dx]
            for dy in range(3) for dx in range(3)
        ]
        cls_target = torch.stack(shifts, dim=0).amax(dim=0)  # [B, num_classes, grid_h, grid_w]

        # ========================================================
        # 2. VECTORIZED DENSE GAUSSIAN FOCAL LOSS (NO DYNAMIC INDEXING)
        # ========================================================
        alpha = 0.25
        probs = cls_logits.sigmoid().clamp(min=1e-6, max=1.0 - 1e-6)
        pos_mask = (cls_target >= 0.99).float()
        neg_mask = 1.0 - pos_mask

        pos_loss = - alpha * ((1.0 - probs) ** 2.0) * torch.log(probs) * pos_mask
        neg_weight = (1.0 - alpha) * ((1.0 - cls_target) ** 4.0) * (probs ** 2.0)
        neg_loss = - neg_weight * torch.log(1.0 - probs) * neg_mask

        total_cls_pos = pos_loss.sum()
        total_cls_neg = neg_loss.sum()
        total_cls_loss = total_cls_pos + total_cls_neg

        with torch.no_grad():
            prob_pos_sum = (probs * pos_mask).sum()
            true_bg_mask = cls_target < 0.05
            prob_neg_max = probs[true_bg_mask].max() if true_bg_mask.any() else torch.tensor(0.0, device=device)

        # Đồng bộ số lượng positive targets qua các GPU trong Distributed / Accelerate Training
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            num_pos_tensor = torch.tensor([num_pos_total], dtype=torch.float32, device=device)
            torch.distributed.all_reduce(num_pos_tensor, op=torch.distributed.ReduceOp.SUM)
            world_size = torch.distributed.get_world_size()
            norm_pos = max(float(num_pos_tensor.item()) / float(world_size), 1.0)
        else:
            norm_pos = max(float(num_pos_total), 1.0)

        total_cls_loss = total_cls_loss / norm_pos
        total_map_loss = total_map_loss / norm_pos
        total_slot_attn_loss = total_slot_attn_loss / norm_pos
        total_slot_correlation_loss = total_slot_correlation_loss / norm_pos
        total_bg_loss = total_bg_loss / norm_pos
        total_box_loss = (total_box_loss / norm_pos) if num_pos_total > 0 else (total_box_loss * 0.0)

        # ========================================================
        # DEFORMABLE SAMPLING LOSSES (IN-BOX, BOUNDARY, OFFSET REG)
        # ========================================================
        total_deform_in_box = (total_deform_in_box / norm_pos) if num_pos_total > 0 else (total_deform_in_box * 0.0)
        total_deform_bound = torch.tensor(0.0, device=device)
        total_deform_reg = torch.tensor(0.0, device=device)

        if sampling_info is not None:
            all_locs = sampling_info["sampling_locations"]  # [B, S, H, W, nh, np, 2]
            all_offsets = sampling_info["offsets"]          # [B, S, H, W, nh, np, 2]

            # 1. Boundary penalty: phạt các điểm lấy mẫu bay ra ngoài biên ảnh [0, 1]
            bound_violation = (
                F.relu(-all_locs[..., 0]) + F.relu(all_locs[..., 0] - 1.0) +
                F.relu(-all_locs[..., 1]) + F.relu(all_locs[..., 1] - 1.0)
            )
            total_deform_bound = bound_violation.mean()

            # 2. Offset Regularization: kiểm soát độ lớn offsets tránh dạt trôi
            total_deform_reg = (all_offsets ** 2).mean()

        loss_deform = 1.0 * total_deform_in_box + 0.2 * total_deform_bound + 0.01 * total_deform_reg

        ortho_loss = torch.tensor(0.0, device=device)
        if slots is not None:
            ortho_loss = self.slot_orthogonality_loss(slots)

        # Đảm bảo 100% decoupled box prediction heads và slot_box_scale luôn kết nối vào computation graph
        # giúp tránh lỗi unused parameter reduction timing mismatch khi chạy DDP / Accelerate
        dummy_box_reg = 0.0 * (pred_boxes.sum() + self.head.slot_box_scale.sum())

        total_loss = (
            2.0 * total_cls_loss 
            + 1.0 * total_box_loss 
            + 0.5 * loss_deform
            + 0.1 * ortho_loss 
            + 0.1 * (total_map_loss + total_slot_attn_loss + total_slot_correlation_loss + total_bg_loss)
            + dummy_box_reg
        )

        return {
            "loss": total_loss,
            "loss_cls": total_cls_loss,
            "loss_box": total_box_loss,
            "loss_deform": loss_deform,
            "loss_deform_in_box": total_deform_in_box,
            "loss_deform_bound": total_deform_bound,
            "loss_deform_reg": total_deform_reg,
            "loss_ortho": ortho_loss,
            "loss_cls_pos": (total_cls_pos / norm_pos).detach(),
            "loss_cls_neg": (total_cls_neg / norm_pos).detach(),
            "loss_prob_pos": (prob_pos_sum / norm_pos).detach(),
            "loss_prob_neg_max": prob_neg_max.detach(),
            "total_map_loss": total_map_loss,
            "total_slot_attn_loss": total_slot_attn_loss,
            "total_slot_correlation_loss": total_slot_correlation_loss,
            "total_bg_loss": total_bg_loss,
        }

    def _post_process(
        self,
        cls_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        conf_threshold: float = 0.10,
        nms_threshold: float = 0.5,
        max_det: int = 100
    ) -> List[Dict[str, torch.Tensor]]:
        """Lọc ngưỡng vector hóa với CenterNet Local Peak Max Pooling và Batched NMS tối ưu tốc độ."""
        batch_size, num_classes, grid_h, grid_w = cls_logits.shape
        cls_scores = cls_logits.sigmoid()
        # 1. Local Peak Max Pooling (3x3): Giữ lại đỉnh cực đại cục bộ, loại bỏ ngay các pixel lân cận sườn đồi
        hmax = F.max_pool2d(cls_scores, kernel_size=3, stride=1, padding=1)
        cls_scores = cls_scores * (cls_scores == hmax).float()

        device = cls_logits.device
        results = []

        for b in range(batch_size):
            b_scores_map = cls_scores[b]  # [C, H, W]
            c_idx, y_idx, x_idx = torch.where(b_scores_map >= conf_threshold)

            if len(c_idx) > 0:
                b_scores = b_scores_map[c_idx, y_idx, x_idx]
                b_labels = c_idx

                # Giới hạn Top-1000 candidates trước khi decode box và chạy NMS
                if len(b_scores) > 1000:
                    topk_idx = torch.topk(b_scores, 1000).indices
                    b_scores = b_scores[topk_idx]
                    b_labels = b_labels[topk_idx]
                    y_idx = y_idx[topk_idx]
                    x_idx = x_idx[topk_idx]

                # Lấy đúng box dự đoán [cx, cy, w, h] cho từng class candidate tại vị trí tương ứng
                sample_boxes = pred_boxes[b, b_labels, y_idx, x_idx]  # [M, 4]

                if nms_threshold is not None and len(sample_boxes) > 0:
                    xyxy = self._cxcywh_to_xyxy(sample_boxes)
                    keep = torchvision.ops.batched_nms(xyxy, b_scores, b_labels, nms_threshold)
                    keep = keep[:max_det]
                    sample_boxes = sample_boxes[keep]
                    b_scores = b_scores[keep]
                    b_labels = b_labels[keep]

                results.append({
                    "boxes": sample_boxes,
                    "scores": b_scores,
                    "labels": b_labels,
                })
            else:
                results.append({
                    "boxes": torch.empty((0, 4), device=device),
                    "scores": torch.empty((0,), device=device),
                    "labels": torch.empty((0,), dtype=torch.long, device=device),
                })
        return results

    def forward_train(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Tính toán Detection Loss trong quá trình huấn luyện với FPN và Deformable Sampling Loss."""
        slots, map_pred, feat_deform, p5, attn = self.extract_features(inputs)
        cls_logits, pred_boxes, sampling_info = self.head(slots, map_pred, feat_deform)

        B, _, H5, W5 = p5.shape
        attn_map = attn.view(B, self.total_slots, H5, W5)

        return self._compute_losses(
            cls_logits, pred_boxes, targets,
            device=inputs.device, slots=slots,
            feat_map=p5, attn_map=attn_map,
            sampling_info=sampling_info,
        )

    def predict(
        self,
        inputs: torch.Tensor,
        conf_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        max_det: int = 100,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        """Dự đoán bounding box, class label và confidence score khi suy luận / đánh giá."""
        slots, map_pred, feat_deform, p5, attn = self.extract_features(inputs)
        cls_logits, pred_boxes, _ = self.head(slots, map_pred, feat_deform)

        return self._post_process(
            cls_logits, pred_boxes, conf_threshold=conf_threshold, nms_threshold=nms_threshold, max_det=max_det
        )

    def forward_val(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        max_det: int = 300,
        **kwargs,
    ) -> Dict[str, Any]:
        """Validation forward pass: tính validation losses và trích xuất preds phục vụ tính mAP."""
        slots, map_pred, feat_deform, p5, attn = self.extract_features(inputs)
        cls_logits, pred_boxes, sampling_info = self.head(slots, map_pred, feat_deform)

        output = {}
        if targets is not None:
            B, _, H5, W5 = p5.shape
            attn_map = attn.view(B, self.total_slots, H5, W5)
            loss_dict = self._compute_losses(
                cls_logits, pred_boxes, targets,
                device=inputs.device, slots=slots,
                feat_map=p5, attn_map=attn_map,
                sampling_info=sampling_info,
            )
            output.update(loss_dict)

            # Chuẩn hóa target boxes sang định dạng [cx, cy, w, h] cho MeanAveragePrecision
            for t in targets:
                if "boxes" in t and len(t["boxes"]) > 0:
                    if not t.get("_is_cxcywh", False):
                        b = t["boxes"]
                        t["boxes"] = torch.stack([
                            b[:, 0] + b[:, 2] / 2.0,
                            b[:, 1] + b[:, 3] / 2.0,
                            b[:, 2],
                            b[:, 3]
                        ], dim=-1)
                        t["_is_cxcywh"] = True

        preds = self._post_process(
            cls_logits, pred_boxes,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            max_det=max_det,
        )
        output["preds"] = preds
        return output

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ):
        """Standard nn.Module forward dispatching to forward_train or forward_val."""
        if self.training:
            return self.forward_train(inputs, targets=targets, **kwargs)
        else:
            return self.forward_val(inputs, targets=targets, **kwargs)

    def analyze_detection_components(self, inputs: torch.Tensor) -> Dict[str, float]:
        """Thống kê chi tiết toàn bộ các chỉ số đặc trưng, slots, attention và đầu ra phân phối phục vụ ExcelTracker."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            slots, map_pred, feat_deform, p5, attn = self.extract_features(inputs)
            cls_logits, pred_boxes, sampling_info = self.head(slots, map_pred, feat_deform)
            cls_scores = cls_logits.sigmoid()

            # Slot Cosine Similarity (độ trực giao giữa các cặp slot)
            slots_norm = F.normalize(slots, p=2, dim=-1)
            cos_matrix = torch.bmm(slots_norm, slots_norm.transpose(1, 2))
            K = slots.shape[1]
            eye = torch.eye(K, device=slots.device).unsqueeze(0)
            off_diag = (cos_matrix - eye).abs()
            avg_cos_sim = off_diag.sum() / max(slots.shape[0] * K * (K - 1), 1)

            # Attention Map Entropy (đo độ sắc nét / phân tán của attention weights trên p5)
            attn_fg = attn[:, :self.num_classes, :]  # [B, num_classes, N]
            attn_clamp = attn_fg.clamp(min=1e-8)
            entropy = -(attn_clamp * torch.log(attn_clamp)).sum(dim=-1).mean()

            # Deformable Sampling Offsets & Boundary Analysis
            all_locs = sampling_info["sampling_locations"]
            all_offsets = sampling_info["offsets"]
            offset_norm = torch.norm(all_offsets, p=2, dim=-1).mean()
            bound_violation = (
                F.relu(-all_locs[..., 0]) + F.relu(all_locs[..., 0] - 1.0) +
                F.relu(-all_locs[..., 1]) + F.relu(all_locs[..., 1] - 1.0)
            ).gt(0).float().mean()

            results = {
                # 1. Phân phối phân loại (Classification distribution)
                "Cls_Logits_Mean": round(cls_logits.mean().item(), 4),
                "Cls_Logits_Max": round(cls_logits.max().item(), 4),
                "Cls_Logits_Min": round(cls_logits.min().item(), 4),
                "Cls_Prob_Mean": round(cls_scores.mean().item(), 4),
                "Cls_Prob_Max": round(cls_scores.max().item(), 4),

                # 2. Phân phối Bounding Box (Box geometry distribution)
                "Box_CX_Mean": round(pred_boxes[..., 0].mean().item(), 4),
                "Box_CY_Mean": round(pred_boxes[..., 1].mean().item(), 4),
                "Box_W_Mean": round(pred_boxes[..., 2].mean().item(), 4),
                "Box_H_Mean": round(pred_boxes[..., 3].mean().item(), 4),

                # 3. Phân tích Slot, Attention Dynamics & FPN Feature Energies
                "Slot_Norm_Mean": round(slots.norm(dim=-1).mean().item(), 4),
                "Slot_Cosine_Sim_Mean": round(avg_cos_sim.item(), 4),
                "Attn_Entropy_Mean": round(entropy.item(), 4),
                "P5_Energy": round(p5.norm(dim=1).mean().item(), 4),
                "PredMap_Energy": round(map_pred.norm(dim=1).mean().item(), 4),
                "DeformFeat_Energy": round(feat_deform.norm(dim=1).mean().item(), 4),

                # 4. Phân tích Lấy mẫu Deformable Attention
                "Deform_Offset_Norm": round(offset_norm.item(), 4),
                "Deform_Out_Bound_Ratio": round(bound_violation.item(), 4),
            }
        if was_training:
            self.train()
        return results
