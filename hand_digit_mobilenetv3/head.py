"""Detection Heads: Standard Pure CNN Head and Deformable Attention Detection Head."""
from typing import Tuple, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from BaseModel.core.registrar import HEADS


@HEADS.register_module(name="MobileNetDetectionHead")
class MobileNetDetectionHead(nn.Module):
    """
    Standard Anchor-Free / CenterNet Detection Head:
    - Nhận feat_map: [B, dim, H, W]
    - Nhánh cls_head: Dự đoán classification heatmap [B, num_classes, H, W]
    - Nhánh box_head: Dự đoán [cx, cy, w, h] chuẩn hóa [0, 1], shape [B, 4, H, W]
    """

    def __init__(
        self,
        num_classes: int = 10,
        dim: int = 64,
        head_conv: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim

        self.cls_head = nn.Sequential(
            nn.Conv2d(dim, head_conv, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, kernel_size=1),
        )

        self.box_head = nn.Sequential(
            nn.Conv2d(dim, head_conv, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 4, kernel_size=1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.box_head[0].weight, a=1)
        nn.init.normal_(self.box_head[-1].weight, std=1e-3)
        nn.init.constant_(self.box_head[-1].bias, 0.0)

        prior_prob = 0.01
        cls_bias_val = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.kaiming_uniform_(self.cls_head[0].weight, a=1)
        nn.init.constant_(self.cls_head[-1].weight, 0.0)
        nn.init.constant_(self.cls_head[-1].bias, cls_bias_val)

    def _get_reference_grid(
        self, H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (H, W, device, dtype)
        if not hasattr(self, "_grid_cache_key") or self._grid_cache_key != cache_key:
            y_grid, x_grid = torch.meshgrid(
                torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device, dtype=dtype),
                torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device, dtype=dtype),
                indexing="ij"
            )
            grid = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0)
            eps = 1e-5
            grid_clamped = grid.clamp(min=eps, max=1.0 - eps)
            grid_logits = torch.log(grid_clamped / (1.0 - grid_clamped))
            self._cached_grid = grid
            self._cached_grid_logits = grid_logits
            self._grid_cache_key = cache_key
        return self._cached_grid, self._cached_grid_logits

    def forward(
        self, feat_map: torch.Tensor, feat_deform: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, _, H, W = feat_map.shape
        cls_logits = self.cls_head(feat_map)
        box_deltas = self.box_head(feat_map)

        _, anchor_logits = self._get_reference_grid(H, W, device=feat_map.device, dtype=feat_map.dtype)
        anchor_logits = anchor_logits.expand(B, -1, -1, -1)

        cxcy = (anchor_logits + box_deltas[:, :2]).sigmoid()
        scale_factor = torch.exp(box_deltas[:, 2:].clamp(min=-4.0, max=4.0))
        wh = (0.12 * scale_factor).clamp(min=1e-3, max=1.0)

        pred_boxes = torch.cat([cxcy, wh], dim=1)
        return cls_logits, pred_boxes, None


@HEADS.register_module(name="DeformableHead")
@HEADS.register_module(name="DeformableDetectionHead")
class DeformableHead(nn.Module):
    """
    Deformable Detection Head cho MobileNetV3:
    Sử dụng Multi-Head Deformable Attention để lấy mẫu đặc trưng chi tiết thích ứng hình dạng đối tượng:
      - feat_map: [B, dim, H_pred, W_pred] (Stride 8) - Spatial Query Tokens
      - feat_deform: [B, dim, H_def, W_def] (Stride 2) - High-resolution Feature Map để lấy mẫu
      - cls_head: Dự đoán classification heatmap [B, num_classes, H, W]
      - box_head: Dự đoán [cx, cy, w, h] chuẩn hóa [0, 1]
    """

    def __init__(
        self,
        num_classes: int = 10,
        dim: int = 64,
        num_heads: int = 4,
        num_points: int = 4,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) phải chia hết cho num_heads ({num_heads})."
        self.num_classes = num_classes
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads

        # 1. Spatial Query Projection
        self.query_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        # 2. Deformable Attention Engine
        self.value_proj = nn.Linear(dim, dim)
        self.sampling_offsets = nn.Linear(dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(dim, num_heads * num_points)
        self.output_proj = nn.Linear(dim, dim)

        # 3. Prediction Heads
        self.cls_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, num_classes, kernel_size=1),
        )

        self.box_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 4, kernel_size=1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        # 1. Sampling Offsets tỏa đều các hướng
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

        # 2. Attention Weights & Linear Projections
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

        # 3. Box Head Init
        nn.init.kaiming_uniform_(self.box_head[0].weight, a=1)
        nn.init.normal_(self.box_head[-1].weight, std=1e-3)
        nn.init.constant_(self.box_head[-1].bias, 0.0)

        # 4. Cls Head Init (Focal Prior p=0.01)
        prior_prob = 0.01
        cls_bias_val = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.kaiming_uniform_(self.cls_head[0].weight, a=1)
        nn.init.constant_(self.cls_head[-1].weight, 0.0)
        nn.init.constant_(self.cls_head[-1].bias, cls_bias_val)

    def _get_reference_grid(
        self, H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (H, W, device, dtype)
        if not hasattr(self, "_grid_cache_key") or self._grid_cache_key != cache_key:
            y_grid, x_grid = torch.meshgrid(
                torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device, dtype=dtype),
                torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device, dtype=dtype),
                indexing="ij"
            )
            # grid shape: [H * W, 2]
            grid = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=-1)
            eps = 1e-5
            grid_clamped = grid.clamp(min=eps, max=1.0 - eps)
            grid_logits = torch.log(grid_clamped / (1.0 - grid_clamped))
            self._cached_grid = grid
            self._cached_grid_logits = grid_logits
            self._grid_cache_key = cache_key
        return self._cached_grid, self._cached_grid_logits

    def forward(
        self,
        feat_map: torch.Tensor,
        feat_deform: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if feat_deform is None:
            feat_deform = feat_map

        B, C, H_pred, W_pred = feat_map.shape
        _, _, H_def, W_def = feat_deform.shape
        device = feat_map.device
        Q = H_pred * W_pred

        # 1. Spatial Queries từ feat_map
        query_map = self.query_proj(feat_map)                           # [B, dim, H_pred, W_pred]
        query_tokens = query_map.flatten(2).permute(0, 2, 1)            # [B, Q, dim]

        # 2. Reference Grid [0, 1]
        ref_grid, grid_logits = self._get_reference_grid(
            H_pred, W_pred, device=device, dtype=query_tokens.dtype
        )
        ref_flat = ref_grid.view(1, Q, 2).expand(B, -1, -1)             # [B, Q, 2]
        anchor_logits = grid_logits.view(1, 2, H_pred, W_pred).expand(B, -1, -1, -1)

        # 3. Chuẩn bị Value Tensor từ feat_deform
        # [B, H_def, W_def, dim] -> [B, H_def, W_def, num_heads, head_dim]
        feat_def_spatial = feat_deform.permute(0, 2, 3, 1)
        value = self.value_proj(feat_def_spatial)
        value = value.view(B, H_def, W_def, self.num_heads, self.head_dim)
        value_permuted = value.permute(0, 3, 4, 1, 2).contiguous().view(
            B * self.num_heads, self.head_dim, H_def, W_def
        )

        # 4. Offsets & Attention Weights
        offsets = self.sampling_offsets(query_tokens).view(B, Q, self.num_heads, self.num_points, 2)
        spatial_scale = torch.tensor([1.0 / W_def, 1.0 / H_def], device=device, dtype=offsets.dtype).view(1, 1, 1, 1, 2)
        sampling_locations = ref_flat.unsqueeze(2).unsqueeze(3) + offsets * spatial_scale

        # Grid sample đoạn [-1, 1]
        sampling_grid = 2.0 * sampling_locations - 1.0
        grid_permuted = sampling_grid.permute(0, 2, 1, 3, 4).contiguous().view(
            B * self.num_heads, Q, self.num_points, 2
        )

        attn_weights = self.attention_weights(query_tokens).view(B, Q, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Bilinear sampling trên feat_deform
        sampled_feat = F.grid_sample(
            value_permuted,
            grid_permuted,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # [B * num_heads, head_dim, Q, num_points]

        sampled_feat = sampled_feat.view(B, self.num_heads, self.head_dim, Q, self.num_points)
        sampled_feat = sampled_feat.permute(0, 3, 1, 4, 2)  # [B, Q, num_heads, num_points, head_dim]

        # Tổng hợp đặc trưng deformable
        out = (sampled_feat * attn_weights.unsqueeze(-1)).sum(dim=3)
        out = self.output_proj(out.view(B, Q, self.dim))  # [B, Q, dim]
        out = out.permute(0, 2, 1).view(B, self.dim, H_pred, W_pred)

        # Residual enhancement
        enhanced_feat = feat_map + out  # [B, dim, H_pred, W_pred]

        # 5. Dự đoán Heatmap & Box
        cls_logits = self.cls_head(enhanced_feat)
        box_deltas = self.box_head(feat_map)

        cxcy = (anchor_logits + box_deltas[:, :2]).sigmoid()
        scale_factor = torch.exp(box_deltas[:, 2:].clamp(min=-4.0, max=4.0))
        wh = (0.12 * scale_factor).clamp(min=1e-3, max=1.0)

        pred_boxes = torch.cat([cxcy, wh], dim=1)  # [B, 4, H_pred, W_pred]

        sampling_info = {
            "sampling_locations": sampling_locations.view(B, H_pred, W_pred, self.num_heads, self.num_points, 2),
            "attn_weights": attn_weights.view(B, H_pred, W_pred, self.num_heads, self.num_points),
            "offsets": offsets.view(B, H_pred, W_pred, self.num_heads, self.num_points, 2),
        }
        return cls_logits, pred_boxes, sampling_info
