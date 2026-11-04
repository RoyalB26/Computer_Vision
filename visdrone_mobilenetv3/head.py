"""Detection Heads: Standard Pure CNN Head and Deformable Attention Detection Head."""
from typing import Tuple, Dict, Optional, Union, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
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

        self.obj_head = nn.Sequential(
            nn.Conv2d(dim, head_conv, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1),
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

        nn.init.kaiming_uniform_(self.obj_head[0].weight, a=1)
        nn.init.constant_(self.obj_head[-1].weight, 0.0)
        nn.init.constant_(self.obj_head[-1].bias, cls_bias_val)

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

    def _forward_single(
        self, feat_map: torch.Tensor, feat_deform: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, _, H, W = feat_map.shape

        cls_logits = self.cls_head(feat_map)
        box_deltas = self.box_head(feat_map)
        obj_logits = self.obj_head(feat_map)

        _, anchor_logits = self._get_reference_grid(H, W, device=feat_map.device, dtype=feat_map.dtype)
        anchor_logits = anchor_logits.expand(B, -1, -1, -1)

        cxcy = (anchor_logits + box_deltas[:, :2]).sigmoid()
        scale_factor = torch.exp(box_deltas[:, 2:].clamp(min=-4.0, max=4.0))
        wh = (0.12 * scale_factor).clamp(min=1e-3, max=1.0)

        pred_boxes = torch.cat([cxcy, wh], dim=1)
        return cls_logits, pred_boxes, obj_logits, None

    def forward(
        self,
        feat_map: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...], list],
        feat_deform: Optional[torch.Tensor] = None,
    ):
        if isinstance(feat_map, dict):
            cls_logits_dict = {}
            pred_boxes_dict = {}
            obj_logits_dict = {}
            for k, v in feat_map.items():
                cls_l, pred_b, obj_l, _ = self._forward_single(v, feat_deform)
                cls_logits_dict[k] = cls_l
                pred_boxes_dict[k] = pred_b
                obj_logits_dict[k] = obj_l
            return cls_logits_dict, pred_boxes_dict, obj_logits_dict, None
        elif isinstance(feat_map, (list, tuple)):
            cls_logits_list = []
            pred_boxes_list = []
            obj_logits_list = []
            for v in feat_map:
                cls_l, pred_b, obj_l, _ = self._forward_single(v, feat_deform)
                cls_logits_list.append(cls_l)
                pred_boxes_list.append(pred_b)
                obj_logits_list.append(obj_l)
            return cls_logits_list, pred_boxes_list, obj_logits_list, None
        return self._forward_single(feat_map, feat_deform)

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

        # self.to_q= nn.Linear(dim, dim)
        # self.to_k= nn.Linear(dim, dim)
        # self.to_v= nn.Linear(dim, dim)
        # self.scale= 1 / math.sqrt(dim)

        self.to_q2= nn.Linear(dim, dim)
        self.to_k2= nn.Linear(dim, dim)
        self.to_v2= nn.Linear(dim, dim)
        self.scale= 1 / math.sqrt(dim)

        self.query_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2)
        )
        # 2. Deformable Attention Engine
        self.value_proj = nn.Linear(dim, dim)
        self.sampling_offsets = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_points * 2)
        )
        self.attention_weights = nn.Linear(dim, num_heads * num_points)
        self.output_proj = nn.Linear(dim, dim)

        # 3. Prediction Heads
        self.cls_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, num_classes, kernel_size=1),
        )

        self.box_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 4, kernel_size=1),
        )

        self.obj_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=1),
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

        nn.init.kaiming_uniform_(self.obj_head[0].weight, a=1)
        nn.init.constant_(self.obj_head[-1].weight, 0.0)
        nn.init.constant_(self.obj_head[-1].bias, cls_bias_val)

        # Trong _reset_parameters() của DeformableHead:
        offset_layer = self.sampling_offsets[-1] if isinstance(self.sampling_offsets, nn.Sequential) else self.sampling_offsets
        nn.init.constant_(offset_layer.weight, 0.0)
        num_rings = 4  # Số vòng tròn đồng tâm
        pts_per_ring = self.num_points // num_rings
        thetas = torch.arange(pts_per_ring, dtype=torch.float32) * (2.0 * math.pi / pts_per_ring)
        ring_pts = torch.stack([thetas.cos(), thetas.sin()], -1)  # [pts_per_ring, 2]
        # Bán kính tăng dần theo cấp số cộng hoặc luỹ thừa 2
        radii = torch.tensor([1.0, 2.5, 5.0, 10.0], dtype=torch.float32)  # hoặc torch.linspace(1.0, 8.0, num_rings)
        grid_init = []
        for r in radii:
            grid_init.append(ring_pts * r)
        grid_init = torch.cat(grid_init, dim=0)  # [num_points, 2]
        with torch.no_grad():
            offset_layer.bias.copy_(grid_init.view(-1))

        query_layer = self.query_proj[-1] if isinstance(self.query_proj, nn.Sequential) else self.query_proj
        nn.init.constant_(query_layer.weight, 0.0)
        nn.init.constant_(query_layer.bias, 0.0)

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

    def _get_pos_embed(self, H: int, W: int, embed_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (H, W, embed_dim, device, dtype)
        if not hasattr(self, "_pos_embed_cache"):
            self._pos_embed_cache = {}
        if cache_key not in self._pos_embed_cache:
            self._pos_embed_cache[cache_key] = build_2d_sincos_position_embedding(
                H=H, W=W, embed_dim=embed_dim, device=device, dtype=dtype
            )
        return self._pos_embed_cache[cache_key]

    def _forward_single(
        self,
        feat_map: torch.Tensor,
        feat_deform: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if feat_deform is None:
            feat_deform = feat_map
        """
        feat_map: (B,C,H,W)
        """
        B, C, H, W = feat_map.shape
        
        feat_map = feat_map + self._get_pos_embed(H, W, C, feat_map.device, feat_map.dtype)
        feat_map = feat_map.view(B, C, H * W).permute(0, 2, 1) # (B,N,C)

        B_def, C_def, H_def, W_def = feat_deform.shape
        feat_deform_with_pos = feat_deform + self._get_pos_embed(H_def, W_def, C_def, feat_map.device, feat_map.dtype)

        _, grid_logits = self._get_reference_grid(H, W, device=feat_map.device, dtype=feat_map.dtype)
        # 2. Dự đoán độ lệch (offset delta) từ đặc trưng v: [B, N, 2]
        delta_ref = self.query_proj(feat_map).view(B, H * W, 2)
        # 3. ref = sigmoid(grid_logits + delta): [B, N, 2]
        ref = (grid_logits.unsqueeze(0) + delta_ref).sigmoid()
        sampling = self.sampling_offsets(feat_map).view(B, H * W, self.num_points, 2) # (B,N, num_points, 2)
        spatial_scale = feat_map.new_tensor([1.0 / W_def, 1.0 / H_def])
        sampling_locations = ref.unsqueeze(-2) + sampling * spatial_scale
        sampling_grid = 2.0 * sampling_locations - 1.0

        sampled_feat = F.grid_sample(
            feat_deform_with_pos,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # (B, C_def, N, num_points)

        sampled_feat = self.value_proj(sampled_feat.permute(0, 2, 3, 1)) # (B, N, num_points, D)

        q = self.to_q2(feat_map) # (B,N,D)
        k = self.to_k2(sampled_feat) # (B, N, num_points, D)
        v2 = self.to_v2(sampled_feat) # (B, N, num_points, D)

        dots = torch.einsum('b n d, b n p d -> b n p', q, k)
        dots = dots * self.scale
        attn = torch.softmax(dots, dim=-1) 
        feat = torch.einsum('b n p d, b n p -> b n d', v2, attn)
        feat = feat.permute(0, 2, 1).view(B, C, H, W)

        cls_logits = self.cls_head(feat)  # (B, num_classes, H, W)
        obj_logits = self.obj_head(feat)  # (B, 1, H, W)
        feat = feat_map.permute(0, 2, 1).view(B, C, H, W) + feat
        box_deltas = self.box_head(feat).permute(0, 2, 3, 1).view(B, H * W, 4)  # (B, N, 4)

        dx = box_deltas[..., 0].sigmoid()
        dy = box_deltas[..., 1].sigmoid()
        dw = box_deltas[..., 2]
        dh = box_deltas[..., 3]

        cx = (ref[..., 0] + dx)
        cy = (ref[..., 1] + dy)
        w = dw.sigmoid()
        h = dh.sigmoid()

        # Gộp lại thành (B, N, 4)
        boxes = torch.stack([cx, cy, w, h], dim=-1)

        # Đưa về shape feature map (B, 4, H, W):
        boxes_map = boxes.permute(0, 2, 1).view(B, 4, H, W)

        ref_map = ref.permute(0, 2, 1).view(B, 2, H, W)
        sampling_info = {
            "ref": ref_map,
            "ref_points": ref,
            "sampling_locations": sampling_locations.view(B, H, W, self.num_points, 2),
            "offsets": sampling.view(B, H, W, self.num_points, 2),
            "obj_logits": obj_logits,
        }
        return cls_logits, boxes_map, obj_logits, sampling_info

    def forward(
        self,
        feat_map: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...], list],
        feat_deform: Optional[torch.Tensor] = None,
    ):
        if isinstance(feat_map, dict):
            cls_logits_dict = {}
            pred_boxes_dict = {}
            obj_logits_dict = {}
            sampling_info_dict = {}
            for k, v in feat_map.items():
                cls_l, pred_b, obj_l, s_info = self._forward_single(v, feat_deform)
                cls_logits_dict[k] = cls_l
                pred_boxes_dict[k] = pred_b
                obj_logits_dict[k] = obj_l
                sampling_info_dict[k] = s_info
            self.last_ref = {k: v["ref"] for k, v in sampling_info_dict.items() if "ref" in v}
            self.last_obj_logits = obj_logits_dict
            return cls_logits_dict, pred_boxes_dict, obj_logits_dict, sampling_info_dict
        elif isinstance(feat_map, (list, tuple)):
            cls_logits_list = []
            pred_boxes_list = []
            obj_logits_list = []
            sampling_info_list = []
            for v in feat_map:
                cls_l, pred_b, obj_l, s_info = self._forward_single(v, feat_deform)
                cls_logits_list.append(cls_l)
                pred_boxes_list.append(pred_b)
                obj_logits_list.append(obj_l)
                sampling_info_list.append(s_info)
            self.last_ref = [s["ref"] for s in sampling_info_list if "ref" in s]
            self.last_obj_logits = obj_logits_list
            return cls_logits_list, pred_boxes_list, obj_logits_list, sampling_info_list
        else:
            cls_l, pred_b, obj_l, s_info = self._forward_single(feat_map, feat_deform)
            self.last_ref = s_info["ref"]
            self.last_obj_logits = obj_l
            return cls_l, pred_b, obj_l, s_info






