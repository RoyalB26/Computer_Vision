"""COCO Object Detection Model predicting bounding boxes and class labels."""
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
import numpy as np
from torch import Tensor
import math
from BaseModel.core import MODELS
from BaseModel.model import BaseModel
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models.feature_extraction import create_feature_extractor
from BaseModel.model.layers_module import (
    AnchorFreeDetHead,
)


class BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        return_nodes = {
            'features.1': 'feat_1',
            'features.3': 'feat_3',
            'features.5': 'feat_5',
            'features.12': 'feat_last',  # Thêm tầng 12
        }
        self.feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    def forward(self, x):
        feats = self.feature_extractor(x)
        return feats['feat_1'], feats['feat_3'], feats['feat_5'], feats['feat_last']

class UpSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1= nn.Conv2d(in_channels=616, out_channels=96, kernel_size= 1, padding= 0)
        self.conv2= nn.Conv2d(in_channels=120, out_channels=48, kernel_size=1, padding= 0)
        self.conv3= nn.Conv2d(in_channels=64, out_channels=dim,kernel_size= 1, padding= 0)

    def forward(self, c1: Tensor,c3: Tensor,c5: Tensor,x: Tensor):
        x= F.interpolate(x, scale_factor=2, mode='nearest') # [B, 576, 40, 40]
        x= torch.cat([x, c5], dim = 1) #[B, 616, 40, 40]
        x= self.conv1(x) #[B, 96, 40, 40]
        x= F.interpolate(x, scale_factor=2, mode='nearest') # [B, 96, 80, 80]
        x=torch.cat([x,c3], dim = 1) #[B, 120, 80, 80]
        x= self.conv2(x) # [B, 48, 80, 80]
        x= F.interpolate(x, scale_factor=2, mode='nearest') # [B, 48, 160, 160]
        x=torch.cat([x,c1], dim = 1) #[B, 64, 160, 160]
        x= self.conv3(x) #[B, dim, 160, 160]
        return x

class DownBlock(nn.Module):
    """Khối Conv 3x3 với stride=2 để downsample giảm 1/2 kích thước không gian"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
    
class DownSample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        
        # 1. Thu nhỏ c1 (160x160 -> 80x80): 16 channels -> 16 channels
        self.down1 = DownBlock(in_c=16, out_c=16)
        # Gộp c1_down(16) + c3(24) = 40 channels -> nén ra 48 channels (giữ nguyên size 80x80)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(40, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        # 2. Thu nhỏ c3 (80x80 -> 40x40): 48 channels -> 48 channels
        self.down2 = DownBlock(in_c=48, out_c=48)
        # Gộp c3_down(48) + c5(40) = 88 channels -> nén ra 96 channels (giữ nguyên size 40x40)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(88, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        # 3. Thu nhỏ c5 (40x40 -> 20x20): 96 channels -> 96 channels
        self.down3 = DownBlock(in_c=96, out_c=96)
        # Gộp c5_down(96) + x(576) = 672 channels -> nén ra dim (giữ nguyên size 20x20)
        self.fuse3 = nn.Sequential(
            nn.Conv2d(672, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, c1: Tensor, c3: Tensor, c5: Tensor, x: Tensor) -> Tensor:
        # Shapes đầu vào giả định:
        # c1: [B, 16, 160, 160]
        # c3: [B, 24, 80, 80]
        # c5: [B, 40, 40, 40]
        # x : [B, 576, 20, 20]
        # 1. 160x160 -> 80x80
        c1_down = self.down1(c1)                    # [B, 16, 80, 80]
        c3 = torch.cat([c1_down, c3], dim=1)        # [B, 16 + 24 = 40, 80, 80]
        c3 = self.fuse1(c3)                         # [B, 48, 80, 80]
        # 2. 80x80 -> 40x40
        c3_down = self.down2(c3)                    # [B, 48, 40, 40]
        c5 = torch.cat([c3_down, c5], dim=1)        # [B, 48 + 40 = 88, 40, 40]
        c5 = self.fuse2(c5)                         # [B, 96, 40, 40]
        # 3. 40x40 -> 20x20
        c5_down = self.down3(c5)                    # [B, 96, 20, 20]
        x = torch.cat([c5_down, x], dim=1)          # [B, 96 + 576 = 672, 20, 20]
        x = self.fuse3(x)                           # [B, dim, 20, 20]
        return x

class Deformable(nn.Module):
    def __init__(self, dim, num_patches, num_reference_point:int= 4):
        super().__init__()
        self.to_reference_x = nn.Linear(num_patches, num_reference_point)
        self.to_reference_y = nn.Linear(num_patches, 1)
        self.to_q= nn.Linear(dim, dim)
        self.to_k= nn.Linear(dim, dim)
        self.to_v= nn.Linear(dim, dim)

    def forward(self, x: Tensor, pure_map: Tensor, feat_map: Tensor, attn: Tensor):
        """
        x: B, C, dim
        feat_map: [B, C, dim, h, w]
        pure_map: [B, dim, H, W]
        attn: [B, C, N]
        """
        B, C, dim, H, W= feat_map.shape
        reference_x= F.sigmoid(self.to_reference_x(attn)) # [B, C, K]
        feat = reference_x[:, :, :, None] + attn[:, :, None, :]
        reference_y= F.sigmoid(self.to_reference_y(feat).squeeze(-1)) # [B, C, K] 
        B, dim, H, W = pure_map.shape
        _, C, K = reference_x.shape
        num_samples = 10  # 10 ô từ mỗi cặp (x, y)
        # 1. Tọa độ gốc chuẩn hóa về dải [-1, 1]
        # Shape ban đầu: [B, C*K, 1]
        base_x = (reference_x.view(B, -1) * 2.0 - 1.0).unsqueeze(-1)  # [B, C*K, 1]
        base_y = (reference_y.view(B, -1) * 2.0 - 1.0).unsqueeze(-1)  # [B, C*K, 1]
        # 2. Định nghĩa offsets (ví dụ: 10 điểm lân cận)
        # Lưu ý: 1 pixel trên trục x ứng với bước nhảy 2.0 / W
        # 1 pixel trên trục y ứng với bước nhảy 2.0 / H
        # Ví dụ 10 offsets (delta_x theo pixel, delta_y theo pixel):
        # (Bao gồm điểm tâm (0,0) và 9 điểm xung quanh hoặc 10 vị trí tùy bạn chọn)
        offsets_pixel = torch.tensor([
            [0.0, 0.0],
            [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0],
            [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0],
            [0.0, 2.0]  # điểm thứ 10 ví dụ
        ], device=pure_map.device)  # [10, 2]
        # Chuyển đổi offset từ pixel sang dải tọa độ [-1, 1]
        offset_x = offsets_pixel[:, 0] * (2.0 / W)  # [10]
        offset_y = offsets_pixel[:, 1] * (2.0 / H)  # [10]
        # 3. Mở rộng và cộng offset vào tọa độ gốc:
        # [B, C*K, 1] + [10] -> [B, C*K, 10]
        sample_x = base_x + offset_x.view(1, 1, num_samples)
        sample_y = base_y + offset_y.view(1, 1, num_samples)
        # 4. Gom lại thành Grid:
        # Ghép (x, y) -> [B, C*K, 10, 2]
        grid = torch.stack([sample_x, sample_y], dim=-1)
        # 5. Lấy mẫu với grid_sample:
        # pure_map: [B, dim, H, W]
        # grid:     [B, C*K, 10, 2]
        # sampled:  [B, dim, C*K, 10]
        sampled = F.grid_sample(pure_map, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        # 6. Định hình lại (Reshape) theo cấu trúc mong muốn:
        # [B, dim, C, K, 10]
        sampled = sampled.view(B, dim, C, K, num_samples)
        sampled = sampled.permute(0, 2, 3, 4, 1) # [B, C, K, 10, dim]
        k= self.to_k(sampled) # [B, C, K, 10, dim]
        v= self.to_v(sampled) # [B, C, K, 10, dim]
        q= self.to_q(x) # [B, C, dim]

        dim = q.shape[-1]
        scale = dim ** -0.5
        attn_scores = torch.einsum('bcd, bcksd -> bcks', q, k) * scale
        attn_weights = F.softmax(attn_scores, dim=-1) # [B, C, K, 10]
        out = torch.einsum('bcks, bcksd -> bckd', attn_weights, v) # [B, C, K, dim]

        B, C, dim, H, W = feat_map.shape
        # 2. Bilinear Splatting (khả vi bậc 1 theo cả out lẫn reference_x, reference_y, chuẩn hóa biên độ đạo hàm tọa độ)
        px = reference_x.detach() * W - 0.5 + (reference_x - reference_x.detach())  # [B, C, K]
        py = reference_y.detach() * H - 0.5 + (reference_y - reference_y.detach())  # [B, C, K]

        x0 = torch.floor(px).detach()
        y0 = torch.floor(py).detach()
        wx1 = px - x0  # Có đạo hàm theo reference_x
        wy1 = py - y0  # Có đạo hàm theo reference_y
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        x0_i = x0.long()
        y0_i = y0.long()
        x1_i = x0_i + 1
        y1_i = y0_i + 1

        scattered_flat = torch.zeros(B, C, dim, H * W, device=feat_map.device, dtype=feat_map.dtype)
        out_perm = out.permute(0, 1, 3, 2)  # [B, C, dim, K]

        for xi, yi, wi in [
            (x0_i, y0_i, wx0 * wy0),
            (x1_i, y0_i, wx1 * wy0),
            (x0_i, y1_i, wx0 * wy1),
            (x1_i, y1_i, wx1 * wy1),
        ]:
            valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)  # [B, C, K]
            idx = (yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1)).unsqueeze(2).expand(B, C, dim, K)
            weighted_out = out_perm * (wi * valid.to(out.dtype)).unsqueeze(2)
            scattered_flat = scattered_flat.scatter_add(-1, idx, weighted_out)

        scattered_feat_map = scattered_flat.view(B, C, dim, H, W)
        updated_feat_map = feat_map + scattered_feat_map

        return updated_feat_map

class ClassSpecificDetHead(nn.Module):
    def __init__(self, dim: int, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim

        # 1. Cls Head: Depthwise Separable Conv theo từng class (groups=num_classes*dim cho Depthwise)
        # Giảm FLOPs gấp ~7 lần so với standard conv 3x3
        self.cls_head = nn.Sequential(
            # Depthwise Conv 3x3: lọc không gian độc lập từng channel
            nn.Conv2d(
                num_classes * dim, num_classes * dim,
                kernel_size=3, padding=1, groups=num_classes * dim, bias=False
            ),
            nn.BatchNorm2d(num_classes * dim),
            nn.ReLU(inplace=True),
            # Pointwise Conv 1x1: phối hợp các channels trong cùng class (groups=num_classes)
            nn.Conv2d(
                num_classes * dim, num_classes * dim,
                kernel_size=1, groups=num_classes, bias=False
            ),
            nn.BatchNorm2d(num_classes * dim),
            nn.ReLU(inplace=True),
            # Final 1x1 Conv: dự đoán heatmap logits cho từng class
            nn.Conv2d(
                num_classes * dim, num_classes,
                kernel_size=1, groups=num_classes
            )
        )

        # 2. Box Head: Depthwise Separable Conv dự đoán offset [dx, dy, w, h] cho từng class
        self.box_head = nn.Sequential(
            # Depthwise Conv 3x3: lọc không gian độc lập từng channel
            nn.Conv2d(
                num_classes * dim, num_classes * dim,
                kernel_size=3, padding=1, groups=num_classes * dim, bias=False
            ),
            nn.BatchNorm2d(num_classes * dim),
            nn.ReLU(inplace=True),
            # Pointwise Conv 1x1: phối hợp các channels trong cùng class (groups=num_classes)
            nn.Conv2d(
                num_classes * dim, num_classes * dim,
                kernel_size=1, groups=num_classes, bias=False
            ),
            nn.BatchNorm2d(num_classes * dim),
            nn.ReLU(inplace=True),
            # Final 1x1 Conv: dự đoán 4 offset cho từng class
            nn.Conv2d(
                num_classes * dim, num_classes * 4,
                kernel_size=1, groups=num_classes
            ),
            nn.Sigmoid()  # dx, dy là offset trong ô [0, 1], w, h chuẩn hóa [0, 1]
        )

        # Khởi tạo bias cho cls_head theo chuẩn Focal Loss (prior_prob = 0.01)
        prior_prob = 0.01
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head[-1].bias, bias_value)

    def forward(self, feat_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat_map: [B, C, dim, H, W]
        B, C, dim, H, W = feat_map.shape

        # Gộp chiều C và dim lại thành: [B, C * dim, H, W]
        feat_grouped = feat_map.view(B, C * dim, H, W)

        # Chạy song song 10 heads cho 10 class cùng lúc:
        cls_logits = self.cls_head(feat_grouped)      # [B, C, H, W]
        pred_offsets = self.box_head(feat_grouped)    # [B, C * 4, H, W]

        # Reshape pred_offsets về [B, C, 4, H, W]
        pred_offsets = pred_offsets.view(B, C, 4, H, W)

        return cls_logits, pred_offsets

class SlotCrossAttention(nn.Module):
    """
    Module Cross-Attention:
    Slots (Query) quét qua Feature Map (Key, Value) để trích xuất đặc trưng chuyên biệt cho từng class.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # Chiếu Q từ Slots
        self.q_proj = nn.Linear(dim, dim, bias=False)
        # Chiếu K, V từ Feature Map (dùng Conv 1x1 siêu nhanh)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        # Output projection & Normalization
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm_feat = nn.BatchNorm2d(dim)
        self.norm_slot = nn.LayerNorm(dim)

    def forward(self, slots: torch.Tensor, feat_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: [B, C, dim] (C = num_classes)
            feat_map: [B, dim, H, W] (từ up_sample FPN, ví dụ 160x160)
        Returns:
            class_feat_map: [B, C, dim, H, W] (Mỗi class có 1 heatmap đặc trưng tương tác sâu)
        """
        B, C, D = slots.shape
        _, _, H, W = feat_map.shape
        N = H * W
        # 1. Chuẩn bị Query từ Slots: [B, C, H_heads, D_head]
        q = self.norm_slot(slots)
        q = self.q_proj(q).view(B, C, self.num_heads, self.head_dim)
        # 2. Chuẩn bị Key và Value từ Feat Map: [B, N, H_heads, D_head]
        k = self.k_proj(feat_map).flatten(2).transpose(1, 2).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(feat_map).flatten(2).transpose(1, 2).view(B, N, self.num_heads, self.head_dim)
        # 3. Tính Attention Map giữa từng Class và từng vị trí pixel:
        # q: [B, C, heads, d]
        # k: [B, N, heads, d]
        # -> attn: [B, C, heads, N]
        attn_scores = torch.einsum('bchd, bnhd -> bchn', q, k) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1) # Softmax trên toàn bộ N pixels
        # 4. Tạo đặc trưng điều biến không gian (Spatial Modulation Map):
        # Nhân attn với Value để tạo trọng số tương tác:
        # attn: [B, C, heads, N] x v: [B, N, heads, d] -> [B, C, heads, d]
        slot_context = torch.einsum('bchn, bnhd -> bchd', attn_weights, v)
        slot_context = slot_context.reshape(B, C, D) # [B, C, dim]
        slots= slot_context
        # 5. Phối hợp đặc trưng (Modulation):
        # Thay vì chỉ cộng broadcast đơn thuần, ta kết hợp cả cộng và nhân tương quan (Gate Modulation):
        slot_context = slot_context.unsqueeze(-1).unsqueeze(-1) # [B, C, dim, 1, 1]
        feat_expanded = feat_map.unsqueeze(1)                   # [B, 1, dim, H, W]
        # Class Feature Map: vừa giữ đặc trưng không gian nét, vừa được lọc theo đặc trưng của class!
        out = feat_expanded * torch.sigmoid(slot_context) + slot_context # [B, C, dim, H, W]
        return slots, out, attn_weights.mean(dim= -2)

@MODELS.register_module()
class VisDroneDetector(BaseModel):
    """Detector for COCO dataset predicting bounding boxes, class labels, and objectness."""

    def __init__(
        self,
        num_classes: int = 80,
        dim: int = 16,
        total_head: int = 4,
        images_size: int = 640,
        num_ref: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.total_head = total_head
        self.dim= dim
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_q = nn.Linear(dim, dim)
        
        # Shape: (num_classes, dim) -> each slot gets unique learned mu and logsigma
        self.mu = nn.Parameter(torch.randn(self.num_classes, dim))
        self.logsigma = nn.Parameter(torch.zeros(self.num_classes, dim))

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(dim)

        # self.FC1 = nn.Linear(dim, dim)
        # self.FC2 = nn.Linear(dim, dim)
        # self.LN = nn.LayerNorm(dim)

        # Backbone setup
        self.backbone = BackBone()

        self.projector = nn.Sequential(
            nn.Conv2d(
                in_channels=576, out_channels=total_head * dim, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(total_head * dim),
            nn.ReLU(inplace=True),
        )


        self.conv = nn.Conv2d(self.total_head, 1, 1, 1, 0)
        self.conv2 = nn.Conv2d(3, dim, 1, 1, 0)

        self.norm_out = nn.LayerNorm(dim)
        self.downsample = DownSample(total_head * dim)
        self.up_sample = UpSample(dim)
        self.deformable = Deformable(dim, (images_size // 4) ** 2, num_ref)
        # Initialization
        nn.init.xavier_uniform_(self.mu)
        nn.init.zeros_(self.logsigma)
        torch.manual_seed(42)

        self.head = ClassSpecificDetHead(
            dim= dim
        )

        self.cross_attn = SlotCrossAttention(dim=dim, num_heads=total_head)

    def initalize_slots(self, batch_size=1, device=None):
        # Handle the case where device is passed as the first positional argument
        if isinstance(batch_size, (torch.device, str)):
            device = batch_size
            batch_size = 1

        device = device if device is not None else self.mu.device

        mu = self.mu.to(device)
        sigma = torch.exp(self.logsigma).to(device)

        noise = torch.randn(batch_size, self.num_classes, self.dim, device=device)

        slots = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise

        return slots

    def positional_encoding(self, H, W, D):
        
        grid_w = np.arange(W, dtype= np.float32) # (W, )
        grid_h = np.arange(H, dtype= np.float32) # (H, )

        X,Y = np.meshgrid(grid_w, grid_h)
        x_flat = X.reshape(-1) # (H * W, )
        y_flat = Y.reshape(-1) # (H * W, )

        emb_x = self.sincos_1D_embedding(D // 2, x_flat)  # [H * W, D // 2]
        emb_y = self.sincos_1D_embedding(D // 2, y_flat)  # [H * W, D // 2]

        # Ghép lại thành vector vị trí 2D hoàn chỉnh: [H * W, D]
        return np.concatenate([emb_x, emb_y], axis=1)

    def sincos_1D_embedding(self, D, grid):
        omega = np.arange(D // 2, dtype= np.float32) # (D // 2, )
        omega/= (D / 2.0)
        omega = 1.0 / (10000 ** omega)

        out = np.einsum('m,d->md', grid, omega) # [N, D // 2]
        return np.concatenate([np.sin(out), np.cos(out)], axis=1) # [N, D]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất đặc trưng qua Backbone và FPN."""
        B, C, H, W= x.shape

        pure_map = self.conv2(x)
        c1,c3,c5,c12= self.backbone(x)

        B, C, H, W= c12.shape
        D= self.total_head * self.dim
        x= self.projector(c12) # [B, D, H, W]
        x= self.downsample(c1,c3,c5,c12) + x
        x= x.permute(0, 2, 3, 1) # [B, H, W, D]
        x= x.reshape(B, H*W, D) # [B, N, D]

        

        position: np.ndarray = self.positional_encoding(H, W, D) 
        position: Tensor = torch.from_numpy(position).to(x.dtype).to(x.device) # [N, D]

        x = x + position # [B, N, D]
        B,N,D= x.shape
        x = x.view(B, N, self.total_head, self.dim) # [B, N, head, dim]
        k = self.to_k(x) # [B, N, dim]
        v = self.to_v(x) #  [B, N, head, dim]
        slots = self.initalize_slots(batch_size= B,device= x.device) # [B, num_classes + 1, dim]
        q = self.to_q(slots) # [B, num_classes + 1, dim]


        dot = torch.einsum("b c d, b n h d -> b c h n", q, k) # [B, num_classes + 1, head, N]
        dot = dot / (self.dim ** 0.5)

        attn = torch.softmax(dot, dim= 1) # [B, num_classes + 1, head, N]
        x = torch.einsum("b c h n, b n h d -> b c h d", attn, v) # [B, num_classes + 1, head, dim]
        x = x.permute(0, 2, 1, 3) # [B, head, num_classes + 1, dim]
        x_slots = self.conv(x).squeeze(1) # [B, num_classes + 1, dim]
        x_slots = self.fc2(F.relu(self.fc1(self.norm_out(x_slots))))
        slots = slots + x_slots
        slots = slots[:, :self.num_classes, :] # [B, num_classes, dim]
        B,D,H,W = c12.shape
        feat_map= self.up_sample(c1, c3, c5, c12) # [B, dim, H, W]

        slots, feat_map, attn_weights = self.cross_attn(slots, feat_map)  # [B, C, dim, H, W]
        return slots, feat_map, attn_weights, pure_map

    def _decode_boxes(self, pred_offsets: torch.Tensor) -> torch.Tensor:
        """Decode offset dự đoán [dx, dy, w, h] tại mỗi ô lưới thành tọa độ box chuẩn hóa [cx, cy, w, h]."""
        device = pred_offsets.device
        dtype = pred_offsets.dtype
        if pred_offsets.dim() == 5:
            B, C, _, H, W = pred_offsets.shape
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device, dtype=dtype),
                torch.arange(W, device=device, dtype=dtype),
                indexing="ij",
            )
            cx = (grid_x.view(1, 1, H, W) + pred_offsets[:, :, 0]) / W
            cy = (grid_y.view(1, 1, H, W) + pred_offsets[:, :, 1]) / H
            w = pred_offsets[:, :, 2]
            h = pred_offsets[:, :, 3]
            return torch.stack([cx, cy, w, h], dim=2)
        else:
            B, _, H, W = pred_offsets.shape
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device, dtype=dtype),
                torch.arange(W, device=device, dtype=dtype),
                indexing="ij",
            )
            cx = (grid_x.view(1, H, W) + pred_offsets[:, 0]) / W
            cy = (grid_y.view(1, H, W) + pred_offsets[:, 1]) / H
            w = pred_offsets[:, 2]
            h = pred_offsets[:, 3]
            return torch.stack([cx, cy, w, h], dim=1)

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Chuyển đổi boxes từ [cx, cy, w, h] sang [x1, y1, x2, y2]."""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _compute_losses(self, cls_logits, pred_offsets, targets, device):
        """Hàm dùng chung tính Focal Loss (cls), DIoU Loss (box từ offset) và các chỉ số tracking chẩn đoán."""
        batch_size, num_classes, grid_h, grid_w = cls_logits.shape

        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_cls_pos = torch.tensor(0.0, device=device)
        total_cls_neg = torch.tensor(0.0, device=device)
        prob_pos_sum = torch.tensor(0.0, device=device)
        prob_neg_max = torch.tensor(0.0, device=device)
        num_pos_total = 0

        for b in range(batch_size):
            img_cls = cls_logits[b]
            img_offset = pred_offsets[b]
            target_dict = targets[b] if targets is not None and b < len(targets) else None
            gt_boxes = target_dict.get("boxes", torch.empty((0, 4), device=device)) if target_dict else torch.empty((0, 4), device=device)
            gt_labels = target_dict.get("labels", torch.empty((0,), device=device, dtype=torch.long)) if target_dict else torch.empty((0,), device=device, dtype=torch.long)
            num_gt = len(gt_boxes)

            cls_target = torch.zeros((num_classes, grid_h, grid_w), device=device)
            if num_gt > 0:
                gt_boxes = gt_boxes.to(device)
                gt_labels = gt_labels.to(device)

                gt_cx = gt_boxes[:, 0].clamp(0.0, 0.999)
                gt_cy = gt_boxes[:, 1].clamp(0.0, 0.999)
                grid_x = (gt_cx * grid_w).long()
                grid_y = (gt_cy * grid_h).long()

                valid_mask = (gt_labels >= 0) & (gt_labels < num_classes)
                valid_labels = gt_labels[valid_mask]
                grid_x = grid_x[valid_mask]
                grid_y = grid_y[valid_mask]
                gt_boxes = gt_boxes[valid_mask]

                num_pos = len(valid_labels)
                num_pos_total += num_pos
                if num_pos > 0:
                    cls_target[valid_labels, grid_y, grid_x] = 1.0
                    if img_offset.dim() == 4:
                        p_off = img_offset[valid_labels, :, grid_y, grid_x]
                    else:
                        p_off = img_offset[:, grid_y, grid_x].permute(1, 0)

                    # Chỉ decode tọa độ box tại các ô lưới dương (positive cells)
                    p_cx = (grid_x.to(p_off.dtype) + p_off[:, 0]) / grid_w
                    p_cy = (grid_y.to(p_off.dtype) + p_off[:, 1]) / grid_h
                    p_box = torch.stack([p_cx, p_cy, p_off[:, 2], p_off[:, 3]], dim=-1)

                    p_xyxy = self._cxcywh_to_xyxy(p_box)
                    gt_xyxy = self._cxcywh_to_xyxy(gt_boxes)
                    box_loss = torchvision.ops.distance_box_iou_loss(
                        p_xyxy, gt_xyxy, reduction="sum"
                    )
                    total_box_loss += box_loss

            focal_map = torchvision.ops.sigmoid_focal_loss(
                img_cls,
                cls_target,
                alpha=0.25,
                gamma=2.0,
                reduction="none",
            )
            total_cls_loss += focal_map.sum()

            with torch.no_grad():
                pos_mask = cls_target > 0.5
                neg_mask = ~pos_mask
                total_cls_pos += focal_map[pos_mask].sum()
                total_cls_neg += focal_map[neg_mask].sum()
                probs = img_cls.detach().sigmoid()
                if pos_mask.any():
                    prob_pos_sum += probs[pos_mask].sum()
                if neg_mask.any():
                    prob_neg_max = torch.maximum(prob_neg_max, probs[neg_mask].max())

        # Chuẩn hóa theo tổng số mẫu dương (num_pos_total), không chia cho toàn bộ số pixel (B * C * H * W)
        norm_pos = max(num_pos_total, 1)
        total_cls_loss = total_cls_loss / norm_pos
        total_box_loss = (total_box_loss / norm_pos) if num_pos_total > 0 else (total_box_loss * 0.0)
        total_loss = 2.0 * total_cls_loss + 1.0 * total_box_loss

        return {
            "loss": total_loss,
            "loss_cls": total_cls_loss,
            "loss_box": total_box_loss,
            "loss_cls_pos": (total_cls_pos / norm_pos).detach(),
            "loss_cls_neg": (total_cls_neg / norm_pos).detach(),
            "loss_prob_pos": (prob_pos_sum / norm_pos).detach(),
            "loss_prob_neg_max": prob_neg_max.detach(),
        }

    def _post_process(self, cls_logits, pred_offsets, conf_threshold=0.10, nms_threshold=0.5, max_det=100):
        """Lọc ngưỡng vector hóa, chỉ decode các candidate vượt ngưỡng và chạy NMS tối ưu tốc độ."""
        batch_size, num_classes, grid_h, grid_w = cls_logits.shape
        cls_scores = cls_logits.sigmoid()
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

                if pred_offsets.dim() == 5:
                    offsets = pred_offsets[b, b_labels, :, y_idx, x_idx]
                else:
                    offsets = pred_offsets[b, :, y_idx, x_idx].permute(1, 0)

                cx = (x_idx.to(offsets.dtype) + offsets[:, 0]) / grid_w
                cy = (y_idx.to(offsets.dtype) + offsets[:, 1]) / grid_h
                b_boxes = torch.stack([cx, cy, offsets[:, 2], offsets[:, 3]], dim=-1)

                if nms_threshold is not None and len(b_boxes) > 0:
                    xyxy = self._cxcywh_to_xyxy(b_boxes)
                    keep = torchvision.ops.batched_nms(xyxy, b_scores, b_labels, nms_threshold)
                    keep = keep[:max_det]
                    b_boxes = b_boxes[keep]
                    b_scores = b_scores[keep]
                    b_labels = b_labels[keep]
            else:
                b_boxes = torch.empty((0, 4), device=device)
                b_scores = torch.empty((0,), device=device)
                b_labels = torch.empty((0,), dtype=torch.long, device=device)

            results.append({
                "boxes": b_boxes,
                "scores": b_scores,
                "labels": b_labels,
            })
        return results

    def forward_train(self, inputs: torch.Tensor, targets=None, **kwargs):
        slots, feat_map, attn, pure_map = self.extract_features(inputs)
        feat_map = self.deformable(slots, pure_map, feat_map, attn)
        cls_logits, pred_offsets = self.head(feat_map)
        return self._compute_losses(cls_logits, pred_offsets, targets, device=inputs.device)
    
    def predict(self, inputs: torch.Tensor, conf_threshold=0.001, nms_threshold=0.5, **kwargs):
        slots, feat_map, attn, pure_map = self.extract_features(inputs)
        feat_map = self.deformable(slots, pure_map, feat_map, attn)
        cls_logits, pred_offsets = self.head(feat_map)
        return self._post_process(cls_logits, pred_offsets, conf_threshold=conf_threshold, nms_threshold=nms_threshold)

    def forward_val(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        conf_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        max_det: int = 300,
        **kwargs,
    ) -> Dict[str, Any]:
        """Validation forward pass: Chỉ chạy mạng 1 LẦN duy nhất cho cả Loss và Predictions."""
        # 1. Chạy mạng 1 lần duy nhất
        slots, feat_map, attn, pure_map = self.extract_features(inputs)
        feat_map = self.deformable(slots, pure_map, feat_map, attn)
        cls_logits, pred_offsets = self.head(feat_map)
        # 2. Tính val loss nếu có targets
        output = {}
        if targets is not None:
            # Gọi trực tiếp hàm tính loss bằng chính logits vừa tính ra
            # (Đảm bảo không phải chạy lại extract_features)
            loss_dict = self._compute_losses(cls_logits, pred_offsets, targets, device=inputs.device)
            output.update(loss_dict)
        # 3. Decode predictions phục vụ tính mAP
        preds = self._post_process(
            cls_logits, pred_offsets,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            max_det=max_det,
        )
        output["preds"] = preds
        return output