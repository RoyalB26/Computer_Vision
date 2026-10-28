"""COCO / VisDrone Object Detection Model WITHOUT Slot Attention.
Kiến trúc tiêu chuẩn (Standard FPN + Anchor-Free Head):
  Backbone (MobileNetV3) -> FPN (UpSample + DownSample) -> Anchor-free Head
"""
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
from torch import Tensor

from BaseModel.core import MODELS
from BaseModel.model import BaseModel
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models.feature_extraction import create_feature_extractor


class BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        return_nodes = {
            'features.1': 'feat_1',
            'features.3': 'feat_3',
            'features.5': 'feat_5',
            'features.12': 'feat_last',
        }
        self.feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    def forward(self, x):
        feats = self.feature_extractor(x)
        return feats['feat_1'], feats['feat_3'], feats['feat_5'], feats['feat_last']


class UpSample_Box(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=616, out_channels=96, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=120, out_channels=48, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=dim, kernel_size=1, padding=0)

    def forward(self, c1: Tensor, c3: Tensor, c5: Tensor, x: Tensor):
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # [B, 576, 40, 40]
        x = torch.cat([x, c5], dim=1)                         # [B, 616, 40, 40]
        x = self.conv1(x)                                     # [B, 96, 40, 40]
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # [B, 96, 80, 80]
        x = torch.cat([x, c3], dim=1)                         # [B, 120, 80, 80]
        x = self.conv2(x)                                     # [B, 48, 80, 80]
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # [B, 48, 160, 160]
        x = torch.cat([x, c1], dim=1)                         # [B, 64, 160, 160]
        x = self.conv3(x)                                     # [B, dim, 160, 160]
        return x

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

class Token_Cross_Attention(nn.Module):
    """
    Multi-Head Cross-Attention module giữa feature map x và slots kèm Positional Encoding.
    - x:     [B, in_dim, H, W]
    - slots: [B, S, slot_dim]
    - output: [B, out_dim, H, W]
    """
    def __init__(self, in_dim: int, slot_dim: int, out_dim: int, use_pos: bool = True):
        super().__init__()
        self.scale = out_dim ** -0.5
        self.use_pos = use_pos
        self.out_dim = out_dim
        
        # Linear chiếu slot thành Query và Value
        self.to_q = nn.Linear(slot_dim, out_dim)
        self.to_v = nn.Linear(slot_dim, out_dim)
        
        # Conv2d chiếu feature x (+ pos_embed nếu có) thành Key
        self.to_k = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.proj = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1)
    def forward(self, slots: Tensor, x: Tensor) -> Tensor:
        """
        slots: [B, S, slot_dim] hoặc [B, S, 1, slot_dim]
        x:     [B, in_dim, H, W]
        """
        B, C, H, W = x.shape
        if slots.dim() == 4:
            slots = slots.flatten(2)
        # 1. Thêm Positional Encoding vào feature x trước khi tạo Key
        if self.use_pos:
            pos_emb = build_2d_sincos_position_embedding(H, W, embed_dim=C, device=x.device, dtype=x.dtype)
            x_with_pos = x + pos_emb
        else:
            x_with_pos = x
        # 2. Chiếu slots và x
        q = self.to_q(slots)                        # [B, S, out_dim]
        v = self.to_v(slots)                        # [B, S, out_dim]
        k = self.to_k(x_with_pos).view(B, self.out_dim, -1)  # [B, out_dim, N], với N = H * W
        # 3. Tính Attention Map giữa slots và các vị trí pixel
        dot = torch.einsum('bsd, bdn -> bsn', q, k) * self.scale
        attn = torch.softmax(dot, dim=1)            # Softmax qua các slots [B, S, N]
        # 4. Trích xuất Value từ slots theo trọng số attention
        out = torch.einsum('bsn, bsd -> bdn', attn, v)  # [B, out_dim, N]
        out = out.view(B, self.out_dim, H, W)
        
        return self.proj(out)
    
class UpSample_Cls(nn.Module):
    def __init__(self, dim: int = 32, slot_dim: int = 64):
        """
        Args:
            dim: Số channels đầu ra cuối cùng (tầng cuối dùng hoàn toàn dim này, nên chia hết cho 4)
            slot_dim: Chiều feature của mỗi slot
        """
        super().__init__()
        assert dim % 4 == 0, f"`dim` ({dim}) phải chia hết cho 4 để áp dụng sin-cos 2D positional encoding."
        
        # --- Block 1: 40x40 ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=616, out_channels=96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.attn_dim1 = 32
        self.attn1 = Token_Cross_Attention(in_dim=96, slot_dim=slot_dim, out_dim=self.attn_dim1, use_pos=True)
        self.fuse1 = nn.Conv2d(96 + self.attn_dim1, 96, kernel_size=1)
        # --- Block 2: 80x80 ---
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=120, out_channels=48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.attn_dim2 = 24
        self.attn2 = Token_Cross_Attention(in_dim=48, slot_dim=slot_dim, out_dim=self.attn_dim2, use_pos=True)
        self.fuse2 = nn.Conv2d(48 + self.attn_dim2, 48, kernel_size=1)
        # --- Block 3: 160x160 (Tầng cuối cùng) ---
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        # Tầng cuối cùng: Sử dụng HOÀN TOÀN dim channels cho Cross-Attention kèm Positional Encoding
        self.attn_last = Token_Cross_Attention(in_dim=dim, slot_dim=slot_dim, out_dim=dim, use_pos=True)

    def forward(self, slots: Tensor, c1: Tensor, c3: Tensor, c5: Tensor, x: Tensor) -> Tensor:
        # ---- Stage 1 (40x40) ----
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # [B, 576, 40, 40]
        x = torch.cat([x, c5], dim=1)                         # [B, 616, 40, 40]
        x = self.conv1(x)                                     # [B, 96, 40, 40]
        
        attn_out1 = self.attn1(slots, x)                      # [B, 32, 40, 40]
        x = self.fuse1(torch.cat([x, attn_out1], dim=1))      # [B, 96, 40, 40]
        # ---- Stage 2 (80x80) ----
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # [B, 96, 80, 80]
        x = torch.cat([x, c3], dim=1)                         # [B, 120, 80, 80]
        x = self.conv2(x)                                     # [B, 48, 80, 80]
        
        attn_out2 = self.attn2(slots, x)                      # [B, 24, 80, 80]
        x = self.fuse2(torch.cat([x, attn_out2], dim=1))      # [B, 48, 80, 80]
        # ---- Stage 3: Tầng cuối (160x160) ----
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # [B, 48, 160, 160]
        x = torch.cat([x, c1], dim=1)                         # [B, 64, 160, 160]
        x = self.conv3(x)                                     # [B, dim, 160, 160]
        
        # Tầng cuối cùng: Sử dụng HOÀN TOÀN dim channels đi qua Token_Cross_Attention
        x = self.attn_last(slots, x)                          # [B, dim, 160, 160]
        
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
        self.down1 = DownBlock(in_c=16, out_c=16)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(40, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.down2 = DownBlock(in_c=48, out_c=48)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(88, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.down3 = DownBlock(in_c=96, out_c=96)
        self.fuse3 = nn.Sequential(
            nn.Conv2d(672, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, c1: Tensor, c3: Tensor, c5: Tensor, x: Tensor) -> Tensor:
        c1_down = self.down1(c1)
        c3 = torch.cat([c1_down, c3], dim=1)
        c3 = self.fuse1(c3)
        c3_down = self.down2(c3)
        c5 = torch.cat([c3_down, c5], dim=1)
        c5 = self.fuse2(c5)
        c5_down = self.down3(c5)
        x = torch.cat([c5_down, x], dim=1)
        x = self.fuse3(x)
        return x


class StandardDetHead(nn.Module):
    """Head phát hiện tiêu chuẩn nhận trực tiếp feat_map [B, dim, H, W] không cần slots."""
    def __init__(self, dim: int, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim

        # Nhánh phân loại: [B, dim, H, W] -> [B, num_classes, H, W]
        self.cls_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, num_classes, kernel_size=1)
        )

        # Nhánh Bounding Box: [B, dim, H, W] -> [B, 4, H, W]
        self.box_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 4, kernel_size=1),
            nn.Sigmoid()  # Dự đoán [cx, cy, w, h] chuẩn hóa [0, 1]
        )

    def forward(self, feat_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_logits = self.cls_head(feat_map)  # [B, num_classes, H, W]
        pred_boxes = self.box_head(feat_map)  # [B, 4, H, W]
        return cls_logits, pred_boxes




@MODELS.register_module()
class VisDroneDetectorNoSlot(BaseModel):
    """Detector for VisDrone dataset without Slot Attention."""

    def __init__(
        self,
        num_classes: int = 10,
        dim: int = 32,
        total_head: int = 4,
        images_size: int = 640,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.total_head = total_head
        self.num_slots= num_classes + 1
        # Backbone & FPN
        self.backbone = BackBone()
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels=576, out_channels=dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.downsample = DownSample(dim)
        self.up_sample_box = UpSample_Box(dim)
        self.up_sample_cls = UpSample_Cls(dim, dim)

        # Shape: (num_classes, dim) -> each slot gets unique learned mu and logsigma
        self.mu = nn.Parameter(torch.randn(self.num_slots, dim))
        self.logsigma = nn.Parameter(torch.zeros(self.num_slots, dim))
        nn.init.xavier_uniform_(self.mu)
        nn.init.zeros_(self.logsigma)
        torch.manual_seed(42)
        # Detection Head tiêu chuẩn
        self.head = StandardDetHead(dim=dim, num_classes=num_classes)

    def initalize_slots(self, batch_size=1, device=None):
        # Handle the case where device is passed as the first positional argument
        if isinstance(batch_size, (torch.device, str)):
            device = batch_size
            batch_size = 1

        device = device if device is not None else self.mu.device

        mu = self.mu.to(device)
        sigma = torch.exp(self.logsigma).to(device)

        noise = torch.randn(batch_size, self.num_slots, self.dim, device=device)

        slots = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise

        return slots

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất đặc trưng qua Backbone và FPN, không dùng slot attention."""
        B,_,_,_= x.shape
        c1, c3, c5, c12 = self.backbone(x)
        slots= self.initalize_slots(B, x.device)
        # c12 có 576 kênh, đưa trực tiếp vào up_sample
        box_feat_map = self.up_sample_box(c1, c3, c5, c12)  # [B, dim, 160, 160]
        cls_feat_map = self.up_sample_cls(slots, c1, c3, c5, c12)  # [B, dim, 160, 160]
        return cls_feat_map, box_feat_map

    def _compute_losses(self, cls_logits, pred_boxes, targets, device):
        """Tính toán Classification Loss và Bounding Box Loss."""
        batch_size, num_classes, grid_h, grid_w = cls_logits.shape
        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        num_pos_total = 0

        for b in range(batch_size):
            img_cls = cls_logits[b]  # [num_classes, H, W]
            img_box = pred_boxes[b]  # [4, H, W]
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
                    p_box = img_box[:, grid_y, grid_x].permute(1, 0)
                    box_loss = F.smooth_l1_loss(p_box, gt_boxes, reduction='sum')
                    total_box_loss += box_loss

            cls_loss = F.binary_cross_entropy_with_logits(
                img_cls, cls_target, 
                pos_weight=torch.tensor([10.0], device=device),
                reduction='sum'
            )
            total_cls_loss += cls_loss

        norm_pixels = max(batch_size * num_classes * grid_h * grid_w, 1)
        total_cls_loss = total_cls_loss / norm_pixels
        total_box_loss = (total_box_loss / num_pos_total) if num_pos_total > 0 else (total_box_loss * 0.0)
        total_loss = 2.0 * total_cls_loss + 1.0 * total_box_loss

        return {
            "loss": total_loss,
            "loss_cls": total_cls_loss,
            "loss_box": total_box_loss,
        }

    def _post_process(self, cls_logits, pred_boxes, conf_threshold=0.10, nms_threshold=0.5, max_det=100):
        """Decode box và chạy NMS tối ưu tốc độ."""
        batch_size, num_classes, grid_h, grid_w = cls_logits.shape
        cls_scores = cls_logits.sigmoid()
        results = []

        for b in range(batch_size):
            b_boxes_list, b_scores_list, b_labels_list = [], [], []
            for c in range(num_classes):
                c_score = cls_scores[b, c]
                mask = c_score >= conf_threshold
                if not mask.any():
                    continue
                scores = c_score[mask]
                labels = torch.full_like(scores, c, dtype=torch.long)
                boxes = pred_boxes[b, :, mask].permute(1, 0)

                b_boxes_list.append(boxes)
                b_scores_list.append(scores)
                b_labels_list.append(labels)

            if len(b_boxes_list) > 0:
                b_boxes = torch.cat(b_boxes_list, dim=0)
                b_scores = torch.cat(b_scores_list, dim=0)
                b_labels = torch.cat(b_labels_list, dim=0)

                if len(b_scores) > 400:
                    topk_idx = torch.topk(b_scores, 400).indices
                    b_boxes = b_boxes[topk_idx]
                    b_scores = b_scores[topk_idx]
                    b_labels = b_labels[topk_idx]

                if nms_threshold is not None and len(b_boxes) > 0:
                    xyxy = torch.stack(
                        [
                            b_boxes[:, 0] - b_boxes[:, 2] / 2,
                            b_boxes[:, 1] - b_boxes[:, 3] / 2,
                            b_boxes[:, 0] + b_boxes[:, 2] / 2,
                            b_boxes[:, 1] + b_boxes[:, 3] / 2,
                        ],
                        dim=1,
                    )
                    keep = torchvision.ops.batched_nms(xyxy, b_scores, b_labels, nms_threshold)
                    keep = keep[:max_det]
                    b_boxes = b_boxes[keep]
                    b_scores = b_scores[keep]
                    b_labels = b_labels[keep]
            else:
                device = cls_logits.device
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
        feat_map = self.extract_features(inputs)
        cls_logits, pred_boxes = self.head(feat_map)
        return self._compute_losses(cls_logits, pred_boxes, targets, device=inputs.device)

    def predict(self, inputs: torch.Tensor, conf_threshold=0.001, nms_threshold=0.5, **kwargs):
        feat_map = self.extract_features(inputs)
        cls_logits, pred_boxes = self.head(feat_map)
        return self._post_process(cls_logits, pred_boxes, conf_threshold=conf_threshold, nms_threshold=nms_threshold)

    def forward_val(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        conf_threshold: float = 0.001,
        nms_threshold: float = 0.5,
        max_det: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
        """Validation forward pass: Chỉ chạy mạng 1 LẦN duy nhất."""
        feat_map = self.extract_features(inputs)
        cls_logits, pred_boxes = self.head(feat_map)

        output = {}
        if targets is not None:
            loss_dict = self._compute_losses(cls_logits, pred_boxes, targets, device=inputs.device)
            output.update(loss_dict)

        preds = self._post_process(
            cls_logits, pred_boxes,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            max_det=max_det,
        )
        output["preds"] = preds
        return output
