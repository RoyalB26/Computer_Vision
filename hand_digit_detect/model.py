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
from .modules import Conv, C3k2, SPPF, C2PSA, Token_Cross_Attention, Slots_Cross_Attention

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


@MODELS.register_module(name="Hand_Digit_Detector")
class Hand_Digit_Detector(BaseModel):
    """Detector for COCO dataset predicting bounding boxes, class labels, and objectness."""

    def __init__(
        self,
        num_classes: int = 80,
        dim: int = 16,
        total_head: int = 16,
        images_size: int = 640,
        num_ref: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.total_head = total_head
        self.dim= dim
        self.num_slots= num_classes + 1
        # Shape: (num_classes, dim) -> each slot gets unique learned mu and logsigma
        self.mu = nn.Parameter(torch.randn(self.num_slots, self.total_head, dim))
        self.logsigma = nn.Parameter(torch.zeros(self.num_slots, self.total_head, dim))

        self.feature = nn.ModuleList([
            Conv(3, dim, k=1, s=1),                           # 0
            Slots_Cross_Attention(self.total_head, dim, dim),
            Conv(dim, 2 * dim, k=3, s=2),                          # 1
            Slots_Cross_Attention(self.total_head, dim, 2 * dim),
            C3k2(2 * dim, 4 * dim, n=1, c3k=False, e=0.25),            # 2
            Conv(4 * dim, 4 * dim, k=3, s=2),                          # 3
            Slots_Cross_Attention(self.total_head, dim, 4 * dim),
            C3k2(4 * dim, 8 * dim, n=1, c3k=False, e=0.25),           # 4
            Conv(8 * dim, 8 * dim, k=3, s=2),                        # 5
            Slots_Cross_Attention(self.total_head, dim, 8 * dim),
            C3k2(8 * dim, 8 * dim, n=1, c3k=True, e=0.5),            # 6
            Conv(8 * dim, 16 * dim, k=3, s=2),                        # 7
            C3k2(16 * dim, 16 * dim, n=1, c3k=True, e=0.5),            # 8
            Slots_Cross_Attention(self.total_head, dim, 16 * dim),
            Token_Cross_Attention(self.total_head, dim, dim)
        ])

        # Initialization
        nn.init.xavier_uniform_(self.mu)
        nn.init.zeros_(self.logsigma)
        torch.manual_seed(42)

        self.head = AnchorFreeDetHead(
            in_channels=dim,
            num_classes=num_classes,
        )

    def initalize_slots(self, batch_size=1, device=None):
        # Handle the case where device is passed as the first positional argument
        if isinstance(batch_size, (torch.device, str)):
            device = batch_size
            batch_size = 1

        device = device if device is not None else self.mu.device

        mu = self.mu.to(device)
        sigma = torch.exp(self.logsigma).to(device)

        noise = torch.randn(batch_size, self.num_slots, self.total_head, self.dim, device=device)

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
        slots = self.initalize_slots(B, x.device)
        feat= None
        for i,f in enumerate(self.feature):
            # print(i, slots.shape)
            if i in [1,3,6,9,13]:
                slots= f(slots, feat)
            elif i == 14:
                feat_map= f(slots, x)
            elif i == 0:
                x = f(x)
                feat= x
            else:
                feat= f(feat)
        return slots, feat_map


    def slot_orthogonality_loss(self, slots, eps=1e-8):
        """
        slots: [B, num_slots, num_heads, head_dim]  (B, K, H, D)
        Loss ép cosine similarity giữa các cặp slot khác nhau về 0 cho từng head.
        """
        B, K, H, D = slots.shape
        
        # 1. Chuyển vị về dạng [B, H, K, D] để xử lý song song các heads
        slots = slots.permute(0, 2, 1, 3)  # [B, H, K, D]
        
        # Gom (B, H) thành batch dimension để tận dụng torch.bmm: [B * H, K, D]
        slots = slots.reshape(B * H, K, D)
        
        # 2. Chuẩn hóa L2 dọc theo chiều feature D
        slots_norm = F.normalize(slots, p=2, dim=-1, eps=eps)  # [B * H, K, D]
        
        # 3. Tính ma trận Cosine Similarity giữa các cặp slot trong từng head
        # [B * H, K, D] x [B * H, D, K] -> [B * H, K, K]
        cos_sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))
        
        # 4. Loại bỏ đường chéo chính (tự so với chính nó)
        eye = torch.eye(K, device=slots.device).unsqueeze(0)  # [1, K, K]
        off_diagonal = cos_sim - eye                          # [B * H, K, K]
        
        # 5. Phạt bình phương các giá trị ngoài đường chéo
        # Số lượng cặp ngoài đường chéo là K * (K - 1)
        loss = (off_diagonal ** 2).sum(dim=(-2, -1)) / (K * (K - 1))  # [B * H]
        
        return loss.mean()


    def compute_loss(
        self,
        cls_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        obj_logits: torch.Tensor,
        slots: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tính toán Detection Loss: Classification + Bounding Box Regression + Objectness + Orthogonality."""
        device = cls_logits.device
        batch_size, _, grid_h, grid_w = cls_logits.shape

        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        pos_weight = torch.tensor([5.0], device=device)

        for b in range(batch_size):
            img_cls = cls_logits[b]      # (num_classes, H, W)
            img_box = pred_boxes[b]      # (4, H, W)
            img_obj = obj_logits[b, 0]   # (H, W)

            target_dict = targets[b] if targets is not None and b < len(targets) else None
            gt_boxes = target_dict.get("boxes", torch.empty((0, 4), device=device)) if target_dict else torch.empty((0, 4), device=device)
            gt_labels = target_dict.get("labels", torch.empty((0,), device=device, dtype=torch.long)) if target_dict else torch.empty((0,), device=device, dtype=torch.long)

            num_gt = len(gt_boxes)
            obj_target = torch.zeros((grid_h, grid_w), device=device)

            if num_gt > 0:
                gt_boxes = gt_boxes.to(device)
                gt_labels = gt_labels.to(device)

                # Gán nhãn vào cell gần tâm bounding box nhất
                gt_cx = gt_boxes[:, 0].clamp(0.0, 0.999)
                gt_cy = gt_boxes[:, 1].clamp(0.0, 0.999)

                grid_x = (gt_cx * grid_w).long()
                grid_y = (gt_cy * grid_h).long()

                obj_target[grid_y, grid_x] = 1.0

                p_cls = img_cls[:, grid_y, grid_x].permute(1, 0)
                p_box = img_box[:, grid_y, grid_x].permute(1, 0)

                valid_labels = gt_labels.clamp(0, self.num_classes - 1)
                total_cls_loss += F.cross_entropy(p_cls, valid_labels)
                total_box_loss += F.smooth_l1_loss(p_box, gt_boxes)

            total_obj_loss += F.binary_cross_entropy_with_logits(
                img_obj, obj_target, pos_weight=pos_weight
            )

        total_cls_loss /= max(batch_size, 1)
        total_box_loss /= max(batch_size, 1)
        total_obj_loss /= max(batch_size, 1)

        ortho_loss = self.slot_orthogonality_loss(slots)

        total_loss = total_cls_loss + 2.0 * total_box_loss + total_obj_loss + ortho_loss
        
        return {
            "loss": total_loss,
            "loss_cls": total_cls_loss,
            "loss_box": total_box_loss,
            "loss_obj": total_obj_loss,
            "ortho_loss": ortho_loss,
        }

    def decode_predictions(
        self,
        cls_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        obj_logits: torch.Tensor,
        conf_threshold: float = 0.05,
        nms_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode bounding box, class label và confidence score từ output head."""
        batch_size = cls_logits.shape[0]
        cls_probs = cls_logits.softmax(dim=1)
        obj_scores = obj_logits.sigmoid()

        results = []
        for b in range(batch_size):
            scores = cls_probs[b] * obj_scores[b]  # (num_classes, H, W)
            max_scores, labels = scores.max(dim=0)  # (H, W)

            mask = max_scores >= conf_threshold
            b_boxes = pred_boxes[b, :, mask].permute(1, 0)  # (K, 4) dạng [cx, cy, w, h]
            b_scores = max_scores[mask]                     # (K,)
            b_labels = labels[mask]                         # (K,)

            if len(b_boxes) > 0 and nms_threshold is not None:
                cx, cy, w, h = b_boxes.unbind(dim=1)
                xyxy = torch.stack(
                    [
                        cx - w / 2,
                        cy - h / 2,
                        cx + w / 2,
                        cy + h / 2,
                    ],
                    dim=1,
                )
                keep = torchvision.ops.batched_nms(xyxy, b_scores, b_labels, nms_threshold)
                b_boxes = b_boxes[keep]
                b_scores = b_scores[keep]
                b_labels = b_labels[keep]

            results.append({
                "boxes": b_boxes,
                "scores": b_scores,
                "labels": b_labels,
            })

        return results

    def forward_train(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Tính toán Detection Loss: Classification + Bounding Box Regression + Objectness."""
        slots, feat_map = self.extract_features(inputs)
        cls_logits, pred_boxes, obj_logits = self.head(feat_map)
        return self.compute_loss(cls_logits, pred_boxes, obj_logits, slots, targets)

    def predict(
        self,
        inputs: torch.Tensor,
        conf_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        """Dự đoán bounding box, class label và confidence score khi suy luận / đánh giá."""
        slots, feat_map = self.extract_features(inputs)
        cls_logits, pred_boxes, obj_logits = self.head(feat_map)
        return self.decode_predictions(
            cls_logits, pred_boxes, obj_logits, conf_threshold=conf_threshold, nms_threshold=nms_threshold
        )

    def forward_val(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        conf_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        **kwargs,
    ) -> Dict[str, Any]:
        """Validation forward pass tối ưu: chỉ chạy forward 1 lần duy nhất."""
        slots, feat_map = self.extract_features(inputs)
        cls_logits, pred_boxes, obj_logits = self.head(feat_map)

        preds = self.decode_predictions(
            cls_logits, pred_boxes, obj_logits, conf_threshold=conf_threshold, nms_threshold=nms_threshold
        )
        output = {"preds": preds}

        if targets is not None:
            losses = self.compute_loss(cls_logits, pred_boxes, obj_logits, slots, targets)
            output.update(losses)

        return output
