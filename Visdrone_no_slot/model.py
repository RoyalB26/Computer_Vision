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


class UpSample(nn.Module):
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

        # Backbone & FPN
        self.backbone = BackBone()
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels=576, out_channels=dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.downsample = DownSample(dim)
        self.up_sample = UpSample(dim)

        # Detection Head tiêu chuẩn
        self.head = StandardDetHead(dim=dim, num_classes=num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất đặc trưng qua Backbone và FPN, không dùng slot attention."""
        c1, c3, c5, c12 = self.backbone(x)
        # c12 có 576 kênh, đưa trực tiếp vào up_sample
        feat_map = self.up_sample(c1, c3, c5, c12)  # [B, dim, 160, 160]
        return feat_map

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

            # Clamp logits để đảm bảo ổn định số học trong AMP FP16
            img_cls_clamped = torch.clamp(img_cls, min=-15.0, max=15.0)
            cls_loss = F.binary_cross_entropy_with_logits(
                img_cls_clamped, cls_target, 
                pos_weight=torch.tensor([10.0], device=device, dtype=img_cls.dtype),
                reduction='sum'
            )
            total_cls_loss += cls_loss

        norm_pixels = max(batch_size * num_classes * grid_h * grid_w, 1)
        total_cls_loss = total_cls_loss / norm_pixels
        total_box_loss = (total_box_loss / max(num_pos_total, 1)) if num_pos_total > 0 else (total_box_loss * 0.0)
        
        # Phòng ngừa triệt để NaN / Inf
        total_cls_loss = torch.nan_to_num(total_cls_loss, nan=0.0, posinf=1.0, neginf=0.0)
        total_box_loss = torch.nan_to_num(total_box_loss, nan=0.0, posinf=1.0, neginf=0.0)
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
