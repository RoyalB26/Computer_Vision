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



@MODELS.register_module(name="CocoDetector")
@MODELS.register_module(name="ConvNetDetector")
class CocoDetector(BaseModel):
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

        self.norm_out = nn.LayerNorm(dim)
        self.downsample = DownSample(total_head * dim)
        self.up_sample = UpSample(dim)
        self.deformable = Deformable(dim, (images_size // 32) ** 2, num_ref)
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
        x = self.conv(x).squeeze(1) # [B, num_classes + 1, dim]
        x = self.fc2(F.relu(self.fc1(self.norm_out(x))))
        slots = slots + x
        slots = slots[:, :self.num_classes, :] # [B, num_classes, dim]

        attn[:, :self.num_classes, :, :]
        attn = attn.permute(0, 2, 1, 3)
        attn = self.conv(attn).squeeze(1) # [B, num_classes, N]
        B,D,H,W = c12.shape
        N = H * W
        c12_flat = c12.view(B, D, N)  # [B, D, N]
        c12 = c12_flat.unsqueeze(1).expand(-1, self.num_classes, -1, -1)  # [B, C, D, N]
        c12= torch.einsum("b c d n, b c n -> b d n", c12, attn)
        c12= c12.view(B, D, H, W)
        feat_map = self.up_sample(c1, c3, c5, c12)
        return slots, feat_map, attn



    def forward_train(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Tính toán Detection Loss: Classification + Bounding Box Regression + Objectness."""
        device = inputs.device
        slots, feat_map, attn = self.extract_features(inputs)
        feat = self.deformable(slots, feat_map, attn)
        B,D,H,W = feat.shape
        cls_logits, pred_boxes, obj_logits= self.head(feat)
        # print(cls_logits.shape)
        # print(pred_boxes.shape)
        # print(obj_logits.shape)

        # cls_logits, pred_boxes, obj_logits = self.head(feat)
        batch_size, _, grid_h, grid_w = cls_logits.shape

        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)

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
                cls_loss = F.cross_entropy(p_cls, valid_labels)
                box_loss = F.smooth_l1_loss(p_box, gt_boxes)

                total_cls_loss += cls_loss
                total_box_loss += box_loss

            obj_loss = F.binary_cross_entropy_with_logits(
                img_obj, obj_target, pos_weight=torch.tensor([5.0], device=device)
            )
            total_obj_loss += obj_loss

        total_cls_loss /= max(batch_size, 1)
        total_box_loss /= max(batch_size, 1)
        total_obj_loss /= max(batch_size, 1)

        total_loss = total_cls_loss + 2.0 * total_box_loss + total_obj_loss

        return {
            "loss": total_loss,
            "loss_cls": total_cls_loss,
            "loss_box": total_box_loss,
            "loss_obj": total_obj_loss,
        }

    def predict(
        self,
        inputs: torch.Tensor,
        conf_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        """Dự đoán bounding box, class label và confidence score khi suy luận / đánh giá."""
        slots, feat_map, attn = self.extract_features(inputs)
        feat = self.deformable(slots, feat_map, attn)
        B,D,H,W = feat.shape
        cls_logits, pred_boxes, obj_logits= self.head(feat)

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
                # Chuyển đổi cxcywh sang xyxy để chạy NMS
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
                b_boxes = b_boxes[keep]
                b_scores = b_scores[keep]
                b_labels = b_labels[keep]

            results.append({
                "boxes": b_boxes,
                "scores": b_scores,
                "labels": b_labels,
            })

        return results

    def forward_val(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Validation forward pass: sinh dự đoán cho mAP và tính val loss."""
        preds = self.predict(inputs, **kwargs)
        output = {"preds": preds}

        if targets is not None:
            train_out = self.forward_train(inputs, targets, **kwargs)
            output.update(train_out)

        return output
