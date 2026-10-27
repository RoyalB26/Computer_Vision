"""COCO Object Detection Model predicting bounding boxes and class labels."""
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
import numpy as np
from torch import Tensor
import math
from attn_kernel import AttnKernel
from BaseModel.core import MODELS
from BaseModel.model import BaseModel
from BaseModel.model.layers_module import (
    ConvModule,
    FPN,
    AnchorFreeDetHead,
    ResNetBackbone,
    SimpleCNNBackbone,
)


class Attention_Slot(nn.Module):
    def __init__(self, dim, num_slot):
        super().__init__()
        self.slot_dim = dim
        self.num_slot = num_slot
        self.total_slots = num_slot  

        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_q = nn.Linear(dim, dim)

        # Shape: (total_slots, dim) -> each slot gets unique learned mu and logsigma
        self.mu = nn.Parameter(torch.randn(self.total_slots, dim))
        self.logsigma = nn.Parameter(torch.zeros(self.total_slots, dim))

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

        # Initialization
        nn.init.xavier_uniform_(self.mu)
        nn.init.zeros_(self.logsigma)

        torch.manual_seed(42)

    def initalize_slots(self, batch_size=1, device=None):
        # Handle the case where device is passed as the first positional argument
        if isinstance(batch_size, (torch.device, str)):
            device = batch_size
            batch_size = 1

        device = device if device is not None else self.mu.device

        mu = self.mu.to(device)
        sigma = torch.exp(self.logsigma).to(device)

        noise = torch.randn(batch_size, self.total_slots, self.slot_dim, device=device)

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
    
    def forward(self, x: Tensor):
        """
        x: [B, 5, C, N]
        """
        B,K,N,D = x.shape
        k = self.to_k(x) # [B, N, dim]

        v = self.to_v(x) # [B, N, dim]
        slots = self.initalize_slots(batch_size= B,device= x.device) # [B, num_slot, dim]
        
        q = self.to_q(slots) # [B, num_slot, dim]
        B, N, D = q.shape
        scale = 1.0 / math.sqrt(D)

        # Đưa N vào batch: (B*N, ...)
        # q: (B*N, 1, D)
        q_reshaped = q.view(B * N, 1, D)

        # k ban đầu: (B, 5, N, D) -> đổi chỗ thành (B, N, 5, D) -> view (B*N, 5, D)
        k_reshaped = k.permute(0, 2, 1, 3).reshape(B * N, 5, D)

        # Tính attention scores: (B*N, 1, D) @ (B*N, D, 5) -> (B*N, 1, 5)
        scores = torch.bmm(q_reshaped, k_reshaped.transpose(1, 2)) * scale
        weights = F.softmax(scores, dim=-1)  # shape: (B*N, 1, 5)

        # Nếu nhân tiếp với v: (B, 5, N, D) -> (B*N, 5, D)
        v_reshaped = v.permute(0, 2, 1, 3).reshape(B * N, 5, D)

        # out: (B*N, 1, 5) @ (B*N, 5, D) -> (B*N, 1, D) -> (B, N, D)
        x = torch.bmm(weights, v_reshaped).view(B, N, D)
        x = self.fc2(F.relu(self.fc1(self.norm_out(x))))
        slots = slots + x
        return slots

class Bottleneck(nn.Module):
    """Khối Bottleneck nén-bung bên trong C3k2."""
    def __init__(self, c, shortcut=True):
        super().__init__()
        # Nén từ c -> c // 2 rồi bung lại c // 2 -> c
        c_hidden = c // 2
        self.cv1 = ConvModule(c, c_hidden, kernel_size=3, padding=1, norm_cfg="bn", act_cfg="Hardswish")
        self.cv2 = ConvModule(c_hidden, c, kernel_size=3, padding=1, norm_cfg="bn", act_cfg="Hardswish")
        self.add = shortcut
    def forward(self, x: Tensor) -> Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out

@MODELS.register_module(name="C3k2")
class C3k2(nn.Module):
    """
    C3k2 chuẩn theo YOLO11:
      - c1: Số kênh đầu vào
      - c2: Số kênh đầu ra mong muốn
    """
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        self.c = c1 // 2  # Số kênh của mỗi nửa sau khi split
        
        # cv1: Tiếp nhận c1 kênh đầu vào
        self.cv1 = ConvModule(c1, c1, kernel_size=1, stride=1, norm_cfg="bn", act_cfg="SiLU")
        
        # m: Chứa n khối Bottleneck xử lý trên self.c kênh
        self.m = nn.ModuleList([Bottleneck(self.c, shortcut=shortcut) for _ in range(n)])
        
        # cv2: Ghép các nhánh lại.
        # Với n=1: gồm [nhánh 1] + [nhánh 2] + [bottleneck] = 3 * self.c kênh
        total_cat_channels = (2 + n) * self.c
        self.cv2 = ConvModule(total_cat_channels, c2, kernel_size=1, stride=1, norm_cfg="bn", act_cfg="SiLU")
    def forward(self, x: Tensor) -> Tensor:
        # Bước 1: Cho qua conv1 rồi chia làm 2 nhánh (mỗi nhánh có self.c kênh)
        # y[0] là nhánh 1 (đi tắt), y[1] là nhánh 2 (đi qua Bottleneck)
        y = list(self.cv1(x).chunk(2, dim=1))
        
        # Bước 2: Cho nhánh 2 đi qua chuỗi Bottleneck
        # Mỗi lần qua một Bottleneck, ta lưu kết quả vào danh sách y để concat
        for b in self.m:
            y.append(b(y[-1]))
            
        # Bước 3: Ghép toàn bộ các nhánh lại dọc theo chiều kênh (dim=1)
        # Tensor sau khi cat có (2 + n) * (c1 // 2) kênh
        out = torch.cat(y, dim=1)
        
        # Bước 4: Chiếu qua conv2 để ra số kênh mong muốn c2
        return self.cv2(out)

  
@MODELS.register_module(name="CocoDetector")
@MODELS.register_module(name="ConvNetDetector")
class CocoDetector(BaseModel):
    """Detector for COCO dataset predicting bounding boxes, class labels, and objectness."""

    def __init__(
        self,
        backbone_type: str = "simple_cnn",
        in_channels: int = 3,
        num_classes: int = 80,
        hidden_dim: int = 32,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_channels = hidden_dim
        self.backbone_type = backbone_type

        
        # 1. Khởi tạo Backbone
        self.features = nn.Sequential(
            ConvModule(in_channels, hidden_dim, kernel_size= 1, stride= 1, padding= 0, norm_cfg= "bn", act_cfg="Hardswish"),
            ConvModule(hidden_dim, hidden_dim, kernel_size= 3, stride= 2, padding= 1, norm_cfg= "bn", act_cfg="Hardswish"),
            # C3k2(hidden_dim, hidden_dim),
            ConvModule( hidden_dim,  hidden_dim, kernel_size= 3, stride= 2, padding= 1, norm_cfg= "bn", act_cfg="Hardswish"),
            # C3k2(hidden_dim, hidden_dim),
            ConvModule( hidden_dim, hidden_dim, kernel_size= 3, stride= 2, padding= 1, norm_cfg= "bn", act_cfg="Hardswish"),
            # C3k2(hidden_dim, hidden_dim),
            ConvModule(hidden_dim, hidden_dim, kernel_size= 3, stride= 2, padding= 1, norm_cfg= "bn", act_cfg="Hardswish"),

            ConvModule(hidden_dim, hidden_dim, kernel_size= 3, stride= 1, padding= 1, norm_cfg= "bn", act_cfg="Hardswish"),
            ConvModule(hidden_dim, 4 * hidden_dim, kernel_size= 3, stride= 1, padding= 1, norm_cfg= "bn", act_cfg="Hardswish"),
            ConvModule(4 * hidden_dim, 4 * hidden_dim, kernel_size= 3, stride= 1, padding= 1, norm_cfg= "bn", act_cfg="Hardswish"),
        )

        self.cls_conv = ConvModule(4 * hidden_dim, num_classes, kernel_size= 1, stride= 1, padding= 0, norm_cfg= "bn", act_cfg="Hardswish")


        self.cls_fc = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, num_classes)
        )

        self.box_conv = ConvModule(4 * hidden_dim, 4, kernel_size= 1, stride= 1, padding= 0, norm_cfg= "bn", act_cfg="Hardswish")
        self.box_fc = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4)
        )

        self.obj_conv = ConvModule(4 * hidden_dim, num_classes, kernel_size= 1, stride= 1, padding= 0, norm_cfg= "bn", act_cfg="Hardswish")
        self.obj_fc = nn.Sequential(
            nn.Linear(num_classes, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.Sigmoid()
        )


        # 3. Khởi tạo Detection Head: Dự đoán đồng thời class, bounding box và objectness
        # self.head = AnchorFreeDetHead(
        #     in_channels=6,
        #     num_classes=num_classes,
        # )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất đặc trưng qua Backbone và FPN."""
        H, W = x.shape[1], x.shape[2]
        feat = self.features(x)
        return feat

    def forward_train(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Tính toán Detection Loss: Classification + Bounding Box Regression + Objectness."""
        device = inputs.device
        feat = self.extract_features(inputs)
        B,C,H,W = feat.shape

        feat_cls = self.cls_conv(feat)
        feat_cls = feat_cls.permute(0, 2, 3, 1)
        cls_logits = self.cls_fc(feat_cls)
        cls_logits = cls_logits.permute(0, 3, 1, 2)

        feat_box = self.box_conv(feat)
        feat_box = feat_box.permute(0, 2, 3, 1)
        pred_boxes = self.box_fc(feat_box)
        pred_boxes = pred_boxes.permute(0, 3, 1, 2)

        feat_obj = self.obj_conv(feat)
        feat_obj = feat_obj.permute(0, 2, 3, 1)
        obj_logits = self.obj_fc(feat_obj)
        obj_logits = obj_logits.permute(0, 3, 1, 2)

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
        feat = self.extract_features(inputs)
        B,C,H,W = feat.shape

        feat_cls = self.cls_conv(feat)
        feat_cls = feat_cls.permute(0, 2, 3, 1)
        cls_logits = self.cls_fc(feat_cls)
        cls_logits = cls_logits.permute(0, 3, 1, 2)

        feat_box = self.box_conv(feat)
        feat_box = feat_box.permute(0, 2, 3, 1)
        pred_boxes = self.box_fc(feat_box)
        pred_boxes = pred_boxes.permute(0, 3, 1, 2)

        feat_obj = self.obj_conv(feat)
        feat_obj = feat_obj.permute(0, 2, 3, 1)
        obj_logits = self.obj_fc(feat_obj)
        obj_logits = obj_logits.permute(0, 3, 1, 2)

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
