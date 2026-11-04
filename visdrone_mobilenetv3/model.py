"""MobileNetV3 Object Detector with DeformableHead for Hand Digit Detection."""
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from BaseModel.core.registrar import MODELS
from BaseModel.model import BaseModel

from .utils import cxcywh_to_xyxy
from .backbone import MobileNetV3Backbone, NormalBackbone
from .neck import MobileNetFPN, MobileNetPANet, PANet
from .head import MobileNetDetectionHead, DeformableHead
from .loss import MobileNetDetectionLoss, compute_multi_ref_loss



@MODELS.register_module(name="VisdroneDetector")
class VisdroneDetector(BaseModel):
    """
    Detector thuần CNN dùng MobileNetV3 backbone kết hợp Deformable Detection Head:
      - Backbone: MobileNetV3-Small (pretrained ImageNet)
      - Neck: Feature Pyramid Network (FPN) đưa về map_pred (Stride 8) & feat_deform (Stride 2)
      - Head: DeformableHead (hoặc MobileNetDetectionHead)
      - Loss: Dense Gaussian Focal Loss + Distance-IoU (DIoU) Loss + Deformable Regularization
    """

    def __init__(
        self,
        num_classes: int = 10,
        dim: int = 64,
        head_type: str = "DeformableHead",  # "DeformableHead" hoặc "MobileNetDetectionHead"
        num_heads: int = 4,
        num_points: int = 4,
        backbone: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
        target_size: int = 160,
        ref_weight: float = 1.0,
        obj_weight: float = 1.0,
        obj_pos_weight: float = 5.0,
        num_ref_per_obj: int = 4,
        ref_repulsion_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.head_type = head_type
        self.ref_weight = ref_weight
        self.obj_weight = obj_weight
        self.obj_pos_weight = obj_pos_weight
        self.num_ref_per_obj = num_ref_per_obj
        self.ref_repulsion_weight = ref_repulsion_weight

        # 1. MobileNetV3 / Custom Backbone
        if backbone is not None:
            if isinstance(backbone, dict):
                from BaseModel.core.registrar import BACKBONES
                self.backbone = BACKBONES.build(backbone)
            elif isinstance(backbone, str):
                from BaseModel.core.registrar import BACKBONES
                self.backbone = BACKBONES.build({"type": backbone})
            else:
                self.backbone = backbone
            in_channels = getattr(
                self.backbone, "out_channels", {'c0': 16, 'c1': 16, 'c3': 24, 'c5': 40, 'c12': 576}
            )
        else:
            self.backbone = MobileNetV3Backbone(pretrained=True)
            in_channels = self.backbone.out_channels

        # 2. Feature Pyramid Network (FPN / PANet)
        if neck is not None:
            if isinstance(neck, dict):
                from BaseModel.core.registrar import NECKS
                self.neck = NECKS.build(neck, default_args={"in_channels": in_channels, "out_dim": dim})
            elif isinstance(neck, str):
                from BaseModel.core.registrar import NECKS
                self.neck = NECKS.build({"type": neck, "in_channels": in_channels, "out_dim": dim})
            else:
                self.neck = neck
        else:
            self.neck = MobileNetFPN(in_channels=in_channels, out_dim=dim, return_tuple=True)

        # 3. Detection Head (DeformableHead mặc định, hoặc Standard Head)
        if head is not None:
            self.head = head
        else:
            if head_type.lower() in ("deformable", "deformablehead", "deformabledetectionhead"):
                self.head = DeformableHead(
                    num_classes=num_classes, dim=dim, num_heads=num_heads, num_points=num_points
                )
            else:
                self.head = MobileNetDetectionHead(num_classes=num_classes, dim=dim, head_conv=dim)

        # 4. Loss Function (Focal Loss + DIoU Loss + Ref Loss + Obj Loss)
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = MobileNetDetectionLoss(
                num_classes=num_classes,
                ref_weight=ref_weight,
                obj_weight=obj_weight,
                obj_pos_weight=obj_pos_weight,
                num_ref_per_obj=num_ref_per_obj,
                ref_repulsion_weight=ref_repulsion_weight,
            )

    def extract_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Trích xuất đặc trưng qua Backbone và Neck FPN."""
        feats = self.backbone(x)
        fpn_out = self.neck(feats)
        if isinstance(fpn_out, tuple):
            map_pred, feat_deform = fpn_out
            return map_pred, feat_deform
        return feats, None

    def compute_ref_loss(
        self,
        ref: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Tính toán Ref Loss đa điểm:
        - Cho phép nhiều ref trỏ về cùng 1 object (phân bố tại nhiều vị trí khác nhau trong box).
        - Có repulsion loss ngăn không cho các ref trùng tọa độ.
        - Bao quát toàn bộ các object trong ảnh theo hạn ngạch công bằng, không bị dồn cục vào 1 object.
        """
        return compute_multi_ref_loss(
            ref_tensor=ref,
            targets=targets,
            device=device,
            num_classes=self.num_classes,
            num_ref_per_obj=getattr(self, "num_ref_per_obj", 4),
            repulsion_weight=getattr(self, "ref_repulsion_weight", 0.5),
        )

    def compute_obj_loss(
        self,
        obj_logits: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]],
        device: torch.device,
        pos_weight: float = 5.0,
    ) -> torch.Tensor:
        """
        Tính toán Objectness Loss (BCEWithLogitsLoss với pos_weight).

        Args:
            obj_logits: Tensor [B, 1, H, W] chứa dự đoán objectness logits.
            targets: Danh sách target dicts chứa bounding boxes.
            device: Thiết bị torch.device.
            pos_weight: Trọng số phạt cho positive target (mặc định 5.0).

        Returns:
            torch.Tensor: Scalar loss cho objectness.
        """
        if obj_logits is None or targets is None:
            return torch.tensor(0.0, device=device)

        batch_size, _, grid_h, grid_w = obj_logits.shape
        obj_target = torch.zeros((batch_size, 1, grid_h, grid_w), device=device, dtype=obj_logits.dtype)

        for b in range(batch_size):
            target_dict = targets[b] if b < len(targets) else None
            if not target_dict:
                continue

            gt_boxes = target_dict.get("boxes", torch.empty((0, 4), device=device))
            if len(gt_boxes) == 0:
                continue

            gt_boxes = gt_boxes.to(device)
            is_cxcywh = target_dict.get("_is_cxcywh", False)
            if is_cxcywh:
                gt_cx = gt_boxes[:, 0].clamp(0.0, 0.999)
                gt_cy = gt_boxes[:, 1].clamp(0.0, 0.999)
            else:
                gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2] / 2.0).clamp(0.0, 0.999)
                gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3] / 2.0).clamp(0.0, 0.999)

            grid_x = (gt_cx * grid_w).long().clamp(0, grid_w - 1)
            grid_y = (gt_cy * grid_h).long().clamp(0, grid_h - 1)
            obj_target[b, 0, grid_y, grid_x] = 1.0

        weight_tensor = torch.tensor([pos_weight], device=device, dtype=obj_logits.dtype)
        return F.binary_cross_entropy_with_logits(obj_logits, obj_target, pos_weight=weight_tensor)

    def forward_train(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        map_pred, feat_deform = self.extract_features(inputs)
        head_out = self.head(map_pred, feat_deform)

        if len(head_out) == 4:
            cls_logits, pred_boxes, obj_logits, sampling_info = head_out
        elif len(head_out) == 3:
            cls_logits, pred_boxes, extra = head_out
            if isinstance(extra, dict):
                sampling_info = extra
                obj_logits = extra.get("obj_logits", None)
            elif isinstance(extra, torch.Tensor) and extra.shape[1] == 1:
                obj_logits = extra
                sampling_info = None
            else:
                sampling_info = extra
                obj_logits = None
        else:
            cls_logits, pred_boxes = head_out
            obj_logits = None
            sampling_info = None

        loss_dict = self.loss_fn(
            cls_logits=cls_logits,
            pred_boxes=pred_boxes,
            targets=targets,
            device=inputs.device,
            sampling_info=sampling_info,
            obj_logits=obj_logits,
        )

        if "loss_ref" not in loss_dict:
            ref = None
            if isinstance(sampling_info, dict) and "ref" in sampling_info:
                ref = sampling_info["ref"]
            elif isinstance(sampling_info, torch.Tensor):
                ref = sampling_info
            elif hasattr(self.head, "last_ref") and self.head.last_ref is not None:
                ref = self.head.last_ref

            if ref is not None and targets is not None:
                ref_loss = self.compute_ref_loss(ref, targets, inputs.device)
                loss_dict["loss_ref"] = ref_loss
                loss_dict["loss"] = loss_dict["loss"] + self.ref_weight * ref_loss

        if "loss_obj" not in loss_dict and obj_logits is not None and targets is not None:
            obj_loss = self.compute_obj_loss(obj_logits, targets, inputs.device, pos_weight=self.obj_pos_weight)
            loss_dict["loss_obj"] = obj_loss
            loss_dict["loss"] = loss_dict["loss"] + self.obj_weight * obj_loss

        return loss_dict

    def _post_process(
        self,
        cls_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        obj_logits: Optional[torch.Tensor] = None,
        conf_threshold: float = 0.10,
        nms_threshold: float = 0.5,
        max_det: int = 100,
        max_candidates: int = 1000,
    ) -> List[Dict[str, torch.Tensor]]:
        """Lọc peak cực đại cục bộ 3x3 và batched NMS."""
        batch_size, num_classes, grid_h, grid_w = cls_logits.shape
        cls_scores = cls_logits.sigmoid()

        if obj_logits is not None:
            obj_scores = obj_logits.sigmoid()
            cls_scores = cls_scores * obj_scores

        # CenterNet Local Peak Max-Pooling (3x3)
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

                if len(b_scores) > max_candidates:
                    topk_idx = torch.topk(b_scores, max_candidates).indices
                    b_scores = b_scores[topk_idx]
                    b_labels = b_labels[topk_idx]
                    y_idx = y_idx[topk_idx]
                    x_idx = x_idx[topk_idx]

                sample_boxes = pred_boxes[b, :, y_idx, x_idx].permute(1, 0)

                if nms_threshold is not None and len(sample_boxes) > 0:
                    xyxy = cxcywh_to_xyxy(sample_boxes)
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

    def predict(
        self,
        inputs: torch.Tensor,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        max_det: int = 100,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        map_pred, feat_deform = self.extract_features(inputs)
        head_out = self.head(map_pred, feat_deform)
        cls_logits, pred_boxes = head_out[0], head_out[1]
        obj_logits = None
        if len(head_out) >= 3 and isinstance(head_out[2], torch.Tensor) and head_out[2].ndim == 4 and head_out[2].shape[1] == 1:
            obj_logits = head_out[2]
        elif len(head_out) >= 4 and isinstance(head_out[3], dict) and "obj_logits" in head_out[3]:
            obj_logits = head_out[3]["obj_logits"]

        return self._post_process(
            cls_logits, pred_boxes,
            obj_logits=obj_logits,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            max_det=max_det,
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
        """Validation forward pass: tính validation losses và sinh preds phục vụ tính mAP."""
        map_pred, feat_deform = self.extract_features(inputs)
        head_out = self.head(map_pred, feat_deform)

        if len(head_out) == 4:
            cls_logits, pred_boxes, obj_logits, sampling_info = head_out
        elif len(head_out) == 3:
            cls_logits, pred_boxes, extra = head_out
            if isinstance(extra, dict):
                sampling_info = extra
                obj_logits = extra.get("obj_logits", None)
            elif isinstance(extra, torch.Tensor) and extra.shape[1] == 1:
                obj_logits = extra
                sampling_info = None
            else:
                sampling_info = extra
                obj_logits = None
        else:
            cls_logits, pred_boxes = head_out
            obj_logits = None
            sampling_info = None

        output = {}
        if targets is not None:
            loss_dict = self.loss_fn(
                cls_logits=cls_logits,
                pred_boxes=pred_boxes,
                targets=targets,
                device=inputs.device,
                sampling_info=sampling_info,
                obj_logits=obj_logits,
            )
            if "loss_ref" not in loss_dict:
                ref = None
                if isinstance(sampling_info, dict) and "ref" in sampling_info:
                    ref = sampling_info["ref"]
                elif isinstance(sampling_info, torch.Tensor):
                    ref = sampling_info
                elif hasattr(self.head, "last_ref") and self.head.last_ref is not None:
                    ref = self.head.last_ref

                if ref is not None:
                    ref_loss = self.compute_ref_loss(ref, targets, inputs.device)
                    loss_dict["loss_ref"] = ref_loss
                    loss_dict["loss"] = loss_dict["loss"] + self.ref_weight * ref_loss

            if "loss_obj" not in loss_dict and obj_logits is not None:
                obj_loss = self.compute_obj_loss(obj_logits, targets, inputs.device, pos_weight=self.obj_pos_weight)
                loss_dict["loss_obj"] = obj_loss
                loss_dict["loss"] = loss_dict["loss"] + self.obj_weight * obj_loss

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
            obj_logits=obj_logits,
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
        if self.training:
            return self.forward_train(inputs, targets=targets, **kwargs)
        else:
            return self.forward_val(inputs, targets=targets, **kwargs)

    def analyze_detection_components(self, inputs: torch.Tensor) -> Dict[str, float]:
        """Thống kê các chỉ số đặc trưng phục vụ ExcelTracker."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            map_pred, feat_deform = self.extract_features(inputs)
            head_out = self.head(map_pred, feat_deform)
            cls_logits, pred_boxes = head_out[0], head_out[1]
            obj_logits = None
            if len(head_out) >= 3 and isinstance(head_out[2], torch.Tensor) and head_out[2].ndim == 4 and head_out[2].shape[1] == 1:
                obj_logits = head_out[2]
                sampling_info = head_out[3] if len(head_out) >= 4 else None
            else:
                sampling_info = head_out[2] if len(head_out) >= 3 else None
                obj_logits = sampling_info.get("obj_logits", None) if isinstance(sampling_info, dict) else None

            if obj_logits is None and hasattr(self.head, "last_obj_logits"):
                obj_logits = self.head.last_obj_logits

            cls_scores = cls_logits.sigmoid()

            results = {
                "Cls_Logits_Mean": round(cls_logits.mean().item(), 4),
                "Cls_Logits_Max": round(cls_logits.max().item(), 4),
                "Cls_Logits_Min": round(cls_logits.min().item(), 4),
                "Cls_Prob_Mean": round(cls_scores.mean().item(), 4),
                "Cls_Prob_Max": round(cls_scores.max().item(), 4),
                "Box_CX_Mean": round(pred_boxes[:, 0].mean().item(), 4),
                "Box_CY_Mean": round(pred_boxes[:, 1].mean().item(), 4),
                "Box_W_Mean": round(pred_boxes[:, 2].mean().item(), 4),
                "Box_H_Mean": round(pred_boxes[:, 3].mean().item(), 4),
                "MapPred_Energy": round(map_pred.norm(dim=1).mean().item(), 4),
            }

            if obj_logits is not None:
                obj_scores = obj_logits.sigmoid()
                results["Obj_Prob_Mean"] = round(obj_scores.mean().item(), 4)
                results["Obj_Prob_Max"] = round(obj_scores.max().item(), 4)
                results["Obj_Logits_Mean"] = round(obj_logits.mean().item(), 4)

            if sampling_info is not None:
                if isinstance(sampling_info, dict):
                    all_locs = sampling_info.get("sampling_locations", None)
                    all_offsets = sampling_info.get("offsets", None)
                    ref_tensor = sampling_info.get("ref", None)
                else:
                    all_locs = None
                    all_offsets = None
                    ref_tensor = sampling_info if isinstance(sampling_info, torch.Tensor) else None

                if ref_tensor is None and hasattr(self.head, "last_ref"):
                    ref_tensor = self.head.last_ref

                if all_locs is not None and all_offsets is not None:
                    offset_norm = torch.norm(all_offsets, p=2, dim=-1).mean()
                    bound_violation = (
                        F.relu(-all_locs[..., 0]) + F.relu(all_locs[..., 0] - 1.0) +
                        F.relu(-all_locs[..., 1]) + F.relu(all_locs[..., 1] - 1.0)
                    ).gt(0).float().mean()
                    results["Deform_Offset_Norm"] = round(offset_norm.item(), 4)
                    results["Deform_Out_Bound_Ratio"] = round(bound_violation.item(), 4)

                if ref_tensor is not None:
                    if ref_tensor.ndim == 4:
                        results["Ref_X_Mean"] = round(ref_tensor[:, 0].mean().item(), 4)
                        results["Ref_Y_Mean"] = round(ref_tensor[:, 1].mean().item(), 4)
                    elif ref_tensor.ndim == 3:
                        results["Ref_X_Mean"] = round(ref_tensor[..., 0].mean().item(), 4)
                        results["Ref_Y_Mean"] = round(ref_tensor[..., 1].mean().item(), 4)

        if was_training:
            self.train()
        return results
