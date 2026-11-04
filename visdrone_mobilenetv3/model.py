"""MobileNetV3 Object Detector with DeformableHead for Hand Digit Detection."""
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from BaseModel.core.registrar import MODELS
from BaseModel.model import BaseModel

from .utils import cxcywh_to_xyxy
from .backbone import MobileNetV3Backbone, NormalBackbone, YOLOv11SmallBackbone, YOLOv11NanoBackbone
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
        ref: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: Optional[List[Dict[str, torch.Tensor]]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Tính toán Ref Loss đa điểm:
        - Cho phép nhiều ref trỏ về cùng 1 object (phân bố tại nhiều vị trí khác nhau trong box).
        - Có repulsion loss ngăn không cho các ref trùng tọa độ.
        - Bao quát toàn bộ các object trong ảnh theo hạn ngạch công bằng, không bị dồn cục vào 1 object.
        """
        if isinstance(ref, dict):
            losses = [
                compute_multi_ref_loss(
                    ref_tensor=v,
                    targets=targets,
                    device=device,
                    num_classes=self.num_classes,
                    num_ref_per_obj=getattr(self, "num_ref_per_obj", 4),
                    repulsion_weight=getattr(self, "ref_repulsion_weight", 0.5),
                    scale_range=getattr(self.loss_fn, "scale_ranges", {}).get(k, None),
                )
                for k, v in ref.items()
            ]
            return sum(losses) / max(len(losses), 1)

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
        obj_logits: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: Optional[List[Dict[str, torch.Tensor]]],
        device: torch.device,
        pos_weight: float = 5.0,
    ) -> torch.Tensor:
        """
        Tính toán Objectness Loss (BCEWithLogitsLoss với pos_weight).

        Args:
            obj_logits: Tensor [B, 1, H, W] hoặc Dict chứa dự đoán objectness logits.
            targets: Danh sách target dicts chứa bounding boxes.
            device: Thiết bị torch.device.
            pos_weight: Trọng số phạt cho positive target (mặc định 5.0).

        Returns:
            torch.Tensor: Scalar loss cho objectness.
        """
        if obj_logits is None or targets is None:
            return torch.tensor(0.0, device=device)

        if isinstance(obj_logits, dict):
            losses = [
                self.compute_obj_loss(v, targets, device, pos_weight)
                for v in obj_logits.values()
            ]
            return sum(losses) / max(len(losses), 1)

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
        cls_logits: Union[torch.Tensor, Dict[str, torch.Tensor]],
        pred_boxes: Union[torch.Tensor, Dict[str, torch.Tensor]],
        obj_logits: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        conf_threshold: float = 0.10,
        nms_threshold: float = 0.5,
        max_det: int = 100,
        max_candidates: int = 1000,
    ) -> List[Dict[str, torch.Tensor]]:
        """Lọc peak cực đại cục bộ 3x3 và batched NMS cho multi-scale predictions (c3, c5)."""
        if isinstance(cls_logits, torch.Tensor):
            cls_dict = {"single": cls_logits}
            boxes_dict = {"single": pred_boxes}
            obj_dict = {"single": obj_logits} if obj_logits is not None else {}
        else:
            cls_dict = cls_logits
            boxes_dict = pred_boxes
            obj_dict = obj_logits if isinstance(obj_logits, dict) else {}

        first_tensor = next(iter(cls_dict.values()))
        batch_size = first_tensor.shape[0]
        device = first_tensor.device

        batch_candidates = [[] for _ in range(batch_size)]

        for k in cls_dict.keys():
            lvl_cls = cls_dict[k]
            lvl_boxes = boxes_dict[k]
            lvl_obj = obj_dict.get(k, None)

            lvl_scores = lvl_cls.sigmoid()
            if lvl_obj is not None:
                lvl_scores = lvl_scores * lvl_obj.sigmoid()

            # CenterNet Local Peak Max-Pooling (3x3) per level
            hmax = F.max_pool2d(lvl_scores, kernel_size=3, stride=1, padding=1)
            lvl_scores = lvl_scores * (lvl_scores == hmax).float()

            for b in range(batch_size):
                b_scores_map = lvl_scores[b]  # [C, H_k, W_k]
                c_idx, y_idx, x_idx = torch.where(b_scores_map >= conf_threshold)

                if len(c_idx) > 0:
                    b_scores = b_scores_map[c_idx, y_idx, x_idx]
                    b_labels = c_idx
                    sample_boxes = lvl_boxes[b, :, y_idx, x_idx].permute(1, 0)
                    batch_candidates[b].append((sample_boxes, b_scores, b_labels))

        results = []
        for b in range(batch_size):
            if len(batch_candidates[b]) > 0:
                all_boxes = torch.cat([c[0] for c in batch_candidates[b]], dim=0)
                all_scores = torch.cat([c[1] for c in batch_candidates[b]], dim=0)
                all_labels = torch.cat([c[2] for c in batch_candidates[b]], dim=0)

                if len(all_scores) > max_candidates:
                    topk_idx = torch.topk(all_scores, max_candidates).indices
                    all_boxes = all_boxes[topk_idx]
                    all_scores = all_scores[topk_idx]
                    all_labels = all_labels[topk_idx]

                if nms_threshold is not None and len(all_boxes) > 0:
                    xyxy = cxcywh_to_xyxy(all_boxes)
                    keep = torchvision.ops.batched_nms(xyxy, all_scores, all_labels, nms_threshold)
                    keep = keep[:max_det]
                    all_boxes = all_boxes[keep]
                    all_scores = all_scores[keep]
                    all_labels = all_labels[keep]

                results.append({
                    "boxes": all_boxes,
                    "scores": all_scores,
                    "labels": all_labels,
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
        conf_threshold: float = 0.001,
        nms_threshold: float = 0.5,
        max_det: int = 100,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        map_pred, feat_deform = self.extract_features(inputs)
        head_out = self.head(map_pred, feat_deform)
        cls_logits, pred_boxes = head_out[0], head_out[1]
        obj_logits = None
        if len(head_out) >= 3 and head_out[2] is not None:
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

        preds = self._post_process(
            cls_logits=cls_logits,
            pred_boxes=pred_boxes,
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
            obj_logits = head_out[2] if len(head_out) >= 3 else None
            sampling_info = head_out[3] if len(head_out) >= 4 else None

            # Nếu là multi-scale dict, lấy tầng chính 'c3' (hoặc phần tử đầu tiên)
            if isinstance(cls_logits, dict):
                k0 = "c3" if "c3" in cls_logits else next(iter(cls_logits.keys()))
                cls_t = cls_logits[k0]
                box_t = pred_boxes[k0]
                map_t = map_pred[k0] if isinstance(map_pred, dict) else map_pred
                obj_t = obj_logits[k0] if isinstance(obj_logits, dict) and k0 in obj_logits else None
                s_info_t = sampling_info[k0] if isinstance(sampling_info, dict) and k0 in sampling_info else None
            else:
                cls_t = cls_logits
                box_t = pred_boxes
                map_t = map_pred
                obj_t = obj_logits
                s_info_t = sampling_info

            if obj_t is None and hasattr(self.head, "last_obj_logits"):
                last_obj = self.head.last_obj_logits
                obj_t = last_obj.get("c3", None) if isinstance(last_obj, dict) else last_obj

            cls_scores = cls_t.sigmoid()

            results = {
                "Cls_Logits_Mean": round(cls_t.mean().item(), 4),
                "Cls_Logits_Max": round(cls_t.max().item(), 4),
                "Cls_Logits_Min": round(cls_t.min().item(), 4),
                "Cls_Prob_Mean": round(cls_scores.mean().item(), 4),
                "Cls_Prob_Max": round(cls_scores.max().item(), 4),
                "Box_CX_Mean": round(box_t[:, 0].mean().item(), 4),
                "Box_CY_Mean": round(box_t[:, 1].mean().item(), 4),
                "Box_W_Mean": round(box_t[:, 2].mean().item(), 4),
                "Box_H_Mean": round(box_t[:, 3].mean().item(), 4),
                "MapPred_Energy": round(map_t.norm(dim=1).mean().item(), 4),
            }

            if obj_t is not None:
                obj_scores = obj_t.sigmoid()
                results["Obj_Prob_Mean"] = round(obj_scores.mean().item(), 4)
                results["Obj_Prob_Max"] = round(obj_scores.max().item(), 4)
                results["Obj_Logits_Mean"] = round(obj_t.mean().item(), 4)

            if s_info_t is not None:
                if isinstance(s_info_t, dict):
                    all_locs = s_info_t.get("sampling_locations", None)
                    all_offsets = s_info_t.get("offsets", None)
                    ref_tensor = s_info_t.get("ref", None)
                else:
                    all_locs = None
                    all_offsets = None
                    ref_tensor = s_info_t if isinstance(s_info_t, torch.Tensor) else None

                if ref_tensor is None and hasattr(self.head, "last_ref"):
                    last_ref = self.head.last_ref
                    ref_tensor = last_ref.get("c3", None) if isinstance(last_ref, dict) else last_ref

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
