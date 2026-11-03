"""VisDrone YOLOv11 Detector Architecture.

Integrates YOLO11 with DIoU Loss and Focal Loss, inheriting BaseModel to ensure
full compatibility with BaseModel training engines, registries, hooks, and metrics.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision.ops

from BaseModel.core import MODELS
from BaseModel.model.base_model import BaseModel
from yolo.yolov11 import YOLOv11, _extract_state_dict
from yolo.head import make_anchors, dist2bbox
from .loss import YOLOv11DetectionLoss, DIoULoss, SigmoidFocalLoss


@MODELS.register_module(name="VisDroneYOLO11", force=True)
@MODELS.register_module(name="VisDroneYOLOv11", force=True)
class VisDroneYOLO11(BaseModel):
    """YOLOv11 Object Detector customized for VisDrone Aerial Imagery.

    Features:
      - YOLO11 Modular Architecture: CSPDarknet Backbone, PANet Neck with C3k2/C2PSA, Decoupled Detection Head.
      - DIoU Loss for accurate bounding box regression on small and dense aerial targets.
      - Sigmoid Focal Loss to effectively suppress abundant background negatives and counter class imbalance.
      - Task-Aligned Assigner (TAL) for dynamic high-quality anchor assignment.
      - Automatic pretrained weight loading from official checkpoints (e.g., yolo11n.pt).

    Args:
        num_classes: Number of target classes (default 10 for VisDrone).
        pretrained: Path to pretrained checkpoint (e.g., 'yolo11n.pt').
        box_weight: Weight for DIoU regression loss.
        cls_weight: Weight for Focal classification loss.
        dfl_weight: Weight for Distribution Focal Loss.
        focal_alpha: Focal loss alpha parameter.
        focal_gamma: Focal loss gamma parameter.
        conf_threshold: Confidence threshold for inference and NMS.
        nms_threshold: IoU threshold for Non-Maximum Suppression.
        max_det: Maximum detections per image.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: Optional[Union[str, Path]] = None,
        box_weight: float = 7.5,
        cls_weight: float = 1.0,
        dfl_weight: float = 1.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        conf_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        max_det: int = 300,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_det = max_det

        # Core YOLOv11 model
        self.net = YOLOv11(nc=num_classes)

        # Loss engine combining DIoU Loss, Focal Loss, and DFL
        self.criterion = YOLOv11DetectionLoss(
            num_classes=num_classes,
            reg_max=self.net.head.reg_max,
            box_weight=box_weight,
            cls_weight=cls_weight,
            dfl_weight=dfl_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )

        # Load pretrained weights if specified
        if pretrained:
            self.load_pretrained(pretrained)

    def load_pretrained(self, weights_path: Union[str, Path]) -> Tuple[List[str], List[str]]:
        """Load pretrained weights into backbone and neck with shape mismatch filtering."""
        weights_file = Path(weights_path)
        if not weights_file.exists():
            print(f"[VisDroneYOLO11] Warning: Pretrained file {weights_file} not found. Skipping.")
            return [], []

        raw_sd = _extract_state_dict(weights_file)
        prefix_map = {
            "model.0.": "backbone.stage0.",
            "model.1.": "backbone.stage1.",
            "model.2.": "backbone.stage2.",
            "model.3.": "backbone.stage3.",
            "model.4.": "backbone.stage4.",
            "model.5.": "backbone.stage5.",
            "model.6.": "backbone.stage6.",
            "model.7.": "backbone.stage7.",
            "model.8.": "backbone.stage8.",
            "model.9.": "backbone.stage9.",
            "model.10.": "backbone.stage10.",
            "model.13.": "neck.c3k2_13.",
            "model.16.": "neck.c3k2_16.",
            "model.17.": "neck.conv17.",
            "model.19.": "neck.c3k2_19.",
            "model.20.": "neck.conv20.",
            "model.22.": "neck.c3k2_22.",
            "model.23.": "head.",
        }

        mapped_sd = {}
        for k, v in raw_sd.items():
            matched = False
            for old_p, new_p in prefix_map.items():
                if k.startswith(old_p):
                    mapped_sd[k.replace(old_p, new_p, 1)] = v
                    matched = True
                    break
            if not matched:
                mapped_sd[k] = v

        # Shape-safe loading: exclude keys whose tensor shape does not match current model (e.g. classification convs)
        model_sd = self.net.state_dict()
        compatible_sd = {
            k: v for k, v in mapped_sd.items()
            if k in model_sd and model_sd[k].shape == v.shape
        }

        missing, unexpected = self.net.load_state_dict(compatible_sd, strict=False)
        print(f"[VisDroneYOLO11] Loaded {len(compatible_sd)} pretrained tensors from {weights_file} (missing: {len(missing)}, skipped/mismatched: {len(mapped_sd) - len(compatible_sd)})")
        return missing, unexpected

    def forward_train(
        self,
        inputs: torch.Tensor,
        targets: Optional[Union[List[Dict[str, torch.Tensor]], torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass computing DIoU, Focal, and DFL losses."""
        # 1. Feature extraction & Multi-scale Head prediction
        feat4, feat6, feat10 = self.net.backbone(inputs)
        p3, p4, p5 = self.net.neck(feat4, feat6, feat10)
        feats = [p3, p4, p5]

        # Explicitly forward through detection head in train mode
        bs = p3.shape[0]
        boxes_raw = torch.cat(
            [self.net.head.cv2[i](feats[i]).view(bs, 4 * self.net.head.reg_max, -1) for i in range(self.net.head.nl)],
            dim=-1,
        )
        scores_raw = torch.cat(
            [self.net.head.cv3[i](feats[i]).view(bs, self.num_classes, -1) for i in range(self.net.head.nl)],
            dim=-1,
        )
        preds = {"boxes": boxes_raw, "scores": scores_raw, "feats": feats}

        # 2. Decode bounding boxes for Task-Aligned Assignment
        a_raw, s_raw = make_anchors(feats, self.net.head.stride, 0.5)
        anchors, strides = (a.transpose(0, 1).to(inputs.device) for a in (a_raw, s_raw))

        pred_dist = self.net.head.dfl(boxes_raw)  # [B, 4, N]
        decoded_xyxy = dist2bbox(pred_dist, anchors.unsqueeze(0), xywh=False, dim=1) * strides  # [B, 4, N] in pixels
        decoded_xyxy = decoded_xyxy.transpose(1, 2)  # [B, N, 4]

        # Anchor points in pixel coordinates for Task-Aligned Assigner
        anchor_points_pixel = (anchors * strides).transpose(0, 1)  # [N, 2]
        stride_tensor = strides.transpose(0, 1)                    # [N, 1]

        # 3. Compute DIoU Loss & Focal Loss
        img_size = (inputs.shape[2], inputs.shape[3])
        loss_dict = self.criterion(
            preds=preds,
            targets=targets,
            decoded_bboxes=decoded_xyxy,
            anchor_points=anchor_points_pixel,
            stride_tensor=stride_tensor,
            img_size=img_size,
        )
        return loss_dict

    def forward_val(
        self,
        inputs: torch.Tensor,
        targets: Optional[Union[List[Dict[str, torch.Tensor]], torch.Tensor]] = None,
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        max_det: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Validation forward pass computing both evaluation predictions and validation losses."""
        c_th = conf_threshold if conf_threshold is not None else self.conf_threshold
        n_th = nms_threshold if nms_threshold is not None else self.nms_threshold
        m_det = max_det if max_det is not None else self.max_det

        preds = self.predict(inputs, conf_threshold=c_th, nms_threshold=n_th, max_det=m_det)
        output = {"preds": preds}

        if targets is not None:
            try:
                # Save training mode state and switch temporarily for loss computation
                was_training = self.training
                self.train()
                with torch.no_grad():
                    losses = self.forward_train(inputs, targets, **kwargs)
                if not was_training:
                    self.eval()

                for k, v in losses.items():
                    output[k] = v
            except Exception as e:
                output["loss"] = torch.tensor(0.0, device=inputs.device)

        return output

    def predict(
        self,
        inputs: torch.Tensor,
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        max_det: Optional[int] = None,
        output_format: str = "xyxy",
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        """Inference forward pass with Non-Maximum Suppression (NMS).

        Args:
            inputs: Batch of image tensors [B, 3, H, W].
            conf_threshold: Confidence score cutoff.
            nms_threshold: IoU cutoff for batched NMS.
            max_det: Maximum detections per image.
            output_format: 'xyxy' [x1, y1, x2, y2] in pixels or 'normalized' [0, 1].

        Returns:
            List of result dicts per image:
                [{'boxes': tensor [K, 4], 'scores': tensor [K], 'labels': tensor [K]}, ...]
        """
        c_th = conf_threshold if conf_threshold is not None else self.conf_threshold
        n_th = nms_threshold if nms_threshold is not None else self.nms_threshold
        m_det = max_det if max_det is not None else self.max_det

        was_training = self.training
        self.eval()

        with torch.no_grad():
            feat4, feat6, feat10 = self.net.backbone(inputs)
            p3, p4, p5 = self.net.neck(feat4, feat6, feat10)
            feats = [p3, p4, p5]

            bs, _, img_h, img_w = inputs.shape
            boxes_raw = torch.cat(
                [self.net.head.cv2[i](feats[i]).view(bs, 4 * self.net.head.reg_max, -1) for i in range(self.net.head.nl)],
                dim=-1,
            )
            scores_raw = torch.cat(
                [self.net.head.cv3[i](feats[i]).view(bs, self.num_classes, -1) for i in range(self.net.head.nl)],
                dim=-1,
            )

            a_raw, s_raw = make_anchors(feats, self.net.head.stride, 0.5)
            anchors, strides = (a.transpose(0, 1).to(inputs.device) for a in (a_raw, s_raw))

            pred_dist = self.net.head.dfl(boxes_raw)
            decoded_boxes = dist2bbox(pred_dist, anchors.unsqueeze(0), xywh=False, dim=1) * strides  # [B, 4, N] in pixels
            decoded_boxes = decoded_boxes.transpose(1, 2)  # [B, N, 4] in xyxy pixel coordinates
            probs = scores_raw.sigmoid().transpose(1, 2)   # [B, N, C]

        if was_training:
            self.train()

        results = []
        for b in range(bs):
            b_boxes = decoded_boxes[b]  # [N, 4]
            b_probs = probs[b]          # [N, C]

            # Find candidates exceeding confidence threshold
            max_scores, labels = b_probs.max(dim=-1)
            valid_mask = max_scores >= c_th

            if not valid_mask.any():
                results.append({
                    "boxes": torch.empty((0, 4), device=inputs.device),
                    "scores": torch.empty((0,), device=inputs.device),
                    "labels": torch.empty((0,), dtype=torch.long, device=inputs.device),
                })
                continue

            cand_boxes = b_boxes[valid_mask]
            cand_scores = max_scores[valid_mask]
            cand_labels = labels[valid_mask]

            # Pre-filter top 1000 candidates to optimize batched NMS speed
            if len(cand_scores) > 1000:
                topk_idx = torch.topk(cand_scores, 1000).indices
                cand_boxes = cand_boxes[topk_idx]
                cand_scores = cand_scores[topk_idx]
                cand_labels = cand_labels[topk_idx]

            # Clamp coordinates to image boundaries
            cand_boxes[:, [0, 2]] = cand_boxes[:, [0, 2]].clamp(0, img_w)
            cand_boxes[:, [1, 3]] = cand_boxes[:, [1, 3]].clamp(0, img_h)

            # Batched NMS
            keep = torchvision.ops.batched_nms(cand_boxes, cand_scores, cand_labels, n_th)
            keep = keep[:m_det]

            final_boxes = cand_boxes[keep]
            final_scores = cand_scores[keep]
            final_labels = cand_labels[keep]

            if output_format == "normalized":
                final_boxes[:, [0, 2]] /= img_w
                final_boxes[:, [1, 3]] /= img_h

            results.append({
                "boxes": final_boxes,
                "scores": final_scores,
                "labels": final_labels,
            })

        return results

    def freeze_backbone(self) -> None:
        """Freeze YOLOv11 backbone weights during fine-tuning."""
        for param in self.net.backbone.parameters():
            param.requires_grad = False
        print("[VisDroneYOLO11] Backbone parameters frozen.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze YOLOv11 backbone weights."""
        for param in self.net.backbone.parameters():
            param.requires_grad = True
        print("[VisDroneYOLO11] Backbone parameters unfrozen.")
