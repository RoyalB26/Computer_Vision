"""VisDrone YOLOv11 Detector Architecture.

Integrates YOLO11 with DIoU Loss and Focal Loss, inheriting BaseModel to ensure
full compatibility with BaseModel training engines, registries, hooks, and metrics.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torchvision.ops

from BaseModel.core import MODELS
from BaseModel.model.base_model import BaseModel
from yolo.yolov11 import YOLOv11, _extract_state_dict
from yolo.head import make_anchors, dist2bbox
from .loss import YOLOv11DetectionLoss, DIoULoss, SigmoidFocalLoss
from .slot_module import PostNeckSlotFiLMRefiner, SlotAuxiliaryLoss


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

        # Initialize classification head bias properly:
        # - For standard YOLOv11 BCE (focal_gamma == 0): Ultralytics per-stride prior (~ -7.5 to -9.5)
        #   so Epoch 1 loss_focal starts around ~5-20 instead of 40,000+!
        # - For RetinaNet Focal Loss (focal_gamma > 0): prior_prob = 0.01 (~ -4.595)
        self._init_cls_bias(focal_gamma=focal_gamma)

    def _init_cls_bias(self, prior_prob: float = 0.01, focal_gamma: float = 2.0) -> None:
        """Initialize Detect.cv3 final conv bias for stable BCE / Focal Loss."""
        strides = self.net.head.stride.tolist()
        for idx, branch in enumerate(self.net.head.cv3):
            last_conv = branch[-1]
            if isinstance(last_conv, nn.Conv2d) and last_conv.bias is not None:
                if focal_gamma <= 0.0:
                    s = strides[idx] if idx < len(strides) else 16.0
                    # Ultralytics YOLOv8/11 formula: log(5 / nc / (640 / stride)^2)
                    bias_val = math.log(5.0 / (self.num_classes * ((640.0 / s) ** 2)))
                else:
                    bias_val = -math.log((1.0 - prior_prob) / prior_prob)
                nn.init.constant_(last_conv.bias, bias_val)

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

        # Match prediction box scale to target box scale (normalized [0, 1] vs pixel)
        out_fmt = "xyxy"
        if isinstance(targets, (list, tuple)):
            for t in targets:
                if isinstance(t, dict) and "boxes" in t and len(t["boxes"]) > 0:
                    if (t["boxes"] <= 1.05).all():
                        out_fmt = "normalized"
                    break

        preds = self.predict(inputs, conf_threshold=c_th, nms_threshold=n_th, max_det=m_det, output_format=out_fmt)
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


@MODELS.register_module(name="VisDroneYOLO11Slot", force=True)
@MODELS.register_module(name="VisDroneYOLOv11Slot", force=True)
class VisDroneYOLO11Slot(VisDroneYOLO11):
    """YOLOv11 + Post-Neck Decoupled Slot Attention & FiLM Refinement (Method B+C).

    Architecture:
      1. YOLOv11 Backbone + YOLOv11 PANet Neck -> [P3 (64), P4 (128), P5 (256)].
      2. PostNeckSlotFiLMRefiner:
         - Extracts K+1 image-adaptive semantic slots via LightweightSlotAttention.
         - Broadcasts slots back via `attn_ori` and modulates ONLY the classification branch
           via Zero-Init FiLM (or Residual), preserving box localization boundaries and
           100% of pretrained yolo11n.pt behavior at epoch 0.
         - Optionally adds zero-init direct semantic logit prior to Detect.cv3 outputs.
      3. Joint loss: YOLOv11 (DIoU + Focal + DFL) + `slot_loss_weight` * SlotAuxiliaryLoss.
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
        # Slot + FiLM (B+C) hyperparameters
        slot_dim: int = 32,
        fusion_mode: str = "film",
        use_bg_slot_write: bool = False,
        use_logit_prior: bool = False,
        assigner_prior_weight: float = 0.25,
        use_slot_checkpoint: bool = False,
        slot_loss_weight: float = 0.3,
        bg_attn_weight: float = 0.5,
        fg_attn_weight: float = 0.5,
        suppress_weight: float = 0.05,
        cls_attn_weight: float = 0.5,
        ortho_weight: float = 0.1,
    ):
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            box_weight=box_weight,
            cls_weight=cls_weight,
            dfl_weight=dfl_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            max_det=max_det,
        )
        self.slot_loss_weight = float(slot_loss_weight)
        self.assigner_prior_weight = float(assigner_prior_weight)

        self.slot_refiner = PostNeckSlotFiLMRefiner(
            in_channels=(64, 128, 256),
            num_classes=num_classes,
            slot_dim=slot_dim,
            fusion_mode=fusion_mode,
            use_bg_slot_write=use_bg_slot_write,
            use_logit_prior=use_logit_prior,
            use_checkpoint=use_slot_checkpoint,
        )
        self.slot_criterion = SlotAuxiliaryLoss(
            num_classes=num_classes,
            bg_attn_weight=bg_attn_weight,
            fg_attn_weight=fg_attn_weight,
            suppress_weight=suppress_weight,
            cls_attn_weight=cls_attn_weight,
            ortho_weight=ortho_weight,
        )

    def _forward_head_with_slots(
        self, inputs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Shared backbone -> neck -> slot FiLM -> decoupled head forward."""
        feat4, feat6, feat10 = self.net.backbone(inputs)
        p3, p4, p5 = self.net.neck(feat4, feat6, feat10)
        feats = [p3, p4, p5]

        # Post-neck Slot Attention + FiLM modulation for classification branch only
        cls_feats, slot_info = self.slot_refiner(feats)

        bs = p3.shape[0]
        # Box regression uses original neck features (sharp boundaries)
        boxes_raw = torch.cat(
            [self.net.head.cv2[i](feats[i]).view(bs, 4 * self.net.head.reg_max, -1) for i in range(self.net.head.nl)],
            dim=-1,
        )
        # Classification uses slot-modulated features
        scores_raw = torch.cat(
            [self.net.head.cv3[i](cls_feats[i]).view(bs, self.num_classes, -1) for i in range(self.net.head.nl)],
            dim=-1,
        )
        if slot_info["logit_prior"] is not None:
            scores_raw = scores_raw + slot_info["logit_prior"]

        return feats, boxes_raw, scores_raw, slot_info

    def forward_train(
        self,
        inputs: torch.Tensor,
        targets: Optional[Union[List[Dict[str, torch.Tensor]], torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass computing YOLOv11 detection loss + Slot auxiliary loss."""
        feats, boxes_raw, scores_raw, slot_info = self._forward_head_with_slots(inputs)
        preds = {"boxes": boxes_raw, "scores": scores_raw, "feats": feats}
        if self.assigner_prior_weight > 0 and "dots" in slot_info:
            # Decoupled Slot-Guided Assignment: guides TAL for fast box learning
            # while keeping `scores` (loss_focal / cv3) clean from logit co-adaptation.
            preds["assign_scores"] = (
                scores_raw + self.assigner_prior_weight * slot_info["dots"][:, : self.num_classes, :].detach()
            )

        # Decode bounding boxes for Task-Aligned Assignment
        a_raw, s_raw = make_anchors(feats, self.net.head.stride, 0.5)
        anchors, strides = (a.transpose(0, 1).to(inputs.device) for a in (a_raw, s_raw))

        pred_dist = self.net.head.dfl(boxes_raw)
        decoded_xyxy = dist2bbox(pred_dist, anchors.unsqueeze(0), xywh=False, dim=1) * strides
        decoded_xyxy = decoded_xyxy.transpose(1, 2)

        anchor_points_pixel = (anchors * strides).transpose(0, 1)
        stride_tensor = strides.transpose(0, 1)
        img_size = (inputs.shape[2], inputs.shape[3])

        # 1. Core YOLOv11 detection loss (DIoU + Focal + DFL)
        loss_dict = self.criterion(
            preds=preds,
            targets=targets,
            decoded_bboxes=decoded_xyxy,
            anchor_points=anchor_points_pixel,
            stride_tensor=stride_tensor,
            img_size=img_size,
        )

        # 2. Auxiliary Slot semantic supervision (unified pixel xyxy GT boxes)
        if self.slot_loss_weight > 0 and targets is not None:
            gt_bboxes, gt_labels, mask_gt = self.criterion._prepare_targets(
                targets, img_size, inputs.device
            )
            slot_losses = self.slot_criterion(
                slots=slot_info["slots"],
                attn_ori=slot_info["attn_ori"],
                obj_queries=self.slot_refiner.slot_attention.obj_queries,
                resolutions=slot_info["resolutions"],
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                mask_gt=mask_gt,
                img_size=img_size,
                dots=slot_info.get("dots"),
            )
            loss_dict["loss"] = loss_dict["loss"] + self.slot_loss_weight * slot_losses["loss_slot"]
            loss_dict.update(slot_losses)

        return loss_dict

    def predict(
        self,
        inputs: torch.Tensor,
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        max_det: Optional[int] = None,
        output_format: str = "xyxy",
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        """Inference forward pass with Slot-FiLM classification refinement + Batched NMS."""
        c_th = conf_threshold if conf_threshold is not None else self.conf_threshold
        n_th = nms_threshold if nms_threshold is not None else self.nms_threshold
        m_det = max_det if max_det is not None else self.max_det

        was_training = self.training
        self.eval()

        with torch.no_grad():
            bs, _, img_h, img_w = inputs.shape
            feats, boxes_raw, scores_raw, _ = self._forward_head_with_slots(inputs)

            a_raw, s_raw = make_anchors(feats, self.net.head.stride, 0.5)
            anchors, strides = (a.transpose(0, 1).to(inputs.device) for a in (a_raw, s_raw))

            pred_dist = self.net.head.dfl(boxes_raw)
            decoded_boxes = dist2bbox(pred_dist, anchors.unsqueeze(0), xywh=False, dim=1) * strides
            decoded_boxes = decoded_boxes.transpose(1, 2)
            probs = scores_raw.sigmoid().transpose(1, 2)

        if was_training:
            self.train()

        results = []
        for b in range(bs):
            b_boxes = decoded_boxes[b]
            b_probs = probs[b]

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

            if len(cand_scores) > 1000:
                topk_idx = torch.topk(cand_scores, 1000).indices
                cand_boxes = cand_boxes[topk_idx]
                cand_scores = cand_scores[topk_idx]
                cand_labels = cand_labels[topk_idx]

            cand_boxes[:, [0, 2]] = cand_boxes[:, [0, 2]].clamp(0, img_w)
            cand_boxes[:, [1, 3]] = cand_boxes[:, [1, 3]].clamp(0, img_h)

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
