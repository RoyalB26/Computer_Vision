"""VisDrone YOLOv11 + Slot Attention + Deformable High-Res Micro-Patch Injection.

Architecture (Single End-to-End Pass):
  1. Accepts high-resolution input `inputs` [B, 3, H_hr, W_hr] (e.g. 1280x1280 or 1440x1440).
  2. Downsamples on GPU to `backbone_size` (e.g. 640x640 or 960x960) -> YOLOv11 Backbone + Neck.
  3. Runs PostNeckSlotFiLMRefiner to obtain class slots, `attn_ori`, and FiLM-modulated `cls_feats`.
  4. Runs DeformableHighResPatchInjector: uses P3 `attn_ori` to select Top-K foreground locations,
     predicts learned deformable offsets/scales, samples K high-res RGB micro-patches directly
     from `inputs` via 1 batched `F.grid_sample`, encodes them with a lightweight Micro-CNN, and
     injects Zero-Init high-res residuals back into P3 (both `box` and `cls` branches).
  5. Decoupled Detection Head (`cv2` box + `cv3` cls) + YOLOv11DetectionLoss + SlotAuxiliaryLoss.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.ops

from BaseModel.core import MODELS
from yolo.head import dist2bbox, make_anchors
from VisDrone_yolov11.model import VisDroneYOLO11Slot
from .deformable_patch import DeformableHighResPatchInjector


@MODELS.register_module(name="VisDroneYOLO11DeformableSlot", force=True)
class VisDroneYOLO11DeformableSlot(VisDroneYOLO11Slot):
    """YOLOv11 + Post-Neck Slot FiLM + Deformable High-Res Micro-Patch Injection."""

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: Optional[Union[str, Path]] = None,
        box_weight: float = 7.5,
        cls_weight: float = 1.0,
        dfl_weight: float = 1.5,
        focal_alpha: float = -1.0,
        focal_gamma: float = 0.0,
        conf_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        max_det: int = 300,
        # Dual-resolution & Deformable High-Res Micro-Patch hyperparameters
        backbone_size: int = 640,
        use_deformable_patch: bool = True,
        num_patches: int = 192,
        patch_size: int = 32,
        base_patch_radius: float = 0.035,
        max_offset_cells: float = 1.5,
        class_focus_weights: Optional[List[float]] = None,
        spread_local_3x3: bool = True,
        patch_dropout: float = 0.1,
        patch_loss_weight: float = 0.2,
        use_uncertainty_routing: bool = False,
        uncertainty_floor: float = 0.2,
        use_sparse_subhead: bool = False,
        # Slot + FiLM hyperparameters (forwarded to VisDroneYOLO11Slot)
        slot_dim: int = 64,
        num_prototypes: Union[int, List[int], Tuple[int, ...]] = (2, 2, 2, 1, 2, 3, 3, 3, 2, 2),
        num_iters: int = 2,
        use_implicit_diff: bool = True,
        fusion_mode: str = "film",
        use_bg_slot_write: bool = False,
        use_logit_prior: bool = False,
        assigner_prior_weight: float = 0.25,
        use_slot_checkpoint: bool = False,
        slot_dropout: float = 0.1,
        use_asa_gate: bool = True,
        asa_gate_floor: float = 0.05,
        asa_logit_weight: float = 0.25,
        asa_topk: int = 16,
        presence_hidden_dim: int = 32,
        presence_dropout: float = 0.3,
        presence_loss_weight: float = 0.25,
        presence_label_smoothing: float = 0.05,
        sparse_thresh: float = 0.05,
        use_cosine_attn: bool = True,
        use_overlap_soft_target: bool = True,
        use_ellipse_mask: bool = True,
        ellipse_min_side: float = 3.0,
        use_class_freq_weight: bool = True,
        class_freq_max_weight: float = 2.0,
        slot_loss_weight: float = 0.3,
        bg_attn_weight: float = 0.3,
        fg_attn_weight: float = 0.7,
        suppress_weight: float = 0.05,
        cls_attn_weight: float = 0.8,
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
            slot_dim=slot_dim,
            num_prototypes=num_prototypes,
            num_iters=num_iters,
            use_implicit_diff=use_implicit_diff,
            fusion_mode=fusion_mode,
            use_bg_slot_write=use_bg_slot_write,
            use_logit_prior=use_logit_prior,
            assigner_prior_weight=assigner_prior_weight,
            use_slot_checkpoint=use_slot_checkpoint,
            slot_dropout=slot_dropout,
            use_asa_gate=use_asa_gate,
            asa_gate_floor=asa_gate_floor,
            asa_logit_weight=asa_logit_weight,
            asa_topk=asa_topk,
            presence_hidden_dim=presence_hidden_dim,
            presence_dropout=presence_dropout,
            presence_loss_weight=presence_loss_weight,
            presence_label_smoothing=presence_label_smoothing,
            sparse_thresh=sparse_thresh,
            use_cosine_attn=use_cosine_attn,
            use_overlap_soft_target=use_overlap_soft_target,
            use_ellipse_mask=use_ellipse_mask,
            ellipse_min_side=ellipse_min_side,
            use_class_freq_weight=use_class_freq_weight,
            class_freq_max_weight=class_freq_max_weight,
            slot_loss_weight=slot_loss_weight,
            bg_attn_weight=bg_attn_weight,
            fg_attn_weight=fg_attn_weight,
            suppress_weight=suppress_weight,
            cls_attn_weight=cls_attn_weight,
            ortho_weight=ortho_weight,
        )
        self.backbone_size = int(backbone_size)
        self.use_deformable_patch = bool(use_deformable_patch)
        self.patch_loss_weight = float(patch_loss_weight)
        self.use_sparse_subhead = bool(use_sparse_subhead)

        if self.use_deformable_patch:
            self.patch_injector = DeformableHighResPatchInjector(
                p3_channels=64,
                num_classes=num_classes,
                num_patches=num_patches,
                patch_size=patch_size,
                base_patch_radius=base_patch_radius,
                max_offset_cells=max_offset_cells,
                class_focus_weights=class_focus_weights,
                spread_local_3x3=spread_local_3x3,
                patch_dropout=patch_dropout,
                use_uncertainty_routing=use_uncertainty_routing,
                uncertainty_floor=uncertainty_floor,
                use_sparse_subhead=use_sparse_subhead,
                reg_max=int(self.net.head.reg_max),
            )
        else:
            self.patch_injector = None

    def _prepare_backbone_input(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Downsample high-res `inputs` on GPU to `backbone_size` while keeping `inputs` intact for micro-patches."""
        in_h, in_w = int(inputs.shape[2]), int(inputs.shape[3])
        if self.backbone_size > 0 and (in_h != self.backbone_size or in_w != self.backbone_size):
            x_lr = F.interpolate(
                inputs,
                size=(self.backbone_size, self.backbone_size),
                mode="bilinear",
                align_corners=False,
            )
            return x_lr, (self.backbone_size, self.backbone_size)
        return inputs, (in_h, in_w)

    def _forward_head_with_slots(
        self, inputs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any], Tuple[int, int]]:
        """Backbone(LR) -> Neck -> Slot FiLM -> Deformable High-Res Micro-Patch Injection(HR->P3) -> Head."""
        x_lr, lr_size = self._prepare_backbone_input(inputs)

        feat4, feat6, feat10 = self.net.backbone(x_lr)
        p3, p4, p5 = self.net.neck(feat4, feat6, feat10)
        feats = [p3, p4, p5]

        # 1. Post-Neck Slot Attention + FiLM modulation
        cls_feats, slot_info = self.slot_refiner(feats)

        # 2. Deformable High-Res Micro-Patch Injection directly from original high-res `inputs` into P3
        box_feats = list(feats)
        if self.patch_injector is not None:
            p3_box_ref, p3_cls_ref, patch_info = self.patch_injector(
                x_hr=inputs,
                p3_box=box_feats[0],
                p3_cls=cls_feats[0],
                attn_ori=slot_info["attn_ori"],
            )
            box_feats[0] = p3_box_ref
            cls_feats[0] = p3_cls_ref
            slot_info["patch_info"] = patch_info

        bs = box_feats[0].shape[0]
        boxes_raw = torch.cat(
            [self.net.head.cv2[i](box_feats[i]).view(bs, 4 * self.net.head.reg_max, -1) for i in range(self.net.head.nl)],
            dim=-1,
        )
        scores_raw = torch.cat(
            [self.net.head.cv3[i](cls_feats[i]).view(bs, self.num_classes, -1) for i in range(self.net.head.nl)],
            dim=-1,
        )
        if slot_info["logit_prior"] is not None:
            scores_raw = scores_raw + slot_info["logit_prior"]

        # Append 4x4 Sparse High-Res Sub-Head predictions (K*16 sub-anchors) when enabled
        p_info = slot_info.get("patch_info")
        if p_info is not None and "sub_boxes_raw" in p_info:
            boxes_raw = torch.cat([boxes_raw, p_info["sub_boxes_raw"]], dim=-1)
            scores_raw = torch.cat([scores_raw, p_info["sub_scores_raw"]], dim=-1)

        return box_feats, boxes_raw, scores_raw, slot_info, lr_size

    def _build_anchors_and_decode(
        self,
        feats: List[torch.Tensor],
        boxes_raw: torch.Tensor,
        slot_info: Dict[str, Any],
        lr_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build anchor points & strides (P3/P4/P5 + optional 4x4 sparse sub-anchors) and decode xyxy boxes."""
        device = boxes_raw.device
        bs = boxes_raw.shape[0]
        a_raw, s_raw = make_anchors(feats, self.net.head.stride, 0.5)
        anchors_p = (a_raw * s_raw).to(device)  # [N_p345, 2] in lr_size pixel coords
        strides_p = s_raw.to(device)            # [N_p345, 1]

        p_info = slot_info.get("patch_info")
        if p_info is not None and "sub_anchors_norm" in p_info:
            lr_h, lr_w = float(lr_size[0]), float(lr_size[1])
            g = float(p_info["sub_grid_size"])
            sub_norm = p_info["sub_anchors_norm"].to(device)  # [B, N_sub, 2] in [-1, 1]
            sub_anchors_px = torch.stack(
                [
                    (sub_norm[..., 0] + 1.0) * 0.5 * lr_w,
                    (sub_norm[..., 1] + 1.0) * 0.5 * lr_h,
                ],
                dim=-1,
            )  # [B, N_sub, 2]
            sub_radii = p_info["sub_radii_norm"].to(device)   # [B, N_sub, 2]
            stride_x = sub_radii[..., 0] * lr_w / g
            stride_y = sub_radii[..., 1] * lr_h / g
            sub_strides = (0.5 * (stride_x + stride_y)).clamp(min=1.0).unsqueeze(-1)  # [B, N_sub, 1]

            anchor_points_pixel = torch.cat(
                [anchors_p.unsqueeze(0).expand(bs, -1, -1), sub_anchors_px], dim=1
            )  # [B, N_total, 2]
            stride_tensor = torch.cat(
                [strides_p.unsqueeze(0).expand(bs, -1, -1), sub_strides], dim=1
            )  # [B, N_total, 1]
        else:
            anchor_points_pixel = anchors_p  # [N_p345, 2]
            stride_tensor = strides_p        # [N_p345, 1]

        pred_dist = self.net.head.dfl(boxes_raw).transpose(1, 2)  # [B, N_total, 4] (l, t, r, b in stride units)
        ap = anchor_points_pixel if anchor_points_pixel.dim() == 3 else anchor_points_pixel.unsqueeze(0)
        st = stride_tensor if stride_tensor.dim() == 3 else stride_tensor.unsqueeze(0)
        lt, rb = pred_dist.chunk(2, dim=-1)
        decoded_xyxy = torch.cat([ap - lt * st, ap + rb * st], dim=-1)  # [B, N_total, 4]
        return decoded_xyxy, anchor_points_pixel, stride_tensor

    def forward_train(
        self,
        inputs: torch.Tensor,
        targets: Optional[Union[List[Dict[str, torch.Tensor]], torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass (boxes & losses computed cleanly in `lr_size` pixel coordinates)."""
        feats, boxes_raw, scores_raw, slot_info, lr_size = self._forward_head_with_slots(inputs)
        preds = {"boxes": boxes_raw, "scores": scores_raw, "feats": feats}
        if self.assigner_prior_weight > 0 and "dots" in slot_info:
            dots_cls = slot_info["dots"][:, : self.num_classes, :].detach()
            p_info = slot_info.get("patch_info")
            if p_info is not None and "sub_scores_raw" in p_info and scores_raw.shape[-1] > dots_cls.shape[-1]:
                h3, w3 = p_info["grid_hw"]
                n3 = h3 * w3
                g = int(p_info["sub_grid_size"])
                top_idx = p_info["top_idx"]  # (B, K)
                dots_p3 = dots_cls[:, :, :n3]  # (B, C, N3)
                idx_exp = top_idx.unsqueeze(1).expand(-1, self.num_classes, -1)  # (B, C, K)
                dots_sub = torch.gather(dots_p3, dim=2, index=idx_exp).repeat_interleave(g * g, dim=-1)
                dots_cls = torch.cat([dots_cls, dots_sub], dim=-1)
            preds["assign_scores"] = scores_raw + self.assigner_prior_weight * dots_cls

        decoded_xyxy, anchor_points_pixel, stride_tensor = self._build_anchors_and_decode(
            feats, boxes_raw, slot_info, lr_size
        )

        # Targets are normalized [0, 1] cxcywh -> scaled cleanly to `lr_size` matching `feats` & `strides`
        loss_dict = self.criterion(
            preds=preds,
            targets=targets,
            decoded_bboxes=decoded_xyxy,
            anchor_points=anchor_points_pixel,
            stride_tensor=stride_tensor,
            img_size=lr_size,
        )

        if (self.slot_loss_weight > 0 or self.patch_loss_weight > 0) and targets is not None:
            gt_bboxes, gt_labels, mask_gt = self.criterion._prepare_targets(targets, lr_size, inputs.device)

            if self.slot_loss_weight > 0:
                slot_losses = self.slot_criterion(
                    slots=slot_info["slots"],
                    attn_ori=slot_info["attn_ori"],
                    obj_queries=self.slot_refiner.slot_attention.obj_queries,
                    resolutions=slot_info["resolutions"],
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels,
                    mask_gt=mask_gt,
                    img_size=lr_size,
                    dots=slot_info.get("dots"),
                    slots_sub=slot_info.get("slots_sub"),
                    presence_logits=slot_info.get("presence_logits"),
                )
                loss_dict["loss"] = loss_dict["loss"] + self.slot_loss_weight * slot_losses["loss_slot"]
                loss_dict.update(slot_losses)

            if self.patch_loss_weight > 0 and self.patch_injector is not None and "patch_info" in slot_info:
                patch_losses = self.patch_injector.compute_patch_loss(
                    patch_info=slot_info["patch_info"],
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels,
                    mask_gt=mask_gt,
                    img_size=lr_size,
                )
                loss_dict["loss"] = loss_dict["loss"] + self.patch_loss_weight * patch_losses["loss_patch"]
                loss_dict.update(patch_losses)

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
        """Inference forward pass returning boxes in `inputs` pixel coordinates (or normalized [0, 1])."""
        c_th = conf_threshold if conf_threshold is not None else self.conf_threshold
        n_th = nms_threshold if nms_threshold is not None else self.nms_threshold
        m_det = max_det if max_det is not None else self.max_det

        was_training = self.training
        self.eval()

        with torch.no_grad():
            bs, _, in_h, in_w = inputs.shape
            feats, boxes_raw, scores_raw, slot_info, (lr_h, lr_w) = self._forward_head_with_slots(inputs)

            decoded_boxes, _, _ = self._build_anchors_and_decode(
                feats, boxes_raw, slot_info, (lr_h, lr_w)
            )  # [B, N_total, 4] in lr_size coordinates
            probs = scores_raw.sigmoid().transpose(1, 2)

        if was_training:
            self.train()

        scale_x = float(in_w) / float(lr_w)
        scale_y = float(in_h) / float(lr_h)

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

            # Map boxes from lr_size to input high-res coordinates
            cand_boxes[:, [0, 2]] = (cand_boxes[:, [0, 2]] * scale_x).clamp(0.0, float(in_w))
            cand_boxes[:, [1, 3]] = (cand_boxes[:, [1, 3]] * scale_y).clamp(0.0, float(in_h))

            keep = torchvision.ops.batched_nms(cand_boxes, cand_scores, cand_labels, n_th)
            keep = keep[:m_det]

            final_boxes = cand_boxes[keep]
            final_scores = cand_scores[keep]
            final_labels = cand_labels[keep]

            if output_format == "normalized":
                final_boxes[:, [0, 2]] /= float(in_w)
                final_boxes[:, [1, 3]] /= float(in_h)

            results.append({
                "boxes": final_boxes,
                "scores": final_scores,
                "labels": final_labels,
            })

        return results
