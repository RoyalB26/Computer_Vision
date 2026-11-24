"""Slot-Guided Glance-and-Focus (SG-GF) for VisDroneYOLO11Slot.

Uses the model's own P3 Slot Attention map (`1 - A_bg` + small/rare class slots) as a zero-RL
spatial router to select K adaptive high-resolution crops directly from the original image,
runs a single batched Focus pass, and fuses Glance + Focus detections via Weighted Boxes Fusion.

Fully compatible with both legacy `epoch_70.pth` (raw-feats GAP+GMP `presence_head`) and
newer checkpoints (compact `proj_in` Top-K `presence_head`, per-class prototypes, MUFASA T=2).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tv_ops
import torchvision.transforms.functional as TF

from VisDrone_yolov11.model import VisDroneYOLO11Slot


def _patch_legacy_presence_head(model: VisDroneYOLO11Slot, sd: Dict[str, torch.Tensor]) -> bool:
    """Detect and restore legacy raw-feats GAP+GMP presence_head (896 -> 128 -> 10) for epoch_70.pth."""
    w0 = sd.get("slot_refiner.presence_head.0.weight")
    if w0 is None or w0.ndim != 2:
        return False
    refiner = model.slot_refiner
    if refiner.presence_head is None:
        return False

    expected_in = 2 * sum(refiner.in_channels)  # 896 for (64, 128, 256)
    if int(w0.shape[1]) != expected_in:
        return False

    hidden_dim = int(w0.shape[0])
    refiner.presence_head = nn.Sequential(
        nn.Linear(expected_in, hidden_dim),
        nn.SiLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, refiner.num_classes),
    )

    # Bind legacy forward that computes presence_head on raw feats_f (GAP + GMP)
    def _legacy_forward(self_ref, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        bs = feats[0].shape[0]
        orig_dtype = feats[0].dtype
        device = feats[0].device

        with torch.amp.autocast(device_type=device.type, enabled=False):
            feats_f = [torch.nan_to_num(f.float(), nan=0.0, posinf=1e4, neginf=-1e4) for f in feats]

            pooled = torch.cat(
                [t for f in feats_f for t in (f.mean(dim=(2, 3)), f.amax(dim=(2, 3)))],
                dim=-1,
            )
            presence_logits = self_ref.presence_head(pooled)
            fl = self_ref.asa_gate_floor
            presence_gate = fl + (1.0 - fl) * torch.sigmoid(presence_logits)

            tokens_list = []
            resolutions: List[Tuple[int, int]] = []
            counts: List[int] = []
            for i, f in enumerate(feats_f):
                h, w = f.shape[2], f.shape[3]
                resolutions.append((h, w))
                counts.append(h * w)
                proj = self_ref.proj_in[i](f)
                proj = proj + self_ref.pos_emb(proj)
                tok = proj.flatten(2).transpose(1, 2) + self_ref.level_embed[i]
                tokens_list.append(tok)

            tokens = torch.cat(tokens_list, dim=1)
            slots_sub, attn_sub, slots, attn_ori, dots = self_ref.slot_attention(
                tokens, presence_gate=presence_gate
            )

            s_write = (
                self_ref.slot_attention.num_slots
                if self_ref.use_bg_slot_write
                else self_ref.slot_attention.num_fg_slots
            )
            attn_write = attn_sub[:, :s_write, :]
            g_sub = presence_gate[:, self_ref.slot_attention.slot_to_cls]
            if self_ref.use_bg_slot_write:
                g_sub = torch.cat([g_sub, torch.ones((bs, 1), device=device, dtype=g_sub.dtype)], dim=1)
            attn_write = attn_write * g_sub.unsqueeze(-1)
            slot_broadcast = torch.einsum("bsd,bsn->bdn", slots_sub[:, :s_write, :], attn_write)

            cls_feats: List[torch.Tensor] = []
            offset = 0
            for i, (f, (h, w), n_i) in enumerate(zip(feats_f, resolutions, counts)):
                slot_map_i = slot_broadcast[:, :, offset : offset + n_i].view(bs, self_ref.slot_dim, h, w)
                offset += n_i
                slot_map_i = self_ref.slot_drop(slot_map_i)
                if self_ref.fusion_mode == "film":
                    delta_gamma, delta_beta = self_ref.modulators[i](slot_map_i).chunk(2, dim=1)
                    f_cls = f * (1.0 + 0.5 * torch.tanh(delta_gamma)) + 0.5 * torch.tanh(delta_beta)
                else:
                    f_cls = f + self_ref.modulators[i](slot_map_i)
                cls_feats.append(f_cls.to(dtype=orig_dtype))

            logit_prior = None
            if self_ref.logit_prior_scale is not None:
                logit_prior = self_ref.logit_prior_scale * dots[:, : self_ref.num_classes, :]

        slot_info = {
            "slots": slots,
            "slots_sub": slots_sub,
            "attn_ori": attn_ori,
            "dots": dots,
            "resolutions": tuple(resolutions),
            "logit_prior": logit_prior,
            "presence_logits": presence_logits,
            "presence_gate": presence_gate,
        }
        return cls_feats, slot_info

    import types
    refiner.forward = types.MethodType(_legacy_forward, refiner)
    return True


def load_slot_model_compat(
    weights_path: str,
    model_cfg: Dict[str, Any],
    device: str = "cuda",
) -> VisDroneYOLO11Slot:
    """Load VisDroneYOLO11Slot checkpoint with automatic legacy epoch_70.pth compatibility."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
    else:
        sd = ckpt
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()

    cfg_kwargs = {k: v for k, v in model_cfg.items() if k not in ("type", "pretrained")}
    cfg_kwargs["pretrained"] = None
    model = VisDroneYOLO11Slot(**cfg_kwargs)

    is_legacy = _patch_legacy_presence_head(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"[SG-GF Load] Loaded {weights_path} "
        f"(legacy_presence={is_legacy}, missing={len(missing)}, unexpected={len(unexpected)})"
    )
    return model.to(device).eval()


def weighted_boxes_fusion_per_class(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    weights: List[float],
    iou_thr: float = 0.55,
    max_det: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast pure-NumPy Class-Aware Weighted Boxes Fusion (WBF) for merging Glance + Focus boxes.

    Unlike hard NMS, WBF averages coordinates of matching detections weighted by confidence,
    and preserves closely packed objects without aggressively deleting adjacent boxes.
    """
    all_b, all_s, all_l, all_w = [], [], [], []
    for b, s, l, w in zip(boxes_list, scores_list, labels_list, weights):
        if len(b) == 0:
            continue
        all_b.append(b.astype(np.float32))
        all_s.append((s * float(w)).astype(np.float32))
        all_l.append(l.astype(np.int64))
        all_w.append(np.full(len(b), float(w), dtype=np.float32))

    if not all_b:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    boxes = np.concatenate(all_b, axis=0)
    scores = np.concatenate(all_s, axis=0)
    labels = np.concatenate(all_l, axis=0)
    src_w = np.concatenate(all_w, axis=0)

    out_boxes: List[np.ndarray] = []
    out_scores: List[float] = []
    out_labels: List[int] = []

    for cls_id in np.unique(labels):
        m = labels == cls_id
        cb = boxes[m]
        cs = scores[m]
        cw = src_w[m]

        order = np.argsort(-cs)
        cb, cs, cw = cb[order], cs[order], cw[order]

        # Maintain running weighted clusters for this class
        cluster_boxes: List[np.ndarray] = []
        cluster_wsum: List[float] = []
        cluster_scores: List[List[float]] = []
        cluster_src_w: List[List[float]] = []

        for i in range(len(cb)):
            box_i = cb[i]
            sc_i = float(cs[i])
            w_i = float(cw[i])

            if not cluster_boxes:
                cluster_boxes.append(box_i.copy())
                cluster_wsum.append(sc_i)
                cluster_scores.append([sc_i])
                cluster_src_w.append([w_i])
                continue

            c_arr = np.stack(cluster_boxes, axis=0)  # (C, 4)
            ix1 = np.maximum(c_arr[:, 0], box_i[0])
            iy1 = np.maximum(c_arr[:, 1], box_i[1])
            ix2 = np.minimum(c_arr[:, 2], box_i[2])
            iy2 = np.minimum(c_arr[:, 3], box_i[3])
            iw = np.maximum(0.0, ix2 - ix1)
            ih = np.maximum(0.0, iy2 - iy1)
            inter = iw * ih
            area_c = np.maximum(0.0, c_arr[:, 2] - c_arr[:, 0]) * np.maximum(0.0, c_arr[:, 3] - c_arr[:, 1])
            area_i = max(0.0, float((box_i[2] - box_i[0]) * (box_i[3] - box_i[1])))
            ious = inter / np.maximum(area_c + area_i - inter, 1e-7)

            best_idx = int(np.argmax(ious))
            if ious[best_idx] >= iou_thr:
                old_wsum = cluster_wsum[best_idx]
                new_wsum = old_wsum + sc_i
                cluster_boxes[best_idx] = (cluster_boxes[best_idx] * old_wsum + box_i * sc_i) / max(new_wsum, 1e-7)
                cluster_wsum[best_idx] = new_wsum
                cluster_scores[best_idx].append(sc_i)
                cluster_src_w[best_idx].append(w_i)
            else:
                cluster_boxes.append(box_i.copy())
                cluster_wsum.append(sc_i)
                cluster_scores.append([sc_i])
                cluster_src_w.append([w_i])

        for b_c, sc_list, w_list in zip(cluster_boxes, cluster_scores, cluster_src_w):
            # Confidence: max-boosted weighted mean (capped at 1.0) so single-pass Focus rescues are NOT penalized!
            mean_s = sum(sc_list) / max(sum(w_list), 1e-7)
            if len(sc_list) > 1:
                # Agreement between Glance & Focus gets a gentle +10% confidence bonus
                fused_s = min(1.0, mean_s * 1.10)
            else:
                fused_s = min(1.0, mean_s)
            out_boxes.append(b_c)
            out_scores.append(fused_s)
            out_labels.append(int(cls_id))

    if not out_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    ob = np.stack(out_boxes, axis=0).astype(np.float32)
    os_ = np.array(out_scores, dtype=np.float32)
    ol = np.array(out_labels, dtype=np.int64)

    order = np.argsort(-os_)[:max_det]
    return ob[order], os_[order], ol[order]


class SlotGuidedGlanceFocus:
    """Two-stage Slot-Guided Glance-and-Focus inference engine using a single VisDroneYOLO11Slot model."""

    def __init__(
        self,
        model: VisDroneYOLO11Slot,
        device: str = "cuda",
        glance_size: int = 960,
        focus_size: int = 640,
        max_crops: int = 5,
        crop_ratio: float = 0.25,
        overlap_suppress: float = 0.85,
        max_crop_iou: float = 0.15,
        min_peak_score: float = 0.12,
        pool_max_weight: float = 0.40,
        small_class_ids: Tuple[int, ...] = (0, 1, 2, 6, 7, 9),
        small_class_boost: float = 1.5,
        class_focus_weights: Optional[List[float]] = None,
        solved_conf_thresh: float = 0.45,
        solved_min_area_frac: float = 0.0015,
        edge_margin_px: int = 6,
        max_focus_box_frac: float = 0.65,
        fusion_mode: str = "wbf",
        wbf_iou_thresh: float = 0.50,
        glance_weight: float = 1.0,
        focus_weight: float = 1.15,
        conf_threshold: float = 0.001,
        nms_threshold: float = 0.5,
        max_det: int = 500,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.glance_size = int(glance_size)
        self.focus_size = int(focus_size)
        self.max_crops = max(0, int(max_crops))
        self.crop_ratio = float(crop_ratio)
        self.overlap_suppress = float(overlap_suppress)
        self.max_crop_iou = float(max_crop_iou)
        self.min_peak_score = float(min_peak_score)
        self.pool_max_weight = float(pool_max_weight)
        self.small_class_ids = tuple(int(c) for c in small_class_ids)
        self.small_class_boost = float(small_class_boost)
        self.class_focus_weights = [float(w) for w in class_focus_weights] if class_focus_weights is not None else None
        self.solved_conf_thresh = float(solved_conf_thresh)
        self.solved_min_area_frac = float(solved_min_area_frac)
        self.edge_margin_px = int(edge_margin_px)
        self.max_focus_box_frac = float(max_focus_box_frac)
        self.fusion_mode = str(fusion_mode).lower()
        self.wbf_iou_thresh = float(wbf_iou_thresh)
        self.glance_weight = float(glance_weight)
        self.focus_weight = float(focus_weight)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.max_det = int(max_det)

    def _build_focus_map(
        self,
        slot_info: Dict[str, Any],
        glance_det: Dict[str, torch.Tensor],
        orig_w: int,
        orig_h: int,
    ) -> torch.Tensor:
        """Build P3-resolution focus priority map M_focus (H3, W3) from Slot Attention `attn_ori`."""
        attn_ori = slot_info["attn_ori"][0].float()  # (K+1, N)
        h3, w3 = slot_info["resolutions"][0]
        n3 = h3 * w3
        attn_p3 = attn_ori[:, :n3].view(-1, h3, w3)  # (K+1, H3, W3)
        k_cls = attn_p3.shape[0] - 1

        fg_sal = (1.0 - attn_p3[-1]).clamp(0.0, 1.0)  # (H3, W3)
        if self.class_focus_weights is not None and len(self.class_focus_weights) == k_cls:
            cw = torch.tensor(self.class_focus_weights, device=attn_p3.device, dtype=torch.float32).view(k_cls, 1, 1)
            weighted_fg = (attn_p3[:k_cls] * cw).sum(dim=0)
            m_focus = fg_sal * (0.25 + weighted_fg)
        elif self.small_class_ids and self.small_class_boost > 0.0:
            small_sum = attn_p3[list(self.small_class_ids)].sum(dim=0).clamp(0.0, 1.0)
            m_focus = fg_sal * (1.0 + self.small_class_boost * small_sum)
        else:
            m_focus = fg_sal

        # Downweight regions already confidently solved by Glance
        boxes = glance_det["boxes"]
        scores = glance_det["scores"]
        if len(boxes) > 0 and self.solved_conf_thresh < 1.0:
            img_area = float(orig_w * orig_h)
            bw = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
            bh = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
            solved = (scores >= self.solved_conf_thresh) & ((bw * bh) >= self.solved_min_area_frac * img_area)
            if solved.any():
                solved_mask = torch.zeros((h3, w3), device=m_focus.device, dtype=torch.float32)
                sb = boxes[solved]
                sx, sy = w3 / float(orig_w), h3 / float(orig_h)
                for b in sb:
                    x1 = max(0, min(w3 - 1, int(b[0].item() * sx)))
                    y1 = max(0, min(h3 - 1, int(b[1].item() * sy)))
                    x2 = max(x1 + 1, min(w3, int(np.ceil(b[2].item() * sx))))
                    y2 = max(y1 + 1, min(h3, int(np.ceil(b[3].item() * sy))))
                    solved_mask[y1:y2, x1:x2] = 0.85
                m_focus = m_focus * (1.0 - solved_mask)

        return m_focus

    def _propose_crops(
        self,
        m_focus: torch.Tensor,
        orig_w: int,
        orig_h: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Select up to `max_crops` non-redundant high-res crop windows via GPU avg_pool2d."""
        if self.max_crops <= 0:
            return []

        h3, w3 = m_focus.shape
        win_h = max(4, min(h3, int(round(h3 * self.crop_ratio))))
        win_w = max(4, min(w3, int(round(w3 * self.crop_ratio))))

        # Actual pixel crop size on original image
        crop_w = max(64, min(orig_w, int(round(orig_w * self.crop_ratio))))
        crop_h = max(64, min(orig_h, int(round(orig_h * self.crop_ratio))))

        work_map = m_focus.clone()
        crops: List[Tuple[int, int, int, int]] = []
        sup_margin_h = int(round(win_h * (1.0 - self.overlap_suppress) * 0.5))
        sup_margin_w = int(round(win_w * (1.0 - self.overlap_suppress) * 0.5))

        ph = h3 - win_h + 1
        pw = w3 - win_w + 1
        # Precompute candidate window grid in P3 coordinates for fast vectorized IoU masking
        gy_grid = torch.arange(ph, device=m_focus.device, dtype=torch.float32).view(ph, 1).expand(ph, pw)
        gx_grid = torch.arange(pw, device=m_focus.device, dtype=torch.float32).view(1, pw).expand(ph, pw)
        win_area = float(win_h * win_w)
        blocked_mask = torch.zeros((ph, pw), device=m_focus.device, dtype=torch.bool)

        for _ in range(self.max_crops):
            inp_4d = work_map.unsqueeze(0).unsqueeze(0)
            avg_p = F.avg_pool2d(inp_4d, kernel_size=(win_h, win_w), stride=1)[0, 0]  # (ph, pw)
            if self.pool_max_weight > 0.0:
                max_p = F.max_pool2d(inp_4d, kernel_size=(win_h, win_w), stride=1)[0, 0]
                pooled = (1.0 - self.pool_max_weight) * avg_p + self.pool_max_weight * max_p
            else:
                pooled = avg_p

            if blocked_mask.any():
                pooled = pooled.masked_fill(blocked_mask, -1.0)

            max_val = float(pooled.max().item())
            if max_val < self.min_peak_score:
                break

            flat_idx = int(pooled.argmax().item())
            gy = flat_idx // pw
            gx = flat_idx % pw

            # Map (gx, gy) on P3 grid to original image pixel crop [x1, y1, x2, y2]
            x1 = int(round((gx / float(w3)) * orig_w))
            y1 = int(round((gy / float(h3)) * orig_h))
            x1 = max(0, min(orig_w - crop_w, x1))
            y1 = max(0, min(orig_h - crop_h, y1))
            crops.append((x1, y1, x1 + crop_w, y1 + crop_h))

            # 1. Suppress inner saliency on work_map
            sy1 = min(h3, gy + sup_margin_h)
            sy2 = max(sy1, gy + win_h - sup_margin_h)
            sx1 = min(w3, gx + sup_margin_w)
            sx2 = max(sx1, gx + win_w - sup_margin_w)
            work_map[sy1:sy2, sx1:sx2] = 0.0

            # 2. Hard Crop-IoU mask so next crops never overlap > max_crop_iou with already chosen crops
            if self.max_crop_iou < 1.0:
                ix1 = torch.clamp(gx_grid, min=float(gx))
                iy1 = torch.clamp(gy_grid, min=float(gy))
                ix2 = torch.clamp(gx_grid + win_w, max=float(gx + win_w))
                iy2 = torch.clamp(gy_grid + win_h, max=float(gy + win_h))
                inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)
                iou = inter / (2.0 * win_area - inter + 1e-6)
                blocked_mask |= (iou > self.max_crop_iou)

        return crops

    @torch.no_grad()
    def predict_image(
        self,
        orig_img: Image.Image,
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        max_det: Optional[int] = None,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        """Run Slot-Guided Glance-and-Focus on a single original-resolution PIL image."""
        c_th = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        n_th = self.nms_threshold if nms_threshold is None else float(nms_threshold)
        m_det = self.max_det if max_det is None else int(max_det)

        orig_w, orig_h = orig_img.size

        # ------------------------------------------------------------------
        # 1. GLANCE PASS (single full-image forward + capture P3 slot attn)
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        g_img = TF.resize(orig_img, [self.glance_size, self.glance_size])
        g_tensor = TF.to_tensor(g_img).unsqueeze(0).to(self.device)

        slot_cache: Dict[str, Any] = {}

        def _slot_hook(_mod, _inp, out):
            _, s_info = out
            slot_cache["slot_info"] = s_info

        handle = self.model.slot_refiner.register_forward_hook(_slot_hook)
        try:
            g_det = self.model.predict(
                g_tensor,
                conf_threshold=c_th,
                nms_threshold=n_th,
                max_det=m_det,
                output_format="xyxy",
            )[0]
        finally:
            handle.remove()

        # Scale Glance boxes to original image coordinates
        g_boxes = g_det["boxes"].clone()
        if len(g_boxes) > 0:
            g_boxes[:, [0, 2]] *= orig_w / float(self.glance_size)
            g_boxes[:, [1, 3]] *= orig_h / float(self.glance_size)
            g_boxes[:, [0, 2]] = g_boxes[:, [0, 2]].clamp(0.0, float(orig_w))
            g_boxes[:, [1, 3]] = g_boxes[:, [1, 3]].clamp(0.0, float(orig_h))
        glance_scaled = {
            "boxes": g_boxes,
            "scores": g_det["scores"],
            "labels": g_det["labels"],
        }
        t_glance_ms = (time.perf_counter() - t0) * 1000.0

        # ------------------------------------------------------------------
        # 2. SLOT-GUIDED CROP PROPOSAL (zero RL, ~0.5 ms on GPU)
        # ------------------------------------------------------------------
        t1 = time.perf_counter()
        m_focus = self._build_focus_map(slot_cache["slot_info"], glance_scaled, orig_w, orig_h)
        crops = self._propose_crops(m_focus, orig_w, orig_h)

        # ------------------------------------------------------------------
        # 3. BATCHED FOCUS PASS on K high-resolution crops from original image
        # ------------------------------------------------------------------
        f_boxes_list: List[np.ndarray] = []
        f_scores_list: List[np.ndarray] = []
        f_labels_list: List[np.ndarray] = []

        if crops:
            crop_tensors = []
            for (cx1, cy1, cx2, cy2) in crops:
                patch = orig_img.crop((cx1, cy1, cx2, cy2))
                patch_r = TF.resize(patch, [self.focus_size, self.focus_size])
                crop_tensors.append(TF.to_tensor(patch_r))
            f_batch = torch.stack(crop_tensors, dim=0).to(self.device)  # (K, 3, focus_size, focus_size)

            f_dets = self.model.predict(
                f_batch,
                conf_threshold=c_th,
                nms_threshold=n_th,
                max_det=m_det,
                output_format="xyxy",
            )

            margin = float(self.edge_margin_px)
            for (cx1, cy1, cx2, cy2), fd in zip(crops, f_dets):
                if len(fd["boxes"]) == 0:
                    continue
                cw = float(cx2 - cx1)
                ch = float(cy2 - cy1)
                crop_area = cw * ch

                fb = fd["boxes"].float().cpu().numpy()
                fs = fd["scores"].float().cpu().numpy()
                fl = fd["labels"].long().cpu().numpy()

                # Scale from focus_size back to crop pixels
                fb[:, [0, 2]] *= cw / float(self.focus_size)
                fb[:, [1, 3]] *= ch / float(self.focus_size)

                # Filter boxes truncated at internal crop edges (keep boxes on true outer image borders!)
                keep = np.ones(len(fb), dtype=bool)
                if cx1 > 0:
                    keep &= fb[:, 0] > margin
                if cy1 > 0:
                    keep &= fb[:, 1] > margin
                if cx2 < orig_w:
                    keep &= fb[:, 2] < (cw - margin)
                if cy2 < orig_h:
                    keep &= fb[:, 3] < (ch - margin)

                # Filter giant partial boxes (Glance already detects full large vehicles cleanly)
                box_areas = np.maximum(0.0, fb[:, 2] - fb[:, 0]) * np.maximum(0.0, fb[:, 3] - fb[:, 1])
                keep &= box_areas <= (self.max_focus_box_frac * crop_area)

                if not keep.any():
                    continue

                fb = fb[keep]
                fb[:, [0, 2]] = np.clip(fb[:, [0, 2]] + cx1, 0.0, float(orig_w))
                fb[:, [1, 3]] = np.clip(fb[:, [1, 3]] + cy1, 0.0, float(orig_h))
                f_boxes_list.append(fb)
                f_scores_list.append(fs[keep])
                f_labels_list.append(fl[keep])

        # ------------------------------------------------------------------
        # 4. FUSION (WBF or Batched NMS)
        # ------------------------------------------------------------------
        gb_np = glance_scaled["boxes"].float().cpu().numpy()
        gs_np = glance_scaled["scores"].float().cpu().numpy()
        gl_np = glance_scaled["labels"].long().cpu().numpy()

        if f_boxes_list:
            fb_all = np.concatenate(f_boxes_list, axis=0)
            fs_all = np.concatenate(f_scores_list, axis=0)
            fl_all = np.concatenate(f_labels_list, axis=0)

            if self.fusion_mode == "wbf":
                fused_b, fused_s, fused_l = weighted_boxes_fusion_per_class(
                    boxes_list=[gb_np, fb_all],
                    scores_list=[gs_np, fs_all],
                    labels_list=[gl_np, fl_all],
                    weights=[self.glance_weight, self.focus_weight],
                    iou_thr=self.wbf_iou_thresh,
                    max_det=m_det,
                )
            else:
                cat_b = torch.from_numpy(np.concatenate([gb_np, fb_all], axis=0))
                cat_s = torch.from_numpy(np.concatenate([gs_np, fs_all], axis=0))
                cat_l = torch.from_numpy(np.concatenate([gl_np, fl_all], axis=0))
                keep = tv_ops.batched_nms(cat_b, cat_s, cat_l, n_th)[:m_det]
                fused_b = cat_b[keep].numpy()
                fused_s = cat_s[keep].numpy()
                fused_l = cat_l[keep].numpy()
        else:
            fused_b, fused_s, fused_l = gb_np[:m_det], gs_np[:m_det], gl_np[:m_det]

        t_focus_ms = (time.perf_counter() - t1) * 1000.0

        result: Dict[str, Any] = {
            "boxes": fused_b,
            "scores": fused_s,
            "labels": fused_l,
        }
        if return_debug:
            result["debug"] = {
                "focus_map": m_focus.detach().cpu().numpy(),
                "crops": crops,
                "glance_boxes": gb_np,
                "glance_scores": gs_np,
                "glance_labels": gl_np,
                "num_crops": len(crops),
                "t_glance_ms": t_glance_ms,
                "t_focus_ms": t_focus_ms,
            }
        return result
