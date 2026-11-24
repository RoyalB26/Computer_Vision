#!/usr/bin/env python3
"""Component-by-Component Validation & Spatiotemporal Feature Inspection of Action-Slot.

Adapts the component-dissection methodology of `VisDrone_yolov11/val_visdrone_components.py`
specifically to Multi-Label Spatiotemporal Atomic Activity Recognition in Traffic Scenes
(Action-Slot, arXiv:2311.17948v2 — Kung et al., TACO / OATS / nuScenes).

Dissects ActionSlotModel into 5 inspectable levels:
  1. 3D Spatiotemporal Backbone (X3D-M block0..block4): spatial FG/BG contrast,
     temporal motion energy, temporal FG-vs-BG motion isolation, spatial/temporal decorrelation.
  2. Projection, Ego Branch & Positional Encodings (conv3d, conv3d_ego, 2D Sin-Cos PE,
     SoftPositionEmbed3D, Token MLP, Keys/Values): PE-to-feature dominance & temporal-vs-spatial PE balance.
  3. Parallel Spatiotemporal Slot Attention (Allocated Action Slots + Background Slot):
     BG slot mask guidance (IoU, Dice, BCE, FG leak), negative slot regularization (L_neg, FA>0.2),
     positive slot localization (FG focus, temporal span/entropy, spatial compactness, co-occurrence overlap),
     competition margins, slot/query orthogonality, slot pos-vs-neg Fisher & global-V divergence.
  4. Classification Heads (ReduceAllocatedHead 32x3 / Allocated_Head 64 + Ego Head) & End-to-End Eval:
     Overall mAP, 6 Road-User Group mAPs (C, K/B, P, C+, K+/B+, P+), Single-vs-Group confusion,
     Topology & Maneuver breakdown, Reverse-Direction confusion, Multi-label Concurrency (k=1..4+),
     Calibration (ECE), Ego-Vehicle Accuracy/F1, Latency & Parameters.
  5. Multi-Task Gradient Conflict & Zero-Retrain Counterfactual Ablations:
     cos_sim(grad_actor, grad_attn / grad_ego), cos_sim(grad_neg, grad_bg),
     Uniform-Attention Ablation (slot localization net mAP gain), and Time-Reversed Ablation (motion sensitivity).

Example:
  python action_slot/val_action_slot_components.py --config action_slot/config.yaml --weights work_dirs/action_slot/best.pth
  python action_slot/val_action_slot_components.py --config action_slot/config.yaml --weights none --max-videos 20
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure both workspace root and action_slot/ are importable (model.py imports model_parts.* & loss)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in (_ROOT_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import action_slot.transform  # noqa: F401 - register Action_Slot_Transform
from action_slot.dataset import ACTION_SLOT_LMDBDataset
from action_slot.metrics.taco_labels import TACO_ACTOR_LABELS
from action_slot.model import ActionSlotModel, build_2d_sincos_position_embedding
from action_slot.model_parts.heads import Allocated_Head, ReduceAllocatedHead


# =============================================================================
# 1. TACO TAXONOMY, TOPOLOGY & SLOT-TO-CLASS MAPPINGS
# =============================================================================

TACO_64_LABELS: List[str] = list(TACO_ACTOR_LABELS[:64])

# Canonical TACO 4 ego-vehicle action classes (Paper Fig. 8 & Appendix A)
TACO_EGO_4_LABELS: List[str] = [
    "Z1-Z1:E (Wait/Slow)",
    "Z1-Z2:E (Right)",
    "Z1-Z3:E (Straight)",
    "Z1-Z4:E (Left)",
]

# Road-user groups matching Paper Table 2 (note: codebase uses 'b'/'b+' for two-wheeler 'K'/'K+')
ROAD_USER_GROUPS: Dict[str, List[int]] = {
    "c": [i for i, n in enumerate(TACO_64_LABELS) if n.startswith("c:")],
    "c+": [i for i, n in enumerate(TACO_64_LABELS) if n.startswith("c+:")],
    "b": [i for i, n in enumerate(TACO_64_LABELS) if n.startswith("b:")],
    "b+": [i for i, n in enumerate(TACO_64_LABELS) if n.startswith("b+:")],
    "p": [i for i, n in enumerate(TACO_64_LABELS) if n.startswith("p:")],
    "p+": [i for i, n in enumerate(TACO_64_LABELS) if n.startswith("p+:")],
}

# Paired 32-slot mapping used by ReduceAllocatedHead (single_cls_idx, group_cls_idx, slot_name, group_key)
REDUCED_SLOT_PAIRS: List[Tuple[int, int, str, str]] = (
    [(i, i + 12, f"c/c+:{TACO_64_LABELS[i].split(':')[1]}", "car") for i in range(12)]
    + [(24 + i, 36 + i, f"b/b+:{TACO_64_LABELS[24 + i].split(':')[1]}", "bike") for i in range(12)]
    + [(48 + i, 56 + i, f"p/p+:{TACO_64_LABELS[48 + i].split(':')[1]}", "ped") for i in range(8)]
)


def _build_reverse_direction_map() -> Dict[int, int]:
    """Map each TACO class (e.g. 'c:z1-z2') to its exact opposite trajectory ('c:z2-z1')."""
    name_to_idx = {n: i for i, n in enumerate(TACO_64_LABELS)}
    rev_map: Dict[int, int] = {}
    for i, name in enumerate(TACO_64_LABELS):
        prefix, traj = name.split(":")
        src, dst = traj.split("-")
        rev_name = f"{prefix}:{dst}-{src}"
        if rev_name in name_to_idx:
            rev_map[i] = name_to_idx[rev_name]
    return rev_map


REVERSE_DIR_MAP: Dict[int, int] = _build_reverse_direction_map()


def _classify_topology(class_name: str) -> Tuple[str, str]:
    """Return (origin_zone_bucket, maneuver_bucket) for a TACO atomic activity class."""
    prefix, traj = class_name.split(":")
    src, dst = traj.split("-")

    # Distance / Origin zone relative to ego vehicle at Z1
    if src in ("z1", "c1", "c4"):
        origin = "near_ego"
    elif src in ("z3", "c2", "c3"):
        origin = "far_opposite"
    else:
        origin = "lateral"

    # Maneuver type in right-hand traffic intersection
    if prefix in ("p", "p+"):
        maneuver = "crosswalk"
    elif traj in ("z1-z3", "z3-z1", "z2-z4", "z4-z2"):
        maneuver = "straight"
    elif traj in ("z1-z2", "z2-z3", "z3-z4", "z4-z1"):
        maneuver = "right_turn"
    else:
        maneuver = "left_turn"
    return origin, maneuver


COMPONENT_ORDER: List[str] = [
    "backbone.block0 (Stem //2)",
    "backbone.block1 (ResStage1 //4)",
    "backbone.block2 (ResStage2 //8)",
    "backbone.block3 (ResStage3 //16)",
    "backbone.block4 (ResStage4 //32)",
    "proj.conv3d (192->256)",
    "proj.conv3d_ego (192->128)",
    "slot.pe_3d (SoftPosEmbed3D)",
    "slot.token_mlp (FC1->FC2)",
    "slot.keys (to_k)",
    "slot.values (to_v)",
    "slot.slots_pre_ffn (Attn Out)",
    "slot.slots_post_ffn (FFN Out)",
    "head.actor (Logits)",
    "head.ego (Logits)",
]


# =============================================================================
# 2. MODEL LOADING & COMPONENT PROBE
# =============================================================================

def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
    else:
        sd = ckpt
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    if not isinstance(sd, dict):
        raise ValueError("Unsupported checkpoint format; expected state_dict mapping.")
    # Strip DDP / Accelerate 'module.' prefix if present
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v
    return cleaned


def load_action_slot_model(
    weights_path: Optional[str],
    device: torch.device,
    model_cfg: Optional[Dict[str, Any]] = None,
    seq_len: int = 16,
) -> Tuple[ActionSlotModel, bool, bool]:
    """Instantiate ActionSlotModel (supporting 32-slot ReduceAllocatedHead and 64-slot Allocated_Head)
    and optionally load checkpoint weights.

    Returns:
        (model, is_reduced_head, has_bg_slot)
    """
    cfg = dict(model_cfg or {})
    sd: Optional[Dict[str, torch.Tensor]] = None

    if weights_path and weights_path.lower() != "none":
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        sd = _extract_state_dict(ckpt)

        # Auto-infer architecture dimensions from state_dict when available
        if "slot_attention.slots_eps" in sd:
            total_slots = int(sd["slot_attention.slots_eps"].shape[1])
            cfg["slot_dim"] = int(sd["slot_attention.slots_eps"].shape[2])
            if total_slots in (33, 65):
                cfg["num_slots"] = total_slots - 1
                cfg["bg_slot"] = True
            else:
                cfg["num_slots"] = total_slots
                cfg["bg_slot"] = False
        if "conv3d_ego.2.weight" in sd:
            cfg["ego_c"] = int(sd["conv3d_ego.2.weight"].shape[0])
        if "head.fc_ego.1.weight" in sd:
            cfg["num_ego_class"] = int(sd["head.fc_ego.1.weight"].shape[0])

    num_ego_class = int(cfg.get("num_ego_class", 4))
    num_actor_class = int(cfg.get("num_actor_class", 64))
    num_slots = int(cfg.get("num_slots", 32))
    slot_dim = int(cfg.get("slot_dim", 256))
    ego_c = int(cfg.get("ego_c", 128))
    bg_slot = bool(cfg.get("bg_slot", True))
    res3d = tuple(cfg.get("resolution3d", (seq_len, 8, 24)))
    dropout = float(cfg.get("dropout", 0.5))

    use_pretrained_bb = (sd is None) and bool(cfg.get("pretrained_backbone", True))
    model = ActionSlotModel(
        num_ego_class=num_ego_class,
        num_actor_class=num_actor_class,
        num_slots=num_slots,
        slot_dim=slot_dim,
        ego_c=ego_c,
        bg_slot=bg_slot,
        resolution3d=res3d,
        dropout=dropout,
        pretrained_backbone=use_pretrained_bb,
        pretrained_backbone_path=cfg.get("pretrained_backbone_path", None),
        finetune_last_n_stages=cfg.get("finetune_last_n_stages", 2),
    )

    # Detect if checkpoint uses 64-slot Allocated_Head (1 logit per slot) instead of ReduceAllocatedHead (3 logits)
    is_reduced_head = True
    if sd is not None and "head.fc_actor.0.1.weight" in sd:
        head_out_dim = int(sd["head.fc_actor.0.1.weight"].shape[0])
        if head_out_dim == 1 and num_slots == num_actor_class:
            is_reduced_head = False
            model.head = Allocated_Head(
                in_channel=slot_dim,
                num_ego_classes=num_ego_class,
                num_actor_classes=num_actor_class,
                ego_channel=ego_c,
            )

    if sd is not None:
        # Filter shape-mismatched keys gracefully (e.g. if evaluating partial/experimental checkpoints)
        model_sd = model.state_dict()
        compat_sd = {}
        shape_mismatch = []
        for k, v in sd.items():
            if k in model_sd:
                if model_sd[k].shape == v.shape:
                    compat_sd[k] = v
                else:
                    shape_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        missing, unexpected = model.load_state_dict(compat_sd, strict=False)
        print(
            f"[Load] ActionSlotModel ({'ReduceAllocatedHead 32x3' if is_reduced_head else 'Allocated_Head 64x1'}, "
            f"slots={num_slots}{'+BG' if bg_slot else ''}) from {weights_path}\n"
            f"       loaded={len(compat_sd)}, missing={len(missing)}, unexpected={len(unexpected)}, "
            f"shape_mismatch={len(shape_mismatch)}"
        )
        if shape_mismatch:
            print(f"  [Warn] Shape mismatches (first 5): {shape_mismatch[:5]}")
    else:
        print(
            f"[Init] ActionSlotModel ({'ReduceAllocatedHead 32x3' if is_reduced_head else 'Allocated_Head 64x1'}, "
            f"slots={num_slots}{'+BG' if bg_slot else ''}) without checkpoint (pretrained_backbone={use_pretrained_bb})"
        )

    return model.to(device).eval(), is_reduced_head, bg_slot


class ActionSlotComponentProbe:
    """Attaches forward hooks across X3D Backbone, Projection/Ego, SlotAttention internals, and Heads."""

    def __init__(self, model: ActionSlotModel):
        self.model = model
        self.features: Dict[str, torch.Tensor] = {}
        self.sa_cache: Dict[str, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _make_out_hook(self, name: str, store: Optional[Dict[str, torch.Tensor]] = None):
        target_dict = self.features if store is None else store

        def hook(_mod, _inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            target_dict[name] = out.detach()
        return hook

    def _register_hooks(self):
        # 1. X3D-M backbone blocks (0..4)
        bb_names = [
            "backbone.block0 (Stem //2)",
            "backbone.block1 (ResStage1 //4)",
            "backbone.block2 (ResStage2 //8)",
            "backbone.block3 (ResStage3 //16)",
            "backbone.block4 (ResStage4 //32)",
        ]
        for idx, blk in enumerate(self.model.backbone.blocks):
            if idx < len(bb_names):
                self.handles.append(blk.register_forward_hook(self._make_out_hook(bb_names[idx])))

        # 2. Spatiotemporal projection & Ego branch
        self.handles.append(self.model.conv3d.register_forward_hook(self._make_out_hook("proj.conv3d (192->256)")))
        if self.model.conv3d_ego is not None:
            self.handles.append(self.model.conv3d_ego.register_forward_hook(self._make_out_hook("proj.conv3d_ego (192->128)")))
        if self.model.pool_ego is not None:
            self.handles.append(self.model.pool_ego.register_forward_hook(self._make_out_hook("ego_pooled", self.sa_cache)))

        # 3. Slot Attention internals
        sa = self.model.slot_attention

        def _pe_hook(_mod, inp, out):
            self.sa_cache["pe_in"] = inp[0].detach()
            self.sa_cache["pe_out"] = out.detach()
            self.features["slot.pe_3d (SoftPosEmbed3D)"] = out.detach().permute(0, 4, 1, 2, 3)

        self.handles.append(sa.pe.register_forward_hook(_pe_hook))

        def _token_mlp_hook(_mod, _inp, out):
            self.sa_cache["token_mlp"] = out.detach()
            self.features["slot.token_mlp (FC1->FC2)"] = out.detach()

        self.handles.append(sa.FC2.register_forward_hook(_token_mlp_hook))
        self.handles.append(sa.to_k.register_forward_hook(self._make_out_hook("k", self.sa_cache)))
        self.handles.append(sa.to_v.register_forward_hook(self._make_out_hook("v", self.sa_cache)))
        self.handles.append(sa.to_q.register_forward_hook(self._make_out_hook("q", self.sa_cache)))

        def _pre_ffn_hook(_mod, inp, _out):
            self.sa_cache["slots_pre_ffn"] = inp[0].detach()
            self.features["slot.slots_pre_ffn (Attn Out)"] = inp[0].detach()

        self.handles.append(sa.norm_pre_ff.register_forward_hook(_pre_ffn_hook))
        self.handles.append(sa.fc2.register_forward_hook(self._make_out_hook("ffn_delta", self.sa_cache)))

        def _sa_out_hook(_mod, _inp, out):
            slots_out, attn_ori = out
            self.sa_cache["slots_post_ffn"] = slots_out.detach()
            self.sa_cache["attn_ori_raw"] = attn_ori.detach()
            self.features["slot.slots_post_ffn (FFN Out)"] = slots_out.detach()

        self.handles.append(sa.register_forward_hook(_sa_out_hook))

        # 4. Classification Head
        def _head_hook(_mod, _inp, out):
            pred_ego, pred_actor = out
            if pred_actor is not None:
                self.features["head.actor (Logits)"] = pred_actor.detach()
                self.sa_cache["pred_actor_raw"] = pred_actor.detach()
            if pred_ego is not None:
                self.features["head.ego (Logits)"] = pred_ego.detach()
                self.sa_cache["pred_ego_raw"] = pred_ego.detach()

        self.handles.append(self.model.head.register_forward_hook(_head_hook))

    def clear(self):
        self.features.clear()
        self.sa_cache.clear()

    def remove(self):
        for hd in self.handles:
            hd.remove()
        self.handles.clear()


# =============================================================================
# 3. SPATIOTEMPORAL FEATURE, SLOT ATTENTION & GRADIENT DIAGNOSTICS
# =============================================================================

def _compute_pca_rgb_gpu(fs: torch.Tensor) -> torch.Tensor:
    """Fast GPU PCA-3D projection for feature map (C, H, W) -> (H, W, 3) in [0, 1]."""
    c, h, w = fs.shape
    flat = fs.reshape(c, -1).transpose(0, 1).float()  # (H*W, C)
    flat_c = flat - flat.mean(dim=0, keepdim=True)
    if c >= 3:
        cov = (flat_c.transpose(0, 1) @ flat_c) / max(1, flat_c.shape[0] - 1)
        _, eigvecs = torch.linalg.eigh(cov)
        V = eigvecs[:, -3:].flip(1)
        proj = flat_c @ V
    else:
        proj = F.pad(flat_c, (0, 3 - c))
    out = torch.empty_like(proj)
    for ch in range(3):
        col = proj[:, ch]
        lo = torch.quantile(col, 0.02)
        hi = torch.quantile(col, 0.98)
        if hi > lo:
            out[:, ch] = torch.clamp((col - lo) / (hi - lo), 0.0, 1.0)
        else:
            out[:, ch] = 0.5
    return out.reshape(h, w, 3)


def _align_bg_mask_3d(bg_seg: torch.Tensor, T: int, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align bg_seg (T_mask, 1, Hm, Wm) to (T, H, W) boolean masks (fg_mask, bg_mask).

    bg_seg = 1 indicates static background (sky/buildings); 0 indicates foreground road corridor / road users.
    """
    # (1, 1, T_mask, Hm, Wm) -> nearest upsample to (T, H, W)
    m5 = bg_seg.float().permute(1, 0, 2, 3).unsqueeze(0)  # (1, 1, T_mask, Hm, Wm)
    m_res = F.interpolate(m5, size=(T, H, W), mode="nearest")[0, 0]  # (T, H, W)
    bg_m = m_res > 0.5
    fg_m = ~bg_m
    return fg_m, bg_m


def measure_spatiotemporal_component_metrics(
    feat: torch.Tensor,
    bg_seg: Optional[torch.Tensor],
    grid_thw: Tuple[int, int, int],
) -> Dict[str, float]:
    """Compute activation health + spatial FG/BG contrast + temporal dynamics on a component tensor.

    Supports:
      - 5D volumetric tensors (1, C, T, H, W) from X3D backbone, conv3d, conv3d_ego, PE3D.
      - 3D token tensors (1, T*H*W, D) from SlotAttention TokenMLP, Keys, Values (reshaped to 5D).
      - 2D/3D slot or logit tensors (1, S, D) or (1, C) -> activation stats only.
    """
    f = feat[0].float()
    out: Dict[str, float] = {
        "act_mean": float(f.mean().item()),
        "act_std": float(f.std().item()) if f.numel() > 1 else 0.0,
        "neg_frac": float((f < 0).float().mean().item()),
        "zero_frac": float((f.abs() < 1e-6).float().mean().item()),
    }

    vol: Optional[torch.Tensor] = None
    T_g, H_g, W_g = grid_thw
    if f.ndim == 4:
        # (C, T, H, W)
        vol = f
    elif f.ndim == 2 and f.shape[0] == T_g * H_g * W_g:
        # (N, D) -> (D, T, H, W)
        vol = f.transpose(0, 1).reshape(f.shape[1], T_g, H_g, W_g)
    elif f.ndim == 2:
        # (S, D) slot representations
        out["l2_mean"] = float(torch.linalg.vector_norm(f, dim=-1).mean().item())
        return out
    elif f.ndim == 1:
        out["l2_mean"] = float(torch.linalg.vector_norm(f).item())
        return out

    if vol is None:
        return out

    C, T, H, W = vol.shape
    l2 = torch.linalg.vector_norm(vol, dim=0)  # (T, H, W)
    l2_mean = float(l2.mean().item())
    out["l2_mean"] = l2_mean

    # Spatial FG vs BG energy contrast using TACO background segmentation mask
    if bg_seg is not None and bg_seg.numel() > 0:
        fg_m, bg_m = _align_bg_mask_3d(bg_seg.to(vol.device), T, H, W)
        if fg_m.any() and bg_m.any():
            fg_e = float(l2[fg_m].mean().item())
            bg_e = float(l2[bg_m].mean().item())
            out["fg_bg_energy_ratio"] = fg_e / (bg_e + 1e-8)
            out["fg_mass_pct"] = float((l2[fg_m].sum() / (l2.sum() + 1e-8)).item() * 100.0)

        # Temporal Motion Energy Isolation: ||F_{t+1} - F_t||_2 inside FG vs BG
        if T >= 2:
            diff_l2 = torch.linalg.vector_norm(vol[:, 1:, :, :] - vol[:, :-1, :, :], dim=0)  # (T-1, H, W)
            fg_d = fg_m[:-1] | fg_m[1:]
            bg_d = bg_m[:-1] & bg_m[1:]
            if fg_d.any() and bg_d.any():
                out["temporal_fg_bg_ratio"] = float(
                    (diff_l2[fg_d].mean() / (diff_l2[bg_d].mean() + 1e-8)).item()
                )

    # Temporal variance energy ratio: how much feature energy varies across time (motion sensitivity)
    if T >= 2:
        t_std = torch.linalg.vector_norm(vol.std(dim=1), dim=0).mean()  # scalar
        out["temporal_energy_ratio"] = float(t_std.item()) / (l2_mean + 1e-8)

        vn = F.normalize(vol, dim=0, eps=1e-8)
        cos_t = (vn[:, :-1, :, :] * vn[:, 1:, :, :]).sum(dim=0)
        out["temporal_decorr"] = float((1.0 - cos_t).mean().item())

        # Spatial decorrelation between adjacent cells
        decorrs = []
        if W >= 2:
            cos_x = (vn[:, :, :, :-1] * vn[:, :, :, 1:]).sum(dim=0)
            decorrs.append(float((1.0 - cos_x).mean().item()))
        if H >= 2:
            cos_y = (vn[:, :, :-1, :] * vn[:, :, 1:, :]).sum(dim=0)
            decorrs.append(float((1.0 - cos_y).mean().item()))
        if decorrs:
            out["spatial_decorr"] = float(np.mean(decorrs))

    return out


def _get_slot_gt_info(
    actor_gt_64: np.ndarray,
    num_action_slots: int,
    is_reduced_head: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (slot_pos_mask[S], slot_3way_target[S]).

    For ReduceAllocatedHead (S=32): 0=single, 1=group, 2=absent.
    For Allocated_Head (S=64): 0=present, 2=absent.
    """
    if is_reduced_head and num_action_slots == 32:
        slot_pos = np.zeros(32, dtype=bool)
        slot_tgt = np.full(32, 2, dtype=np.int64)
        for s_idx, (c_single, c_group, _, _) in enumerate(REDUCED_SLOT_PAIRS):
            s_on = actor_gt_64[c_single] > 0.5
            g_on = actor_gt_64[c_group] > 0.5
            if s_on or g_on:
                slot_pos[s_idx] = True
                slot_tgt[s_idx] = 0 if s_on else 1
        return slot_pos, slot_tgt
    else:
        slot_pos = actor_gt_64[:num_action_slots] > 0.5
        slot_tgt = np.where(slot_pos, 0, 2).astype(np.int64)
        return slot_pos, slot_tgt


def measure_slot_attention_clip_metrics(
    model: ActionSlotModel,
    sa_cache: Dict[str, torch.Tensor],
    conv3d_feat: torch.Tensor,
    actor_gt_64: np.ndarray,
    bg_seg: torch.Tensor,
    is_reduced_head: bool,
    has_bg_slot: bool,
    grid_thw: Tuple[int, int, int],
    attn_thresh: float = 0.2,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, List[float]]], Dict[str, Any]]:
    """Compute per-clip Action-Slot diagnostics directly matching the 4 paper design pillars.

    Returns:
        (clip_metrics, per_slot_updates, vis_extras)
    """
    sa = model.slot_attention
    T, H, W = grid_thw
    N = T * H * W
    S_act = model.num_slots
    S_all = sa.num_slots

    q = sa_cache["q"][0].float()           # (S_all, D)
    k = sa_cache["k"][0].float()           # (N, D)
    v = sa_cache["v"][0].float()           # (N, D)
    slots_pre = sa_cache["slots_pre_ffn"][0].float()    # (S_kept, D)
    slots_post = sa_cache["slots_post_ffn"][0].float()  # (S_kept, D)
    ffn_delta = sa_cache["ffn_delta"][0].float()        # (S_kept, D)

    # Exact SlotAttention logits and normalized attention maps
    dots = (q @ k.transpose(0, 1)) * sa.scale           # (S_all, N)
    attn_ori = dots.softmax(dim=0) + sa.eps             # (S_all, N) slot-wise competition
    attn_tok = attn_ori / attn_ori.sum(dim=-1, keepdim=True)  # (S_all, N) token-normalized aggregation weights

    slot_pos_np, _ = _get_slot_gt_info(actor_gt_64, S_act, is_reduced_head)
    slot_pos_t = torch.from_numpy(slot_pos_np).to(dots.device)
    slot_neg_t = ~slot_pos_t

    # Align BG mask at slot resolution (T, H, W) and also at supervision resolution (T_mask, 32, 96)
    fg_m3d, bg_m3d = _align_bg_mask_3d(bg_seg.to(dots.device), T, H, W)
    fg_flat = fg_m3d.reshape(-1)  # (N,)
    bg_flat = bg_m3d.reshape(-1)  # (N,)

    out: Dict[str, float] = {
        "dots_std": float(dots.std().item()),
        "dots_abs_max": float(dots.abs().max().item()),
    }
    per_slot: Dict[int, Dict[str, List[float]]] = {
        s: {
            "pos_attn_mean": [], "pos_attn_max": [], "pos_area_gt02_pct": [],
            "neg_attn_mean": [], "neg_fa_gt02_pct": [],
            "pos_fg_focus_pct": [], "pos_temporal_entropy": [],
            "dot_pos_peak": [], "gap_vs_bg_peak": [],
            "pos_vs_global_v_cos": [], "neg_vs_global_v_cos": [],
        }
        for s in range(S_act)
    }

    # 1. PE-to-Feature Ratios (2D Sin-Cos PE + 3D SoftPositionEmbed3D)
    pe_in = sa_cache.get("pe_in")    # (1, T, H, W, D) = conv3d + PE_2D
    pe_out = sa_cache.get("pe_out")  # (1, T, H, W, D) = conv3d + PE_2D + PE_3D
    if pe_in is not None and pe_out is not None:
        vis_raw = conv3d_feat[0].float().permute(1, 2, 3, 0)  # (T, H, W, D)
        pe2d = pe_in[0].float() - vis_raw
        pe3d = pe_out[0].float() - pe_in[0].float()
        vis_norm = float(torch.linalg.vector_norm(vis_raw, dim=-1).mean().item())
        out["pe2d_to_feat_ratio"] = float(torch.linalg.vector_norm(pe2d, dim=-1).mean().item()) / (vis_norm + 1e-8)
        out["pe3d_to_feat_ratio"] = float(torch.linalg.vector_norm(pe3d, dim=-1).mean().item()) / (vis_norm + 1e-8)

    # 2. Slot-wise softmax entropy & competition margin on FG vs BG
    p_safe = attn_ori.clamp(min=1e-7)
    ent_all = -(p_safe * p_safe.log()).sum(dim=0) / float(np.log(S_all))  # (N,) in [0, 1]
    if fg_flat.any():
        out["attn_entropy_fg"] = float(ent_all[fg_flat].mean().item())
        top2_fg = torch.topk(dots[:, fg_flat], k=min(2, S_all), dim=0).values
        if top2_fg.shape[0] == 2:
            out["dots_margin_fg"] = float((top2_fg[0] - top2_fg[1]).mean().item())
    if bg_flat.any():
        out["attn_entropy_bg"] = float(ent_all[bg_flat].mean().item())

    # 3. Background Slot Guidance (L_bg) at native + upsampled supervision resolution (32x96 every 4 frames)
    bg_idx = S_all - 1 if has_bg_slot else -1
    if has_bg_slot:
        bg_attn_flat = attn_ori[bg_idx]  # (N,)
        if bg_flat.any():
            out["bg_slot_on_bg_mean"] = float(bg_attn_flat[bg_flat].mean().item())
        if fg_flat.any():
            out["bg_slot_on_fg_mean"] = float(bg_attn_flat[fg_flat].mean().item())
        if bg_flat.any() and fg_flat.any():
            out["bg_slot_fg_bg_logit_gap"] = float(
                (dots[bg_idx, bg_flat].mean() - dots[bg_idx, fg_flat].mean()).item()
            )

        # Exact supervision-grid evaluation (4 frames x 32 x 96) matching ActionSlotLoss
        bg_attn_3d = bg_attn_flat.view(T, H, W)
        T_mask, _, Hm, Wm = bg_seg.shape
        step_t = max(1, T // T_mask)
        bg_sub = bg_attn_3d[::step_t][:T_mask].unsqueeze(1)  # (T_mask, 1, H, W)
        bg_up = F.interpolate(bg_sub, size=(Hm, Wm), mode="bilinear", align_corners=False).clamp(1e-6, 1.0 - 1e-6)
        bg_gt = bg_seg.to(bg_up.device).float().clamp(0.0, 1.0)

        out["bg_slot_bce"] = float(F.binary_cross_entropy(bg_up, bg_gt).item())
        pred_bin = bg_up > 0.5
        gt_bin = bg_gt > 0.5
        inter = float((pred_bin & gt_bin).sum().item())
        union = float((pred_bin | gt_bin).sum().item())
        out["bg_slot_iou"] = inter / max(union, 1.0)
        out["bg_slot_dice"] = float((2.0 * (bg_up * bg_gt).sum() / (bg_up.sum() + bg_gt.sum() + 1e-8)).item())

        # Who wins argmax on foreground road-corridor tokens? (BG vs Positive Slots vs Negative Slots)
        if fg_flat.any():
            winners_fg = attn_ori[:, fg_flat].argmax(dim=0)  # (N_fg,)
            n_fg = float(fg_flat.sum().item())
            out["bg_slot_leak_fg_pct"] = float((winners_fg == bg_idx).sum().item()) / n_fg * 100.0
            pos_ids = torch.where(slot_pos_t)[0]
            neg_ids = torch.where(slot_neg_t)[0]
            if len(pos_ids) > 0:
                won_pos = torch.isin(winners_fg, pos_ids).sum().item()
                out["fg_claimed_by_pos_pct"] = float(won_pos) / n_fg * 100.0
            if len(neg_ids) > 0:
                won_neg = torch.isin(winners_fg, neg_ids).sum().item()
                out["neg_slot_win_fg_pct"] = float(won_neg) / n_fg * 100.0

    # 4. Positive vs Negative Action Slots (L_neg, localization, temporal dynamics, global-V divergence)
    act_attn = attn_ori[:S_act]  # (S_act, N)
    v_global = F.normalize(v.mean(dim=0), dim=0, eps=1e-8)  # (D,)
    slots_pre_act = F.normalize(slots_pre[:S_act], dim=-1, eps=1e-8)
    slots_post_act = F.normalize(slots_post[:S_act], dim=-1, eps=1e-8)
    cos_to_vglob = (slots_pre_act @ v_global)  # (S_act,)

    if slot_pos_t.any():
        pos_attn = act_attn[slot_pos_t]  # (K_pos, N)
        out["pos_slot_attn_mean"] = float(pos_attn.mean().item())
        out["pos_slot_attn_max"] = float(pos_attn.amax(dim=-1).mean().item())
        out["pos_slot_active_area_gt02_pct"] = float((pos_attn > attn_thresh).float().mean().item() * 100.0)
        out["pos_slot_vs_global_v_cos"] = float(cos_to_vglob[slot_pos_t].mean().item())

        # Positive slot FG focus, temporal span/entropy, spatial compactness, and peak margin vs BG
        fg_focus_list, t_span_list, t_ent_list, s_ent_list, margin_bg_list = [], [], [], [], []
        pos_indices = torch.where(slot_pos_t)[0].tolist()
        for s_idx in pos_indices:
            a_s = act_attn[s_idx]              # (N,)
            a_3d = a_s.view(T, H, W)
            tot_a = float(a_s.sum().item()) + 1e-8
            fg_foc = float(a_s[fg_flat].sum().item()) / tot_a * 100.0 if fg_flat.any() else 0.0
            fg_focus_list.append(fg_foc)

            # Temporal distribution of slot attention across T frames
            t_mass = a_3d.sum(dim=(1, 2))  # (T,)
            t_prob = (t_mass / (t_mass.sum() + 1e-8)).clamp(min=1e-7)
            t_ent = float(-(t_prob * t_prob.log()).sum().item() / np.log(max(T, 2)))
            t_active_frames = float((a_3d.amax(dim=(1, 2)) > attn_thresh).float().mean().item() * 100.0)
            t_span_list.append(t_active_frames)
            t_ent_list.append(t_ent)

            # Spatial compactness (entropy of token-normalized spatial map averaged over active frames)
            s_mass = a_3d.sum(dim=0).reshape(-1)  # (H*W,)
            s_prob = (s_mass / (s_mass.sum() + 1e-8)).clamp(min=1e-7)
            s_ent = float(-(s_prob * s_prob.log()).sum().item() / np.log(max(H * W, 2)))
            s_ent_list.append(s_ent)

            # Peak token dot logit and gap vs BG
            pk_idx = int(a_s.argmax().item())
            d_pk = float(dots[s_idx, pk_idx].item())
            gap_bg = float((dots[s_idx, pk_idx] - dots[bg_idx, pk_idx]).item()) if has_bg_slot else 0.0
            margin_bg_list.append(gap_bg)

            ps = per_slot[s_idx]
            ps["pos_attn_mean"].append(float(a_s.mean().item()))
            ps["pos_attn_max"].append(float(a_s.max().item()))
            ps["pos_area_gt02_pct"].append(float((a_s > attn_thresh).float().mean().item() * 100.0))
            ps["pos_fg_focus_pct"].append(fg_foc)
            ps["pos_temporal_entropy"].append(t_ent)
            ps["dot_pos_peak"].append(d_pk)
            ps["gap_vs_bg_peak"].append(gap_bg)
            ps["pos_vs_global_v_cos"].append(float(cos_to_vglob[s_idx].item()))

        out["pos_slot_fg_focus_pct"] = float(np.mean(fg_focus_list))
        out["pos_slot_temporal_span_pct"] = float(np.mean(t_span_list))
        out["pos_slot_temporal_entropy"] = float(np.mean(t_ent_list))
        out["pos_slot_spatial_entropy"] = float(np.mean(s_ent_list))
        if has_bg_slot:
            out["dots_margin_pos_vs_bg"] = float(np.mean(margin_bg_list))

        # Co-occurring positive slot spatiotemporal overlap (Paper Appendix F occlusion/competition check)
        if len(pos_indices) >= 2:
            pa_norm = F.normalize(pos_attn, dim=-1, eps=1e-8)
            ov_mat = pa_norm @ pa_norm.T
            off_ov = ov_mat[~torch.eye(len(pos_indices), dtype=torch.bool, device=ov_mat.device)]
            out["pos_cooccur_attn_overlap"] = float(off_ov.mean().item())

    if slot_neg_t.any():
        neg_attn = act_attn[slot_neg_t]  # (K_neg, N)
        out["neg_slot_attn_mean"] = float(neg_attn.mean().item())
        out["neg_slot_attn_max"] = float(neg_attn.amax(dim=-1).mean().item())
        out["neg_slot_fa_gt02_pct"] = float((neg_attn > attn_thresh).float().mean().item() * 100.0)
        out["neg_slot_bce"] = float(
            F.binary_cross_entropy(neg_attn.clamp(1e-6, 1.0 - 1e-6), torch.zeros_like(neg_attn)).item()
        )
        out["neg_slot_vs_global_v_cos"] = float(cos_to_vglob[slot_neg_t].mean().item())

        neg_indices = torch.where(slot_neg_t)[0].tolist()
        for s_idx in neg_indices:
            a_s = act_attn[s_idx]
            ps = per_slot[s_idx]
            ps["neg_attn_mean"].append(float(a_s.mean().item()))
            ps["neg_fa_gt02_pct"].append(float((a_s > attn_thresh).float().mean().item() * 100.0))
            ps["neg_vs_global_v_cos"].append(float(cos_to_vglob[s_idx].item()))

    if "pos_slot_attn_mean" in out and "neg_slot_attn_mean" in out:
        out["pos_neg_attn_ratio"] = out["pos_slot_attn_mean"] / (out["neg_slot_attn_mean"] + 1e-8)

    # 5. Output Slot Geometry & Residual FFN Strength
    sim_s = slots_post_act @ slots_post_act.T
    off_s = sim_s[~torch.eye(S_act, dtype=torch.bool, device=sim_s.device)]
    out["slot_cos_sep"] = float((1.0 - off_s).mean().item())

    pre_norm = torch.linalg.vector_norm(slots_pre[:S_act], dim=-1).mean().item()
    ffn_norm = torch.linalg.vector_norm(ffn_delta[:S_act], dim=-1).mean().item()
    out["slot_ffn_delta_ratio"] = float(ffn_norm / (pre_norm + 1e-8))

    # Positive vs Negative slot vector L2 norm ratio
    if slot_pos_t.any() and slot_neg_t.any():
        s_norms = torch.linalg.vector_norm(slots_post[:S_act], dim=-1)
        out["slot_pos_neg_l2_ratio"] = float(
            (s_norms[slot_pos_t].mean() / (s_norms[slot_neg_t].mean() + 1e-8)).item()
        )

    vis_extras = {
        "attn_ori_3d": attn_ori.view(S_all, T, H, W).detach().cpu().numpy(),
        "slot_pos_mask": slot_pos_np,
    }
    return out, per_slot, vis_extras


def _compute_uniform_attn_ablation_logits(
    model: ActionSlotModel,
    sa_cache: Dict[str, torch.Tensor],
    is_reduced_head: bool,
) -> np.ndarray:
    """Zero-retrain counterfactual ablation: replace SlotAttention weights with uniform 1/N pooling
    (every slot receives the identical global mean Value vector), then pass through the exact same
    FFN + classification head to isolate the net mAP gain from spatiotemporal slot localization.
    """
    sa = model.slot_attention
    v = sa_cache["v"].float()                      # (1, N, D)
    slots_post = sa_cache["slots_post_ffn"]        # (1, S_kept, D)
    ego_pooled = sa_cache.get("ego_pooled")
    ego_x = ego_pooled.reshape(1, -1) if ego_pooled is not None else None

    S_kept = slots_post.shape[1]
    v_mean = v.mean(dim=1, keepdim=True).expand(1, S_kept, -1)
    slots_unif = v_mean + sa.fc2(F.relu(sa.fc1(sa.norm_pre_ff(v_mean))))
    slots_unif = slots_unif[:, :model.num_actor_class]
    _, pred_actor_raw = model.head(slots_unif, ego_x=ego_x)
    if is_reduced_head:
        logits_64 = model._process_output(pred_actor_raw)
    else:
        logits_64 = pred_actor_raw
    return torch.sigmoid(logits_64[0].float()).cpu().numpy()


def measure_multitask_gradient_conflict(
    model: ActionSlotModel,
    video_t: torch.Tensor,
    target_dict: Dict[str, Any],
    is_reduced_head: bool,
    device: torch.device,
) -> Dict[str, float]:
    """Measure multi-task gradient cosine similarity and magnitude ratios between:
      - L_actor vs L_attn (L_neg + w_bg * L_bg) at conv3d and backbone.block4
      - L_neg vs L_bg at conv3d
      - L_actor vs L_ego at backbone.block4
    """
    was_training = model.training
    model.eval()
    out: Dict[str, float] = {}

    with torch.enable_grad():
        with torch.no_grad():
            vid = video_t
            if vid.dtype == torch.uint8:
                vid = vid.float().mul_(1.0 / (255.0 * 0.225)).add_(-0.45 / 0.225)
            vid = vid.permute(0, 2, 1, 3, 4)  # (1, 3, T, H, W)
            seq_len = vid.shape[2]
            bb_feats_detached = model.backbone(vid)

        bb_feats = bb_feats_detached.detach().requires_grad_(True)

        # Ego branch
        ego_x = None
        if model.conv3d_ego is not None:
            ego_feat = model.pool_ego(model.conv3d_ego(bb_feats))
            ego_x = ego_feat.reshape(1, model.ego_c)

        _, _, new_seq_len, new_h, new_w = bb_feats.shape
        x_proj = model.conv3d(bb_feats)
        x_proj.retain_grad()

        pe2d = build_2d_sincos_position_embedding(new_h, new_w, model.slot_dim, device=x_proj.device, dtype=x_proj.dtype).unsqueeze(2)
        x = (x_proj + pe2d).permute(0, 2, 3, 4, 1).reshape(1, new_seq_len, new_h, new_w, model.slot_dim)

        slots, attn_masks = model.slot_attention(x)
        b, n, _ = attn_masks.shape
        attn_masks = attn_masks.view(b * n, 1, new_seq_len, new_h, new_w)
        if seq_len > new_seq_len:
            attn_masks = F.interpolate(attn_masks, size=(seq_len, new_h, new_w), mode="trilinear")
        attn_masks = attn_masks.reshape(b, n, seq_len, new_h, new_w).permute(0, 2, 1, 3, 4)

        slots = slots[:, :model.num_actor_class]
        pred_ego, pred_actor = model.head(slots, ego_x=ego_x)

        lbl = {
            "ego": target_dict["ego"].to(device).long().unsqueeze(0),
            "actor": target_dict["actor"].to(device).float().unsqueeze(0),
            "bg_seg": target_dict["bg_seg"].to(device).float().unsqueeze(0),
        }

        # Compute individual loss terms
        loss_ego = model.loss_fn.ego_loss(pred_ego, lbl["ego"])
        if is_reduced_head:
            actor_gt = lbl["actor"]
            p1 = actor_gt[:, 0:24].reshape(1, 2, 12).transpose(1, 2)
            p2 = actor_gt[:, 24:48].reshape(1, 2, 12).transpose(1, 2)
            p3 = actor_gt[:, 48:64].reshape(1, 2, 8).transpose(1, 2)
            fg = torch.cat([p1, p2, p3], dim=1)
            targets_3 = torch.where(fg.sum(dim=-1) == 0, 2, fg.argmax(dim=-1)).long()
            loss_actor = model.loss_fn.actor_loss_reduce(pred_actor, targets_3)
        else:
            loss_actor = F.binary_cross_entropy_with_logits(pred_actor, lbl["actor"])

        attn_dict = model.loss_fn.attn_loss(attn_masks, lbl, validate=False)
        loss_neg = attn_dict.get("attn_loss")
        loss_bg = attn_dict.get("bg_attn_loss")
        loss_attn = None
        if loss_neg is not None and loss_bg is not None:
            loss_attn = loss_neg + loss_bg
        elif loss_neg is not None:
            loss_attn = loss_neg
        elif loss_bg is not None:
            loss_attn = loss_bg

        def _cos_and_ratio(g1: Optional[torch.Tensor], g2: Optional[torch.Tensor]) -> Optional[Tuple[float, float]]:
            if g1 is None or g2 is None:
                return None
            v1 = g1.reshape(-1).float()
            v2 = g2.reshape(-1).float()
            n1 = float(v1.norm().item())
            n2 = float(v2.norm().item())
            if n1 < 1e-9 or n2 < 1e-9:
                return None
            cos = float(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item())
            return cos, n1 / (n2 + 1e-9)

        # Compute gradients cleanly without accumulating into model parameters
        if loss_actor.requires_grad and loss_attn is not None and loss_attn.requires_grad:
            g_act_x, g_act_bb = torch.autograd.grad(loss_actor, (x_proj, bb_feats), retain_graph=True, allow_unused=True)
            g_att_x, g_att_bb = torch.autograd.grad(loss_attn, (x_proj, bb_feats), retain_graph=True, allow_unused=True)

            res = _cos_and_ratio(g_act_x, g_att_x)
            if res is not None:
                out["grad_cos_actor_vs_attn_conv3d"], out["grad_ratio_actor_over_attn_conv3d"] = res
            res_bb = _cos_and_ratio(g_act_bb, g_att_bb)
            if res_bb is not None:
                out["grad_cos_actor_vs_attn_backbone"], out["grad_ratio_actor_over_attn_backbone"] = res_bb

            if loss_neg is not None and loss_bg is not None and loss_neg.requires_grad and loss_bg.requires_grad:
                g_neg_x = torch.autograd.grad(loss_neg, x_proj, retain_graph=True, allow_unused=True)[0]
                g_bg_x = torch.autograd.grad(loss_bg, x_proj, retain_graph=True, allow_unused=True)[0]
                res_nb = _cos_and_ratio(g_neg_x, g_bg_x)
                if res_nb is not None:
                    out["grad_cos_neg_vs_bg_conv3d"], out["grad_ratio_neg_over_bg_conv3d"] = res_nb

            if loss_ego is not None and loss_ego.requires_grad:
                g_ego_bb = torch.autograd.grad(0.5 * loss_ego, bb_feats, retain_graph=False, allow_unused=True)[0]
                res_ae = _cos_and_ratio(g_act_bb, g_ego_bb)
                if res_ae is not None:
                    out["grad_cos_actor_vs_ego_backbone"], out["grad_ratio_actor_over_ego_backbone"] = res_ae

    if was_training:
        model.train()
    return out


# =============================================================================
# 4. MULTI-LABEL, CALIBRATION & REPRESENTATION SEPARABILITY METRICS
# =============================================================================

def compute_binary_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) flattened across all clips x 64 multi-label binary outputs."""
    p = probs.reshape(-1)
    y = targets.reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_tot = max(len(p), 1)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (p >= lo) & (p <= hi if b == n_bins - 1 else p < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            acc = float(y[mask].mean())
            conf = float(p[mask].mean())
            ece += (cnt / n_tot) * abs(acc - conf)
    return float(ece)


def compute_fisher_and_cos_sep(vecs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Compute multi-class Fisher Discriminant Ratio and inter-class cosine distance."""
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan"), float("nan")
    grand_mean = vecs.mean(axis=0)
    sb, sw = 0.0, 0.0
    protos = []
    for c in uniq:
        sub = vecs[labels == c]
        if len(sub) == 0:
            continue
        mu_c = sub.mean(axis=0)
        sb += len(sub) * float(np.sum((mu_c - grand_mean) ** 2))
        sw += float(np.sum((sub - mu_c) ** 2))
        norm_mu = np.linalg.norm(mu_c)
        if norm_mu > 1e-8:
            protos.append(mu_c / norm_mu)
    fisher = sb / (sw + 1e-8)
    cos_sep = float("nan")
    if len(protos) >= 2:
        P = np.stack(protos, axis=0)
        sim = P @ P.T
        off = sim[~np.eye(len(protos), dtype=bool)]
        cos_sep = float(1.0 - off.mean())
    return float(fisher), cos_sep


def compute_taco_multilabel_metrics(
    scores_64: np.ndarray,
    targets_64: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], np.ndarray]:
    """Compute full TACO multi-label metrics (overall mAP, 6 groups, topology, concurrency, per-class AP)."""
    N, C = scores_64.shape
    aps = np.zeros(C, dtype=np.float64)
    active_mask = np.zeros(C, dtype=bool)
    per_class_rows: List[Dict[str, Any]] = []

    # Precompute reverse-direction confusion per class
    rev_conf_list = []

    for c in range(C):
        y_t = targets_64[:, c]
        y_s = scores_64[:, c]
        supp = int((y_t == 1).sum())
        pred_pos = y_s >= threshold
        tp = int(((y_t == 1) & pred_pos).sum())
        fp = int(((y_t == 0) & pred_pos).sum())
        fn = int(((y_t == 1) & (~pred_pos)).sum())
        tn = int(((y_t == 0) & (~pred_pos)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if supp > 0:
            active_mask[c] = True
            try:
                ap = float(average_precision_score(y_t, y_s))
            except Exception:
                ap = 0.0
        else:
            ap = 0.0
        aps[c] = ap

        pos_prob_m = float(y_s[y_t == 1].mean()) if supp > 0 else float("nan")
        neg_prob_m = float(y_s[y_t == 0].mean()) if (N - supp) > 0 else float("nan")

        # Reverse-direction confusion: when class c is active and reverse(c) is absent
        rev_c = REVERSE_DIR_MAP.get(c, -1)
        rev_prob = float("nan")
        if rev_c >= 0:
            m_rev = (y_t == 1) & (targets_64[:, rev_c] == 0)
            if m_rev.any():
                rev_prob = float(scores_64[m_rev, rev_c].mean())
                rev_conf_list.append(rev_prob)

        cname = TACO_64_LABELS[c]
        grp = cname.split(":")[0]
        origin_z, maneuver_t = _classify_topology(cname)

        # Find paired slot_id in 32-slot layout
        slot_id = c
        for s_idx, (cs, cg, _, _) in enumerate(REDUCED_SLOT_PAIRS):
            if c in (cs, cg):
                slot_id = s_idx
                break

        per_class_rows.append({
            "class_id": c,
            "class_name": cname,
            "road_user_group": grp,
            "origin_zone": origin_z,
            "maneuver": maneuver_t,
            "paired_slot_id": slot_id,
            "support": supp,
            "predicted_pos": int(pred_pos.sum()),
            "ap": round(ap, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "pos_prob_mean": round(pos_prob_m, 4) if np.isfinite(pos_prob_m) else np.nan,
            "neg_prob_mean": round(neg_prob_m, 4) if np.isfinite(neg_prob_m) else np.nan,
            "reverse_dir_conf_prob": round(rev_prob, 4) if np.isfinite(rev_prob) else np.nan,
        })

    def _mean_ap(idxs: List[int]) -> float:
        valid = [aps[i] for i in idxs if active_mask[i]]
        return float(np.mean(valid)) if valid else 0.0

    # Overall mAP (matching TACOActionMetric: mean over all 64 classes; also report active-only mAP)
    map_all_64 = float(np.mean(aps))
    map_active = float(np.mean(aps[active_mask])) if active_mask.any() else 0.0

    # Road-user group mAPs (Paper Table 2)
    map_c = float(np.mean([aps[i] for i in ROAD_USER_GROUPS["c"]]))
    map_b = float(np.mean([aps[i] for i in ROAD_USER_GROUPS["b"]]))
    map_p = float(np.mean([aps[i] for i in ROAD_USER_GROUPS["p"]]))
    map_c_plus = float(np.mean([aps[i] for i in ROAD_USER_GROUPS["c+"]]))
    map_b_plus = float(np.mean([aps[i] for i in ROAD_USER_GROUPS["b+"]]))
    map_p_plus = float(np.mean([aps[i] for i in ROAD_USER_GROUPS["p+"]]))

    map_single = _mean_ap(ROAD_USER_GROUPS["c"] + ROAD_USER_GROUPS["b"] + ROAD_USER_GROUPS["p"])
    map_group = _mean_ap(ROAD_USER_GROUPS["c+"] + ROAD_USER_GROUPS["b+"] + ROAD_USER_GROUPS["p+"])

    # Topology & maneuver mAPs
    origin_buckets: Dict[str, List[int]] = {"near_ego": [], "lateral": [], "far_opposite": []}
    maneuver_buckets: Dict[str, List[int]] = {"straight": [], "right_turn": [], "left_turn": [], "crosswalk": []}
    for c, name in enumerate(TACO_64_LABELS):
        oz, mv = _classify_topology(name)
        origin_buckets[oz].append(c)
        maneuver_buckets[mv].append(c)

    # Sample-level multi-label metrics at threshold
    preds_bin = (scores_64 >= threshold).astype(np.int32)
    gt_bin = (targets_64 >= 0.5).astype(np.int32)
    tp_tot = int(((preds_bin == 1) & (gt_bin == 1)).sum())
    fp_tot = int(((preds_bin == 1) & (gt_bin == 0)).sum())
    fn_tot = int(((preds_bin == 0) & (gt_bin == 1)).sum())
    micro_p = tp_tot / max(tp_tot + fp_tot, 1)
    micro_r = tp_tot / max(tp_tot + fn_tot, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)
    macro_f1 = float(np.mean([r["f1"] for r in per_class_rows if r["support"] > 0])) if active_mask.any() else 0.0
    hamming = float((preds_bin != gt_bin).mean())
    exact_match = float((preds_bin == gt_bin).all(axis=1).mean() * 100.0)

    # Optimal threshold search (micro-F1)
    best_thr, best_opt_f1 = threshold, micro_f1
    for thr_cand in np.linspace(0.1, 0.9, 17):
        pb = (scores_64 >= thr_cand).astype(np.int32)
        tp_c = int(((pb == 1) & (gt_bin == 1)).sum())
        fp_c = int(((pb == 1) & (gt_bin == 0)).sum())
        fn_c = int(((pb == 0) & (gt_bin == 1)).sum())
        f1_c = (2 * tp_c) / max(2 * tp_c + fp_c + fn_c, 1)
        if f1_c > best_opt_f1:
            best_opt_f1 = float(f1_c)
            best_thr = float(thr_cand)

    # Concurrency breakdown: clip-level F1 bucketed by # active GT atomic activities k = 1, 2, 3, 4+
    clip_gt_cnt = gt_bin.sum(axis=1)
    clip_tp = ((preds_bin == 1) & (gt_bin == 1)).sum(axis=1)
    clip_fp = ((preds_bin == 1) & (gt_bin == 0)).sum(axis=1)
    clip_fn = ((preds_bin == 0) & (gt_bin == 1)).sum(axis=1)
    clip_f1 = (2.0 * clip_tp) / np.maximum(2.0 * clip_tp + clip_fp + clip_fn, 1.0)

    def _bucket_f1(mask_b: np.ndarray) -> float:
        return float(clip_f1[mask_b].mean()) if mask_b.any() else float("nan")

    overall = {
        "mAP": map_all_64,
        "mAP_active_classes": map_active,
        "active_classes": int(active_mask.sum()),
        "total_clips": int(N),
        "avg_gt_pos_per_clip": float(gt_bin.sum(axis=1).mean()),
        "avg_pred_pos_per_clip": float(preds_bin.sum(axis=1).mean()),
        "mAP_c": map_c,
        "mAP_b": map_b,
        "mAP_p": map_p,
        "mAP_c_plus": map_c_plus,
        "mAP_b_plus": map_b_plus,
        "mAP_p_plus": map_p_plus,
        "mAP_single": map_single,
        "mAP_group": map_group,
        "mAP_near_ego": _mean_ap(origin_buckets["near_ego"]),
        "mAP_lateral": _mean_ap(origin_buckets["lateral"]),
        "mAP_far_opposite": _mean_ap(origin_buckets["far_opposite"]),
        "mAP_straight": _mean_ap(maneuver_buckets["straight"]),
        "mAP_right_turn": _mean_ap(maneuver_buckets["right_turn"]),
        "mAP_left_turn": _mean_ap(maneuver_buckets["left_turn"]),
        "mAP_crosswalk": _mean_ap(maneuver_buckets["crosswalk"]),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "best_micro_f1": float(best_opt_f1),
        "best_micro_f1_thr": float(best_thr),
        "exact_match_pct": float(exact_match),
        "hamming_loss": float(hamming),
        "ece": compute_binary_ece(scores_64, targets_64),
        "pos_prob_mean": float(scores_64[gt_bin == 1].mean()) if (gt_bin == 1).any() else float("nan"),
        "neg_prob_mean": float(scores_64[gt_bin == 0].mean()) if (gt_bin == 0).any() else float("nan"),
        "reverse_dir_conf_prob_mean": float(np.mean(rev_conf_list)) if rev_conf_list else float("nan"),
        "concurrency_f1_k1": _bucket_f1(clip_gt_cnt == 1),
        "concurrency_f1_k2": _bucket_f1(clip_gt_cnt == 2),
        "concurrency_f1_k3": _bucket_f1(clip_gt_cnt == 3),
        "concurrency_f1_k4plus": _bucket_f1(clip_gt_cnt >= 4),
    }
    return overall, per_class_rows, aps


# =============================================================================
# 5. VISUALIZATION SUITE (6 PNG CHARTS + HTML DASHBOARD)
# =============================================================================

def _plot_component_progression(comp_rows: List[Dict[str, object]], save_path: str) -> None:
    """Chart 1: Spatiotemporal component progression (FG/BG contrast, Motion energy, Decorrelation, Health)."""
    sp_rows = [r for r in comp_rows if np.isfinite(float(r.get("fg_bg_energy_ratio", np.nan)))]
    if not sp_rows:
        return

    names = [str(r["component"]).split(" (")[0] for r in sp_rows]
    x = np.arange(len(names))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))

    # (a) Spatial FG/BG Energy Ratio & FG Mass %
    ax = axes[0, 0]
    fg_bg = [float(r["fg_bg_energy_ratio"]) for r in sp_rows]
    fg_mass = [float(r["fg_mass_pct"]) for r in sp_rows]
    ax.plot(x, fg_bg, "o-", color="#1f77b4", lw=2.2, label="FG/BG Energy Ratio")
    ax.axhline(1.0, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax.set_ylabel("FG/BG Energy Ratio", color="#1f77b4", fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(x, fg_mass, "s--", color="#ff7f0e", lw=2.0, label="FG Mass %")
    ax2.set_ylabel("FG Mass (%)", color="#ff7f0e", fontweight="bold")
    ax.set_title("Spatial Focus: Road-Corridor FG vs Static BG", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=32, ha="right", fontsize=8.5)
    ax.grid(alpha=0.25)

    # (b) Temporal Motion Energy & Temporal FG/BG Motion Isolation
    ax = axes[0, 1]
    t_en = [float(r.get("temporal_energy_ratio", np.nan)) for r in sp_rows]
    t_fg = [float(r.get("temporal_fg_bg_ratio", np.nan)) for r in sp_rows]
    ax.plot(x, t_en, "o-", color="#2ca02c", lw=2.2, label="Temporal Energy Ratio")
    ax.set_ylabel("Temporal Energy Ratio", color="#2ca02c", fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(x, t_fg, "d--", color="#d62728", lw=2.0, label="Temporal FG/BG Motion Ratio")
    ax2.axhline(1.0, color="#d62728", ls=":", lw=1.0, alpha=0.6)
    ax2.set_ylabel("Temporal FG/BG Motion Ratio", color="#d62728", fontweight="bold")
    ax.set_title("Temporal Dynamics & Motion Isolation (Road vs BG)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=32, ha="right", fontsize=8.5)
    ax.grid(alpha=0.25)

    # (c) Spatial vs Temporal Decorrelation
    ax = axes[1, 0]
    s_dec = [float(r.get("spatial_decorr", np.nan)) for r in sp_rows]
    t_dec = [float(r.get("temporal_decorr", np.nan)) for r in sp_rows]
    ax.plot(x, s_dec, "o-", color="#9467bd", lw=2.2, label="Spatial Decorrelation (1 - cos)")
    ax.plot(x, t_dec, "s-", color="#17becf", lw=2.2, label="Temporal Decorrelation (t -> t+1)")
    ax.set_title("Spatial vs Temporal Feature Diversity", fontweight="bold")
    ax.set_ylabel("Cosine Distance (1 - cos)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=32, ha="right", fontsize=8.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    # (d) Activation L2 Norm & Dead/Negative Fraction across ALL components
    ax = axes[1, 1]
    all_names = [str(r["component"]).split(" (")[0] for r in comp_rows]
    x_all = np.arange(len(all_names))
    l2_vals = [float(r.get("l2_mean", np.nan)) for r in comp_rows]
    zero_vals = [float(r.get("zero_frac", np.nan)) * 100.0 for r in comp_rows]
    ax.bar(x_all, l2_vals, color="#4c78a8", alpha=0.75, label="L2 Norm Mean")
    ax.set_ylabel("L2 Norm Mean", color="#4c78a8", fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(x_all, zero_vals, "o-", color="#e45756", lw=1.8, label="Zero Frac (%)")
    ax2.set_ylabel("Zero Frac (%)", color="#e45756", fontweight="bold")
    ax.set_title("Activation Energy & Sparsity (All Components)", fontweight="bold")
    ax.set_xticks(x_all)
    ax.set_xticklabels(all_names, rotation=38, ha="right", fontsize=7.5)
    ax.grid(alpha=0.25)

    plt.suptitle("Action-Slot Component-by-Component Spatiotemporal Progression", fontsize=13.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_group_and_topology_map(
    overall: Dict[str, float],
    grad_ablation: Dict[str, float],
    save_path: str,
) -> None:
    """Chart 2: TACO 6-Group mAP, Topology/Maneuver breakdown, Concurrency, and Counterfactual Ablations."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))

    # (a) 6 Road-User Group mAPs (Paper Table 2)
    ax = axes[0, 0]
    grp_labels = ["C\n(Car)", "C+\n(Car Grp)", "K/B\n(Bike)", "K+/B+\n(Bike Grp)", "P\n(Ped)", "P+\n(Ped Grp)", "mAP\n(All 64)"]
    grp_vals = [
        overall["mAP_c"], overall["mAP_c_plus"],
        overall["mAP_b"], overall["mAP_b_plus"],
        overall["mAP_p"], overall["mAP_p_plus"],
        overall["mAP"],
    ]
    colors = ["#4c78a8", "#1f4e79", "#f58518", "#c25900", "#54a24b", "#2d6a2e", "#e45756"]
    bars = ax.bar(grp_labels, grp_vals, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, grp_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(0.2, max(grp_vals) * 1.18))
    ax.set_title("TACO Road-User Group mAP (Single vs Group)", fontweight="bold")
    ax.set_ylabel("mAP")
    ax.grid(axis="y", alpha=0.25)

    # (b) Topology & Maneuver Breakdown
    ax = axes[0, 1]
    topo_labels = ["Single\n(C,K,P)", "Group\n(C+,K+,P+)", "Near\n(Z1/C1/C4)", "Lateral\n(Z2/Z4)", "Far\n(Z3/C2/C3)", "Straight", "Right\nTurn", "Left\nTurn", "Crosswalk"]
    topo_vals = [
        overall["mAP_single"], overall["mAP_group"],
        overall["mAP_near_ego"], overall["mAP_lateral"], overall["mAP_far_opposite"],
        overall["mAP_straight"], overall["mAP_right_turn"], overall["mAP_left_turn"], overall["mAP_crosswalk"],
    ]
    t_cols = ["#72b7b2", "#2a7f7a", "#4c78a8", "#88aacc", "#c44e52", "#54a24b", "#eeca3b", "#f58518", "#b279a2"]
    bars = ax.bar(topo_labels, topo_vals, color=t_cols, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, topo_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylim(0, max(0.2, max(topo_vals) * 1.18))
    ax.set_title("Topology Distance & Maneuver Type mAP", fontweight="bold")
    ax.set_ylabel("mAP (Active Classes)")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.25)

    # (c) Concurrency Breakdown (Clip-level F1 vs # simultaneous GT activities)
    ax = axes[1, 0]
    k_labels = ["k = 1\nActivity", "k = 2\nActivities", "k = 3\nActivities", "k >= 4\nActivities"]
    k_vals = [
        overall.get("concurrency_f1_k1", np.nan),
        overall.get("concurrency_f1_k2", np.nan),
        overall.get("concurrency_f1_k3", np.nan),
        overall.get("concurrency_f1_k4plus", np.nan),
    ]
    k_plot = [v if np.isfinite(v) else 0.0 for v in k_vals]
    bars = ax.bar(k_labels, k_plot, color=["#54a24b", "#72b7b2", "#f58518", "#e45756"], edgecolor="black", linewidth=0.6)
    for b, raw_v in zip(bars, k_vals):
        txt = f"{raw_v:.3f}" if np.isfinite(raw_v) else "N/A"
        ax.text(b.get_x() + b.get_width() / 2, (raw_v if np.isfinite(raw_v) else 0.0) + 0.015, txt, ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, max(0.25, max(k_plot) * 1.2))
    ax.set_title("Multi-Label Concurrency Robustness (Clip F1 vs # Active Actions)", fontweight="bold")
    ax.set_ylabel("Mean Clip F1")
    ax.grid(axis="y", alpha=0.25)

    # (d) Zero-Retrain Counterfactual Ablations
    ax = axes[1, 1]
    ab_labels = ["Normal\nAction-Slot"]
    ab_vals = [grad_ablation.get("ablation_normal_mAP", overall["mAP"])]
    ab_cols = ["#1f77b4"]
    if np.isfinite(grad_ablation.get("ablation_uniform_attn_mAP", np.nan)):
        ab_labels.append("Uniform Attn\n(No Slot Loc.)")
        ab_vals.append(grad_ablation["ablation_uniform_attn_mAP"])
        ab_cols.append("#ff7f0e")
    if np.isfinite(grad_ablation.get("ablation_time_reversed_mAP", np.nan)):
        ab_labels.append("Time-Reversed\n(Motion Flip)")
        ab_vals.append(grad_ablation["ablation_time_reversed_mAP"])
        ab_cols.append("#d62728")
    bars = ax.bar(ab_labels, ab_vals, color=ab_cols, edgecolor="black", linewidth=0.6, width=0.52)
    for b, v in zip(bars, ab_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, max(0.2, max(ab_vals) * 1.22))
    ax.set_title("Zero-Retrain Counterfactual Ablations (mAP)", fontweight="bold")
    ax.set_ylabel("mAP (64 Classes)")
    ax.grid(axis="y", alpha=0.25)

    plt.suptitle("TACO Multi-Label Atomic Activity Breakdown & Ablations", fontsize=13.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_per_class_ap_heatmap(aps_64: np.ndarray, per_class_rows: List[Dict[str, Any]], save_path: str) -> None:
    """Chart 3: Structured 64-class TACO AP Heatmap mirroring Paper Table 10."""
    support_64 = np.array([int(r["support"]) for r in per_class_rows])

    veh_trajs = ["Z1-Z2", "Z1-Z3", "Z1-Z4", "Z2-Z1", "Z2-Z3", "Z2-Z4", "Z3-Z1", "Z3-Z2", "Z3-Z4", "Z4-Z1", "Z4-Z2", "Z4-Z3"]
    ped_trajs = ["C1-C2", "C1-C4", "C2-C1", "C2-C3", "C3-C2", "C3-C4", "C4-C1", "C4-C3"]

    veh_mat = np.zeros((4, 12), dtype=np.float64)
    veh_sup = np.zeros((4, 12), dtype=np.int64)
    for r_idx, gkey in enumerate(["c", "c+", "b", "b+"]):
        idxs = ROAD_USER_GROUPS[gkey]
        veh_mat[r_idx] = aps_64[idxs]
        veh_sup[r_idx] = support_64[idxs]

    ped_mat = np.zeros((2, 8), dtype=np.float64)
    ped_sup = np.zeros((2, 8), dtype=np.int64)
    for r_idx, gkey in enumerate(["p", "p+"]):
        idxs = ROAD_USER_GROUPS[gkey]
        ped_mat[r_idx] = aps_64[idxs]
        ped_sup[r_idx] = support_64[idxs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7.5), gridspec_kw={"height_ratios": [4, 2.2]})

    im1 = ax1.imshow(veh_mat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(veh_trajs, fontsize=9.5, fontweight="bold")
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(["Car (C)", "Car Group (C+)", "Two-Wheeler (K/B)", "Two-Wheeler Grp (K+/B+)"], fontsize=9.5, fontweight="bold")
    ax1.set_title("Vehicle & Two-Wheeler Atomic Activities (48 Classes: AP / [support])", fontweight="bold")
    for r in range(4):
        for c in range(12):
            v = veh_mat[r, c]
            s = veh_sup[r, c]
            txt = f"{v:.2f}\n(n={s})" if s > 0 else "-\n(n=0)"
            ax1.text(c, r, txt, ha="center", va="center", fontsize=8, color="white" if v > 0.55 else "black")
    fig.colorbar(im1, ax=ax1, fraction=0.02, pad=0.02, label="AP")

    im2 = ax2.imshow(ped_mat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax2.set_xticks(range(8))
    ax2.set_xticklabels(ped_trajs, fontsize=9.5, fontweight="bold")
    ax2.set_yticks(range(2))
    ax2.set_yticklabels(["Pedestrian (P)", "Pedestrian Grp (P+)"], fontsize=9.5, fontweight="bold")
    ax2.set_title("Pedestrian Crosswalk Atomic Activities (16 Classes: AP / [support])", fontweight="bold")
    for r in range(2):
        for c in range(8):
            v = ped_mat[r, c]
            s = ped_sup[r, c]
            txt = f"{v:.2f}\n(n={s})" if s > 0 else "-\n(n=0)"
            ax2.text(c, r, txt, ha="center", va="center", fontsize=8.5, color="white" if v > 0.55 else "black")
    fig.colorbar(im2, ax=ax2, fraction=0.02, pad=0.02, label="AP")

    plt.suptitle("Per-Class TACO Atomic Activity AP Grid (Paper Table 10 Format)", fontsize=13.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_slot_attention_diagnostics(
    slot_summary: Dict[str, float],
    per_slot_rows: List[Dict[str, Any]],
    ego_conf_mat: np.ndarray,
    grad_ablation: Dict[str, float],
    save_path: str,
) -> None:
    """Chart 4: Slot Attention Pos vs Neg contrast, FG Token Claim, Fisher vs AP scatter, Ego & Gradients."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))

    # (a) Per-Slot Positive vs Negative Mean Attention
    ax = axes[0, 0]
    s_ids = [int(r["slot_id"]) for r in per_slot_rows]
    pos_m = [float(r["pos_attn_mean"]) if np.isfinite(float(r.get("pos_attn_mean", np.nan))) else 0.0 for r in per_slot_rows]
    neg_m = [float(r["neg_attn_mean"]) if np.isfinite(float(r.get("neg_attn_mean", np.nan))) else 0.0 for r in per_slot_rows]
    w = 0.42
    x = np.arange(len(s_ids))
    ax.bar(x - w / 2, pos_m, width=w, color="#2ca02c", label="Positive Clips (Active)")
    ax.bar(x + w / 2, neg_m, width=w, color="#d62728", alpha=0.85, label="Negative Clips (L_neg target=0)")
    ax.set_title("Per-Slot Mean Attention: Active vs Absent Clips", fontweight="bold")
    ax.set_xlabel("Slot ID")
    ax.set_ylabel("Mean Attention Weight (attn_ori)")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)

    # (b) Slot Representation Separability (Fisher) vs Presence AP
    ax = axes[0, 1]
    fish_vals = [float(r["slot_pos_neg_fisher"]) for r in per_slot_rows if np.isfinite(float(r.get("slot_pos_neg_fisher", np.nan))) and np.isfinite(float(r.get("presence_ap", np.nan)))]
    ap_vals = [float(r["presence_ap"]) for r in per_slot_rows if np.isfinite(float(r.get("slot_pos_neg_fisher", np.nan))) and np.isfinite(float(r.get("presence_ap", np.nan)))]
    grp_cols_map = {"car": "#1f77b4", "bike": "#ff7f0e", "ped": "#2ca02c"}
    pt_cols = [grp_cols_map.get(str(r["road_user_group"]), "#555") for r in per_slot_rows if np.isfinite(float(r.get("slot_pos_neg_fisher", np.nan))) and np.isfinite(float(r.get("presence_ap", np.nan)))]
    if fish_vals:
        ax.scatter(fish_vals, ap_vals, c=pt_cols, s=55, alpha=0.85, edgecolors="black", linewidth=0.5)
        for r in per_slot_rows:
            fv = float(r.get("slot_pos_neg_fisher", np.nan))
            av = float(r.get("presence_ap", np.nan))
            if np.isfinite(fv) and np.isfinite(av):
                ax.annotate(str(r["slot_id"]), (fv, av), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set_title("Slot Vector Fisher Ratio (Pos vs Neg) vs Presence AP", fontweight="bold")
    ax.set_xlabel("Slot Vector Fisher Ratio (Higher = Cleaner Pos/Neg Separation)")
    ax.set_ylabel("Slot Presence AP")
    ax.grid(alpha=0.25)

    # (c) Multi-Task Gradient Cosine Similarities & FG Token Claim
    ax = axes[1, 0]
    g_keys = [
        ("Actor vs Attn\n(conv3d)", "grad_cos_actor_vs_attn_conv3d"),
        ("L_neg vs L_bg\n(conv3d)", "grad_cos_neg_vs_bg_conv3d"),
        ("Actor vs Attn\n(Backbone)", "grad_cos_actor_vs_attn_backbone"),
        ("Actor vs Ego\n(Backbone)", "grad_cos_actor_vs_ego_backbone"),
    ]
    g_lbls = [k[0] for k in g_keys]
    g_vals = [grad_ablation.get(k[1], np.nan) for k in g_keys]
    g_plot = [v if np.isfinite(v) else 0.0 for v in g_vals]
    g_cols = ["#2ca02c" if v >= 0 else "#d62728" for v in g_plot]
    bars = ax.bar(g_lbls, g_plot, color=g_cols, edgecolor="black", linewidth=0.6, width=0.55)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.axhline(-0.1, color="#d62728", ls="--", lw=0.9, alpha=0.6, label="Conflict Threshold (-0.1)")
    for b, raw_v in zip(bars, g_vals):
        txt = f"{raw_v:+.3f}" if np.isfinite(raw_v) else "N/A"
        y_off = 0.03 if (np.isfinite(raw_v) and raw_v >= 0) else -0.07
        ax.text(b.get_x() + b.get_width() / 2, (raw_v if np.isfinite(raw_v) else 0.0) + y_off, txt, ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Multi-Task Gradient Cosine Similarity (Harmony vs Conflict)", fontweight="bold")
    ax.set_ylabel("Cosine Similarity")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # (d) Ego-Vehicle Confusion Matrix (4x4 row-normalized %)
    ax = axes[1, 1]
    if ego_conf_mat.sum() > 0:
        row_s = np.maximum(ego_conf_mat.sum(axis=1, keepdims=True), 1.0)
        pct = ego_conf_mat / row_s * 100.0
        im = ax.imshow(pct, cmap="Blues", vmin=0.0, vmax=100.0)
        short_ego = ["Z1-Z1\n(Wait)", "Z1-Z2\n(Right)", "Z1-Z3\n(Str.)", "Z1-Z4\n(Left)"][:ego_conf_mat.shape[0]]
        ax.set_xticks(range(len(short_ego)))
        ax.set_yticks(range(len(short_ego)))
        ax.set_xticklabels(short_ego, fontsize=9)
        ax.set_yticklabels(short_ego, fontsize=9)
        ax.set_xlabel("Predicted Ego Action")
        ax.set_ylabel("GT Ego Action")
        for r in range(len(short_ego)):
            for c in range(len(short_ego)):
                v = pct[r, c]
                ax.text(c, r, f"{v:.1f}%\n({int(ego_conf_mat[r, c])})", ha="center", va="center",
                        fontsize=8, color="white" if v > 55 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of GT")
        ax.set_title(f"Ego-Vehicle Confusion Matrix (Acc={grad_ablation.get('ego_acc_pct', 0):.1f}%)", fontweight="bold")
    else:
        ax.axis("off")

    plt.suptitle("Slot Attention Competition, Separability, Gradient Harmony & Ego Branch", fontsize=13.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_visual_spatiotemporal_inspection(vis_sample: Dict[str, Any], save_path: str) -> None:
    """Chart 5: 4-Keyframe Spatiotemporal Visual Inspection (RGB, BG Mask vs BG Slot, Backbone PCA, Pos & Neg Slots)."""
    frames = vis_sample["frames_rgb"]     # (T, H, W, 3) uint8
    bg_seg = vis_sample["bg_seg"]         # (4, Hm, Wm) float
    attn_3d = vis_sample["attn_ori_3d"]   # (S_all, T, h, w) float
    bb4_pca = vis_sample["bb4_pca"]       # (T, h, w, 3) float
    pos_slots = vis_sample["pos_slots"]   # list of (slot_id, slot_name, score)
    neg_slot = vis_sample["top_neg_slot"] # (slot_id, slot_name, max_attn)
    has_bg = vis_sample["has_bg_slot"]

    T, H, W, _ = frames.shape
    t_idxs = [0, T // 4, T // 2, (3 * T) // 4]
    n_rows = 4 + min(2, len(pos_slots)) + (1 if neg_slot is not None else 0)

    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 2.7 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for col, ti in enumerate(t_idxs):
        rgb = frames[ti]
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"RGB Frame t={ti}", fontsize=10, fontweight="bold")
        axes[0, col].axis("off")

        # Row 1: GT Background Mask (overlay)
        m_idx = min(col, bg_seg.shape[0] - 1)
        gt_bg_up = cv2.resize(bg_seg[m_idx].astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        axes[1, col].imshow(rgb, alpha=0.55)
        axes[1, col].imshow(gt_bg_up, cmap="Purples", alpha=0.55, vmin=0.0, vmax=1.0)
        axes[1, col].set_title(f"GT BG Mask M_bg (t={ti})" if col == 0 else f"t={ti}", fontsize=9.5)
        axes[1, col].axis("off")

        # Row 2: Predicted BG Slot Attention (or Backbone L2 if no BG slot)
        if has_bg:
            bg_a = cv2.resize(attn_3d[-1, ti].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            axes[2, col].imshow(rgb, alpha=0.55)
            axes[2, col].imshow(bg_a, cmap="magma", alpha=0.60, vmin=0.0, vmax=1.0)
            axes[2, col].set_title(f"Pred BG Slot Attn (t={ti})" if col == 0 else f"mean={bg_a.mean():.2f}", fontsize=9.5)
        else:
            axes[2, col].imshow(rgb)
        axes[2, col].axis("off")

        # Row 3: Backbone block4 PCA-RGB
        pca_up = cv2.resize(bb4_pca[ti].astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        axes[3, col].imshow(pca_up)
        axes[3, col].set_title("Backbone block4 PCA-RGB" if col == 0 else f"t={ti}", fontsize=9.5)
        axes[3, col].axis("off")

        # Row 4..: Top-2 Positive Action Slots
        row_cursor = 4
        for p_idx in range(min(2, len(pos_slots))):
            s_id, s_name, s_prob = pos_slots[p_idx]
            sa_map = cv2.resize(attn_3d[s_id, ti].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            axes[row_cursor, col].imshow(rgb, alpha=0.55)
            axes[row_cursor, col].imshow(sa_map, cmap="jet", alpha=0.60, vmin=0.0, vmax=max(0.4, float(attn_3d[s_id].max())))
            if col == 0:
                axes[row_cursor, col].set_title(f"POS Slot {s_id}: {s_name} (p={s_prob:.2f})", fontsize=9, fontweight="bold", color="#0b6623")
            else:
                axes[row_cursor, col].set_title(f"max={sa_map.max():.2f}", fontsize=9)
            axes[row_cursor, col].axis("off")
            row_cursor += 1

        # Last row: Top-1 Negative Slot (worst false-alarm attention)
        if neg_slot is not None:
            ns_id, ns_name, _ = neg_slot
            na_map = cv2.resize(attn_3d[ns_id, ti].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            axes[row_cursor, col].imshow(rgb, alpha=0.55)
            axes[row_cursor, col].imshow(na_map, cmap="hot", alpha=0.60, vmin=0.0, vmax=max(0.25, float(attn_3d[ns_id].max())))
            if col == 0:
                axes[row_cursor, col].set_title(f"NEG Slot {ns_id} (worst FA): {ns_name}", fontsize=9, fontweight="bold", color="#a30000")
            else:
                axes[row_cursor, col].set_title(f"max={na_map.max():.2f}", fontsize=9)
            axes[row_cursor, col].axis("off")

    gt_txt = ", ".join(vis_sample["gt_active_names"][:6])
    plt.suptitle(
        f"Spatiotemporal Dissection | {vis_sample['scenario_key']} | Ego={vis_sample['ego_gt_name']} | GT=[{gt_txt}]",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


def _plot_slot_similarity_and_single_group(
    q_sim: np.ndarray,
    per_slot_rows: List[Dict[str, Any]],
    is_reduced_head: bool,
    save_path: str,
) -> None:
    """Chart 6: Query Cosine Similarity Matrix + Single-vs-Group Disambiguation across 32 Paired Slots."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5))

    # (a) Query Cosine Similarity Matrix (detects slot query collapse / trajectory correlation)
    ax = axes[0]
    im = ax.imshow(q_sim, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_title("Learned Action Query Cosine Similarity Matrix (q_i . q_j)", fontweight="bold")
    ax.set_xlabel("Action Slot Index")
    ax.set_ylabel("Action Slot Index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine Similarity")

    # (b) Single vs Group AP & Cardinality Disambiguation (for ReduceAllocatedHead)
    ax = axes[1]
    if is_reduced_head and len(per_slot_rows) == 32:
        s_names = [str(r["slot_name"]) for r in per_slot_rows]
        pres_ap = [float(r["presence_ap"]) if np.isfinite(float(r.get("presence_ap", np.nan))) else 0.0 for r in per_slot_rows]
        sg_acc = [float(r["single_vs_group_acc_pct"]) / 100.0 if np.isfinite(float(r.get("single_vs_group_acc_pct", np.nan))) else 0.0 for r in per_slot_rows]
        y = np.arange(len(s_names))
        ax.barh(y - 0.18, pres_ap, height=0.36, color="#1f77b4", label="Presence AP (Single OR Group)")
        ax.barh(y + 0.18, sg_acc, height=0.36, color="#2ca02c", alpha=0.85, label="Single-vs-Group Acc (when Present)")
        ax.set_yticks(y)
        ax.set_yticklabels(s_names, fontsize=7.2)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Score [0, 1]")
        ax.set_title("32 Paired Slots: Trajectory Presence AP vs Single/Group Acc", fontweight="bold")
        ax.legend(fontsize=8.5, loc="lower right")
        ax.grid(axis="x", alpha=0.25)
    else:
        s_ids = [int(r["slot_id"]) for r in per_slot_rows]
        q_norms = [float(r["q_norm"]) for r in per_slot_rows]
        ax.bar(s_ids, q_norms, color="#4c78a8")
        ax.set_title("Action Slot Query L2 Norm ||q_s||_2", fontweight="bold")
        ax.set_xlabel("Slot ID")
        ax.set_ylabel("||q_s||_2")
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _generate_action_slot_dashboard(
    out_dir: str,
    overall: Dict[str, float],
    comp_rows: List[Dict[str, object]],
    slot_summary: Dict[str, float],
    grad_ablation: Dict[str, float],
    per_slot_rows: List[Dict[str, Any]],
) -> None:
    """Generate interactive HTML dashboard mirroring VisDrone's dashboard style."""
    comp_headers = "".join(f"<th>{html.escape(k)}</th>" for k in comp_rows[0].keys())
    comp_body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r.values()) + "</tr>"
        for r in comp_rows
    )

    slot_items = "".join(
        f"<div class='kpi'><div class='lbl'>{html.escape(k)}</div><div class='val'>{v:.4f}</div></div>"
        for k, v in slot_summary.items()
        if isinstance(v, (int, float)) and np.isfinite(v)
    )
    grad_items = "".join(
        f"<div class='kpi'><div class='lbl'>{html.escape(k)}</div><div class='val'>{v:.4f}</div></div>"
        for k, v in grad_ablation.items()
        if isinstance(v, (int, float)) and np.isfinite(v)
    )

    ps_headers = "".join(f"<th>{html.escape(k)}</th>" for k in per_slot_rows[0].keys()) if per_slot_rows else ""
    ps_body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r.values()) + "</tr>"
        for r in per_slot_rows
    )

    chart_files = [
        ("1. Component-by-Component Spatiotemporal Progression", "chart_01_component_progression.png"),
        ("2. TACO 6-Group, Topology, Concurrency & Ablation Summary", "chart_02_group_and_topology_map.png"),
        ("3. Per-Class TACO Atomic Activity AP Grid (Paper Table 10)", "chart_03_per_class_ap_heatmap.png"),
        ("4. Slot Attention Competition, Separability, Gradients & Ego Matrix", "chart_04_slot_attention_diagnostics.png"),
        ("5. Visual Spatiotemporal Dissection (RGB, BG Slot, Backbone PCA, Pos/Neg Slots)", "chart_05_visual_spatiotemporal_inspection.png"),
        ("6. Query Similarity & Paired Single-vs-Group Disambiguation", "chart_06_slot_similarity_and_single_group.png"),
    ]
    charts_html = "".join(
        f"<div class='card'><h3>{html.escape(title)}</h3><img src='{fname}' style='max-width:100%;'></div>"
        for title, fname in chart_files
        if os.path.isfile(os.path.join(out_dir, fname))
    )

    html_str = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Action-Slot Component Validation Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; background: #f5f7fa; color: #222; }}
  h1, h2 {{ color: #1f3a60; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .kpi {{ background: #fff; padding: 14px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .kpi .lbl {{ font-size: 11px; color: #666; text-transform: uppercase; }}
  .kpi .val {{ font-size: 20px; font-weight: 700; color: #1f3a60; margin-top: 4px; }}
  .card {{ background: #fff; padding: 18px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 24px; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
  th, td {{ border: 1px solid #e1e5eb; padding: 6px 9px; text-align: right; }}
  th {{ background: #1f3a60; color: #fff; text-align: center; }}
  td:first-child, td:nth-child(2) {{ text-align: left; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f9fbfd; }}
</style>
</head>
<body>
  <h1>Action-Slot Component-by-Component Validation (TACO)</h1>
  <div class="kpi-grid">
    <div class="kpi"><div class="lbl">Actor mAP (64 cls)</div><div class="val">{overall['mAP']:.4f}</div></div>
    <div class="kpi"><div class="lbl">mAP C / K(B) / P</div><div class="val">{overall['mAP_c']:.3f} / {overall['mAP_b']:.3f} / {overall['mAP_p']:.3f}</div></div>
    <div class="kpi"><div class="lbl">mAP C+ / K+(B+) / P+</div><div class="val">{overall['mAP_c_plus']:.3f} / {overall['mAP_b_plus']:.3f} / {overall['mAP_p_plus']:.3f}</div></div>
    <div class="kpi"><div class="lbl">Micro F1 / Macro F1</div><div class="val">{overall['micro_f1']:.3f} / {overall['macro_f1']:.3f}</div></div>
    <div class="kpi"><div class="lbl">Ego Acc / Macro F1</div><div class="val">{grad_ablation.get('ego_acc_pct', 0):.1f}% / {grad_ablation.get('ego_macro_f1', 0):.3f}</div></div>
    <div class="kpi"><div class="lbl">BG Slot IoU / Dice</div><div class="val">{slot_summary.get('bg_slot_iou', 0):.3f} / {slot_summary.get('bg_slot_dice', 0):.3f}</div></div>
    <div class="kpi"><div class="lbl">Pos/Neg Attn Ratio</div><div class="val">{slot_summary.get('pos_neg_attn_ratio', 0):.2f}x</div></div>
    <div class="kpi"><div class="lbl">Uniform-Attn &Delta;mAP</div><div class="val">{grad_ablation.get('ablation_uniform_attn_delta_mAP', 0):+.4f}</div></div>
    <div class="kpi"><div class="lbl">Time-Reverse &Delta;mAP</div><div class="val">{grad_ablation.get('ablation_time_reversed_delta_mAP', 0):+.4f}</div></div>
    <div class="kpi"><div class="lbl">Latency / Clip FPS</div><div class="val">{overall['latency_ms']:.1f} ms ({overall['frames_per_sec']:.1f} FPS)</div></div>
    <div class="kpi"><div class="lbl">Total Params</div><div class="val">{overall['params_total_m']:.2f}M</div></div>
  </div>
  <h2>Slot Attention Diagnostics (L_bg, L_neg, Competition, Geometry)</h2>
  <div class="kpi-grid">{slot_items}</div>
  <h2>Spatiotemporal Gradients, Ego &amp; Counterfactual Ablations</h2>
  <div class="kpi-grid">{grad_items}</div>
  <div class="card">
    <h2>Component-by-Component Breakdown</h2>
    <table><thead><tr>{comp_headers}</tr></thead><tbody>{comp_body}</tbody></table>
  </div>
  <div class="card">
    <h2>Per-Slot Diagnostics</h2>
    <table><thead><tr>{ps_headers}</tr></thead><tbody>{ps_body}</tbody></table>
  </div>
  {charts_html}
</body>
</html>"""
    with open(os.path.join(out_dir, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html_str)


# =============================================================================
# 6. MAIN VALIDATION PIPELINE
# =============================================================================

@torch.no_grad()
def run_action_slot_component_validation(
    weights_path: Optional[str] = None,
    config_path: Optional[str] = "action_slot/config.yaml",
    data_roots: Optional[List[str]] = None,
    out_dir: str = "action_slot/output_val_components",
    split: str = "val",
    max_videos: int = 0,
    threshold: float = 0.5,
    attn_thresh: float = 0.2,
    grad_samples: int = 20,
    skip_time_reverse: bool = False,
    seed: Optional[int] = None,
    device_str: str = "auto",
) -> None:
    cfg: Dict[str, Any] = {}
    model_cfg: Dict[str, Any] = {}
    ds_cfg: Dict[str, Any] = {}

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        model_cfg = cfg.get("model", {}) or {}
        ds_cfg = cfg.get("data", {}).get(split, {}).get("dataset", {}) or {}
        print(f"[Config] Loaded {config_path} (section: data.{split}.dataset)")

    resolved_roots = data_roots if data_roots else ds_cfg.get("data_root", ["D:/apps/Datasets/TACO/data"])
    if isinstance(resolved_roots, str):
        resolved_roots = [resolved_roots]
    resolved_seq_len = int(ds_cfg.get("sequence_len", 16))
    resolved_seed = seed if seed is not None else int(ds_cfg.get("seed", cfg.get("seed", 42)))

    os.makedirs(out_dir, exist_ok=True)
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 84)
    print(f"Action-Slot Component-by-Component Validation | Split: {split} | Device: {device} | T={resolved_seq_len}")
    print("=" * 84)

    model, is_reduced_head, has_bg_slot = load_action_slot_model(
        weights_path=weights_path,
        device=device,
        model_cfg=model_cfg,
        seq_len=resolved_seq_len,
    )
    probe = ActionSlotComponentProbe(model)
    S_act = model.num_slots

    # Parameter accounting per component
    sa = model.slot_attention
    comp_modules: Dict[str, Optional[nn.Module]] = {
        "backbone.block0 (Stem //2)": model.backbone.blocks[0],
        "backbone.block1 (ResStage1 //4)": model.backbone.blocks[1],
        "backbone.block2 (ResStage2 //8)": model.backbone.blocks[2],
        "backbone.block3 (ResStage3 //16)": model.backbone.blocks[3],
        "backbone.block4 (ResStage4 //32)": model.backbone.blocks[4],
        "proj.conv3d (192->256)": model.conv3d,
        "proj.conv3d_ego (192->128)": model.conv3d_ego,
        "slot.pe_3d (SoftPosEmbed3D)": sa.pe,
        "slot.token_mlp (FC1->FC2)": nn.ModuleList([sa.LN, sa.FC1, sa.FC2]),
        "slot.keys (to_k)": sa.to_k,
        "slot.values (to_v)": sa.to_v,
        "slot.slots_pre_ffn (Attn Out)": nn.ModuleList([sa.norm_input, sa.norm_slots, sa.to_q]),
        "slot.slots_post_ffn (FFN Out)": nn.ModuleList([sa.norm_pre_ff, sa.fc1, sa.fc2]),
        "head.actor (Logits)": model.head.fc_actor,
        "head.ego (Logits)": getattr(model.head, "fc_ego", None),
    }
    comp_params = {
        k: (sum(p.numel() for p in m.parameters()) if m is not None else 0)
        for k, m in comp_modules.items()
    }
    p_total = sum(p.numel() for p in model.parameters()) / 1e6
    p_backbone = sum(p.numel() for p in model.backbone.parameters()) / 1e6
    p_proj_ego = (
        sum(p.numel() for p in model.conv3d.parameters())
        + (sum(p.numel() for p in model.conv3d_ego.parameters()) if model.conv3d_ego is not None else 0)
    ) / 1e6
    p_slot = sum(p.numel() for p in model.slot_attention.parameters()) / 1e6
    p_head = sum(p.numel() for p in model.head.parameters()) / 1e6

    # Load dataset
    ds = ACTION_SLOT_LMDBDataset(
        data_root=resolved_roots,
        split=split,
        sequence_len=resolved_seq_len,
        seed=resolved_seed,
        pipeline=[{"type": "Action_Slot_Transform"}],
        num_samples=max_videos if max_videos > 0 else None,
        test_mode=True,
    )
    num_samples = len(ds)
    if num_samples == 0:
        raise RuntimeError(f"No valid video clips found in data_roots={resolved_roots} for split='{split}'.")
    print(f"[Dataset] Loaded {num_samples} clips from {resolved_roots}")

    # Accumulators
    comp_metric_keys = [
        "act_mean", "act_std", "neg_frac", "zero_frac", "l2_mean",
        "fg_bg_energy_ratio", "fg_mass_pct", "temporal_energy_ratio",
        "temporal_fg_bg_ratio", "spatial_decorr", "temporal_decorr",
    ]
    comp_acc: Dict[str, Dict[str, List[float]]] = {c: {k: [] for k in comp_metric_keys} for c in COMPONENT_ORDER}

    slot_metric_keys = [
        "dots_std", "dots_abs_max", "dots_margin_fg", "dots_margin_pos_vs_bg",
        "attn_entropy_fg", "attn_entropy_bg",
        "bg_slot_iou", "bg_slot_dice", "bg_slot_bce",
        "bg_slot_on_bg_mean", "bg_slot_on_fg_mean", "bg_slot_leak_fg_pct", "bg_slot_fg_bg_logit_gap",
        "pos_slot_attn_mean", "pos_slot_attn_max", "pos_slot_active_area_gt02_pct",
        "pos_slot_fg_focus_pct", "pos_slot_temporal_span_pct", "pos_slot_temporal_entropy",
        "pos_slot_spatial_entropy", "fg_claimed_by_pos_pct", "pos_cooccur_attn_overlap",
        "neg_slot_attn_mean", "neg_slot_attn_max", "neg_slot_fa_gt02_pct",
        "neg_slot_win_fg_pct", "neg_slot_bce", "pos_neg_attn_ratio",
        "slot_cos_sep", "slot_ffn_delta_ratio", "slot_pos_neg_l2_ratio",
        "pos_slot_vs_global_v_cos", "neg_slot_vs_global_v_cos",
        "pe2d_to_feat_ratio", "pe3d_to_feat_ratio",
    ]
    slot_acc: Dict[str, List[float]] = {k: [] for k in slot_metric_keys}

    per_slot_acc: Dict[int, Dict[str, List[float]]] = {
        s: {
            "pos_attn_mean": [], "pos_attn_max": [], "pos_area_gt02_pct": [],
            "neg_attn_mean": [], "neg_fa_gt02_pct": [],
            "pos_fg_focus_pct": [], "pos_temporal_entropy": [],
            "dot_pos_peak": [], "gap_vs_bg_peak": [],
            "pos_vs_global_v_cos": [], "neg_vs_global_v_cos": [],
        }
        for s in range(S_act)
    }

    grad_acc: Dict[str, List[float]] = {
        "grad_cos_actor_vs_attn_conv3d": [], "grad_ratio_actor_over_attn_conv3d": [],
        "grad_cos_neg_vs_bg_conv3d": [], "grad_ratio_neg_over_bg_conv3d": [],
        "grad_cos_actor_vs_attn_backbone": [], "grad_ratio_actor_over_attn_backbone": [],
        "grad_cos_actor_vs_ego_backbone": [], "grad_ratio_actor_over_ego_backbone": [],
    }
    loss_acc: Dict[str, List[float]] = {"loss_actor": [], "loss_ego": [], "attn_loss": [], "bg_attn_loss": []}

    all_scores_64: List[np.ndarray] = []
    all_targets_64: List[np.ndarray] = []
    all_scores_unif_64: List[np.ndarray] = []
    all_scores_rev_64: List[np.ndarray] = []

    all_slot_pres_scores: List[np.ndarray] = []   # (N, S_act)
    all_slot_pres_targets: List[np.ndarray] = []  # (N, S_act)
    all_slot_raw_3way: List[np.ndarray] = []      # (N, 32, 3) when is_reduced_head
    all_slot_3way_targets: List[np.ndarray] = []  # (N, S_act)
    all_slot_vecs: List[np.ndarray] = []          # (N, S_act, D)

    all_ego_preds: List[int] = []
    all_ego_preds_rev: List[int] = []
    all_ego_gts: List[int] = []
    all_ego_feats: List[np.ndarray] = []

    latencies_ms: List[float] = []
    vis_sample: Optional[Dict[str, Any]] = None
    n_grad_done = 0

    t0_all = time.perf_counter()
    for idx in range(num_samples):
        item = ds[idx]
        vid_cpu: torch.Tensor = item["inputs"]      # (T, 3, H, W) uint8
        tgt: Dict[str, Any] = item["targets"]
        bg_seg_cpu: torch.Tensor = tgt["bg_seg"]    # (4, 1, 32, 96) float
        actor_gt_64 = tgt["actor"].float().numpy()  # (64,)
        ego_gt = int(tgt["ego"].item())
        scen_key = str(tgt.get("path", {}).get("scenario_key", f"clip_{idx:04d}"))

        vid_t = vid_cpu.unsqueeze(0).to(device, non_blocking=True)

        probe.clear()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        pred_out = model.predict(vid_t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t_start) * 1000.0)

        # Predictions (64-class probabilities)
        actor_logits_64 = pred_out["actor"][0].float()
        probs_64 = torch.sigmoid(actor_logits_64).cpu().numpy()
        all_scores_64.append(probs_64)
        all_targets_64.append(actor_gt_64)

        # Ego prediction & feature vector
        if pred_out.get("ego") is not None:
            ego_logits = pred_out["ego"][0].float()
            all_ego_preds.append(int(ego_logits.argmax().item()))
            all_ego_gts.append(ego_gt)
            if "ego_pooled" in probe.sa_cache:
                all_ego_feats.append(probe.sa_cache["ego_pooled"][0].float().reshape(-1).cpu().numpy())

        # Validation loss terms on this clip
        raw_pred_for_loss = {
            "actor": probe.sa_cache["pred_actor_raw"],
            "ego": probe.sa_cache.get("pred_ego_raw"),
            "attn": pred_out["attn"],
        }
        tgt_dev = {
            "ego": tgt["ego"].to(device).long(),
            "actor": tgt["actor"].to(device).float(),
            "bg_seg": bg_seg_cpu.to(device).float(),
        }
        if is_reduced_head:
            l_dict = model._compute_losses(raw_pred_for_loss, [tgt_dev], validate=False)
        else:
            l_dict = {
                "loss_actor": F.binary_cross_entropy_with_logits(probe.sa_cache["pred_actor_raw"], tgt_dev["actor"].unsqueeze(0)),
                "loss_ego": 0.5 * model.loss_fn.ego_loss(probe.sa_cache.get("pred_ego_raw"), tgt_dev["ego"].unsqueeze(0)) if probe.sa_cache.get("pred_ego_raw") is not None else None,
            }
            l_dict.update(model.loss_fn.attn_loss(pred_out["attn"], {"actor": tgt_dev["actor"].unsqueeze(0), "bg_seg": tgt_dev["bg_seg"].unsqueeze(0)}, validate=False))
        for lk in loss_acc:
            lv = l_dict.get(lk)
            if lv is not None and torch.is_tensor(lv):
                loss_acc[lk].append(float(lv.detach().item()))

        # Grid resolution (T_f, H_f, W_f) at backbone output
        bb4 = probe.features["backbone.block4 (ResStage4 //32)"]
        grid_thw = (int(bb4.shape[2]), int(bb4.shape[3]), int(bb4.shape[4]))

        # Component-by-component metrics
        probe.features["slot.keys (to_k)"] = probe.sa_cache["k"]
        probe.features["slot.values (to_v)"] = probe.sa_cache["v"]
        for cname in COMPONENT_ORDER:
            feat = probe.features.get(cname)
            if feat is None:
                continue
            cm = measure_spatiotemporal_component_metrics(feat, bg_seg_cpu, grid_thw)
            for mk, mv in cm.items():
                comp_acc[cname][mk].append(mv)

        # Slot Attention clip diagnostics
        s_clip, ps_upd, vis_ex = measure_slot_attention_clip_metrics(
            model=model,
            sa_cache=probe.sa_cache,
            conv3d_feat=probe.features["proj.conv3d (192->256)"],
            actor_gt_64=actor_gt_64,
            bg_seg=bg_seg_cpu,
            is_reduced_head=is_reduced_head,
            has_bg_slot=has_bg_slot,
            grid_thw=grid_thw,
            attn_thresh=attn_thresh,
        )
        for mk, mv in s_clip.items():
            if mk in slot_acc:
                slot_acc[mk].append(mv)
        for s_id, s_dict in ps_upd.items():
            for mk, mlist in s_dict.items():
                per_slot_acc[s_id][mk].extend(mlist)

        # Per-slot targets, presence scores, 3-way logits, and slot output vectors
        slot_pos_np, slot_3tgt = _get_slot_gt_info(actor_gt_64, S_act, is_reduced_head)
        all_slot_pres_targets.append(slot_pos_np.astype(np.int32))
        all_slot_3way_targets.append(slot_3tgt)
        all_slot_vecs.append(probe.sa_cache["slots_post_ffn"][0, :S_act].float().cpu().numpy())

        if is_reduced_head:
            raw3 = probe.sa_cache["pred_actor_raw"][0].float()  # (32, 3)
            p3 = raw3.softmax(dim=-1).cpu().numpy()
            all_slot_raw_3way.append(p3)
            all_slot_pres_scores.append(1.0 - p3[:, 2])
        else:
            all_slot_pres_scores.append(probs_64[:S_act])

        # Counterfactual Ablation 1: Uniform Attention (no extra backbone pass)
        all_scores_unif_64.append(_compute_uniform_attn_ablation_logits(model, probe.sa_cache, is_reduced_head))

        # Capture visual sample (prefer clip with >=2 active activities for rich visualization)
        if vis_sample is None or (len(vis_sample["pos_slots"]) < 2 and int(slot_pos_np.sum()) >= 2):
            frames_rgb = vid_cpu.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, 3) uint8
            bb4_0 = bb4[0].float()  # (C, T, h, w)
            bb4_pca = np.stack([_compute_pca_rgb_gpu(bb4_0[:, t]).cpu().numpy() for t in range(bb4_0.shape[1])], axis=0)

            pos_slot_entries = []
            for s_id in np.where(slot_pos_np)[0].tolist():
                s_name = REDUCED_SLOT_PAIRS[s_id][2] if (is_reduced_head and S_act == 32) else TACO_64_LABELS[s_id]
                s_p = float(all_slot_pres_scores[-1][s_id])
                pos_slot_entries.append((s_id, s_name, s_p))
            pos_slot_entries.sort(key=lambda z: float(vis_ex["attn_ori_3d"][z[0]].max()), reverse=True)

            neg_ids = np.where(~slot_pos_np)[0].tolist()
            top_neg = None
            if neg_ids:
                worst_neg = max(neg_ids, key=lambda sid: float(vis_ex["attn_ori_3d"][sid].max()))
                ns_name = REDUCED_SLOT_PAIRS[worst_neg][2] if (is_reduced_head and S_act == 32) else TACO_64_LABELS[worst_neg]
                top_neg = (worst_neg, ns_name, float(vis_ex["attn_ori_3d"][worst_neg].max()))

            vis_sample = {
                "scenario_key": scen_key,
                "ego_gt_name": TACO_EGO_4_LABELS[ego_gt] if ego_gt < len(TACO_EGO_4_LABELS) else str(ego_gt),
                "gt_active_names": [TACO_64_LABELS[i] for i in range(64) if actor_gt_64[i] > 0.5],
                "frames_rgb": frames_rgb,
                "bg_seg": bg_seg_cpu[:, 0].numpy(),
                "attn_ori_3d": vis_ex["attn_ori_3d"],
                "bb4_pca": bb4_pca,
                "pos_slots": pos_slot_entries,
                "top_neg_slot": top_neg,
                "has_bg_slot": has_bg_slot,
            }

        # Counterfactual Ablation 2: Time-Reversed Video (tests true temporal motion direction sensitivity)
        if not skip_time_reverse:
            probe.clear()
            pred_rev = model.predict(vid_t.flip(dims=[1]))
            all_scores_rev_64.append(torch.sigmoid(pred_rev["actor"][0].float()).cpu().numpy())
            if pred_rev.get("ego") is not None:
                all_ego_preds_rev.append(int(pred_rev["ego"][0].float().argmax().item()))

        # Multi-task Gradient Conflict on first `grad_samples` clips
        if n_grad_done < grad_samples:
            probe.clear()
            gc_m = measure_multitask_gradient_conflict(model, vid_t, tgt, is_reduced_head, device)
            for gk, gv in gc_m.items():
                if gk in grad_acc:
                    grad_acc[gk].append(gv)
            n_grad_done += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == num_samples:
            print(f"  Processed {idx + 1}/{num_samples} clips...")

    probe.remove()
    total_sec = time.perf_counter() - t0_all

    # =========================================================================
    # AGGREGATE METRICS
    # =========================================================================
    scores_arr = np.stack(all_scores_64, axis=0)      # (N, 64)
    targets_arr = np.stack(all_targets_64, axis=0)    # (N, 64)
    scores_unif_arr = np.stack(all_scores_unif_64, axis=0)

    overall, per_class_rows, aps_64 = compute_taco_multilabel_metrics(scores_arr, targets_arr, threshold=threshold)
    unif_overall, _, _ = compute_taco_multilabel_metrics(scores_unif_arr, targets_arr, threshold=threshold)

    overall["latency_ms"] = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    overall["clips_per_sec"] = 1000.0 / max(1e-6, overall["latency_ms"])
    overall["frames_per_sec"] = overall["clips_per_sec"] * resolved_seq_len
    overall["total_eval_sec"] = total_sec
    overall["params_total_m"] = p_total
    overall["params_backbone_m"] = p_backbone
    overall["params_proj_ego_m"] = p_proj_ego
    overall["params_slot_attn_m"] = p_slot
    overall["params_head_m"] = p_head
    for lk, lv in loss_acc.items():
        overall[lk] = float(np.mean(lv)) if lv else float("nan")

    # Component breakdown table
    comp_rows: List[Dict[str, object]] = []
    for cname in COMPONENT_ORDER:
        acc = comp_acc[cname]
        row: Dict[str, object] = {
            "component": cname,
            "module_group": cname.split(".")[0],
            "params_k": round(comp_params.get(cname, 0) / 1000.0, 2),
        }
        for mk in comp_metric_keys:
            row[mk] = round(float(np.mean(acc[mk])), 4) if acc[mk] else np.nan
        comp_rows.append(row)

    # Static query geometry, weight norms, and SoftPositionEmbed3D temporal vs spatial norm ratio
    slot_summary: Dict[str, float] = {}
    for mk in slot_metric_keys:
        if slot_acc[mk]:
            slot_summary[mk] = round(float(np.mean(slot_acc[mk])), 4)

    with torch.no_grad():
        mu = sa.slots_mu.expand(1, sa.num_slots, -1)
        sig = sa.slots_sigma.expand(1, sa.num_slots, -1)
        init_slots = (mu + sig * sa.slots_eps).float()
        q_static = sa.to_q(sa.norm_slots(init_slots))[0]  # (S_all, D)
        q_norms = torch.linalg.vector_norm(q_static, dim=-1).cpu().numpy()
        q_fg_n = F.normalize(q_static[:S_act], dim=-1, eps=1e-8)
        q_sim_mat = (q_fg_n @ q_fg_n.T).cpu().numpy()
        q_off = q_sim_mat[~np.eye(S_act, dtype=bool)]
        slot_summary["query_cos_sep"] = round(float(1.0 - q_off.mean()), 4)
        slot_summary["q_norm_fg_min"] = round(float(q_norms[:S_act].min()), 4)
        slot_summary["q_norm_fg_mean"] = round(float(q_norms[:S_act].mean()), 4)
        slot_summary["q_norm_fg_max"] = round(float(q_norms[:S_act].max()), 4)
        if has_bg_slot:
            slot_summary["q_norm_bg"] = round(float(q_norms[-1]), 4)

        # Temporal vs Spatial weight norm ratio inside SoftPositionEmbed3D (6-dim grid: t, h, w, 1-t, 1-h, 1-w)
        pe_w = sa.pe.embedding.weight.detach().float()  # (D, 6)
        t_col_norm = float(torch.linalg.vector_norm(pe_w[:, [0, 3]]).item())
        s_col_norm = float(torch.linalg.vector_norm(pe_w[:, [1, 2, 4, 5]]).item())
        slot_summary["pe3d_temporal_vs_spatial_ratio"] = round(t_col_norm / (s_col_norm + 1e-8), 4)

    def _wnorm(mod: nn.Module) -> float:
        sq = sum(float(p.detach().float().pow(2).sum().item()) for p in mod.parameters())
        return round(float(np.sqrt(sq)), 4)

    slot_summary["wnorm_conv3d"] = _wnorm(model.conv3d)
    slot_summary["wnorm_slot_attn"] = _wnorm(model.slot_attention)
    slot_summary["wnorm_head"] = _wnorm(model.head)

    # Cross-video slot vector geometry: cross-video std, pos-vs-neg cosine dist & Fisher ratio
    slot_vecs_arr = np.stack(all_slot_vecs, axis=0)              # (N, S_act, D)
    slot_pres_scores_arr = np.stack(all_slot_pres_scores, axis=0)  # (N, S_act)
    slot_pres_tgts_arr = np.stack(all_slot_pres_targets, axis=0)   # (N, S_act)
    slot_3tgt_arr = np.stack(all_slot_3way_targets, axis=0)        # (N, S_act)
    slot_raw3_arr = np.stack(all_slot_raw_3way, axis=0) if all_slot_raw_3way else None

    if len(slot_vecs_arr) >= 2:
        sv_norm = slot_vecs_arr / (np.linalg.norm(slot_vecs_arr, axis=-1, keepdims=True) + 1e-8)
        slot_summary["slot_cross_vid_std"] = round(float(sv_norm.std(axis=0).mean()), 4)

    per_slot_rows: List[Dict[str, Any]] = []
    fish_list, cos_pn_list, pres_ap_list, sg_acc_list, sg_conf_list = [], [], [], [], []

    for s_id in range(S_act):
        if is_reduced_head and S_act == 32:
            c_single, c_group, s_name, grp_key = REDUCED_SLOT_PAIRS[s_id]
            single_ap = float(aps_64[c_single])
            group_ap = float(aps_64[c_group])
        else:
            s_name = TACO_64_LABELS[s_id]
            grp_key = s_name.split(":")[0]
            single_ap = float(aps_64[s_id])
            group_ap = float("nan")

        y_pres = slot_pres_tgts_arr[:, s_id]
        s_pres = slot_pres_scores_arr[:, s_id]
        pos_clips = int(y_pres.sum())
        if pos_clips > 0 and pos_clips < len(y_pres):
            pres_ap = float(average_precision_score(y_pres, s_pres))
            pres_ap_list.append(pres_ap)
        else:
            pres_ap = float("nan")

        # Single vs Group disambiguation on present clips (ReduceAllocatedHead)
        sg_acc = float("nan")
        sg_conf = float("nan")
        if is_reduced_head and slot_raw3_arr is not None and pos_clips > 0:
            m_p = y_pres == 1
            gt_sg = slot_3tgt_arr[m_p, s_id]          # 0=single, 1=group
            pred_sg = np.argmax(slot_raw3_arr[m_p, s_id, :2], axis=-1)
            sg_acc = float((pred_sg == gt_sg).mean() * 100.0)
            sg_conf = 100.0 - sg_acc
            sg_acc_list.append(sg_acc)
            sg_conf_list.append(sg_conf)

        # Slot vector Fisher & Pos-vs-Neg prototype cosine distance
        s_vecs = slot_vecs_arr[:, s_id, :]  # (N, D)
        s_fish, s_cos_pn = compute_fisher_and_cos_sep(s_vecs, y_pres)
        if np.isfinite(s_fish):
            fish_list.append(s_fish)
        if np.isfinite(s_cos_pn):
            cos_pn_list.append(s_cos_pn)

        psa = per_slot_acc[s_id]

        def _m(k_name: str) -> float:
            return float(np.mean(psa[k_name])) if psa[k_name] else float("nan")

        pos_am = _m("pos_attn_mean")
        neg_am = _m("neg_attn_mean")
        pn_ratio = pos_am / (neg_am + 1e-8) if (np.isfinite(pos_am) and np.isfinite(neg_am)) else float("nan")

        per_slot_rows.append({
            "slot_id": s_id,
            "slot_name": s_name,
            "road_user_group": grp_key,
            "pos_clips": pos_clips,
            "presence_ap": round(pres_ap, 4) if np.isfinite(pres_ap) else np.nan,
            "single_ap": round(single_ap, 4) if np.isfinite(single_ap) else np.nan,
            "group_ap": round(group_ap, 4) if np.isfinite(group_ap) else np.nan,
            "single_vs_group_acc_pct": round(sg_acc, 2) if np.isfinite(sg_acc) else np.nan,
            "single_group_conf_pct": round(sg_conf, 2) if np.isfinite(sg_conf) else np.nan,
            "pos_attn_mean": round(pos_am, 5) if np.isfinite(pos_am) else np.nan,
            "pos_attn_max": round(_m("pos_attn_max"), 4) if np.isfinite(_m("pos_attn_max")) else np.nan,
            "pos_area_gt02_pct": round(_m("pos_area_gt02_pct"), 2) if np.isfinite(_m("pos_area_gt02_pct")) else np.nan,
            "neg_attn_mean": round(neg_am, 5) if np.isfinite(neg_am) else np.nan,
            "neg_fa_gt02_pct": round(_m("neg_fa_gt02_pct"), 3) if np.isfinite(_m("neg_fa_gt02_pct")) else np.nan,
            "pos_neg_attn_ratio": round(pn_ratio, 2) if np.isfinite(pn_ratio) else np.nan,
            "pos_fg_focus_pct": round(_m("pos_fg_focus_pct"), 2) if np.isfinite(_m("pos_fg_focus_pct")) else np.nan,
            "pos_temporal_entropy": round(_m("pos_temporal_entropy"), 4) if np.isfinite(_m("pos_temporal_entropy")) else np.nan,
            "q_norm": round(float(q_norms[s_id]), 3),
            "dot_pos_peak": round(_m("dot_pos_peak"), 3) if np.isfinite(_m("dot_pos_peak")) else np.nan,
            "gap_vs_bg_peak": round(_m("gap_vs_bg_peak"), 3) if np.isfinite(_m("gap_vs_bg_peak")) else np.nan,
            "slot_pos_neg_fisher": round(s_fish, 4) if np.isfinite(s_fish) else np.nan,
            "slot_pos_neg_cos_dist": round(s_cos_pn, 4) if np.isfinite(s_cos_pn) else np.nan,
            "pos_vs_global_v_cos": round(_m("pos_vs_global_v_cos"), 4) if np.isfinite(_m("pos_vs_global_v_cos")) else np.nan,
            "neg_vs_global_v_cos": round(_m("neg_vs_global_v_cos"), 4) if np.isfinite(_m("neg_vs_global_v_cos")) else np.nan,
        })

    if fish_list:
        slot_summary["slot_pos_neg_fisher"] = round(float(np.mean(fish_list)), 4)
    if cos_pn_list:
        slot_summary["slot_pos_neg_cos_dist"] = round(float(np.mean(cos_pn_list)), 4)
    if pres_ap_list:
        overall["presence_mAP_slots"] = float(np.mean(pres_ap_list))
    if sg_acc_list:
        overall["single_vs_group_acc_pct"] = float(np.mean(sg_acc_list))
        overall["single_group_conf_pct"] = float(np.mean(sg_conf_list))

    # Ego branch, multi-task gradients & counterfactual ablations
    n_ego = max(model.num_ego_class, 1)
    ego_conf_mat = np.zeros((n_ego, n_ego), dtype=np.int64)
    grad_ablation: Dict[str, float] = {
        k: round(float(np.mean(v)), 4) for k, v in grad_acc.items() if len(v) > 0
    }
    grad_ablation["ablation_normal_mAP"] = round(float(overall["mAP"]), 4)
    grad_ablation["ablation_uniform_attn_mAP"] = round(float(unif_overall["mAP"]), 4)
    grad_ablation["ablation_uniform_attn_delta_mAP"] = round(float(overall["mAP"] - unif_overall["mAP"]), 4)

    if all_scores_rev_64:
        scores_rev_arr = np.stack(all_scores_rev_64, axis=0)
        rev_overall, _, _ = compute_taco_multilabel_metrics(scores_rev_arr, targets_arr, threshold=threshold)
        grad_ablation["ablation_time_reversed_mAP"] = round(float(rev_overall["mAP"]), 4)
        grad_ablation["ablation_time_reversed_delta_mAP"] = round(float(overall["mAP"] - rev_overall["mAP"]), 4)

    if all_ego_gts:
        ego_gt_arr = np.array(all_ego_gts, dtype=np.int64)
        ego_pr_arr = np.array(all_ego_preds, dtype=np.int64)
        for gt_e, pr_e in zip(ego_gt_arr, ego_pr_arr):
            if 0 <= gt_e < n_ego and 0 <= pr_e < n_ego:
                ego_conf_mat[gt_e, pr_e] += 1
        grad_ablation["ego_acc_pct"] = round(float((ego_gt_arr == ego_pr_arr).mean() * 100.0), 2)
        grad_ablation["ego_macro_f1"] = round(float(f1_score(ego_gt_arr, ego_pr_arr, average="macro", zero_division=0)), 4)
        overall["ego_acc_pct"] = grad_ablation["ego_acc_pct"]
        overall["ego_macro_f1"] = grad_ablation["ego_macro_f1"]
        if all_ego_preds_rev:
            ego_rev_arr = np.array(all_ego_preds_rev, dtype=np.int64)
            grad_ablation["ablation_time_reversed_ego_acc_pct"] = round(float((ego_gt_arr == ego_rev_arr).mean() * 100.0), 2)
        if all_ego_feats:
            ef_fish, ef_cos = compute_fisher_and_cos_sep(np.stack(all_ego_feats, axis=0), ego_gt_arr)
            if np.isfinite(ef_fish):
                grad_ablation["ego_feat_fisher"] = round(ef_fish, 4)
            if np.isfinite(ef_cos):
                grad_ablation["ego_feat_cos_sep"] = round(ef_cos, 4)

    # =========================================================================
    # SAVE CSVs, CHARTS & HTML DASHBOARD
    # =========================================================================
    def _save_dict_csv(path: str, d: Dict[str, Any]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(d.keys()))
            w.writeheader()
            w.writerow({k: (round(v, 5) if isinstance(v, float) and np.isfinite(v) else v) for k, v in d.items()})

    def _save_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _save_dict_csv(os.path.join(out_dir, "action_overall_metrics.csv"), overall)
    _save_rows_csv(os.path.join(out_dir, "per_class_ap.csv"), sorted(per_class_rows, key=lambda r: r["ap"], reverse=True))
    _save_rows_csv(os.path.join(out_dir, "component_breakdown.csv"), comp_rows)
    _save_dict_csv(os.path.join(out_dir, "slot_attention_diagnostics.csv"), slot_summary)
    _save_rows_csv(os.path.join(out_dir, "slot_per_slot_diagnostics.csv"), per_slot_rows)
    _save_dict_csv(os.path.join(out_dir, "spatiotemporal_grad_ablation.csv"), grad_ablation)

    _plot_component_progression(comp_rows, os.path.join(out_dir, "chart_01_component_progression.png"))
    _plot_group_and_topology_map(overall, grad_ablation, os.path.join(out_dir, "chart_02_group_and_topology_map.png"))
    _plot_per_class_ap_heatmap(aps_64, per_class_rows, os.path.join(out_dir, "chart_03_per_class_ap_heatmap.png"))
    _plot_slot_attention_diagnostics(
        slot_summary, per_slot_rows, ego_conf_mat, grad_ablation,
        os.path.join(out_dir, "chart_04_slot_attention_diagnostics.png"),
    )
    if vis_sample is not None:
        _plot_visual_spatiotemporal_inspection(vis_sample, os.path.join(out_dir, "chart_05_visual_spatiotemporal_inspection.png"))
    _plot_slot_similarity_and_single_group(
        q_sim_mat, per_slot_rows, is_reduced_head,
        os.path.join(out_dir, "chart_06_slot_similarity_and_single_group.png"),
    )
    _generate_action_slot_dashboard(out_dir, overall, comp_rows, slot_summary, grad_ablation, per_slot_rows)

    # Terminal Summary
    print("\n" + "=" * 84)
    print("ACTION-SLOT COMPONENT VALIDATION SUMMARY:")
    print(f"  Actor mAP (64 cls / active)    : {overall['mAP']:.4f} / {overall['mAP_active_classes']:.4f} ({overall['active_classes']}/64 active)")
    print(f"  Group mAP (C / K(B) / P)       : {overall['mAP_c']:.4f} / {overall['mAP_b']:.4f} / {overall['mAP_p']:.4f}")
    print(f"  Group mAP (C+ / K+(B+) / P+)   : {overall['mAP_c_plus']:.4f} / {overall['mAP_b_plus']:.4f} / {overall['mAP_p_plus']:.4f}")
    print(f"  Single vs Group mAP            : {overall['mAP_single']:.4f} vs {overall['mAP_group']:.4f}"
          + (f" | Single/Group Acc: {overall.get('single_vs_group_acc_pct', 0):.1f}%" if "single_vs_group_acc_pct" in overall else ""))
    print(f"  Topology (Near/Lat/Far) mAP    : {overall['mAP_near_ego']:.4f} / {overall['mAP_lateral']:.4f} / {overall['mAP_far_opposite']:.4f}")
    print(f"  Multi-label Micro/Macro F1     : {overall['micro_f1']:.4f} / {overall['macro_f1']:.4f} (opt thr={overall['best_micro_f1_thr']:.2f} -> {overall['best_micro_f1']:.4f}) | ECE: {overall['ece']:.4f}")
    print(f"  Reverse-Dir Conf Prob          : {overall.get('reverse_dir_conf_prob_mean', float('nan')):.4f}")
    print("  --- Slot Attention Diagnostics (BG Slot, L_neg, Competition, Geometry) ---")
    print(f"  BG Slot IoU / Dice / BCE       : {slot_summary.get('bg_slot_iou', 0):.4f} / {slot_summary.get('bg_slot_dice', 0):.4f} / {slot_summary.get('bg_slot_bce', 0):.4f}")
    print(f"  FG Claim (Pos / Neg / BG)      : {slot_summary.get('fg_claimed_by_pos_pct', 0):.1f}% / {slot_summary.get('neg_slot_win_fg_pct', 0):.1f}% / {slot_summary.get('bg_slot_leak_fg_pct', 0):.1f}%")
    print(f"  Pos vs Neg Attn (Ratio)        : {slot_summary.get('pos_slot_attn_mean', 0):.4f} vs {slot_summary.get('neg_slot_attn_mean', 0):.4f} ({slot_summary.get('pos_neg_attn_ratio', 0):.2f}x) | Neg FA(>0.2): {slot_summary.get('neg_slot_fa_gt02_pct', 0):.3f}%")
    print(f"  Pos Focus FG / T-Span / T-Ent  : {slot_summary.get('pos_slot_fg_focus_pct', 0):.1f}% / {slot_summary.get('pos_slot_temporal_span_pct', 0):.1f}% / {slot_summary.get('pos_slot_temporal_entropy', 0):.3f}")
    print(f"  Query Sep / Slot Sep / CrossStd: {slot_summary.get('query_cos_sep', 0):.4f} / {slot_summary.get('slot_cos_sep', 0):.4f} / {slot_summary.get('slot_cross_vid_std', 0):.4f}")
    print(f"  Slot Fisher / Pos-Neg CosDist  : {slot_summary.get('slot_pos_neg_fisher', 0):.4f} / {slot_summary.get('slot_pos_neg_cos_dist', 0):.4f}")
    print(f"  Cos(Slot, Global-V) Pos / Neg  : {slot_summary.get('pos_slot_vs_global_v_cos', 0):.4f} / {slot_summary.get('neg_slot_vs_global_v_cos', 0):.4f}")
    print(f"  PE Ratio (2D / 3D | T/S PE3D)  : {slot_summary.get('pe2d_to_feat_ratio', 0):.3f} / {slot_summary.get('pe3d_to_feat_ratio', 0):.3f} | {slot_summary.get('pe3d_temporal_vs_spatial_ratio', 0):.3f}")
    print("  --- Multi-Task Gradients, Ego & Counterfactual Ablations ---")
    print(f"  Ego Acc / Macro F1 / Fisher    : {grad_ablation.get('ego_acc_pct', 0):.2f}% / {grad_ablation.get('ego_macro_f1', 0):.4f} / {grad_ablation.get('ego_feat_fisher', 0):.3f}")
    print(f"  Uniform-Attn Ablation mAP      : {grad_ablation.get('ablation_normal_mAP', 0):.4f} -> {grad_ablation.get('ablation_uniform_attn_mAP', 0):.4f} (slot loc gain: {grad_ablation.get('ablation_uniform_attn_delta_mAP', 0):+.4f})")
    if "ablation_time_reversed_mAP" in grad_ablation:
        print(f"  Time-Reversed Ablation mAP     : {grad_ablation['ablation_normal_mAP']:.4f} -> {grad_ablation['ablation_time_reversed_mAP']:.4f} (motion drop: {grad_ablation['ablation_time_reversed_delta_mAP']:+.4f})")
    print(f"  Grad Cos(Actor, Attn) c3d / bb : {grad_ablation.get('grad_cos_actor_vs_attn_conv3d', float('nan')):+.3f} / {grad_ablation.get('grad_cos_actor_vs_attn_backbone', float('nan')):+.3f}")
    print(f"  Grad Cos(Neg, BG) / (Act, Ego) : {grad_ablation.get('grad_cos_neg_vs_bg_conv3d', float('nan')):+.3f} / {grad_ablation.get('grad_cos_actor_vs_ego_backbone', float('nan')):+.3f}")
    print(f"  Latency / Effective FPS        : {overall['latency_ms']:.2f} ms/clip ({overall['clips_per_sec']:.1f} clips/s | {overall['frames_per_sec']:.1f} FPS)")
    print(f"  Params (Total / BB / Slot / Hd): {p_total:.2f}M ({p_backbone:.2f}M / {p_slot:.2f}M / {p_head:.2f}M)")
    print(f"  Output saved                   : {os.path.abspath(out_dir)}")
    print("=" * 84)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Component-by-component validation of Action-Slot on TACO.")
    p.add_argument("--config", type=str, default="action_slot/config.yaml", help="YAML config path.")
    p.add_argument("--weights", type=str, default="none", help="Path to checkpoint (.pth), or 'none' for init/pretrained backbone.")
    p.add_argument("--data-root", type=str, nargs="+", default=None, help="One or more TACO LMDB root dirs (overrides --config).")
    p.add_argument("--out-dir", type=str, default="action_slot/output_val_components", help="Output directory for CSVs, charts & HTML.")
    p.add_argument("--split", type=str, default="val", choices=["val", "train"], help="Dataset split to evaluate.")
    p.add_argument("--max-videos", type=int, default=0, help="Max video clips to evaluate (0 = all).")
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for multi-label Precision/Recall/F1.")
    p.add_argument("--attn-thresh", type=float, default=0.2, help="Attention threshold for active/FA area (Paper Sec 4.3: 0.2 on TACO).")
    p.add_argument("--grad-samples", type=int, default=20, help="Number of clips for multi-task gradient conflict estimation.")
    p.add_argument("--skip-time-reverse", action="store_true", help="Skip time-reversed counterfactual pass for faster validation.")
    p.add_argument("--seed", type=int, default=None, help="Random seed.")
    p.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', or 'cpu'.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_action_slot_component_validation(
        weights_path=args.weights,
        config_path=args.config,
        data_roots=args.data_root,
        out_dir=args.out_dir,
        split=args.split,
        max_videos=args.max_videos,
        threshold=args.threshold,
        attn_thresh=args.attn_thresh,
        grad_samples=args.grad_samples,
        skip_time_reverse=args.skip_time_reverse,
        seed=args.seed,
        device_str=args.device,
    )
