"""PyTorch/XLA (TPU) Training Engine for VisDrone YOLOv11 & VisDroneYOLO11Slot.

Activated via:
    python VisDrone_yolov11/run.py --tpu --config VisDrone_yolov11/config_kaggle.yaml

Features:
  - 100% static tensor shapes (fixed `max_gt`, `drop_last=True`, masked losses).
  - CPU-side DataLoader collate for GT padding + `bg_mask`/`cls_mask` rasterization.
  - Native PyTorch/XLA multi-core (`xmp.spawn`), `MpDeviceLoader`, `xm.optimizer_step`, `bfloat16`.
  - Clean fallback to CUDA/CPU when `torch_xla` is unavailable (for local dry-run testing).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import yaml

from BaseModel.core import DATASETS, MODELS
from yolo.head import dist2bbox, make_anchors
from .metrics import VisDroneMAP
from .tpu_loss import TPUSlotAuxiliaryLoss, TPUYOLOv11DetectionLoss


def _detect_tpu_backend() -> str:
    """Detect available TPU backend: 'torch_tpu' (Colab PrivateUse1), 'torch_xla', or 'fallback'."""
    try:
        import torch_tpu  # noqa: F401
        if hasattr(torch, "tpu"):
            # Verify this process can actually access the TPU (not locked by another PID)
            _ = torch.zeros(1, device="tpu")
            return "torch_tpu"
    except Exception as e:
        print(f"[TPU Trainer] `torch_tpu` detected but unavailable in this process: {e}")
    try:
        import torch_xla  # noqa: F401
        return "torch_xla"
    except Exception:
        pass
    return "fallback"


class TPUVisDroneCollate:
    """CPU-side static-shape collate_fn for TPU/XLA.

    Pads GT boxes to fixed `max_gt` and pre-rasterizes `bg_mask` & `cls_mask` on CPU workers
    so TPU graphs never recompile and never sync with CPU.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (960, 960),
        num_classes: int = 10,
        max_gt: int = 300,
        strides: Tuple[int, int, int] = (8, 16, 32),
        build_slot_masks: bool = True,
    ):
        self.img_h, self.img_w = int(img_size[0]), int(img_size[1])
        self.num_classes = int(num_classes)
        self.max_gt = int(max_gt)
        self.resolutions = tuple((self.img_h // s, self.img_w // s) for s in strides)
        self.total_tokens = sum(h * w for h, w in self.resolutions)
        self.build_slot_masks = bool(build_slot_masks)

    def __call__(self, batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        bs = len(batch)
        images = torch.stack([item[0] for item in batch], dim=0)

        gt_bboxes = torch.zeros((bs, self.max_gt, 4), dtype=torch.float32)
        gt_labels = torch.zeros((bs, self.max_gt, 1), dtype=torch.int64)
        mask_gt = torch.zeros((bs, self.max_gt, 1), dtype=torch.float32)
        gt_counts = torch.zeros((bs, self.num_classes), dtype=torch.float32)

        bg_mask = torch.ones((bs, self.total_tokens), dtype=torch.float32)
        cls_mask = torch.zeros((bs, self.num_classes, self.total_tokens), dtype=torch.float32)

        for b, (_, target) in enumerate(batch):
            boxes = target.get("boxes", torch.empty((0, 4), dtype=torch.float32)).float()
            labels = target.get("labels", torch.empty((0,), dtype=torch.int64)).long()

            n = min(len(boxes), self.max_gt)
            if n > 0:
                boxes = boxes[:n]
                labels = labels[:n]
                # Convert normalized cxcywh (or xywh) to pixel xyxy
                if (boxes <= 1.05).all():
                    if target.get("_is_cxcywh", True):
                        cx, cy = boxes[:, 0] * self.img_w, boxes[:, 1] * self.img_h
                        bw, bh = boxes[:, 2] * self.img_w, boxes[:, 3] * self.img_h
                        xyxy = torch.stack([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], dim=-1)
                    else:
                        x1, y1 = boxes[:, 0] * self.img_w, boxes[:, 1] * self.img_h
                        bw, bh = boxes[:, 2] * self.img_w, boxes[:, 3] * self.img_h
                        xyxy = torch.stack([x1, y1, x1 + bw, y1 + bh], dim=-1)
                else:
                    xyxy = boxes

                xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0.0, float(self.img_w))
                xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0.0, float(self.img_h))

                gt_bboxes[b, :n] = xyxy
                gt_labels[b, :n, 0] = labels
                mask_gt[b, :n, 0] = 1.0

                valid_lbl = labels[(labels >= 0) & (labels < self.num_classes)]
                if len(valid_lbl) > 0:
                    gt_counts[b] = torch.bincount(valid_lbl, minlength=self.num_classes).float()

                if self.build_slot_masks:
                    # Reuse pre-computed bg_mask from sidecar LMDB if shape matches
                    pre_bg = target.get("bg_mask", None)
                    use_pre_bg = pre_bg is not None and pre_bg.numel() == self.total_tokens
                    if use_pre_bg:
                        bg_mask[b] = pre_bg.reshape(-1).float()

                    offset = 0
                    for h, w in self.resolutions:
                        n_tok = h * w
                        sx = w / max(float(self.img_w), 1.0)
                        sy = h / max(float(self.img_h), 1.0)
                        x1 = (xyxy[:, 0] * sx).floor().long().clamp(0, w - 1)
                        y1 = (xyxy[:, 1] * sy).floor().long().clamp(0, h - 1)
                        x2 = torch.maximum(x1 + 1, (xyxy[:, 2] * sx).ceil().long().clamp(0, w))
                        y2 = torch.maximum(y1 + 1, (xyxy[:, 3] * sy).ceil().long().clamp(0, h))

                        bg_grid = None if use_pre_bg else torch.ones((h, w), dtype=torch.float32)
                        cls_grid = torch.zeros((self.num_classes, h, w), dtype=torch.float32)
                        for i in range(n):
                            if bg_grid is not None:
                                bg_grid[y1[i] : y2[i], x1[i] : x2[i]] = 0.0
                            c = int(labels[i].item())
                            if 0 <= c < self.num_classes:
                                cls_grid[c, y1[i] : y2[i], x1[i] : x2[i]] = 1.0

                        if bg_grid is not None:
                            bg_mask[b, offset : offset + n_tok] = bg_grid.reshape(-1)
                        cls_mask[b, :, offset : offset + n_tok] = cls_grid.reshape(self.num_classes, -1)
                        offset += n_tok

        return {
            "images": images,
            "gt_bboxes": gt_bboxes,
            "gt_labels": gt_labels,
            "mask_gt": mask_gt,
            "gt_counts": gt_counts,
            "bg_mask": bg_mask,
            "cls_mask": cls_mask,
        }


class TPUYOLO11Wrapper(nn.Module):
    """Wraps VisDroneYOLO11 or VisDroneYOLO11Slot with static-shape TPU losses."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model
        self.has_slots = hasattr(base_model, "slot_refiner")

        crit = base_model.criterion
        self.tpu_det_loss = TPUYOLOv11DetectionLoss(
            num_classes=base_model.num_classes,
            reg_max=base_model.net.head.reg_max,
            box_weight=crit.box_weight,
            cls_weight=crit.cls_weight,
            dfl_weight=crit.dfl_weight,
            focal_alpha=crit.focal_loss.alpha,
            focal_gamma=crit.focal_loss.gamma,
        )

        if self.has_slots:
            sc = base_model.slot_criterion
            self.slot_loss_weight = float(base_model.slot_loss_weight)
            self.tpu_slot_loss = TPUSlotAuxiliaryLoss(
                num_classes=base_model.num_classes,
                bg_attn_weight=sc.bg_attn_weight,
                fg_attn_weight=sc.fg_attn_weight,
                suppress_weight=sc.suppress_weight,
                cls_attn_weight=sc.cls_attn_weight,
                ortho_weight=sc.ortho_weight,
            )
        else:
            self.slot_loss_weight = 0.0
            self.tpu_slot_loss = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images = batch["images"]
        if self.has_slots:
            feats, boxes_raw, scores_raw, slot_info = self.model._forward_head_with_slots(images)
        else:
            feat4, feat6, feat10 = self.model.net.backbone(images)
            p3, p4, p5 = self.model.net.neck(feat4, feat6, feat10)
            feats = [p3, p4, p5]
            bs = p3.shape[0]
            boxes_raw = torch.cat(
                [self.model.net.head.cv2[i](feats[i]).view(bs, 4 * self.model.net.head.reg_max, -1) for i in range(self.model.net.head.nl)],
                dim=-1,
            )
            scores_raw = torch.cat(
                [self.model.net.head.cv3[i](feats[i]).view(bs, self.model.num_classes, -1) for i in range(self.model.net.head.nl)],
                dim=-1,
            )
            slot_info = None

        preds = {"boxes": boxes_raw, "scores": scores_raw}
        if self.has_slots and getattr(self.model, "assigner_prior_weight", 0.0) > 0 and slot_info is not None and "dots" in slot_info:
            preds["assign_scores"] = (
                scores_raw + self.model.assigner_prior_weight * slot_info["dots"][:, : self.model.num_classes, :].detach()
            )
        a_raw, s_raw = make_anchors(feats, self.model.net.head.stride, 0.5)
        anchors, strides = (a.transpose(0, 1).to(images.device) for a in (a_raw, s_raw))

        pred_dist = self.model.net.head.dfl(boxes_raw)
        decoded_xyxy = (dist2bbox(pred_dist, anchors.unsqueeze(0), xywh=False, dim=1) * strides).transpose(1, 2)
        anchor_points_pixel = (anchors * strides).transpose(0, 1)
        stride_tensor = strides.transpose(0, 1)

        loss_dict = self.tpu_det_loss(
            preds=preds,
            gt_bboxes=batch["gt_bboxes"],
            gt_labels=batch["gt_labels"],
            mask_gt=batch["mask_gt"],
            decoded_bboxes=decoded_xyxy,
            anchor_points=anchor_points_pixel,
            stride_tensor=stride_tensor,
        )

        if self.has_slots and self.slot_loss_weight > 0 and slot_info is not None:
            slot_losses = self.tpu_slot_loss(
                slots=slot_info["slots"],
                attn_ori=slot_info["attn_ori"],
                obj_queries=self.model.slot_refiner.slot_attention.obj_queries,
                resolutions=slot_info["resolutions"],
                bg_mask=batch["bg_mask"],
                cls_mask=batch["cls_mask"],
                gt_counts=batch["gt_counts"],
            )
            loss_dict["loss"] = loss_dict["loss"] + self.slot_loss_weight * slot_losses["loss_slot"]
            loss_dict.update(slot_losses)

        return loss_dict


def _extract_img_size(ds_cfg: Dict[str, Any]) -> Tuple[int, int]:
    for step in ds_cfg.get("pipeline", []):
        if step.get("type") == "Resize" and "size" in step:
            sz = step["size"]
            return (int(sz[0]), int(sz[1]))
    return (960, 960)


def _train_worker(rank: int, cfg: Dict[str, Any], max_gt: int, backend: str) -> None:
    use_xla = backend == "torch_xla"
    use_torch_tpu = backend == "torch_tpu"

    if use_xla:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl

        device = xm.xla_device()
        world_size = xm.xrt_world_size()
        is_master = xm.is_master_ordinal()
        log_fn = xm.master_print
    elif use_torch_tpu:
        device = torch.device("tpu")
        world_size = 1
        is_master = True
        log_fn = print
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        is_master = True
        log_fn = print

    work_dir = Path(cfg.get("work_dir", "./work_dirs/visdrone_yolov11_tpu"))
    if is_master:
        work_dir.mkdir(parents=True, exist_ok=True)
    log_fn(f"[TPU Trainer] Backend: {backend} | Device: {device} | World size: {world_size} | max_gt: {max_gt}")

    # 1. Build model & wrap with static-shape TPU losses
    base_model = MODELS.build(cfg["model"])
    wrapper = TPUYOLO11Wrapper(base_model).to(device)

    # 2. Build datasets & static-shape CPU collate
    train_data_cfg = cfg["data"]["train"]
    val_data_cfg = cfg["data"]["val"]
    img_size = _extract_img_size(train_data_cfg["dataset"])

    train_ds = DATASETS.build(train_data_cfg["dataset"])
    val_ds = DATASETS.build(val_data_cfg["dataset"])

    collate_fn = TPUVisDroneCollate(
        img_size=img_size,
        num_classes=base_model.num_classes,
        max_gt=max_gt,
        build_slot_masks=wrapper.has_slots,
    )

    # On `torch_tpu` (PrivateUse1 PjRt), DataLoader multiprocessing workers trigger
    # `torch.manual_seed` -> `torch_tpu.manual_seed_all` -> "libtpu.so already in use".
    # Force num_workers=0 to avoid PjRt lock conflicts.
    train_workers = 0 if use_torch_tpu else train_data_cfg.get("num_workers", 4)
    val_workers = 0 if use_torch_tpu else val_data_cfg.get("num_workers", 2)
    if use_torch_tpu and (train_data_cfg.get("num_workers", 0) > 0 or val_data_cfg.get("num_workers", 0) > 0):
        log_fn("[TPU Trainer] `torch_tpu` detected: setting DataLoader num_workers=0 to prevent PjRt lock conflict.")

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_data_cfg.get("batch_size", 16),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_workers,
        collate_fn=collate_fn,
        drop_last=True,  # Critical for TPU static batch shape
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_data_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=val_workers,
        collate_fn=lambda b: (torch.stack([x[0] for x in b], dim=0), [x[1] for x in b]),
        drop_last=False,
    )

    # 3. Optimizer & LR scheduler
    opt_cfg = cfg.get("optimizer", {"lr": 0.001, "weight_decay": 0.0005})
    optimizer = torch.optim.AdamW(
        wrapper.parameters(),
        lr=float(opt_cfg.get("lr", 0.001)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0005)),
    )
    train_cfg = cfg.get("train_cfg", {})
    max_epochs = int(train_cfg.get("max_epochs", 100))
    val_interval = int(train_cfg.get("val_interval", 10))
    grad_clip = float(train_cfg.get("grad_clip", 10.0))
    use_bf16 = bool(train_cfg.get("use_amp", True)) and (use_xla or use_torch_tpu)
    autocast_device = "xla" if use_xla else "tpu"
    log_interval = int(train_cfg.get("log_interval", 50))

    sched_cfg = cfg.get("lr_scheduler", {})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(sched_cfg.get("T_max", max_epochs)),
        eta_min=float(sched_cfg.get("eta_min", 1e-5)),
    )

    best_map50 = -1.0

    for epoch in range(1, max_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        wrapper.train()
        t0 = time.perf_counter()
        loader = pl.MpDeviceLoader(train_loader, device) if use_xla else train_loader
        running_loss = 0.0
        steps = 0

        for step_idx, batch in enumerate(loader, start=1):
            if not use_xla:
                batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.amp.autocast(autocast_device, dtype=torch.bfloat16):
                    out = wrapper(batch)
                    loss = out["loss"]
            else:
                out = wrapper(batch)
                loss = out["loss"]

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), grad_clip)

            if use_xla:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()

            steps += 1
            if step_idx % log_interval == 0 or step_idx == len(train_loader):
                loss_val = float(loss.detach().cpu().item())
                running_loss += loss_val
                elapsed = time.perf_counter() - t0
                log_fn(
                    f"[TPU] Epoch [{epoch}/{max_epochs}] Iter [{step_idx}/{len(train_loader)}] "
                    f"lr: {optimizer.param_groups[0]['lr']:.6f} | loss: {loss_val:.4f} | time: {elapsed:.1f}s"
                )

        scheduler.step()

        # Validation & checkpointing on master
        if is_master and (epoch % val_interval == 0 or epoch == max_epochs):
            wrapper.eval()
            metric = VisDroneMAP(num_classes=base_model.num_classes)
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = imgs.to(device)
                    preds = base_model.predict(imgs, output_format="normalized")
                    metric.update(preds, targets)
            res = metric.compute()
            log_fn(
                f"[TPU Val] Epoch [{epoch}/{max_epochs}] "
                f"mAP_50: {res['mAP_50']:.4f} | mAP_75: {res['mAP_75']:.4f} | mAP_50_95: {res['mAP_50_95']:.4f}"
            )
            state = {"epoch": epoch, "model": base_model.state_dict(), "metrics": res}
            save_fn = xm.save if use_xla else torch.save
            save_fn(state, str(work_dir / "last.pth"))
            if res["mAP_50"] >= best_map50:
                best_map50 = res["mAP_50"]
                save_fn(state, str(work_dir / "best.pth"))
                log_fn(f"[TPU Checkpoint] New best mAP_50: {best_map50:.4f} -> {work_dir / 'best.pth'}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="TPU Trainer for VisDrone YOLOv11 (torch_tpu & torch_xla)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--tpu", action="store_true", help="Enable TPU static-shape training mode")
    parser.add_argument("--tpu-cores", type=int, default=8, help="Number of TPU cores (torch_xla multi-core)")
    parser.add_argument("--max-gt", type=int, default=300, help="Fixed static GT padding length per image")
    args, _ = parser.parse_known_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    backend = _detect_tpu_backend()
    if backend == "torch_xla" and args.tpu_cores > 1:
        import torch_xla.distributed.xla_multiprocessing as xmp

        xmp.spawn(_train_worker, args=(cfg, args.max_gt, backend), nprocs=args.tpu_cores, start_method="fork")
    else:
        if backend == "fallback":
            print("[TPU Trainer] Warning: Neither `torch_tpu` nor `torch_xla` found. Running static-shape pipeline on local CUDA/CPU.")
        _train_worker(0, cfg, args.max_gt, backend)
