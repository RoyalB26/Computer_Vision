"""Training Pipeline for Action Depth (2D Oxz Ground Plane Slot Tracking).

Trains ActionDepthModel with pure Conv2d Oxz2DMapEncoder to simultaneously:
  1. Learn multi-label spatiotemporal action classification ('far', 'near', 'left', 'right')
  2. Track object trajectories directly in 2D ground plane Oxz without Conv3d
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Tuple

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from action_depth.dataset import ActionDepthLMDBDataset
from action_depth.generate_data import (
    ACTIONS,
    ALL_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    SYMBOLS,
    generate_lmdb_dataset,
)
from action_depth.model import ActionDepthModel


def collate_fn(batch: List[Dict]) -> Dict:
    images = torch.stack([b["image"] for b in batch], dim=0)
    indices = [b["idx"] for b in batch]
    label_strs = [b["label_str"] for b in batch]

    targets = {}
    if "actor" in batch[0]["target"]:
        targets["actor"] = torch.stack([b["target"]["actor"] for b in batch], dim=0)
    if "oxz_bev" in batch[0]["target"]:
        targets["oxz_bev"] = torch.stack([b["target"]["oxz_bev"] for b in batch], dim=0)
    if "oxz_coords" in batch[0]["target"]:
        targets["oxz_coords"] = torch.stack([b["target"]["oxz_coords"] for b in batch], dim=0)

    return {
        "indices": indices,
        "images": images,
        "targets": targets,
        "label_strs": label_strs,
    }


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """Computes binary multi-label Accuracy, Precision, Recall, and F1 score."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets >= 0.5).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()
    tn = ((1.0 - preds) * (1.0 - targets)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_loss_actor = 0.0
    total_loss_bev = 0.0
    all_logits = []
    all_targets = []
    bev_mse_list = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            losses = model.forward_train(images, targets)
            total_loss += losses["loss"].item() * images.size(0)
            total_loss_actor += losses["loss_actor"].item() * images.size(0)
            total_loss_bev += losses["loss_bev_oxz"].item() * images.size(0)

            all_logits.append(losses["pred_actor"].cpu())
            all_targets.append(losses["target_actor"].cpu())

    n_samples = len(dataloader.dataset)
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_logits, all_targets)

    metrics["val_loss"] = total_loss / n_samples
    metrics["val_loss_actor"] = total_loss_actor / n_samples
    metrics["val_loss_bev"] = total_loss_bev / n_samples
    return metrics


def get_lmdb_sample_count(lmdb_path: str) -> int:
    """Safely inspects num-samples without keeping env open."""
    import lmdb
    if not os.path.exists(lmdb_path):
        return 0
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            val = txn.get(b"num-samples")
            count = int(val.decode("utf-8")) if val is not None else 0
        env.close()
        return count
    except Exception:
        return 0


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"=== Running Action Depth Training on {device} ===")

    # 1. Dataset verification / generation
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    train_path = os.path.join(args.data_dir, "train_lmdb")
    val_path = os.path.join(args.data_dir, "val_lmdb")

    train_count = get_lmdb_sample_count(train_path)
    if train_count < args.num_train:
        print(f"Generating {args.num_train} training samples -> {train_path}...")
        generate_lmdb_dataset(train_path, num_samples=args.num_train)
    else:
        print(f"Found existing train LMDB with {train_count} samples.")

    val_count = get_lmdb_sample_count(val_path)
    if val_count < args.num_val:
        print(f"Generating {args.num_val} validation samples -> {val_path}...")
        generate_lmdb_dataset(val_path, num_samples=args.num_val)
    else:
        print(f"Found existing val LMDB with {val_count} samples.")

    train_dataset = ActionDepthLMDBDataset(train_path)
    val_dataset = ActionDepthLMDBDataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # 2. Model Initialization
    model = ActionDepthModel(
        num_slots=6,
        slot_dim=args.slot_dim,
        num_classes=NUM_CLASSES,
        bg_slot=True,
        z_bins=args.z_bins,
        x_bins=args.x_bins,
        vis_dim=args.vis_dim,
        bev_loss_weight=args.bev_weight,
        attn_suppress_weight=args.suppress_weight,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized ActionDepthModel with {total_params:,} trainable parameters.")
    print(f"  - Conv2d Oxz Ground Encoder: {args.z_bins}x{args.x_bins} bins")
    print(f"  - Total slots: 6 object slots + 1 bg slot")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_f1 = -1.0
    best_loss = 1e9
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")

    print("\nStarting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_actor = 0.0
        running_bev = 0.0
        running_suppress = 0.0

        for batch in train_loader:
            images = batch["images"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            optimizer.zero_grad()
            losses = model.forward_train(images, targets)
            loss = losses["loss"]
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = images.size(0)
            running_loss += loss.item() * bs
            running_actor += losses["loss_actor"].item() * bs
            running_bev += losses["loss_bev_oxz"].item() * bs
            running_suppress += losses["loss_suppress"].item() * bs

        scheduler.step()
        epoch_time = time.time() - t0
        n_train = len(train_dataset)

        train_loss = running_loss / n_train
        train_actor = running_actor / n_train
        train_bev = running_bev / n_train
        train_suppress = running_suppress / n_train

        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) "
            f"Train Loss: {train_loss:.4f} (Act: {train_actor:.3f}, BEV: {train_bev:.4f}, Sup: {train_suppress:.3f}) | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['acc']:.4f}"
        )

        # Checkpoint based on composite metric (F1 + low loss)
        if val_metrics["f1"] > best_f1 or (abs(val_metrics["f1"] - best_f1) < 0.01 and val_metrics["val_loss"] < best_loss):
            best_f1 = val_metrics["f1"]
            best_loss = val_metrics["val_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"  >>> Saved New Best Model Checkpoint (F1: {best_f1:.4f}, Val Loss: {best_loss:.4f})")

    print(f"\n[DONE] Training complete! Best Checkpoint saved at: {best_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Action Depth Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--slot-dim", type=int, default=128, help="Slot latent dimension")
    parser.add_argument("--z-bins", type=int, default=16, help="Ground plane depth bins")
    parser.add_argument("--x-bins", type=int, default=16, help="Ground plane lateral bins")
    parser.add_argument("--vis-dim", type=int, default=16, help="Visual feature dimension")
    parser.add_argument("--bev-weight", type=float, default=2.0, help="Weight for BEV Oxz MSE loss")
    parser.add_argument("--suppress-weight", type=float, default=0.5, help="Weight for inactive suppression")
    parser.add_argument("--num-train", type=int, default=250, help="Number of training samples")
    parser.add_argument("--num-val", type=int, default=50, help="Number of validation samples")
    parser.add_argument("--data-dir", type=str, default="action_depth/data", help="Directory for datasets")
    parser.add_argument("--checkpoint-dir", type=str, default="action_depth/checkpoints", help="Directory for models")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")

    args = parser.parse_args()
    train(args)
