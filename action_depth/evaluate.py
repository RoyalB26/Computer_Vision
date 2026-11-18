"""Evaluation and BEV Qualitative Visualizer for action_depth.

Loads trained ActionDepthModel checkpoint, evaluates on validation set,
and generates side-by-side Ground Truth vs Predicted 2D Oxz Ground Plane Trajectories.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import matplotlib.pyplot as plt
import numpy as np
import torch
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
)
from action_depth.model import ActionDepthModel
from action_depth.train import collate_fn, compute_metrics


def evaluate_model(
    model: ActionDepthModel,
    val_dataset: ActionDepthLMDBDataset,
    device: torch.device,
    output_dir: str,
    num_vis: int = 4,
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    all_logits = []
    all_targets = []
    all_bev_mse = []
    all_tracking_errors = []

    print("\n--- Running Quantitative Evaluation ---")
    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            out = model.predict(images)
            logits = out["logits"]
            pred_bev = out["pred_oxz_bev"]  # (B, 16, 6, z_bins, x_bins)

            all_logits.append(logits.cpu())
            all_targets.append(targets["actor"].cpu())

            if "oxz_bev" in targets:
                gt_bev = targets["oxz_bev"]
                b, t, s, zb, xb = pred_bev.shape
                gt_bev_down = F.interpolate(
                    gt_bev.view(b * t * s, 1, 32, 32),
                    size=(zb, xb),
                    mode="area",
                ).view(b, t, s, zb, xb)
                mse = F.mse_loss(pred_bev, gt_bev_down, reduction="none").mean(dim=(1, 2, 3, 4))
                all_bev_mse.extend(mse.cpu().tolist())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_logits, all_targets)

    print(f"Validation Multi-label Classification:")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"  Accuracy : {metrics['acc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    if all_bev_mse:
        print(f"  BEV Oxz MSE: {np.mean(all_bev_mse):.5f}")

    # Per-action category metrics
    print("\nPer-Action Recognition Performance:")
    probs = torch.sigmoid(all_logits)
    for act in ACTIONS:
        act_indices = [i for i, c in enumerate(ALL_CLASSES) if c.endswith(f":{act}")]
        sub_probs = probs[:, act_indices].flatten()
        sub_targets = all_targets[:, act_indices].flatten()
        sub_preds = (sub_probs >= 0.5).float()

        tp = (sub_preds * sub_targets).sum().item()
        fp = (sub_preds * (1.0 - sub_targets)).sum().item()
        fn = ((1.0 - sub_preds) * sub_targets).sum().item()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        print(f"  Action '{act:5s}' -> Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")

    # Generate Qualitative Visualizations
    print(f"\n--- Generating Qualitative Visualizations ({num_vis} samples) ---")
    vis_indices = list(range(min(num_vis, len(val_dataset))))

    for sample_idx in vis_indices:
        data = val_dataset.get_data_info(sample_idx)
        img_tensor = data["image"].unsqueeze(0).to(device)  # (1, 3, 16, 224, 224)
        gt_target = data["target"]
        gt_label_str = data["label_str"]

        with torch.no_grad():
            out = model.predict(img_tensor)
            pred_logits = out["logits"][0].cpu()
            pred_probs = out["probs"][0].cpu()
            pred_oxz_bev = out["pred_oxz_bev"][0].cpu()  # (16, 6, z_bins, x_bins)

        # Decode predicted classes (prob >= 0.45)
        pred_classes = [ALL_CLASSES[i] for i, p in enumerate(pred_probs) if p >= 0.45]

        # Select 4 representative frames: t=0, 5, 10, 15
        key_frames = [0, 5, 10, 15]

        fig = plt.figure(figsize=(18, 10), constrained_layout=True)
        gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 1.4])

        # 1. Perspective RGB Frames
        for col_idx, t in enumerate(key_frames):
            ax = fig.add_subplot(gs[0, col_idx])
            frame_rgb = data["image"][:, t, :, :].permute(1, 2, 0).numpy()
            ax.imshow(frame_rgb)
            ax.set_title(f"Camera View (t={t})", fontsize=10)
            ax.axis("off")

        # 2. Predicted 2D Oxz Ground Plane Map (Max over object slots)
        for col_idx, t in enumerate(key_frames):
            ax = fig.add_subplot(gs[1, col_idx])
            bev_pred_t = pred_oxz_bev[t].amax(dim=0).numpy()  # (z_bins, x_bins)
            im = ax.imshow(bev_pred_t, cmap="magma", origin="lower", extent=[-1.0, 1.0, 0.0, 1.0])
            ax.set_title(f"Pred Oxz Map (t={t})", fontsize=10)
            ax.set_xlabel("X (left/right)")
            if col_idx == 0:
                ax.set_ylabel("Z (near/far)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 3. Ground Truth 2D Oxz Ground Plane Map
        if "oxz_bev" in gt_target:
            gt_bev_raw = gt_target["oxz_bev"]  # (16, 6, 32, 32)
            for col_idx, t in enumerate(key_frames):
                ax = fig.add_subplot(gs[2, col_idx])
                gt_bev_t = gt_bev_raw[t].amax(dim=0).numpy()
                im = ax.imshow(gt_bev_t, cmap="viridis", origin="lower", extent=[-1.0, 1.0, 0.0, 1.0])
                ax.set_title(f"GT Oxz Map (t={t})", fontsize=10)
                ax.set_xlabel("X (left/right)")
                if col_idx == 0:
                    ax.set_ylabel("Z (near/far)")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 4. Trajectory Plot in metric Oxz ground plane on right column
        ax_traj = fig.add_subplot(gs[:, 4])
        ax_traj.set_xlim(-1.1, 1.1)
        ax_traj.set_ylim(0.0, 1.1)
        ax_traj.set_xlabel("Ground X (lateral in meters/norm)")
        ax_traj.set_ylabel("Ground Z (depth ahead in meters/norm)")
        ax_traj.set_title("2D Ground Plane Oxz Trajectories", fontsize=12, fontweight="bold")
        ax_traj.grid(True, linestyle="--", alpha=0.5)

        # Plot GT trajectories
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
        if "oxz_coords" in gt_target:
            coords = gt_target["oxz_coords"].numpy()  # (16, 6, 2)
            for s in range(coords.shape[1]):
                traj_s = coords[:, s, :]  # (16, 2)
                # Check if this object is present (not dummy zeros)
                if np.ptp(traj_s[:, 0]) > 0.01 or np.ptp(traj_s[:, 1]) > 0.01 or traj_s[0, 1] > 0.05:
                    c = colors[s % len(colors)]
                    sym = SYMBOLS[s] if s < len(SYMBOLS) else str(s)
                    ax_traj.plot(traj_s[:, 0], traj_s[:, 1], "o-", color=c, label=f"GT Slot {s} ('{sym}')", linewidth=2, markersize=4)
                    ax_traj.annotate("Start", (traj_s[0, 0], traj_s[0, 1]), fontsize=8, color=c)
                    ax_traj.annotate("End", (traj_s[-1, 0], traj_s[-1, 1]), fontsize=8, color=c)

        # Plot predicted slot peak trajectories
        z_grid = np.linspace(0.0, 1.0, pred_oxz_bev.shape[2])
        x_grid = np.linspace(-1.0, 1.0, pred_oxz_bev.shape[3])

        for s in range(pred_oxz_bev.shape[1]):
            slot_bev = pred_oxz_bev[:, s].numpy()  # (16, z_bins, x_bins)
            # Find peak (Z, X) over time
            pred_x_list = []
            pred_z_list = []
            max_val = np.max(slot_bev)
            if max_val > 0.05:  # Active slot threshold
                for t in range(16):
                    flat_idx = np.argmax(slot_bev[t])
                    zi, xi = np.unravel_index(flat_idx, slot_bev[t].shape)
                    pred_z_list.append(z_grid[zi])
                    pred_x_list.append(x_grid[xi])

                c = colors[s % len(colors)]
                sym = SYMBOLS[s] if s < len(SYMBOLS) else str(s)
                ax_traj.plot(pred_x_list, pred_z_list, "--^", color=c, alpha=0.8, label=f"Pred Slot {s} ('{sym}')", linewidth=1.5, markersize=5)

        ax_traj.legend(loc="upper left", fontsize=8)

        # Text summary overlay
        summary_text = (
            f"Sample {sample_idx:03d}\n"
            f"Ground Truth Actions:\n  {gt_label_str if gt_label_str else 'None'}\n\n"
            f"Predicted Actions:\n  {', '.join(pred_classes) if pred_classes else 'None'}"
        )
        fig.text(0.72, 0.05, summary_text, fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#999999"))

        save_path = os.path.join(output_dir, f"oxz_eval_sample_{sample_idx}.png")
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"Saved visualization artifact: {save_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Action Depth Model")
    parser.add_argument("--checkpoint", type=str, default="action_depth/checkpoints/best_model.pth")
    parser.add_argument("--val-dir", type=str, default="action_depth/data/val_lmdb")
    parser.add_argument("--output-dir", type=str, default="C:/Users/hbaog/.gemini/antigravity/brain/a3bfaf0a-3396-4e7f-99fb-957f36dc6ae3")
    parser.add_argument("--num-vis", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model_args = ckpt.get("args", {})
    model = ActionDepthModel(
        num_slots=6,
        slot_dim=model_args.get("slot_dim", 128),
        num_classes=NUM_CLASSES,
        bg_slot=True,
        z_bins=model_args.get("z_bins", 16),
        x_bins=model_args.get("x_bins", 16),
        vis_dim=model_args.get("vis_dim", 16),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    val_dataset = ActionDepthLMDBDataset(args.val_dir)

    evaluate_model(model, val_dataset, device, args.output_dir, num_vis=args.num_vis)
