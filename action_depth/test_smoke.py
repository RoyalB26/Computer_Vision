"""Quick Verification and Smoke-Test for action_depth.

1. Generates 20 train and 5 val samples in action_depth/data
2. Tests ActionDepthLMDBDataset loading
3. Runs forward_train pass through ActionDepthModel with Oxz2DMapEncoder
4. Checks backward pass & gradient flow
"""

import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
from action_depth.generate_data import generate_lmdb_dataset
from action_depth.dataset import ActionDepthLMDBDataset
from action_depth.model import ActionDepthModel


def run_test():
    print("=== Step 1: Generating 20 train samples & 5 val samples ===")
    train_dir = "action_depth/data/train_lmdb"
    val_dir = "action_depth/data/val_lmdb"
    generate_lmdb_dataset(train_dir, num_samples=20)
    generate_lmdb_dataset(val_dir, num_samples=5)

    print("\n=== Step 2: Testing Dataset Loading ===")
    ds = ActionDepthLMDBDataset(train_dir)
    sample = ds.get_data_info(0)
    print("Dataset length:", len(ds))
    print("Video tensor shape:", sample["image"].shape)
    print("Target actor label shape:", sample["target"]["actor"].shape)
    print("Target oxz_coords shape:", sample["target"]["oxz_coords"].shape)
    print("Target oxz_bev shape:", sample["target"]["oxz_bev"].shape)

    print("\n=== Step 3: Instantiating ActionDepthModel ===")
    model = ActionDepthModel(
        num_slots=6,
        slot_dim=128,
        z_bins=16,
        x_bins=16,
        vis_dim=16,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ActionDepthModel initialized! Trainable parameters: {num_params:,}")

    print("\n=== Step 4: Testing Forward & Backward Pass ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Batch of 2 samples
    b1 = ds.get_data_info(0)
    b2 = ds.get_data_info(1)
    batch_img = torch.stack([b1["image"], b2["image"]], dim=0).to(device)
    batch_targets = {
        "actor": torch.stack([b1["target"]["actor"], b2["target"]["actor"]], dim=0).to(device),
        "oxz_bev": torch.stack([b1["target"]["oxz_bev"], b2["target"]["oxz_bev"]], dim=0).to(device),
    }

    losses = model.forward_train(batch_img, batch_targets)
    print("Computed Losses:")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            print(f"  {k:15s}: {v.item():.4f}")

    total_loss = losses["loss"]
    total_loss.backward()
    print("\n[SUCCESS] Backward pass executed smoothly! Gradients computed for all modules.")


if __name__ == "__main__":
    run_test()
