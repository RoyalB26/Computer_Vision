"""Command-line argument parser for training and evaluation."""
import argparse
from typing import Optional


def parse_args(description: str = "Computer Vision Pipeline") -> argparse.Namespace:
    """Parse standard command line arguments."""
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML or JSON configuration file.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Directory to save logs and checkpoints. Overrides config.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file (.pth) to resume training from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g. 'cuda', 'cuda:0', 'cpu', 'mps', 'tpu'). Overrides config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Overrides config.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Whether to run evaluation only without training.",
    )
    parser.add_argument(
        "--accumulate",
        type=int,
        default=None,
        help="Gradient accumulation steps. Overrides train_cfg.accumulate in config.",
    )
    parser.add_argument(
        "--custom-imports",
        nargs="+",
        help="Custom Python modules or packages to import before building modules.",
    )
    parser.add_argument(
        "--accelerate",
        action="store_true",
        help="Enable Hugging Face Accelerate for distributed, multi-GPU, or mixed precision training.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases (WandB) experiment tracking and metric logging.",
    )
    parser.add_argument(
        "--tqdm",
        action="store_true",
        help="Enable interactive tqdm progress bars for training and validation iterations.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging for losses and scalar metrics.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        help="Override configuration options using key=value syntax (e.g. optimizer.lr=0.001 train.epochs=50).",
    )

    return parser.parse_args()
