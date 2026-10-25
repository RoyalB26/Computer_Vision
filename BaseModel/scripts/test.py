"""Inference script for running predictions on images or test datasets."""
import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from BaseModel.model import build_model
from BaseModel.utils import Config, get_device, get_logger, load_checkpoint


def parse_test_args():
    parser = argparse.ArgumentParser(description="Run inference using trained CV model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (.pth)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu', 'tpu')")
    parser.add_argument("--input", type=str, default=None, help="Path to test image or folder of images")
    parser.add_argument(
        "--custom-imports",
        nargs="+",
        help="Custom Python modules or packages to import before building modules.",
    )
    return parser.parse_args()


def main():
    args = parse_test_args()
    cfg = Config.fromfile(args.config)

    # Import custom modules if specified via CLI or config
    custom_imports = args.custom_imports or getattr(cfg, "custom_imports", None)
    if custom_imports:
        import importlib
        for mod in custom_imports:
            importlib.import_module(mod)

    device_str = args.device or getattr(cfg, "device", "cuda")
    device = get_device(device_str)
    logger = get_logger(name="BaseModel.Inference")

    # Build and load model
    model = build_model(cfg.model)
    load_checkpoint(args.checkpoint, model=model, device=device)
    model.to(device)
    model.eval()

    logger.info(f"Model successfully loaded on {device}. Ready for inference.")
    return model


if __name__ == "__main__":
    main()
