"""Dedicated validation and evaluation script for benchmarking checkpoints."""
import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from BaseModel.dataset import build_dataloader
from BaseModel.loss import build_loss
from BaseModel.model import build_model
from BaseModel.utils import Config, get_device, get_logger, load_checkpoint
from BaseModel.utils.metrics import build_metric


def parse_val_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CV model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (.pth)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu', 'mps', 'tpu')")
    parser.add_argument("--work-dir", type=str, default=None, help="Directory to save evaluation results")
    parser.add_argument(
        "--custom-imports",
        nargs="+",
        help="Custom Python modules or packages to import before building modules.",
    )
    return parser.parse_args()


def main():
    args = parse_val_args()
    cfg = Config.fromfile(args.config)

    # Import custom modules if specified via CLI or config
    custom_imports = args.custom_imports or getattr(cfg, "custom_imports", None)
    if custom_imports:
        import importlib
        for mod in custom_imports:
            importlib.import_module(mod)

    device_str = args.device or getattr(cfg, "device", "cuda")
    device = get_device(device_str)

    work_dir = args.work_dir or getattr(cfg, "work_dir", "./work_dirs/eval")
    os.makedirs(work_dir, exist_ok=True)
    logger = get_logger(name="BaseModel.Val", log_file=os.path.join(work_dir, "eval.log"))

    logger.info(f"Loaded config: {args.config}")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading checkpoint: {args.checkpoint}")

    # Build Model
    model = build_model(cfg.model)
    load_checkpoint(args.checkpoint, model=model, device=device)
    model.to(device)
    model.eval()

    # Build Val DataLoader
    val_data_cfg = cfg.data.val
    val_loader = build_dataloader(
        dataset=val_data_cfg.dataset,
        batch_size=val_data_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=val_data_cfg.get("num_workers", 4),
        pin_memory=val_data_cfg.get("pin_memory", True),
        drop_last=False,
    )

    # Build Metrics
    metrics = {}
    if "metrics" in cfg:
        for name, metric_cfg in cfg.metrics.items():
            metrics[name] = build_metric(metric_cfg)

    # Build Loss (optional)
    loss_fn = build_loss(cfg.loss).to(device) if "loss" in cfg else None

    # Run evaluation
    logger.info(f"Evaluating across {len(val_loader)} batches...")
    for m in metrics.values():
        if hasattr(m, "reset"):
            m.reset()

    val_loss_total = 0.0
    val_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if isinstance(batch, dict):
                inputs = batch.get("image", batch.get("inputs"))
                targets = batch.get("target", batch.get("targets"))
            elif isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None

            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device)

            if hasattr(model, "forward_val"):
                outputs = model.forward_val(inputs, targets)
            else:
                preds = model(inputs)
                outputs = {"preds": preds}

            preds = outputs.get("preds", outputs.get("pred", None))

            # Loss computation
            if "loss" in outputs:
                loss_val = outputs["loss"].item() if hasattr(outputs["loss"], "item") else float(outputs["loss"])
                val_loss_total += loss_val
                val_count += 1
            elif loss_fn is not None and targets is not None:
                loss_val = loss_fn(preds, targets).item()
                val_loss_total += loss_val
                val_count += 1

            # Update metrics
            if preds is not None and targets is not None:
                for metric in metrics.values():
                    if hasattr(metric, "update"):
                        metric.update(preds, targets)

    results = {}
    if val_count > 0:
        results["val_loss"] = val_loss_total / val_count

    for name, metric in metrics.items():
        if hasattr(metric, "compute"):
            val = metric.compute()
            if isinstance(val, dict):
                results.update(val)
            else:
                results[name] = float(val)

    metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in results.items()])
    logger.info(f"Evaluation Results -> {metric_str}")
    return results


if __name__ == "__main__":
    main()
