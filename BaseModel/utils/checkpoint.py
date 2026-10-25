"""Checkpoint saving and loading utilities."""
import os
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    best_filepath: Optional[str] = None,
    use_tpu: bool = False,
) -> None:
    """Save model checkpoint dictionary to disk.

    Args:
        state: State dictionary to save.
        filepath: Destination filepath.
        is_best: Whether this checkpoint represents the best metric achieved.
        best_filepath: Destination path for best checkpoint if is_best is True.
        use_tpu: Whether running on TPU (uses torch_xla.core.xla_model.save for distributed safety).
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    if use_tpu:
        try:
            import torch_xla.core.xla_model as xm
            xm.save(state, filepath)
            if is_best and best_filepath is not None:
                xm.save(state, best_filepath)
            return
        except (ImportError, Exception):
            pass

    torch.save(state, filepath)
    if is_best and best_filepath is not None:
        torch.save(state, best_filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model weights and optional optimizer/scheduler state from checkpoint file.

    Args:
        filepath: Path to checkpoint file.
        model: PyTorch model into which weights should be loaded.
        optimizer: Optional optimizer to restore state.
        lr_scheduler: Optional scheduler to restore state.
        device: Device mapping.
        strict: Whether to strictly enforce matching keys in state_dict.

    Returns:
        The loaded checkpoint dictionary.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No checkpoint found at '{filepath}'")

    checkpoint = torch.load(filepath, map_location=device)

    # Extract model state dict
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip potential 'module.' prefixes from DataParallel / DDP
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[name] = v

    model.load_state_dict(cleaned_state_dict, strict=strict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if lr_scheduler is not None and "lr_scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    return checkpoint
