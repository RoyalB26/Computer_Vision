"""Hardware accelerator and device management."""
from typing import Any, Optional, Union
import torch


def is_tpu_available() -> bool:
    """Check if PyTorch XLA (Google TPU support) is installed and available."""
    try:
        import torch_xla.core.xla_model as xm
        return True
    except (ImportError, Exception):
        return False


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Detect and return appropriate torch.device (supporting GPU, TPU, MPS, and CPU).

    Args:
        device_str: Specific device name ('cuda', 'cuda:0', 'mps', 'cpu', 'tpu', 'xla').
                    If None or 'auto', automatically chooses the fastest available device.
    """
    if device_str and device_str.lower() in ["tpu", "xla"]:
        if not is_tpu_available():
            raise ImportError(
                "TPU execution was requested (device='tpu'), but 'torch_xla' is not installed or available.\n"
                "To run on Google Cloud TPU (or Google Colab / Kaggle TPU runtimes), install torch_xla:\n"
                "  pip install torch_xla\n"
                "For more details, see: https://github.com/pytorch/xla"
            )
        import torch_xla.core.xla_model as xm
        return xm.xla_device()

    if device_str is not None and device_str != "auto":
        return torch.device(device_str)

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def to_device(data: Any, device: torch.device) -> Any:
    """Move data recursively to the target device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)([to_device(v, device) for v in data])
    return data
