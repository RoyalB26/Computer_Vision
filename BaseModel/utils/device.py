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


def all_gather_numpy(arr: Any, device: Optional[Union[torch.device, str]] = None) -> Any:
    """Gather and concatenate a NumPy array across all distributed ranks safely.

    Forces C-contiguous memory layout before transferring to a tensor and before
    collective ``all_gather`` calls (preventing ``ValueError: Tensors must be contiguous``),
    and pads unequal leading dimensions across ranks to the global max length.
    """
    import numpy as np

    arr_c = np.ascontiguousarray(arr)
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return arr_c

    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return arr_c

    backend = str(torch.distributed.get_backend()).lower()
    if "nccl" in backend and torch.cuda.is_available():
        target_dev = torch.device(device) if isinstance(device, (torch.device, str)) else None
        if target_dev is not None and target_dev.type == "cuda" and target_dev.index is not None:
            curr_dev = target_dev
        else:
            curr_dev = torch.device("cuda", torch.cuda.current_device())
    else:
        curr_dev = torch.device("cpu")

    tensor = torch.from_numpy(arr_c).to(curr_dev).contiguous()
    local_len = int(tensor.shape[0]) if tensor.ndim > 0 else 1

    local_len_t = torch.tensor([local_len], dtype=torch.long, device=curr_dev)
    len_list = [torch.zeros(1, dtype=torch.long, device=curr_dev) for _ in range(world_size)]
    torch.distributed.all_gather(len_list, local_len_t)
    lengths = [int(x.item()) for x in len_list]
    max_len = max(lengths)

    if max_len == 0:
        return arr_c

    if local_len < max_len:
        pad_shape = (max_len - local_len, *tensor.shape[1:])
        pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=curr_dev)
        tensor = torch.cat([tensor, pad_tensor], dim=0).contiguous()

    gathered = [
        torch.zeros((max_len, *tensor.shape[1:]), dtype=tensor.dtype, device=curr_dev)
        for _ in range(world_size)
    ]
    torch.distributed.all_gather(gathered, tensor)

    valid_chunks = [chunk[:n] for chunk, n in zip(gathered, lengths) if n > 0]
    if not valid_chunks:
        return arr_c
    return np.ascontiguousarray(torch.cat(valid_chunks, dim=0).cpu().numpy())
