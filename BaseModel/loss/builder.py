"""Loss builder using Registry."""
from typing import Any, Dict, Optional, Union
import torch.nn as nn
from ..core.registrar import LOSSES
from ..core.registry import build_from_cfg


def build_loss(cfg: Union[Dict[str, Any], nn.Module], default_args: Optional[Dict[str, Any]] = None) -> nn.Module:
    """Build a loss module from config dict or return existing module."""
    if isinstance(cfg, nn.Module):
        return cfg
    return build_from_cfg(cfg, LOSSES, default_args=default_args)
