"""Model builder using Registry."""
from typing import Any, Dict, Optional, Union
import torch.nn as nn
from ..core.registrar import MODELS
from ..core.registry import build_from_cfg


def build_model(cfg: Union[Dict[str, Any], nn.Module], default_args: Optional[Dict[str, Any]] = None) -> nn.Module:
    """Build a model instance from configuration dict or return existing model."""
    if isinstance(cfg, nn.Module):
        return cfg
    return build_from_cfg(cfg, MODELS, default_args=default_args)
