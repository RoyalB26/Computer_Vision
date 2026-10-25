"""Metrics builder using Registry."""
from typing import Any, Dict, Optional, Union
from ...core.registrar import METRICS
from ...core.registry import build_from_cfg
from .base_metric import BaseMetric


def build_metric(cfg: Union[Dict[str, Any], BaseMetric], default_args: Optional[Dict[str, Any]] = None) -> BaseMetric:
    """Build a metric instance from config dict or return existing metric."""
    if isinstance(cfg, BaseMetric):
        return cfg
    return build_from_cfg(cfg, METRICS, default_args=default_args)
