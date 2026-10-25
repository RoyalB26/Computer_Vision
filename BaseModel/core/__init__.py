"""Core module providing the modular registry, hook system, and base trainer engine."""
from .hooks import (
    BaseHook,
    CheckpointHook,
    EarlyStoppingHook,
    LoggerHook,
    LrSchedulerHook,
    TensorboardHook,
    WandbHook,
)
from .registrar import (
    BACKBONES,
    COLLATE_FUNCTIONS,
    DATASETS,
    HEADS,
    HOOKS,
    LAYERS,
    LOSSES,
    METRICS,
    MODELS,
    NECKS,
    OPTIMIZERS,
    SCHEDULERS,
    TRANSFORMS,
)
from .registry import Registry, build_from_cfg
from .trainer import BaseTrainer

__all__ = [
    "Registry",
    "build_from_cfg",
    "BaseTrainer",
    "BaseHook",
    "CheckpointHook",
    "LoggerHook",
    "LrSchedulerHook",
    "EarlyStoppingHook",
    "TensorboardHook",
    "WandbHook",
    "MODELS",
    "BACKBONES",
    "NECKS",
    "HEADS",
    "LAYERS",
    "DATASETS",
    "TRANSFORMS",
    "LOSSES",
    "METRICS",
    "HOOKS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "COLLATE_FUNCTIONS",
]
