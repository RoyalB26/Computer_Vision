"""Toy Action Package.

Toy dataset generator, LMDB dataset reader, video recognition model,
hooks, and evaluation metrics compatible with BaseModel.
"""

from .dataset import (
    ALL_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    ToyActionLMDBDataset,
    decode_target,
    encode_label,
)
from .hooks import TrainToyActionMAPHook
from .metrics import (
    SYMBOL_GROUP_LABELS,
    SYMBOL_SINGLE_LABELS,
    TOY_ACTION_LABELS,
    TOY_ACTIONS,
    TOY_SYMBOLS,
    TOY_SYMBOL_VARIANTS,
    ToyActionMetric,
)
from .model import MultiLabelMetric, ToyActionNet

__all__ = [
    "ToyActionLMDBDataset",
    "ToyActionNet",
    "MultiLabelMetric",
    "ToyActionMetric",
    "TrainToyActionMAPHook",
    "ALL_CLASSES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "NUM_CLASSES",
    "encode_label",
    "decode_target",
    "TOY_ACTION_LABELS",
    "TOY_SYMBOLS",
    "TOY_ACTIONS",
    "TOY_SYMBOL_VARIANTS",
    "SYMBOL_SINGLE_LABELS",
    "SYMBOL_GROUP_LABELS",
]
