"""Dataset module providing abstract BaseDataset, transform pipelines, collate functions, and helper utilities.

Users are expected to define task-specific datasets by inheriting BaseDataset.
"""
from .base_dataset import BaseDataset
from .builder import build_collate_fn, build_dataloader, build_dataset
from .collate import DetectionCollate, PadCollate, SmartCollate
from .helpers import (
    IMG_EXTENSIONS,
    close_lmdb,
    create_lmdb,
    find_images,
    load_image,
    open_lmdb,
    parse_csv,
    read_lmdb_bytes,
)
from .transforms import Compose, Normalize, Resize, ToTensor

__all__ = [
    # Abstract Base
    "BaseDataset",
    # Builders
    "build_dataset",
    "build_dataloader",
    "build_collate_fn",
    # Collate Functions
    "SmartCollate",
    "PadCollate",
    "DetectionCollate",
    # Transforms
    "Compose",
    "ToTensor",
    "Normalize",
    "Resize",
    # Helpers for building custom datasets
    "load_image",
    "find_images",
    "parse_csv",
    "open_lmdb",
    "close_lmdb",
    "read_lmdb_bytes",
    "create_lmdb",
    "IMG_EXTENSIONS",
]
