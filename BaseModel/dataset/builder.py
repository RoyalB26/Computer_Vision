"""Dataset and DataLoader builders using Registry."""
from typing import Any, Callable, Dict, Optional, Union
from torch.utils.data import DataLoader, Dataset
from ..core.registrar import COLLATE_FUNCTIONS, DATASETS
from ..core.registry import build_from_cfg
from .base_dataset import BaseDataset
import torch

def build_collate_fn(cfg: Union[str, Dict[str, Any], Callable]) -> Callable:
    """Build a collate function from name string, config dict, or return callable."""
    if callable(cfg):
        return cfg
    if isinstance(cfg, str):
        fn_cls = COLLATE_FUNCTIONS.get(cfg)
        if fn_cls is None:
            raise KeyError(
                f"Collate function '{cfg}' not found in COLLATE_FUNCTIONS registry. "
                f"Available: {list(COLLATE_FUNCTIONS.module_dict.keys())}"
            )
        return fn_cls()
    if isinstance(cfg, dict):
        return build_from_cfg(cfg, COLLATE_FUNCTIONS)
    raise TypeError(f"collate_fn must be str, dict, or callable, got {type(cfg)}")


def build_dataset(cfg: Union[Dict[str, Any], Dataset], default_args: Optional[Dict[str, Any]] = None) -> Dataset:
    """Build a dataset instance from config dict or return existing dataset."""
    if isinstance(cfg, Dataset):
        return cfg
    return build_from_cfg(cfg, DATASETS, default_args=default_args)


def build_dataloader(
    dataset: Union[Dict[str, Any], Dataset],
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Union[str, Dict[str, Any], Callable]] = None,
    **kwargs,
) -> DataLoader:
    """Build a PyTorch DataLoader from dataset or dataset config.

    If collate_fn is not provided, defaults to dataset.collate_fn or BaseDataset.collate_fn.
    """
    if isinstance(dataset, dict):
        dataset = build_dataset(dataset)

    if collate_fn is not None:
        collate_fn = build_collate_fn(collate_fn)
    elif hasattr(dataset, "collate_fn") and callable(dataset.collate_fn):
        collate_fn = dataset.collate_fn
    else:
        collate_fn = BaseDataset.collate_fn

    sampler = kwargs.pop("sampler", None)
    if sampler is None and torch.distributed.is_available() and torch.distributed.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        shuffle = False  # Shuffle được xử lý bởi DistributedSampler

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        **kwargs,
    )
