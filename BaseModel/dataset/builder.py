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


def _dataloader_worker_init_fn(worker_id: int):
    """Worker initialization to prevent CPU thread thrashing and ensure reproducibility."""
    import os
    import random
    import numpy as np
    import torch

    # Tắt đa luồng ẩn của OpenCV/NumPy/Torch trong từng worker để không tranh chấp CPU cores
    try:
        import cv2
        cv2.setNumThreads(0)
    except ImportError:
        pass

    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Đảm bảo worker có seed ngẫu nhiên độc lập
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    dataset: Union[Dict[str, Any], Dataset],
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Union[str, Dict[str, Any], Callable]] = None,
    worker_init_fn: Optional[Callable] = None,
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

    if num_workers <= 0:
        kwargs.pop("persistent_workers", None)
        kwargs.pop("prefetch_factor", None)
        init_fn = None
    else:
        init_fn = worker_init_fn if worker_init_fn is not None else _dataloader_worker_init_fn
        # Mặc định cấu hình persistent_workers và prefetch_factor để pipeline nạp liên tục
        if "persistent_workers" not in kwargs:
            kwargs["persistent_workers"] = True
        if "prefetch_factor" not in kwargs:
            kwargs["prefetch_factor"] = 2

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        worker_init_fn=init_fn,
        **kwargs,
    )
