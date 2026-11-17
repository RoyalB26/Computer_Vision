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


def _dataloader_worker_init_fn(worker_id: int, threads_per_worker: int = 1):
    """Worker initialization to prevent OpenCV/OpenMP thread oversubscription and keep RNG independent."""
    import os
    import random
    import numpy as np
    import torch

    # Ngăn OpenCV tạo thêm thread pool ngầm trong mỗi worker gây nghẽn context-switch trên CPU nhiều worker
    try:
        import cv2
        cv2.setNumThreads(0)
    except ImportError:
        pass

    threads = max(1, int(threads_per_worker))
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)

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
        kwargs.pop("multiprocessing_context", None)
        init_fn = None
    else:
        if worker_init_fn is not None:
            init_fn = worker_init_fn
        else:
            import os
            from functools import partial
            world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
            total_cpus = os.cpu_count() or 4
            threads_per_worker = max(1, total_cpus // max(1, num_workers * world_size))
            init_fn = partial(_dataloader_worker_init_fn, threads_per_worker=threads_per_worker)
        # Chỉ giữ persistent_workers mặc định cho train loader (hoặc khi được cấu hình tường minh)
        if "persistent_workers" not in kwargs:
            kwargs["persistent_workers"] = bool(shuffle or sampler is not None)
        if "prefetch_factor" not in kwargs:
            kwargs["prefetch_factor"] = 2
        # Tránh deadlock futex_do_wait do fork sau khi CUDA / OpenCV đã khởi tạo
        if "multiprocessing_context" not in kwargs and hasattr(torch.multiprocessing, "get_context"):
            import platform
            if platform.system().lower() == "linux":
                # 'forkserver' an toàn hơn 'fork' vì tạo process sạch không dính lock CUDA/OpenMP
                kwargs["multiprocessing_context"] = "forkserver"

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
