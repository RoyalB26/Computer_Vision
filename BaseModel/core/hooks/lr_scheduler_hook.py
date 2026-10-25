"""LR Scheduler hook to update learning rate automatically."""
from typing import TYPE_CHECKING
from ..registrar import HOOKS
from .base_hook import BaseHook

if TYPE_CHECKING:
    from ..trainer import BaseTrainer


@HOOKS.register_module()
class LrSchedulerHook(BaseHook):
    """Step the learning rate scheduler per epoch or per iteration.

    Args:
        by_epoch: Whether to step the scheduler per epoch (True) or per iteration (False).
    """

    def __init__(self, by_epoch: bool = True):
        self.by_epoch = by_epoch

    def after_epoch(self, trainer: "BaseTrainer") -> None:
        if self.by_epoch and trainer.lr_scheduler is not None:
            # If scheduler is ReduceLROnPlateau, pass monitored metric
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            if isinstance(trainer.lr_scheduler, ReduceLROnPlateau):
                monitor_val = getattr(trainer, "last_val_metric", None)
                if monitor_val is not None:
                    trainer.lr_scheduler.step(monitor_val)
            else:
                trainer.lr_scheduler.step()

    def after_train_iter(self, trainer: "BaseTrainer", batch_idx: int, outputs: dict) -> None:
        if not self.by_epoch and trainer.lr_scheduler is not None:
            trainer.lr_scheduler.step()
