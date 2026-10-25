"""Early stopping hook to avoid overfitting when validation metrics plateau."""
from typing import TYPE_CHECKING, Dict
from ..registrar import HOOKS
from .base_hook import BaseHook

if TYPE_CHECKING:
    from ..trainer import BaseTrainer


@HOOKS.register_module()
class EarlyStoppingHook(BaseHook):
    """Stops training when a monitored metric has stopped improving.

    Args:
        monitor: Metric name to monitor (e.g. 'val_loss', 'acc').
        patience: Number of epochs with no improvement after which training is stopped.
        min_delta: Minimum change to qualify as an improvement.
        mode: One of {'min', 'max'}.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        assert self.mode in ["min", "max"], "mode must be 'min' or 'max'"

        self.best_score = float("inf") if self.mode == "min" else float("-inf")
        self.wait_epochs = 0
        self.stopped_epoch = 0

    def after_val(self, trainer: "BaseTrainer", val_metrics: Dict[str, float]) -> None:
        if self.monitor not in val_metrics:
            return

        current_val = val_metrics[self.monitor]
        improved = False

        if self.mode == "min":
            if current_val < (self.best_score - self.min_delta):
                improved = True
        else:
            if current_val > (self.best_score + self.min_delta):
                improved = True

        if improved:
            self.best_score = current_val
            self.wait_epochs = 0
        else:
            self.wait_epochs += 1
            if trainer.logger:
                trainer.logger.info(
                    f"[EarlyStoppingHook] Patience: {self.wait_epochs}/{self.patience} (Best: {self.best_score:.4f})"
                )
            if self.wait_epochs >= self.patience:
                trainer.should_stop = True
                self.stopped_epoch = trainer.epoch + 1
                if trainer.logger:
                    trainer.logger.warning(
                        f"[EarlyStoppingHook] Early stopping triggered at epoch {self.stopped_epoch}!"
                    )
