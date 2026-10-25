"""Logger hook for console, file and experiment tracking."""
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from ..registrar import HOOKS
from .base_hook import BaseHook

if TYPE_CHECKING:
    from ..trainer import BaseTrainer


@HOOKS.register_module()
class LoggerHook(BaseHook):
    """Logs iteration stats and epoch summaries to console and files.

    Args:
        interval: Iteration interval for logging training status.
        reset_flag: Whether to reset running averages after each log interval.
        log_keys: Optional list of keys to print (e.g. ['loss', 'loss_cls', 'lr']).
            If None, logs all detected loss metrics and learning rate.
        ignore_keys: Optional list of keys to exclude from logging.
        show_time: Whether to show elapsed time per interval. Default: True.
        show_lr: Whether to show learning rate. Default: True.
    """

    def __init__(
        self,
        interval: int = 20,
        reset_flag: bool = True,
        log_keys: Optional[List[str]] = None,
        ignore_keys: Optional[List[str]] = None,
        show_time: bool = True,
        show_lr: bool = True,
    ):
        self.interval = interval
        self.reset_flag = reset_flag
        self.log_keys = log_keys
        self.ignore_keys = ignore_keys or []
        self.show_time = show_time
        self.show_lr = show_lr
        self.iter_time = 0.0
        self.start_time = 0.0
        self.running_loss: Dict[str, float] = {}
        self.running_count = 0

    def before_epoch(self, trainer: "BaseTrainer") -> None:
        self.start_time = time.time()
        self.running_loss.clear()
        self.running_count = 0

    def after_train_iter(self, trainer: "BaseTrainer", batch_idx: int, outputs: Dict[str, Any]) -> None:
        # Aggregate batch metrics
        for k, v in outputs.items():
            if k in self.ignore_keys:
                continue
            if self.log_keys is not None and k not in self.log_keys:
                continue
            if "loss" in k.lower() or (self.log_keys and k in self.log_keys):
                val = v.item() if hasattr(v, "item") else float(v)
                self.running_loss[k] = self.running_loss.get(k, 0.0) + val
        self.running_count += 1

        curr_iter = batch_idx + 1
        total_iters = len(trainer.train_loader)

        if curr_iter % self.interval == 0 or curr_iter == total_iters:
            elapsed = time.time() - self.start_time
            avg_losses = {k: v / self.running_count for k, v in self.running_loss.items()}

            parts = [f"Epoch [{trainer.epoch + 1}/{trainer.max_epochs}] Iter [{curr_iter}/{total_iters}]"]

            if self.show_lr and (self.log_keys is None or "lr" in self.log_keys):
                current_lr = trainer.get_current_lr()
                lr_str = f"lr: {current_lr:.6f}" if isinstance(current_lr, float) else f"lr: {current_lr}"
                parts.append(lr_str)

            if avg_losses:
                # Format specific keys if log_keys specified, or all
                loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
                parts.append(loss_str)

            if self.show_time:
                parts.append(f"time: {elapsed:.2f}s")

            msg = " - ".join(parts)
            if trainer.logger:
                trainer.logger.info(msg)

            if self.reset_flag:
                self.running_loss.clear()
                self.running_count = 0

    def after_val(self, trainer: "BaseTrainer", val_metrics: Dict[str, float]) -> None:
        filtered = {}
        for k, v in val_metrics.items():
            if k in self.ignore_keys:
                continue
            if self.log_keys is not None and k not in self.log_keys:
                continue
            filtered[k] = v

        metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in filtered.items()])
        msg = f"[Val Summary] Epoch [{trainer.epoch + 1}/{trainer.max_epochs}] - {metric_str}"
        if trainer.logger:
            trainer.logger.info(msg)
