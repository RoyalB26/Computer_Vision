"""Checkpoint hook to automatically save model states and best metrics."""
import os
from typing import TYPE_CHECKING, Dict, Optional
from ..registrar import HOOKS
from .base_hook import BaseHook

if TYPE_CHECKING:
    from ..trainer import BaseTrainer


@HOOKS.register_module()
class CheckpointHook(BaseHook):
    """Save checkpoints periodically and track the best model weights.

    Args:
        interval: Epoch interval for saving regular checkpoints.
        save_best: Metric name to track for best checkpoint (e.g. 'acc', 'val_loss', 'iou').
        mode: Optimization direction ('min' or 'max').
        filename_tmpl: File name template for periodic checkpoints.
        max_keep_ckpts: Maximum number of recent checkpoints to keep (-1 for all).
    """

    def __init__(
        self,
        interval: int = 1,
        save_best: Optional[str] = "acc",
        mode: str = "max",
        filename_tmpl: str = "epoch_{}.pth",
        max_keep_ckpts: int = 3,
    ):
        self.interval = interval
        self.save_best = save_best
        self.mode = mode.lower()
        self.filename_tmpl = filename_tmpl
        self.max_keep_ckpts = max_keep_ckpts

        assert self.mode in ["min", "max"], "mode must be either 'min' or 'max'"
        self.best_score = float("inf") if self.mode == "min" else float("-inf")
        self.saved_ckpts = []

    def after_epoch(self, trainer: "BaseTrainer") -> None:
        epoch = trainer.epoch

        # Save periodic checkpoint
        if self.interval > 0 and (epoch + 1) % self.interval == 0:
            ckpt_name = self.filename_tmpl.format(epoch + 1)
            ckpt_path = os.path.join(trainer.work_dir, ckpt_name)
            trainer.save_checkpoint(ckpt_path)
            self.saved_ckpts.append(ckpt_path)

            # Cleanup older checkpoints if max_keep_ckpts reached
            if self.max_keep_ckpts > 0 and len(self.saved_ckpts) > self.max_keep_ckpts:
                oldest = self.saved_ckpts.pop(0)
                if os.path.exists(oldest):
                    os.remove(oldest)

        # Always update last.pth
        last_ckpt_path = os.path.join(trainer.work_dir, "last.pth")
        trainer.save_checkpoint(last_ckpt_path)

    def after_val(self, trainer: "BaseTrainer", val_metrics: Dict[str, float]) -> None:
        if self.save_best is None or self.save_best not in val_metrics:
            return

        current_score = val_metrics[self.save_best]
        is_best = False

        if self.mode == "max" and current_score > self.best_score:
            is_best = True
        elif self.mode == "min" and current_score < self.best_score:
            is_best = True

        if is_best:
            self.best_score = current_score
            best_ckpt_path = os.path.join(trainer.work_dir, "best.pth")
            trainer.save_checkpoint(best_ckpt_path, meta={"best_score": self.best_score, "metric": self.save_best})
            if trainer.logger:
                trainer.logger.info(
                    f"[CheckpointHook] New best {self.save_best}: {self.best_score:.4f}. Saved to {best_ckpt_path}"
                )
