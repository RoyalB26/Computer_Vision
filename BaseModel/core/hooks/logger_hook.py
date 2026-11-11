"""Logger hook for console, file and experiment tracking."""
import time
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import numpy as np
import torch
from ..registrar import HOOKS
from ...utils.device import all_gather_numpy
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
        self._train_pred_actors: List[Any] = []
        self._train_target_actors: List[Any] = []

    def before_epoch(self, trainer: "BaseTrainer") -> None:
        self.start_time = time.time()
        self.running_loss.clear()
        self.running_count = 0
        self._train_pred_actors.clear()
        self._train_target_actors.clear()

    def after_train_iter(self, trainer: "BaseTrainer", batch_idx: int, outputs: Dict[str, Any]) -> None:
        # Collect train actor predictions & targets for epoch-level mAP calculation
        if "pred_actor" in outputs and "target_actor" in outputs:
            pred_act = outputs["pred_actor"]
            tgt_act = outputs["target_actor"]
            if pred_act is not None and tgt_act is not None:
                if hasattr(pred_act, "detach"):
                    scores = torch.sigmoid(pred_act.detach().float()).contiguous().cpu().numpy()
                else:
                    scores = np.ascontiguousarray(pred_act, dtype=np.float32)

                if hasattr(tgt_act, "detach"):
                    targets = tgt_act.detach().float().contiguous().cpu().numpy()
                else:
                    targets = np.ascontiguousarray(tgt_act, dtype=np.float32)

                self._train_pred_actors.append(scores)
                self._train_target_actors.append(targets)

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

    def after_train_epoch(self, trainer: "BaseTrainer") -> None:
        """Invoked immediately after train_epoch() finishes (before validation)."""
        self._log_train_actor_map(trainer)

    def after_epoch(self, trainer: "BaseTrainer") -> None:
        """Invoked after the entire epoch cycle (train + val if scheduled) finishes."""
        self._log_train_actor_map(trainer)

    def _log_train_actor_map(self, trainer: "BaseTrainer") -> None:
        """Compute and display (train) mAP of the actor rounded to 6 decimal places."""
        if not self._train_pred_actors or not self._train_target_actors:
            return

        scores_arr = np.concatenate(self._train_pred_actors, axis=0)
        targets_arr = np.concatenate(self._train_target_actors, axis=0)
        self._train_pred_actors.clear()
        self._train_target_actors.clear()

        # Multi-GPU Distributed Synchronization if applicable
        trainer_dev = getattr(trainer, "device", None)
        scores_arr = all_gather_numpy(scores_arr, device=trainer_dev)
        targets_arr = all_gather_numpy(targets_arr, device=trainer_dev)

        try:
            from sklearn.metrics import average_precision_score
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mAP = average_precision_score(targets_arr, scores_arr.astype(np.float32))
                train_map = float(mAP)
        except Exception:
            train_map = 0.0

        # Store on trainer for downstream hooks (ExcelTracker, WandbHook)
        trainer.train_actor_map = train_map

        is_main = getattr(trainer, "is_main_process", True)
        if is_main:
            msg = f"(train) mAP of the actor: {train_map:.6f}"
            print(msg)
            if trainer.logger:
                trainer.logger.info(msg)

