"""Hooks specific to action_slot (e.g. training actor mAP computation and logging)."""
from typing import TYPE_CHECKING, Any, Dict, List
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from BaseModel.core.hooks.base_hook import BaseHook
from BaseModel.core.registrar import HOOKS
from BaseModel.utils.device import all_gather_numpy

if TYPE_CHECKING:
    from BaseModel.core.trainer import BaseTrainer


@HOOKS.register_module(name="TrainActorMAPHook")
@HOOKS.register_module(name="ActionSlotTrainActorHook")
class TrainActorMAPHook(BaseHook):
    """Tracks training actor predictions and ground truths to compute and log

    actor mAP across the entire training epoch.
    """

    def __init__(self, log_name: str = "(train) mAP of the actor"):
        self.log_name = log_name
        self._train_pred_actors: List[np.ndarray] = []
        self._train_target_actors: List[np.ndarray] = []

    def before_epoch(self, trainer: "BaseTrainer") -> None:
        self._train_pred_actors.clear()
        self._train_target_actors.clear()

    def after_train_iter(self, trainer: "BaseTrainer", batch_idx: int, outputs: Dict[str, Any]) -> None:
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

    def after_train_epoch(self, trainer: "BaseTrainer") -> None:
        self._log_train_actor_map(trainer)

    def _log_train_actor_map(self, trainer: "BaseTrainer") -> None:
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
            num_classes = scores_arr.shape[1]
            aps = []
            for c in range(num_classes):
                y_true = targets_arr[:, c]
                y_score = scores_arr[:, c]
                if np.sum(y_true == 1) > 0:
                    ap = float(average_precision_score(y_true, y_score))
                    aps.append(ap)

            train_map = float(np.mean(aps)) if aps else 0.0
        except Exception:
            train_map = 0.0

        # Store on trainer for downstream hooks (ExcelTracker, WandbHook)
        trainer.train_actor_map = train_map

        is_main = getattr(trainer, "is_main_process", True)
        if is_main:
            msg = f"{self.log_name}: {train_map:.6f}"
            print(msg)
            if trainer.logger:
                trainer.logger.info(msg)
