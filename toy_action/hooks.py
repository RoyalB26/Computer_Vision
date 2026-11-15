"""Hooks specific to toy_action (e.g. training action mAP computation and logging).

Corresponds to and adapts action_slot/hooks.py for the toy_action dataset and models.
"""

from typing import TYPE_CHECKING, Any, Dict, List
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from BaseModel.core.hooks.base_hook import BaseHook
from BaseModel.core.registrar import HOOKS
from BaseModel.utils.device import all_gather_numpy

if TYPE_CHECKING:
    from BaseModel.core.trainer import BaseTrainer


@HOOKS.register_module(name="TrainToyActionMAPHook")
@HOOKS.register_module(name="ToyActionTrainHook")
@HOOKS.register_module(name="TrainActorMAPHook")
class TrainToyActionMAPHook(BaseHook):
    """Tracks training action predictions and ground truths to compute and log

    action mAP across the entire training epoch.
    """

    def __init__(self, log_name: str = "(train) mAP of toy actions"):
        self.log_name = log_name
        self._train_pred_actions: List[np.ndarray] = []
        self._train_target_actions: List[np.ndarray] = []

    def before_epoch(self, trainer: "BaseTrainer") -> None:
        self._train_pred_actions.clear()
        self._train_target_actions.clear()

    def after_train_iter(self, trainer: "BaseTrainer", batch_idx: int, outputs: Dict[str, Any]) -> None:
        pred_act = None
        for key in ("pred_actor", "preds", "logits", "actor"):
            if key in outputs and outputs[key] is not None:
                pred_act = outputs[key]
                break

        tgt_act = None
        for key in ("target_actor", "targets", "target"):
            if key in outputs and outputs[key] is not None:
                tgt_act = outputs[key]
                break

        if pred_act is not None and tgt_act is not None:
            if hasattr(pred_act, "detach"):
                # If logits, apply sigmoid
                if pred_act.min() < 0.0 or pred_act.max() > 1.0:
                    scores = torch.sigmoid(pred_act.detach().float()).contiguous().cpu().numpy()
                else:
                    scores = pred_act.detach().float().contiguous().cpu().numpy()
            else:
                scores = np.ascontiguousarray(pred_act, dtype=np.float32)

            if hasattr(tgt_act, "detach"):
                targets = tgt_act.detach().float().contiguous().cpu().numpy()
            else:
                targets = np.ascontiguousarray(tgt_act, dtype=np.float32)

            self._train_pred_actions.append(scores)
            self._train_target_actions.append(targets)

    def after_train_epoch(self, trainer: "BaseTrainer") -> None:
        self._log_train_action_map(trainer)

    def _log_train_action_map(self, trainer: "BaseTrainer") -> None:
        if not self._train_pred_actions or not self._train_target_actions:
            return

        scores_arr = np.concatenate(self._train_pred_actions, axis=0)
        targets_arr = np.concatenate(self._train_target_actions, axis=0)
        self._train_pred_actions.clear()
        self._train_target_actions.clear()

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

        # Store on trainer for downstream hooks
        trainer.train_action_map = train_map
        trainer.train_actor_map = train_map

        is_main = getattr(trainer, "is_main_process", True)
        if is_main:
            msg = f"{self.log_name}: {train_map:.6f}"
            print(msg)
            if trainer.logger:
                trainer.logger.info(msg)
