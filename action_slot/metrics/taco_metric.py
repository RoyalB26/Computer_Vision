"""TACO multi-label action recognition evaluation metrics.

Computes class-wise Average Precision (AP), support, predicted positive count,
precision, recall, f1, confusion matrix values (tp, fp, fn, tn), and mAP
across overall classes and semantic sub-groups (c, b, p, c+, b+, p+).
"""
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from BaseModel.core.registrar import METRICS
from BaseModel.utils.device import all_gather_numpy
from BaseModel.utils.metrics.base_metric import BaseMetric
from .taco_labels import TACO_ACTOR_LABELS, TACO_EGO_LABELS


@METRICS.register_module(name="TACOActionMetric")
@METRICS.register_module(name="TACOMetric")
@METRICS.register_module(name="ActionSlotMetric")
class TACOActionMetric(BaseMetric):
    """Evaluation metric for TACO actor action recognition.

    Computes:
      - Overall mAP (mean of AP across all 64 actor action classes)
      - Group mAPs:
          mAP of c   (four-wheeler)
          mAP of b   (two-wheeler / bike)
          mAP of p   (pedestrian)
          mAP of c+  (four-wheeler interaction)
          mAP of b+  (two-wheeler interaction)
          mAP of p+  (pedestrian interaction)
      - Per-class AP arrays printed after each eval epoch.
      - Per-class detailed statistics table tracked in Excel sheet 'Per_Class_Lastest':
          Epoch, class_id, name, support, predicted_pos, ap, precision, recall, f1, tp, fp, fn, tn
    """

    def __init__(
        self,
        num_classes: int = 64,
        class_names: Optional[List[str]] = None,
        threshold: float = 0.5,
        logger: Optional[Any] = None,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or TACO_ACTOR_LABELS[:num_classes]
        self.threshold = threshold
        self.logger = logger
        self.best_map: Optional[float] = None
        self.epoch: int = 1
        self.trainer: Optional[Any] = None
        self.per_class_records: List[Dict[str, Any]] = []

        # Determine indices for semantic class groups from label prefixes
        self.c_idx = [i for i, n in enumerate(self.class_names) if n.startswith("c:")]
        self.b_idx = [i for i, n in enumerate(self.class_names) if n.startswith("b:")]
        self.p_idx = [i for i, n in enumerate(self.class_names) if n.startswith("p:")]
        self.c_plus_idx = [i for i, n in enumerate(self.class_names) if n.startswith("c+:")]
        self.b_plus_idx = [i for i, n in enumerate(self.class_names) if n.startswith("b+:")]
        self.p_plus_idx = [i for i, n in enumerate(self.class_names) if n.startswith("p+:")]

        self.reset()

    def reset(self) -> None:
        """Reset batch accumulators at start of evaluation."""
        self.all_scores: List[np.ndarray] = []
        self.all_targets: List[np.ndarray] = []

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch index."""
        self.epoch = int(epoch)

    def set_trainer(self, trainer: Any) -> None:
        """Attach trainer instance to allow logging and hook coordination."""
        self.trainer = trainer
        if hasattr(trainer, "logger") and trainer.logger is not None:
            self.logger = trainer.logger

    def get_per_class_records(self, epoch: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return per-class records formatted for Excel sheet Per_Class_Lastest."""
        if epoch is not None:
            for r in self.per_class_records:
                r["Epoch"] = int(epoch)
        return self.per_class_records

    def update(self, preds: Any, targets: Any) -> None:
        """Accumulate predictions and ground truths for a batch."""
        if isinstance(preds, dict):
            pred_actor = preds.get("actor", preds.get("pred_actor", None))
        elif isinstance(preds, torch.Tensor):
            pred_actor = preds
        else:
            return

        if pred_actor is None:
            return

        if isinstance(targets, (list, tuple)):
            target_actor = torch.stack([t["actor"] for t in targets], dim=0)
        elif isinstance(targets, dict):
            target_actor = targets.get("actor", None)
        elif isinstance(targets, torch.Tensor):
            target_actor = targets
        else:
            return

        if target_actor is None:
            return

        # Ensure correct shapes: [B, num_classes]
        if pred_actor.ndim != 2 or target_actor.ndim != 2:
            pred_actor = pred_actor.reshape(-1, self.num_classes)
            target_actor = target_actor.reshape(-1, self.num_classes)

        # Convert logits to probabilities via sigmoid (C-contiguous for fast concat & DDP)
        scores = torch.sigmoid(pred_actor.detach().float()).contiguous().cpu().numpy()
        gts = target_actor.detach().float().contiguous().cpu().numpy()

        self.all_scores.append(scores)
        self.all_targets.append(gts)

    def compute(self) -> Dict[str, float]:
        """Compute evaluation metrics, print terminal report, and prepare Excel rows."""
        if len(self.all_scores) == 0 or len(self.all_targets) == 0:
            return {
                "mAP": 0.0,
                "best_mAP": 0.0,
                "mAP_50": 0.0,
                "mAP_c": 0.0,
                "mAP_b": 0.0,
                "mAP_p": 0.0,
                "mAP_c_plus": 0.0,
                "mAP_b_plus": 0.0,
                "mAP_p_plus": 0.0,
            }

        scores_arr = np.concatenate(self.all_scores, axis=0)    # [N, num_classes]
        targets_arr = np.concatenate(self.all_targets, axis=0)  # [N, num_classes]

        # Multi-GPU Distributed Synchronization if applicable
        scores_arr = all_gather_numpy(scores_arr)
        targets_arr = all_gather_numpy(targets_arr)

        records: List[Dict[str, Any]] = []
        aps: List[float] = []

        for c in range(self.num_classes):
            y_true = targets_arr[:, c]
            y_score = scores_arr[:, c]

            support = int(np.sum(y_true == 1))
            pred_pos_mask = (y_score >= self.threshold)
            predicted_pos = int(np.sum(pred_pos_mask))

            tp = int(np.sum((y_true == 1) & pred_pos_mask))
            fp = int(np.sum((y_true == 0) & pred_pos_mask))
            fn = int(np.sum((y_true == 1) & (~pred_pos_mask)))
            tn = int(np.sum((y_true == 0) & (~pred_pos_mask)))

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            if support == 0:
                ap = 0.0
            else:
                try:
                    ap = float(average_precision_score(y_true, y_score))
                except Exception:
                    ap = 0.0

            aps.append(ap)
            records.append({
                "Epoch": int(self.epoch),
                "class_id": int(c),
                "name": self.class_names[c] if c < len(self.class_names) else f"class_{c}",
                "support": support,
                "predicted_pos": predicted_pos,
                "ap": round(ap, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            })

        self.per_class_records = records

        # Mean AP across all 64 classes
        current_map = float(np.mean(aps)) if aps else 0.0
        if self.best_map is None or current_map > self.best_map:
            self.best_map = current_map
        best_map = self.best_map

        # Sub-group mAPs
        map_c = float(np.mean([aps[i] for i in self.c_idx])) if self.c_idx else 0.0
        map_b = float(np.mean([aps[i] for i in self.b_idx])) if self.b_idx else 0.0
        map_p = float(np.mean([aps[i] for i in self.p_idx])) if self.p_idx else 0.0
        map_c_plus = float(np.mean([aps[i] for i in self.c_plus_idx])) if self.c_plus_idx else 0.0
        map_b_plus = float(np.mean([aps[i] for i in self.b_plus_idx])) if self.b_plus_idx else 0.0
        map_p_plus = float(np.mean([aps[i] for i in self.p_plus_idx])) if self.p_plus_idx else 0.0

        c_per_class_str = " ".join(f"{aps[i]:.4f}" for i in self.c_idx)
        b_per_class_str = " ".join(f"{aps[i]:.4f}" for i in self.b_idx)
        c_plus_per_class_str = " ".join(f"{aps[i]:.4f}" for i in self.c_plus_idx)
        b_plus_per_class_str = " ".join(f"{aps[i]:.4f}" for i in self.b_plus_idx)
        p_per_class_str = " ".join(f"{aps[i]:.4f}" for i in self.p_idx)
        p_plus_per_class_str = " ".join(f"{aps[i]:.4f}" for i in self.p_plus_idx)

        # Output to terminal on main process
        is_main = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_main = (torch.distributed.get_rank() == 0)

        if is_main:
            terminal_lines = [
                f"best mAP: {best_map:.4f}",
                f"mAP: {current_map:.4f}",
                f"mAP of c: {map_c:.4f}",
                f"mAP of b: {map_b:.4f}",
                f"mAP of p: {map_p:.4f}",
                f"mAP of c+: {map_c_plus:.4f}",
                f"mAP of b+: {map_b_plus:.4f}",
                f"mAP of p+: {map_p_plus:.4f}",
                "c per class:",
                c_per_class_str,
                "b per class:",
                b_per_class_str,
                "c+ per class:",
                c_plus_per_class_str,
                "b+ per class:",
                b_plus_per_class_str,
                "p per class:",
                p_per_class_str,
                "p+ per class:",
                p_plus_per_class_str,
            ]
            terminal_text = "\n".join(terminal_lines)
            print(terminal_text)
            if self.logger:
                self.logger.info("\n" + terminal_text)

        # Direct synchronization with ExcelTracker in trainer hooks
        if self.trainer is not None:
            for hook in getattr(self.trainer, "hooks", []):
                if hasattr(hook, "sheets_data"):
                    hook.sheets_data["Per_Class_Lastest"] = records

        return {
            "mAP": round(current_map, 4),
            "best_mAP": round(best_map, 4),
            "mAP_50": round(current_map, 4),
            "mAP_c": round(map_c, 4),
            "mAP_b": round(map_b, 4),
            "mAP_p": round(map_p, 4),
            "mAP_c_plus": round(map_c_plus, 4),
            "mAP_b_plus": round(map_b_plus, 4),
            "mAP_p_plus": round(map_p_plus, 4),
        }
