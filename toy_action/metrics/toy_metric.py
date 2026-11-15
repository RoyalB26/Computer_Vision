"""Toy Action multi-label action recognition evaluation metrics.

Computes class-wise Average Precision (AP), support, predicted positive count,
precision, recall, f1, confusion matrix values (tp, fp, fn, tn), and mAP
across overall classes and semantic sub-groups (x, y, o, T, 8, O, x+, y+, o+, T+, 8+, O+).
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from BaseModel.core.registrar import METRICS
from BaseModel.utils.device import all_gather_numpy
from BaseModel.utils.metrics.base_metric import BaseMetric
from .toy_labels import TOY_ACTION_LABELS, TOY_SYMBOLS, TOY_ACTIONS


@METRICS.register_module(name="ToyActionMetric")
@METRICS.register_module(name="ToyMetric")
@METRICS.register_module(name="ToyActionActionMetric")
class ToyActionMetric(BaseMetric):
    """Evaluation metric for Toy Action recognition.

    Computes:
      - Overall mAP (mean of AP across all 48 action-symbol classes)
      - Sub-group mAPs:
          mAP of x, y, o, T, 8, O (single objects)
          mAP of x+, y+, o+, T+, 8+, O+ (multiple objects interaction)
          mAP of gotop, gobottom, goleft, goright (directional actions)
          mAP_singles vs mAP_groups
      - Per-class AP arrays printed after each eval epoch.
      - Per-class detailed statistics table tracked for logs/Excel:
          Epoch, class_id, name, support, predicted_pos, ap, precision, recall, f1, tp, fp, fn, tn
    """

    def __init__(
        self,
        num_classes: int = len(TOY_ACTION_LABELS),
        class_names: Optional[List[str]] = None,
        threshold: float = 0.5,
        logger: Optional[Any] = None,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or TOY_ACTION_LABELS[:num_classes]
        self.threshold = threshold
        self.logger = logger
        self.best_map: Optional[float] = None
        self.epoch: int = 1
        self.trainer: Optional[Any] = None
        self.per_class_records: List[Dict[str, Any]] = []

        # Sub-group indices by symbol
        self.symbol_indices: Dict[str, List[int]] = {}
        for s in TOY_SYMBOLS:
            self.symbol_indices[s] = [
                i for i, n in enumerate(self.class_names) if n.startswith(f"{s}:")
            ]
            self.symbol_indices[f"{s}+"] = [
                i for i, n in enumerate(self.class_names) if n.startswith(f"{s}+:")
            ]

        # Singles vs Groups
        self.single_indices = [
            i for i, n in enumerate(self.class_names) if not n.split(":")[0].endswith("+")
        ]
        self.group_indices = [
            i for i, n in enumerate(self.class_names) if n.split(":")[0].endswith("+")
        ]

        # Directional actions
        self.action_indices: Dict[str, List[int]] = {}
        for act in TOY_ACTIONS:
            self.action_indices[act] = [
                i for i, n in enumerate(self.class_names) if n.endswith(f":{act}")
            ]

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
        """Return per-class records formatted for tracking."""
        if epoch is not None:
            for r in self.per_class_records:
                r["Epoch"] = int(epoch)
        return self.per_class_records

    def update(self, preds: Any, targets: Any) -> None:
        """Accumulate predictions and ground truths for a batch."""
        pred_tensor = None
        if isinstance(preds, dict):
            for k in ("preds", "pred_actor", "probs", "logits", "actor"):
                if k in preds and preds[k] is not None:
                    pred_tensor = preds[k]
                    break
        elif isinstance(preds, torch.Tensor):
            pred_tensor = preds
        else:
            return

        if pred_tensor is None:
            return

        target_tensor = None
        if isinstance(targets, dict):
            for k in ("actor", "target", "targets", "target_actor"):
                if k in targets and targets[k] is not None:
                    target_tensor = targets[k]
                    break
        elif isinstance(targets, torch.Tensor):
            target_tensor = targets
        elif isinstance(targets, (list, tuple)) and len(targets) > 0 and isinstance(targets[0], torch.Tensor):
            target_tensor = torch.stack(targets, dim=0)
        else:
            return

        if target_tensor is None:
            return

        # Ensure correct shapes: [B, num_classes]
        if pred_tensor.ndim != 2 or target_tensor.ndim != 2:
            pred_tensor = pred_tensor.reshape(-1, self.num_classes)
            target_tensor = target_tensor.reshape(-1, self.num_classes)

        # Convert logits to probabilities if not already in [0, 1]
        if pred_tensor.min() < 0.0 or pred_tensor.max() > 1.0:
            scores = torch.sigmoid(pred_tensor.detach().float()).contiguous().cpu().numpy()
        else:
            scores = pred_tensor.detach().float().contiguous().cpu().numpy()

        gts = target_tensor.detach().float().contiguous().cpu().numpy()

        self.all_scores.append(scores)
        self.all_targets.append(gts)

    def compute(self) -> Dict[str, float]:
        """Compute evaluation metrics, print terminal report, and prepare records."""
        if len(self.all_scores) == 0 or len(self.all_targets) == 0:
            return {
                "mAP": 0.0,
                "best_mAP": 0.0,
                "mAP_singles": 0.0,
                "mAP_groups": 0.0,
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

        # Mean AP across all classes
        current_map = float(np.mean(aps)) if aps else 0.0
        if self.best_map is None or current_map > self.best_map:
            self.best_map = current_map
        best_map = self.best_map

        # Sub-group mAPs (by symbol singles and groups)
        subgroup_maps: Dict[str, float] = {}
        for sym_name, indices in self.symbol_indices.items():
            subgroup_maps[f"mAP_{sym_name}"] = float(np.mean([aps[i] for i in indices])) if indices else 0.0

        # Singles vs Groups mAP
        map_singles = float(np.mean([aps[i] for i in self.single_indices])) if self.single_indices else 0.0
        map_groups = float(np.mean([aps[i] for i in self.group_indices])) if self.group_indices else 0.0

        # Action direction mAPs
        action_maps: Dict[str, float] = {}
        for act_name, indices in self.action_indices.items():
            action_maps[f"mAP_{act_name}"] = float(np.mean([aps[i] for i in indices])) if indices else 0.0

        # Terminal output formatting
        is_main = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_main = (torch.distributed.get_rank() == 0)

        if is_main:
            terminal_lines = [
                f"best mAP: {best_map:.4f}",
                f"mAP: {current_map:.4f}",
                f"mAP of singles: {map_singles:.4f} | mAP of groups (+): {map_groups:.4f}",
            ]
            for s in TOY_SYMBOLS:
                s_map = subgroup_maps.get(f"mAP_{s}", 0.0)
                sp_map = subgroup_maps.get(f"mAP_{s}+", 0.0)
                terminal_lines.append(f"mAP of {s}: {s_map:.4f} | mAP of {s}+: {sp_map:.4f}")

            terminal_lines.append("Action directions:")
            act_line = " | ".join(f"{act}: {action_maps.get(f'mAP_{act}', 0.0):.4f}" for act in TOY_ACTIONS)
            terminal_lines.append(act_line)

            terminal_text = "\n".join(terminal_lines)
            print(terminal_text)
            if self.logger:
                self.logger.info("\n" + terminal_text)

        # Direct synchronization with ExcelTracker in trainer hooks if available
        if self.trainer is not None:
            for hook in getattr(self.trainer, "hooks", []):
                if hasattr(hook, "sheets_data"):
                    hook.sheets_data["Per_Class_Lastest"] = records

        metrics_out = {
            "mAP": round(current_map, 4),
            "best_mAP": round(best_map, 4),
            "mAP_singles": round(map_singles, 4),
            "mAP_groups": round(map_groups, 4),
        }
        for k, v in subgroup_maps.items():
            metrics_out[k] = round(v, 4)
        for k, v in action_maps.items():
            metrics_out[k] = round(v, 4)

        return metrics_out
