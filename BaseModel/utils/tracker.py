"""Excel metrics and gradient tracker module.

Logs training statistics, epoch overviews, gradient norms, and custom metrics
directly to a multi-sheet Microsoft Excel (.xlsx) file after each epoch.
Runs completely silent on the console/terminal.
"""
from collections import defaultdict
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

try:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

from ..core.hooks.base_hook import BaseHook
from ..core.registrar import HOOKS

if TYPE_CHECKING:
    from ..core.trainer import BaseTrainer


def compute_gradient_norms(
    model: nn.Module,
    submodules: Optional[Sequence[str]] = None,
    norm_type: float = 2.0,
) -> Dict[str, float]:
    """Calculate the global L2 gradient norm and per-submodule gradient norms.

    Args:
        model: PyTorch model with computed parameter gradients.
        submodules: Optional list of submodule names (e.g. ['backbone', 'neck', 'head']).
            If None, automatically inspects the direct named children of `model`.
        norm_type: Type of norm (default: 2.0 for standard Euclidean L2 norm).

    Returns:
        Dict mapping metric names to their gradient norm values, e.g.:
        {
            'Total_Grad_Norm': 12.824306,
            'grad_norm_backbone': 8.041145,
            'grad_norm_neck': 6.402350,
            'grad_norm_head': 7.668946,
        }
    """
    underlying = getattr(model, "module", model)
    norms: Dict[str, float] = {}

    # 1. Total Gradient Norm over all parameters with gradients
    all_params = [p for p in underlying.parameters() if p.grad is not None]
    if all_params:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in all_params]),
            norm_type,
        ).item()
    else:
        total_norm = 0.0
    norms["Total_Grad_Norm"] = float(total_norm)

    # 2. Per-submodule Gradient Norms
    if submodules is not None:
        for name in submodules:
            target_mod = getattr(underlying, name, None)
            if target_mod is None:
                # Search in named_modules if nested
                for mod_name, mod in underlying.named_modules():
                    if mod_name == name:
                        target_mod = mod
                        break

            if target_mod is not None:
                mod_params = [p for p in target_mod.parameters() if p.grad is not None]
                if mod_params:
                    mod_norm = torch.norm(
                        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in mod_params]),
                        norm_type,
                    ).item()
                else:
                    mod_norm = 0.0
                norms[f"grad_norm_{name}"] = float(mod_norm)
    else:
        # Auto-discover direct children of the model
        for child_name, child_mod in underlying.named_children():
            child_params = [p for p in child_mod.parameters() if p.grad is not None]
            if child_params:
                child_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), norm_type) for p in child_params]),
                    norm_type,
                ).item()
                norms[f"grad_norm_{child_name}"] = float(child_norm)

    return norms


@HOOKS.register_module()
class ExcelTracker(BaseHook):
    """Excel-based training and gradient statistics tracker.

    Tracks metrics across multiple sheets in an .xlsx file:
    1. 'Epoch_Overview': Epoch, Train_Loss, Val_Loss, Lr, sub-losses, val metrics, time, etc.
    2. 'Gradients_Detail': Epoch, Total_Grad_Norm, grad_norm_backbone, grad_norm_neck, etc.
    3. 'Detection': Detailed decomposition of feat_map into components (FPN, Slot, Deform_Local,
       Deform_Global) and their influence ratios on cls_logits and pred_offsets.
    4. Custom Sheets: Any additional sheets added by user.

    Features:
    - Completely silent: Does not print or emit logs to the terminal/console.
    - Updated after every epoch.
    - Modular & extensible: Supports custom metric extractors and custom sheets.
    - Resumes gracefully if file already exists.
    - Safe writing: Avoids crash if file is temporarily locked by Microsoft Excel.

    Args:
        filename: Name or path to the .xlsx file (default: 'training_metrics.xlsx').
            If relative, automatically placed into trainer's `work_dir`.
        submodules: List of specific submodule names to track gradients for.
            If None, automatically inspects model's top-level children.
        track_gradients: Whether to compute and record gradient norms (default: True).
        grad_interval: Frequency (in batches) to sample gradients (default: 1, every batch).
        track_detection: Whether to decompose detection features and analyze influence on
            cls_logits and pred_offsets in a dedicated 'Detection' sheet (default: True).
        detection_interval: Frequency (in batches) to sample detection components (default: 10).
        auto_save: Whether to automatically save after each epoch (default: True).
        resume: Whether to load existing sheets if the file exists (default: True).
    """

    def __init__(
        self,
        filename: str = "training_metrics.xlsx",
        submodules: Optional[List[str]] = None,
        track_gradients: bool = True,
        grad_interval: int = 1,
        track_detection: bool = True,
        detection_interval: int = 10,
        auto_save: bool = True,
        resume: bool = True,
    ):
        if not HAS_OPENPYXL:
            raise ImportError(
                "The 'openpyxl' package is required for ExcelTracker.\n"
                "Please install it using: pip install openpyxl"
            )

        self.filename = filename
        self.submodules = list(submodules) if submodules else None
        self.track_gradients = track_gradients
        self.grad_interval = max(1, int(grad_interval))
        self.track_detection = track_detection
        self.detection_interval = max(1, int(detection_interval))
        self.auto_save = auto_save
        self.resume = resume

        # Storage for records per sheet: {sheet_name: [row_dict, ...]}
        self.sheets_data: Dict[str, List[Dict[str, Any]]] = {
            "Epoch_Overview": [],
            "Gradients_Detail": [],
        }
        if self.track_detection:
            self.sheets_data["Detection"] = []

        # Registered custom metric callbacks: {sheet_name: {metric_name: callable(trainer)}}
        self.custom_metrics: Dict[str, Dict[str, Callable[[Any], Any]]] = defaultdict(dict)

        # Temporary epoch state
        self._epoch_losses: Dict[str, float] = defaultdict(float)
        self._epoch_loss_count = 0
        self._epoch_grad_norms: Dict[str, float] = defaultdict(float)
        self._epoch_grad_count = 0
        self._epoch_detection_metrics: Dict[str, float] = defaultdict(float)
        self._epoch_detection_count = 0
        self._epoch_start_time = 0.0
        self._last_val_metrics: Dict[str, float] = {}
        self.filepath: str = filename

    # =========================================================================
    # Modular Extensibility & Custom Metrics API
    # =========================================================================

    def add_sheet(self, sheet_name: str, initial_data: Optional[List[Dict[str, Any]]] = None) -> None:
        """Register a new custom sheet in the workbook."""
        if sheet_name not in self.sheets_data:
            self.sheets_data[sheet_name] = initial_data or []

    def register_metric(
        self,
        sheet_name: str,
        metric_name: str,
        callback: Callable[[Any], Any],
    ) -> "ExcelTracker":
        """Register a dynamic metric calculator for a specific sheet.

        Args:
            sheet_name: Target sheet (e.g. 'Epoch_Overview' or 'Gradients_Detail').
            metric_name: Column header in the sheet.
            callback: Function accepting `trainer` and returning a scalar value.

        Returns:
            self (for method chaining).
        """
        self.custom_metrics[sheet_name][metric_name] = callback
        if sheet_name not in self.sheets_data:
            self.add_sheet(sheet_name)
        return self

    def log_to_sheet(self, sheet_name: str, row_dict: Dict[str, Any]) -> None:
        """Manually append a row of metrics to a specified sheet."""
        if sheet_name not in self.sheets_data:
            self.add_sheet(sheet_name)
        self.sheets_data[sheet_name].append(row_dict.copy())

    def log_epoch_overview(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        lr: Optional[Union[float, List[float]]] = None,
        **extra_metrics,
    ) -> None:
        """Helper to manually log an entry to 'Epoch_Overview' without BaseTrainer."""
        row: Dict[str, Any] = {"Epoch": epoch}
        if train_loss is not None:
            row["Train_Loss"] = train_loss
        if val_loss is not None:
            row["Val_Loss"] = val_loss
        if lr is not None:
            row["Lr"] = lr[0] if isinstance(lr, (list, tuple)) else lr

        for k, v in extra_metrics.items():
            row[k] = v

        self.sheets_data["Epoch_Overview"].append(row)

    def log_gradients_detail(
        self,
        epoch: int,
        model: Optional[nn.Module] = None,
        total_grad_norm: Optional[float] = None,
        grad_norms: Optional[Dict[str, float]] = None,
        **extra_gradients,
    ) -> None:
        """Helper to manually log an entry to 'Gradients_Detail' without BaseTrainer."""
        row: Dict[str, Any] = {"Epoch": epoch}

        if model is not None:
            computed = compute_gradient_norms(model, submodules=self.submodules)
            row.update(computed)

        if total_grad_norm is not None:
            row["Total_Grad_Norm"] = total_grad_norm

        if grad_norms:
            row.update(grad_norms)

        for k, v in extra_gradients.items():
            row[k] = v

        self.sheets_data["Gradients_Detail"].append(row)

    def log_detection(
        self,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None,
        **extra_metrics,
    ) -> None:
        """Helper to manually log an entry to 'Detection' sheet without BaseTrainer."""
        row: Dict[str, Any] = {"Epoch": epoch}
        if metrics:
            row.update(metrics)
        for k, v in extra_metrics.items():
            row[k] = v
        if "Detection" not in self.sheets_data:
            self.sheets_data["Detection"] = []
        self.sheets_data["Detection"].append(row)

    # =========================================================================
    # Lifecycle Hook Methods (BaseHook Integration)
    # =========================================================================

    def before_train(self, trainer: "BaseTrainer") -> None:
        """Initialize target file path and load previous history if resuming."""
        if not getattr(trainer, "is_main_process", True):
            return

        if not os.path.isabs(self.filename):
            if os.path.dirname(self.filename) == "" and hasattr(trainer, "work_dir") and trainer.work_dir:
                self.filepath = os.path.join(trainer.work_dir, self.filename)
            else:
                self.filepath = os.path.abspath(self.filename)
        else:
            self.filepath = self.filename

        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)

        # Load existing file if resume is requested
        if self.resume and os.path.exists(self.filepath):
            self._load_existing_workbook(self.filepath)

    def before_epoch(self, trainer: "BaseTrainer") -> None:
        """Reset temporary accumulators at the beginning of each epoch."""
        if not getattr(trainer, "is_main_process", True):
            return

        self._epoch_losses.clear()
        self._epoch_loss_count = 0
        self._epoch_grad_norms.clear()
        self._epoch_grad_count = 0
        self._epoch_detection_metrics.clear()
        self._epoch_detection_count = 0
        self._epoch_start_time = time.time()
        self._last_val_metrics = {}

    def _sample_detection_from_batch(self, trainer: "BaseTrainer", data_batch: Any) -> None:
        """Sample detection component decomposition metrics from the model."""
        if not self.track_detection:
            return
        underlying = getattr(trainer.model, "module", trainer.model)
        if hasattr(underlying, "analyze_detection_components"):
            inputs = None
            if isinstance(data_batch, dict):
                inputs = data_batch.get("image", data_batch.get("inputs", None))
            elif isinstance(data_batch, (list, tuple)) and len(data_batch) > 0:
                inputs = data_batch[0]
            elif isinstance(data_batch, torch.Tensor):
                inputs = data_batch

            if inputs is not None and isinstance(inputs, torch.Tensor):
                sample = inputs[:min(4, len(inputs))]
                try:
                    stats = underlying.analyze_detection_components(sample)
                    for k, v in stats.items():
                        self._epoch_detection_metrics[k] += v
                    self._epoch_detection_count += 1
                except Exception:
                    pass

    def before_train_iter(self, trainer: "BaseTrainer", batch_idx: int, data_batch: Any) -> None:
        """Sample detection component influence at configured intervals during training."""
        if not getattr(trainer, "is_main_process", True):
            return

        if self.track_detection and ((batch_idx + 1) % self.detection_interval == 0 or batch_idx == 0):
            self._sample_detection_from_batch(trainer, data_batch)

    def before_val_iter(self, trainer: "BaseTrainer", batch_idx: int, data_batch: Any) -> None:
        """Sample detection component influence during validation if no train batches were sampled."""
        if not getattr(trainer, "is_main_process", True):
            return

        if self.track_detection and (self._epoch_detection_count == 0 or (batch_idx + 1) % self.detection_interval == 0):
            self._sample_detection_from_batch(trainer, data_batch)

    def after_train_iter(self, trainer: "BaseTrainer", batch_idx: int, outputs: Dict[str, Any]) -> None:
        """Accumulate train batch losses and sample gradient norms."""
        if not getattr(trainer, "is_main_process", True):
            return

        # 1. Accumulate losses
        for k, v in outputs.items():
            if "loss" in k.lower():
                val = v.item() if hasattr(v, "item") else float(v)
                self._epoch_losses[k] += val
        self._epoch_loss_count += 1

        # 2. Track Gradients if enabled
        if self.track_gradients and ((batch_idx + 1) % self.grad_interval == 0):
            norms = compute_gradient_norms(trainer.model, submodules=self.submodules)
            for k, v in norms.items():
                self._epoch_grad_norms[k] += v
            self._epoch_grad_count += 1

    def after_val(self, trainer: "BaseTrainer", val_metrics: Dict[str, float]) -> None:
        """Record validation metrics for the current epoch."""
        if not getattr(trainer, "is_main_process", True):
            return

        self._last_val_metrics = val_metrics.copy() if val_metrics else {}

        # Update Per_Class_Lastest sheet if provided by any metric
        epoch_idx = (getattr(trainer, "epoch", 0) or 0) + 1
        if hasattr(trainer, "metrics"):
            for m in trainer.metrics.values():
                if hasattr(m, "get_per_class_records"):
                    records = m.get_per_class_records(epoch=epoch_idx)
                    if records:
                        self.sheets_data["Per_Class_Lastest"] = records

    def after_epoch(self, trainer: "BaseTrainer") -> None:
        """Compile epoch data and update Excel workbook."""
        if not getattr(trainer, "is_main_process", True):
            return

        epoch_idx = trainer.epoch + 1

        # ---------------------------------------------------------
        # 1. Sheet: Epoch_Overview
        # ---------------------------------------------------------
        overview_row: Dict[str, Any] = {"Epoch": epoch_idx}

        # Train Losses
        avg_count = max(1, self._epoch_loss_count)
        if "loss" in self._epoch_losses:
            overview_row["Train_Loss"] = round(self._epoch_losses["loss"] / avg_count, 6)
        for k, v in self._epoch_losses.items():
            if k != "loss":
                overview_row[f"Train_{k}"] = round(v / avg_count, 6)

        # Train Actor mAP if tracked
        if hasattr(trainer, "train_actor_map") and trainer.train_actor_map is not None:
            overview_row["Train_Actor_mAP"] = round(float(trainer.train_actor_map), 6)

        # Validation Losses & Metrics
        if "val_loss" in self._last_val_metrics:
            overview_row["Val_Loss"] = round(float(self._last_val_metrics["val_loss"]), 6)
        for k, v in self._last_val_metrics.items():
            if k != "val_loss":
                overview_row[f"Val_{k}"] = round(float(v), 6) if isinstance(v, (int, float)) else v

        # Learning Rate
        current_lr = trainer.get_current_lr()
        if isinstance(current_lr, list):
            overview_row["Lr"] = float(current_lr[0])
        else:
            overview_row["Lr"] = float(current_lr)

        # Elapsed Time
        overview_row["Time_Seconds"] = round(time.time() - self._epoch_start_time, 2)

        # Execute registered custom metric callbacks for Epoch_Overview
        if "Epoch_Overview" in self.custom_metrics:
            for col_name, callback in self.custom_metrics["Epoch_Overview"].items():
                try:
                    overview_row[col_name] = callback(trainer)
                except Exception:
                    overview_row[col_name] = None

        self.sheets_data["Epoch_Overview"].append(overview_row)

        # ---------------------------------------------------------
        # 2. Sheet: Gradients_Detail
        # ---------------------------------------------------------
        grad_row: Dict[str, Any] = {"Epoch": epoch_idx}
        grad_count = max(1, self._epoch_grad_count)
        for k, v in self._epoch_grad_norms.items():
            grad_row[k] = round(v / grad_count, 6)

        # Execute registered custom metric callbacks for Gradients_Detail
        if "Gradients_Detail" in self.custom_metrics:
            for col_name, callback in self.custom_metrics["Gradients_Detail"].items():
                try:
                    grad_row[col_name] = callback(trainer)
                except Exception:
                    grad_row[col_name] = None

        self.sheets_data["Gradients_Detail"].append(grad_row)

        # ---------------------------------------------------------
        # 3. Sheet: Detection (Head Input Components & Output Impact)
        # ---------------------------------------------------------
        if self.track_detection and self._epoch_detection_count > 0:
            det_row: Dict[str, Any] = {"Epoch": epoch_idx}
            det_count = max(1, self._epoch_detection_count)
            for k, v in self._epoch_detection_metrics.items():
                if "Ratio" in k or "Impact" in k:
                    det_row[k] = round(v / det_count, 2)
                else:
                    det_row[k] = round(v / det_count, 5)

            # Execute registered custom metric callbacks for Detection
            if "Detection" in self.custom_metrics:
                for col_name, callback in self.custom_metrics["Detection"].items():
                    try:
                        det_row[col_name] = callback(trainer)
                    except Exception:
                        det_row[col_name] = None

            if "Detection" not in self.sheets_data:
                self.sheets_data["Detection"] = []
            self.sheets_data["Detection"].append(det_row)

        # ---------------------------------------------------------
        # 4. Custom Registered Sheets
        # ---------------------------------------------------------
        for sheet_name, metric_dict in self.custom_metrics.items():
            if sheet_name not in ["Epoch_Overview", "Gradients_Detail", "Detection", "Per_Class_Lastest"]:
                custom_row: Dict[str, Any] = {"Epoch": epoch_idx}
                for col_name, callback in metric_dict.items():
                    try:
                        custom_row[col_name] = callback(trainer)
                    except Exception:
                        custom_row[col_name] = None
                self.sheets_data[sheet_name].append(custom_row)

        # Update Per_Class_Lastest sheet if provided by any metric
        if hasattr(trainer, "metrics"):
            for m in trainer.metrics.values():
                if hasattr(m, "get_per_class_records"):
                    records = m.get_per_class_records(epoch=epoch_idx)
                    if records:
                        self.sheets_data["Per_Class_Lastest"] = records

        # 4. Save to Excel file
        if self.auto_save:
            self.save()

    def after_train(self, trainer: "BaseTrainer") -> None:
        """Final save upon training completion."""
        if not getattr(trainer, "is_main_process", True):
            return

        epoch_idx = (getattr(trainer, "epoch", 0) or 0) + 1
        if hasattr(trainer, "metrics"):
            for m in trainer.metrics.values():
                if hasattr(m, "get_per_class_records"):
                    records = m.get_per_class_records(epoch=epoch_idx)
                    if records:
                        self.sheets_data["Per_Class_Lastest"] = records

        self.save()

    # =========================================================================
    # Excel Serialization & Formatting
    # =========================================================================

    def save(self, filepath: Optional[str] = None) -> str:
        """Write all tracked sheets to the .xlsx file with professional styling."""
        target_path = filepath or self.filepath
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

        wb = openpyxl.Workbook()
        # Remove default sheet
        wb.remove(wb.active)

        # Styling constants
        header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
        header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
        regular_font = Font(name="Segoe UI", size=10)
        center_align = Alignment(horizontal="center", vertical="center")
        right_align = Alignment(horizontal="right", vertical="center")
        thin_border = Border(
            left=Side(style="thin", color="D9D9D9"),
            right=Side(style="thin", color="D9D9D9"),
            top=Side(style="thin", color="D9D9D9"),
            bottom=Side(style="thin", color="D9D9D9"),
        )

        for sheet_name, rows in self.sheets_data.items():
            ws = wb.create_sheet(title=sheet_name)
            ws.views.sheetView[0].showGridLines = True

            if not rows:
                ws.append(["Epoch"])
                continue

            # Collect unified column headers preserving insertion order
            headers: List[str] = []
            for r in rows:
                for k in r.keys():
                    if k not in headers:
                        headers.append(k)

            # 1. Write Header Row
            ws.append(headers)
            for col_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=1, column=col_idx)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = center_align

            # 2. Write Data Rows
            for row_dict in rows:
                row_vals = [row_dict.get(h, None) for h in headers]
                ws.append(row_vals)

                row_idx = ws.max_row
                for col_idx, val in enumerate(row_vals, 1):
                    cell = ws.cell(row=row_idx, column=col_idx)
                    cell.font = regular_font
                    cell.border = thin_border

                    if isinstance(val, int) or headers[col_idx - 1] == "Epoch":
                        cell.alignment = center_align
                        cell.number_format = "0"
                    elif isinstance(val, float):
                        cell.alignment = right_align
                        header_name = headers[col_idx - 1]
                        if header_name.lower() == "lr" or val < 1e-4:
                            cell.number_format = "0.000000"
                        elif "Ratio" in header_name or "Impact" in header_name:
                            cell.number_format = '0.00"%"'
                        else:
                            cell.number_format = "0.0000"
                    else:
                        cell.alignment = center_align

            # 3. Auto-fit column widths
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                col_letter = get_column_letter(col[0].column)
                ws.column_dimensions[col_letter].width = max(max_len + 4, 14)

        # Safe save (handles Windows file locking gracefully)
        try:
            wb.save(target_path)
        except PermissionError:
            # File might be temporarily open in MS Excel; write to backup copy
            backup_path = target_path.replace(".xlsx", "_backup.xlsx")
            try:
                wb.save(backup_path)
            except Exception:
                pass
        except Exception:
            pass
        finally:
            wb.close()

        return target_path

    def _load_existing_workbook(self, filepath: str) -> None:
        """Load rows from an existing Excel file to support experiment resume."""
        try:
            wb = openpyxl.load_workbook(filepath, data_only=True)
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                rows = list(ws.iter_rows(values_only=True))
                if len(rows) > 1:
                    headers = [str(h) for h in rows[0]]
                    sheet_records = []
                    for row_vals in rows[1:]:
                        row_dict = {headers[i]: row_vals[i] for i in range(min(len(headers), len(row_vals)))}
                        sheet_records.append(row_dict)
                    self.sheets_data[sheet_name] = sheet_records
            wb.close()
        except Exception:
            pass


# Alias for consistent naming
ExcelTrackerHook = ExcelTracker
HOOKS.register_module(module=ExcelTracker, name="ExcelTrackerHook")
