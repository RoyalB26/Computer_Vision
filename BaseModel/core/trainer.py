"""Training Engine implementing the Template Method Pattern.

Manages training lifecycle, validation, optimizer, mixed precision, and hooks.
"""
import os
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .hooks.base_hook import BaseHook
from .hooks.checkpoint_hook import CheckpointHook
from .hooks.logger_hook import LoggerHook
from .hooks.lr_scheduler_hook import LrSchedulerHook


class BaseTrainer:
    """Base Trainer engine for Computer Vision tasks.

    Subclasses can customize steps like `train_step` or `val_step` if special handling is needed,
    while inheriting the complete training lifecycle, hook notifications, and state management.

    Args:
        model: PyTorch model (typically inheriting BaseModel).
        optimizer: PyTorch optimizer.
        train_loader: Training DataLoader.
        val_loader: Optional validation DataLoader.
        lr_scheduler: Optional learning rate scheduler.
        loss_fn: Optional loss function (if not computed inside the model).
        metrics: Optional dictionary of evaluation metrics.
        max_epochs: Total number of training epochs.
        work_dir: Working directory to save checkpoints and logs.
        device: Target execution device ('cuda', 'cpu', 'mps', or torch.device).
        grad_clip: Optional max norm for gradient clipping.
        use_amp: Whether to use Automatic Mixed Precision (AMP).
        val_interval: Number of epochs between validation runs.
        logger: Optional logger instance.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr_scheduler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        metrics: Optional[Dict[str, Any]] = None,
        max_epochs: int = 100,
        work_dir: str = "./work_dirs",
        device: Union[str, torch.device] = "cuda",
        grad_clip: Optional[float] = None,
        use_amp: bool = False,
        val_interval: int = 1,
        logger: Optional[Any] = None,
        use_tqdm: bool = False,
        use_accelerate: bool = False,
        log_interval: int = 20,
        accumulate: int = 1,
    ):
        self.use_tqdm = use_tqdm
        self.use_accelerate = use_accelerate
        self.log_interval = log_interval
        self.accumulate = max(1, int(accumulate))
        self.accelerator = None

        if self.use_accelerate:
            try:
                from accelerate import Accelerator
            except ImportError:
                raise ImportError(
                    "Accelerate is not installed. To use --accelerate, please install it via:\n"
                    "  pip install accelerate"
                )
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
                model, optimizer, train_loader
            )

            self.val_loader = self.accelerator.prepare(val_loader) if val_loader is not None else None
            self.lr_scheduler = self.accelerator.prepare(lr_scheduler) if lr_scheduler is not None else None

            if not self.accelerator.is_main_process:
                self.logger = None
            else:
                self.logger = logger
        else:
            self.device = torch.device(device if isinstance(device, str) else device)
            self.model = model.to(self.device)
            self.optimizer = optimizer
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.lr_scheduler = lr_scheduler
            self.logger = logger

        self.is_tpu = (getattr(self.device, "type", "") == "xla") or ("xla" in str(self.device))
        self.loss_fn = loss_fn.to(self.device) if loss_fn is not None and hasattr(loss_fn, "to") else loss_fn
        self.metrics = metrics or {}
        self.max_epochs = max_epochs
        self.work_dir = work_dir
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.val_interval = val_interval

        # State tracking
        self.epoch = 0
        self.iter = 0
        self.should_stop = False
        self.last_val_metric: Optional[float] = None
        self.hooks: List[BaseHook] = []

        # AMP Scaler setup:
        if self.use_amp and not self.is_tpu and not self.use_accelerate and torch.cuda.is_available():
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            else:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None

        os.makedirs(self.work_dir, exist_ok=True)
        self._register_default_hooks()

    def _register_default_hooks(self) -> None:
        """Register essential hooks by default."""
        self.register_hook(LoggerHook(interval=self.log_interval))
        self.register_hook(CheckpointHook(interval=1, save_best="acc" if "acc" in self.metrics else None))
        if self.lr_scheduler is not None:
            self.register_hook(LrSchedulerHook(by_epoch=True))

    def register_hook(self, hook: BaseHook) -> None:
        """Register a new hook into the trainer."""
        if not isinstance(hook, BaseHook):
            raise TypeError(f"hook must be an instance of BaseHook, but got {type(hook)}")
        self.hooks.append(hook)

    def call_hook(self, fn_name: str, *args, **kwargs) -> None:
        """Invoke a specific hook method across all registered hooks."""
        for hook in self.hooks:
            method = getattr(hook, fn_name, None)
            if callable(method):
                method(self, *args, **kwargs)

    def get_current_lr(self) -> Union[float, List[float]]:
        """Retrieve current learning rate(s) from the optimizer."""
        lrs = [group["lr"] for group in self.optimizer.param_groups]
        return lrs[0] if len(lrs) == 1 else lrs

    def to_device(self, data: Any) -> Any:
        """Recursively move tensors or nested containers of tensors to the target device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            res = [self.to_device(v) for v in data]
            return type(data)(res)
        return data

    def should_validate(self, epoch: int) -> bool:
        """Check if validation should run on the given epoch."""
        if self.val_loader is None:
            return False
        return (epoch + 1) % self.val_interval == 0 or (epoch + 1) == self.max_epochs

    def train(self) -> None:
        """The Master Template Method orchestrating the full training process."""
        if self.logger:
            num_train = len(self.train_loader.dataset) if hasattr(self.train_loader, "dataset") else len(self.train_loader)
            num_val = len(self.val_loader.dataset) if (self.val_loader and hasattr(self.val_loader, "dataset")) else (len(self.val_loader) if self.val_loader else 0)
            self.logger.info(
                f"Starting training on device: {self.device} | Total Epochs: {self.max_epochs} | "
                f"Num Train Data: {num_train:,} | Num Val Data: {num_val:,}"
            )

        self.call_hook("before_train")

        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            self.call_hook("before_epoch")

            # Train one epoch
            self.train_epoch()

            # Run validation if scheduled
            if self.should_validate(epoch):
                val_metrics = self.val_epoch()
                self.last_val_metric = val_metrics.get("val_loss", None)

            self.call_hook("after_epoch")

            if self.should_stop:
                if self.logger:
                    self.logger.info("Training stopped early by hook.")
                break

        self.call_hook("after_train")
        if self.logger:
            self.logger.info("Training completed successfully.")

    def train_epoch(self) -> None:
        """Loop over all batches in the training dataloader."""
        self.model.train()

        loader = self.train_loader
        if self.use_tqdm:
            from tqdm import tqdm
            loader = tqdm(
                self.train_loader,
                desc=f"Epoch [{self.epoch + 1}/{self.max_epochs}] [Train]",
                leave=False,
            )

        for batch_idx, batch in enumerate(loader):
            self.iter += 1
            if not self.use_accelerate:
                batch = self.to_device(batch)
            self.call_hook("before_train_iter", batch_idx=batch_idx, data_batch=batch)

            is_last_batch = (batch_idx + 1 == len(self.train_loader))
            outputs = self.train_step(batch, batch_idx, is_last_batch=is_last_batch)

            if self.use_tqdm:
                total_loss = outputs.get("loss", 0.0)
                loss_val = total_loss.item() if hasattr(total_loss, "item") else float(total_loss)
                loader.set_postfix({"loss": f"{loss_val:.4f}"})

            self.call_hook("after_train_iter", batch_idx=batch_idx, outputs=outputs)

    def _get_autocast(self):
        """Helper to get compatible AMP autocast context manager across GPU, TPU, Accelerate, and CPU."""
        if self.use_accelerate:
            return self.accelerator.autocast()

        if self.is_tpu:
            # Google TPU bfloat16 autocast via torch_xla / PyTorch 2.x
            if hasattr(torch, "autocast"):
                try:
                    return torch.autocast(device_type="xla", dtype=torch.bfloat16, enabled=self.use_amp)
                except Exception:
                    pass
            import contextlib
            return contextlib.nullcontext()

        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            return torch.amp.autocast(device_type=device_type, enabled=self.use_amp)
        return torch.cuda.amp.autocast(enabled=self.use_amp)

    def train_step(self, batch: Any, batch_idx: int, is_last_batch: bool = False) -> Dict[str, Any]:
        """Single training step: forward, loss calculation, backward, and optimizer update.

        Supports gradient accumulation via `self.accumulate`.
        """
        # Only zero gradients at the start of an accumulation window
        if batch_idx % self.accumulate == 0:
            self.optimizer.zero_grad()

        with self._get_autocast():
            # Polymorphic input handling
            if isinstance(batch, dict):
                # Standard dictionary format: {'image': ..., 'target': ...} or keyword arguments
                if "image" in batch and "target" in batch:
                    inputs, targets = batch["image"], batch["target"]
                elif "inputs" in batch and "targets" in batch:
                    inputs, targets = batch["inputs"], batch["targets"]
                else:
                    inputs, targets = batch, None
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None

            # Model Forward
            if hasattr(self.model, "forward_train") and callable(self.model.forward_train):
                outputs = self.model.forward_train(inputs, targets)
            else:
                outputs = self.model(inputs)

            # Compute loss
            if isinstance(outputs, dict) and "loss" in outputs:
                total_loss = outputs["loss"]
                loss_dict = outputs
            elif isinstance(outputs, dict) and any("loss" in k for k in outputs):
                # Sum individual loss components
                loss_items = [v for k, v in outputs.items() if "loss" in k]
                total_loss = sum(loss_items)
                loss_dict = outputs.copy()
                loss_dict["loss"] = total_loss
            elif self.loss_fn is not None and targets is not None:
                total_loss = self.loss_fn(outputs, targets)
                loss_dict = {"loss": total_loss}
            elif isinstance(outputs, torch.Tensor) and outputs.ndim == 0:
                total_loss = outputs
                loss_dict = {"loss": total_loss}
            else:
                raise RuntimeError(
                    "Unable to determine training loss. Either model must return a loss dict/tensor, "
                    "or trainer.loss_fn must be provided."
                )

        # Scale loss down by accumulation steps
        scaled_loss = total_loss / self.accumulate

        should_step = ((batch_idx + 1) % self.accumulate == 0) or is_last_batch

        # Backward pass
        if self.use_accelerate:
            self.accelerator.backward(scaled_loss)
            if should_step:
                if self.grad_clip is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
        elif self.is_tpu:
            import torch_xla.core.xla_model as xm
            scaled_loss.backward()
            if should_step:
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                xm.optimizer_step(self.optimizer)
                xm.mark_step()
        elif self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
            if should_step:
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            scaled_loss.backward()
            if should_step:
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

        return loss_dict

    def val_epoch(self) -> Dict[str, float]:
        """Run validation over the entire val_loader."""
        self.model.eval()
        self.call_hook("before_val")

        # Reset all metrics
        for metric in self.metrics.values():
            if hasattr(metric, "reset"):
                metric.reset()

        val_losses: Dict[str, float] = {}
        val_samples = 0

        loader = self.val_loader
        if self.use_tqdm:
            from tqdm import tqdm
            loader = tqdm(
                self.val_loader,
                desc=f"Epoch [{self.epoch + 1}/{self.max_epochs}] [Val]",
                leave=False,
            )

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if not self.use_accelerate:
                    batch = self.to_device(batch)
                self.call_hook("before_val_iter", batch_idx=batch_idx, data_batch=batch)

                outputs = self.val_step(batch, batch_idx)

                # Accumulate loss
                if "loss" in outputs:
                    b_size = len(batch["image"]) if isinstance(batch, dict) and "image" in batch else 1
                    val_samples += b_size
                    for k, v in outputs.items():
                        if "loss" in k:
                            val = v.item() if hasattr(v, "item") else float(v)
                            val_losses[f"val_{k}"] = val_losses.get(f"val_{k}", 0.0) + val * b_size

                self.call_hook("after_val_iter", batch_idx=batch_idx, outputs=outputs)

        # Average losses
        metrics_summary = {}
        if val_samples > 0:
            for k, v in val_losses.items():
                metrics_summary[k] = v / val_samples

        # Compute evaluation metrics
        for name, metric in self.metrics.items():
            if hasattr(metric, "compute"):
                val = metric.compute()
                if isinstance(val, dict):
                    metrics_summary.update(val)
                else:
                    metrics_summary[name] = float(val)

        self.call_hook("after_val", val_metrics=metrics_summary)
        return metrics_summary

    def val_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Single validation step."""
        with self._get_autocast():
            if isinstance(batch, dict):
                if "image" in batch and "target" in batch:
                    inputs, targets = batch["image"], batch["target"]
                elif "inputs" in batch and "targets" in batch:
                    inputs, targets = batch["inputs"], batch["targets"]
                else:
                    inputs, targets = batch, None
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None

            if hasattr(self.model, "forward_val") and callable(self.model.forward_val):
                outputs = self.model.forward_val(inputs, targets)
            else:
                preds = self.model(inputs)
                outputs = {"preds": preds}

            # If model didn't calculate loss internally, compute it via trainer.loss_fn
            if "loss" not in outputs and self.loss_fn is not None and targets is not None:
                preds_for_loss = outputs.get("preds", outputs)
                outputs["loss"] = self.loss_fn(preds_for_loss, targets)

            # Update metrics
            preds = outputs.get("preds", outputs.get("pred", None))
            if preds is not None and targets is not None:
                for metric in self.metrics.values():
                    if hasattr(metric, "update"):
                        metric.update(preds, targets)

        return outputs

    def save_checkpoint(self, path: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """Save training state checkpoint."""
        # Unwrap model if Accelerate is used
        raw_model = self.accelerator.unwrap_model(self.model) if self.use_accelerate else self.model

        checkpoint = {
            "epoch": self.epoch,
            "iter": self.iter,
            "state_dict": raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()
        if meta:
            checkpoint["meta"] = meta

        if self.is_tpu:
            try:
                import torch_xla.core.xla_model as xm
                xm.save(checkpoint, path)
                return
            except (ImportError, Exception):
                pass

        torch.save(checkpoint, path)

    def resume(self, checkpoint_path: str) -> None:
        """Resume training state from a checkpoint file."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])

        if "optimizer" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.epoch = checkpoint.get("epoch", 0) + 1
        self.iter = checkpoint.get("iter", 0)

        if self.logger:
            self.logger.info(f"Resumed successfully from '{checkpoint_path}' at epoch {self.epoch}")
