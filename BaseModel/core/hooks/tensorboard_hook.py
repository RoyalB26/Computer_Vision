"""TensorBoard logger hook for visualization."""
import os
from typing import TYPE_CHECKING, Any, Dict, Optional
from ..registrar import HOOKS
from .base_hook import BaseHook

if TYPE_CHECKING:
    from ..trainer import BaseTrainer


@HOOKS.register_module()
class TensorboardHook(BaseHook):
    """Log training losses, learning rates, and validation metrics to TensorBoard.

    Args:
        log_dir: Directory where TensorBoard event files are saved. Defaults to `<work_dir>/tf_logs`.
    """

    def __init__(self, log_dir: Optional[str] = None):
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError:
            raise ImportError(
                "TensorBoard is not installed or available.\n"
                "To use --tensorboard, please install it via:\n"
                "  pip install tensorboard"
            )
        self.log_dir = log_dir
        self.writer = None

    def before_train(self, trainer: "BaseTrainer") -> None:
        if not getattr(trainer, "is_main_process", True):
            return

        from torch.utils.tensorboard import SummaryWriter

        tb_dir = self.log_dir or os.path.join(trainer.work_dir, "tf_logs")
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_dir)

        if trainer.logger:
            trainer.logger.info(f"[TensorboardHook] Writing TensorBoard events to: {tb_dir}")

    def after_train_iter(self, trainer: "BaseTrainer", batch_idx: int, outputs: Dict[str, Any]) -> None:
        if not getattr(trainer, "is_main_process", True):
            return

        if self.writer is None:
            return

        step = trainer.iter
        # Log all training losses and custom training scalars returned by forward_train
        for k, v in outputs.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"train/{k}", float(v), global_step=step)
            elif hasattr(v, "item") and getattr(v, "ndim", 0) == 0:
                self.writer.add_scalar(f"train/{k}", float(v.item()), global_step=step)

        # Log learning rate
        lr = trainer.get_current_lr()
        if isinstance(lr, (int, float)):
            self.writer.add_scalar("train/lr", float(lr), global_step=step)
        elif isinstance(lr, list) and lr:
            self.writer.add_scalar("train/lr", float(lr[0]), global_step=step)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Manually log any custom scalar indicator."""
        if self.writer is not None:
            self.writer.add_scalar(tag, float(value), global_step=step)

    def log_image(self, tag: str, img_tensor: Any, step: int) -> None:
        """Log an image or image grid tensor to TensorBoard."""
        if self.writer is not None:
            self.writer.add_image(tag, img_tensor, global_step=step)

    def after_val(self, trainer: "BaseTrainer", val_metrics: Dict[str, float]) -> None:
        if not getattr(trainer, "is_main_process", True):
            return

        if self.writer is None:
            return

        epoch = trainer.epoch + 1
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"val/{k}", float(v), global_step=epoch)

    def after_train(self, trainer: "BaseTrainer") -> None:
        if not getattr(trainer, "is_main_process", True):
            return

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
