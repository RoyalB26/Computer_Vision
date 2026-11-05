"""Weights & Biases (WandB) experiment tracking hook."""
import os
from typing import TYPE_CHECKING, Any, Dict, Optional
from ..registrar import HOOKS
from .base_hook import BaseHook

if TYPE_CHECKING:
    from ..trainer import BaseTrainer


@HOOKS.register_module()
class WandbHook(BaseHook):
    """Log training stats, learning rates, and validation metrics to Weights & Biases (WandB).

    Args:
        project: Project name on wandb.ai.
        name: Run name. Defaults to the folder name of work_dir.
        config: Dictionary containing hyperparameters and configuration.
        entity: WandB entity (username or team name).
    """

    def __init__(
        self,
        project: str = "BaseModel_CV",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        **kwargs,
    ):
        try:
            import wandb  # noqa: F401
        except ImportError:
            raise ImportError(
                "wandb is not installed or available.\n"
                "To use --wandb, please install it via:\n"
                "  pip install wandb"
            )
        self.project = project
        self.name = name
        self.config = config
        self.entity = entity
        self.extra_kwargs = kwargs

    def before_train(self, trainer: "BaseTrainer") -> None:
        if not getattr(trainer, "is_main_process", True):
            return

        import wandb

        run_name = self.name or os.path.basename(os.path.normpath(trainer.work_dir))
        cfg_dict = self.config
        if cfg_dict is None and hasattr(trainer, "cfg_dict"):
            cfg_dict = trainer.cfg_dict

        wandb.init(
            project=self.project,
            name=run_name,
            config=cfg_dict,
            entity=self.entity,
            dir=trainer.work_dir,
            **self.extra_kwargs,
        )

        if trainer.logger:
            trainer.logger.info(f"[WandbHook] Initialized WandB run '{run_name}' in project '{self.project}'")

    def after_train_iter(self, trainer: "BaseTrainer", batch_idx: int, outputs: Dict[str, Any]) -> None:
        if not getattr(trainer, "is_main_process", True):
            return

        import wandb

        log_dict = {}
        # Log all training losses and custom training scalars returned by forward_train
        for k, v in outputs.items():
            if isinstance(v, (int, float)):
                log_dict[f"train/{k}"] = float(v)
            elif hasattr(v, "item") and getattr(v, "ndim", 0) == 0:
                log_dict[f"train/{k}"] = float(v.item())

        lr = trainer.get_current_lr()
        if isinstance(lr, (int, float)):
            log_dict["train/lr"] = float(lr)
        elif isinstance(lr, list) and lr:
            log_dict["train/lr"] = float(lr[0])

        log_dict["trainer/iter"] = trainer.iter
        wandb.log(log_dict, step=trainer.iter)

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """Manually log any custom scalar to WandB."""
        import wandb
        wandb.log({tag: float(value)}, step=step)

    def log_image(self, tag: str, image: Any, caption: Optional[str] = None, step: Optional[int] = None) -> None:
        """Log an image or visualization to WandB."""
        import wandb
        wandb.log({tag: wandb.Image(image, caption=caption)}, step=step)

    def after_val(self, trainer: "BaseTrainer", val_metrics: Dict[str, float]) -> None:
        if not getattr(trainer, "is_main_process", True):
            return

        import wandb

        log_dict = {f"val/{k}": float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))}
        log_dict["trainer/epoch"] = trainer.epoch + 1
        wandb.log(log_dict, step=trainer.iter)

    def after_train(self, trainer: "BaseTrainer") -> None:
        if not getattr(trainer, "is_main_process", True):
            return

        import wandb

        wandb.finish()
