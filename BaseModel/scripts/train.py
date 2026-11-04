"""Unified training script for all Computer Vision tasks inheriting from BaseModel."""
import os
import sys
from typing import Any, Dict

# Ensure project root is in sys.path when running script directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from BaseModel.core import (
    BaseTrainer,
    HOOKS,
    OPTIMIZERS,
    SCHEDULERS,
    build_from_cfg,
)
from BaseModel.dataset import build_dataloader
from BaseModel.loss import build_loss
from BaseModel.model import build_model
from BaseModel.scripts.parser import parse_args
from BaseModel.utils import Config, get_device, get_logger, set_random_seed
from BaseModel.utils.metrics import build_metric


def main():
    args = parse_args(description="Train a Computer Vision model")

    # 1. Load configuration file
    cfg = Config.fromfile(args.config)

    # 2. Apply CLI overrides
    if args.cfg_options:
        cfg.merge_from_options(args.cfg_options)

    # Import custom modules if specified via CLI or config
    custom_imports = args.custom_imports or getattr(cfg, "custom_imports", None)
    if custom_imports:
        import importlib
        for mod in custom_imports:
            importlib.import_module(mod)

    # Overwrite from command line arguments if explicitly specified
    work_dir = args.work_dir or getattr(cfg, "work_dir", "./work_dirs/exp")
    device_name = args.device or getattr(cfg, "device", "cuda")
    device = get_device(device_name)
    seed = args.seed if args.seed is not None else getattr(cfg, "seed", 42)

    is_main_process = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0

    if is_main_process:
        os.makedirs(work_dir, exist_ok=True)
        # Dump the final configuration to work_dir for record and reproducibility
        cfg.dump(os.path.join(work_dir, "config_archived.yaml"))

    # 3. Setup Logger (only main process writes to console and train.log)
    logger = None
    if is_main_process:
        logger = get_logger(name="BaseModel.Train", log_file=os.path.join(work_dir, "train.log"))
        logger.info(f"Loaded config from: {args.config}")
        logger.info(f"Output directory: {work_dir}")
        logger.info("=" * 60)
        logger.info(f"EXPERIMENT CONFIGURATION:\n{cfg}")
        logger.info("=" * 60)

    def log_info(msg: str) -> None:
        if logger is not None:
            logger.info(msg)

    # 4. Set Seed
    deterministic = getattr(cfg, "deterministic", False)
    set_random_seed(seed=seed, deterministic=deterministic)
    log_info(f"Set random seed to {seed} (deterministic={deterministic})")

    # 5. Build Model
    log_info("Building model...")
    model = build_model(cfg.model)
    if hasattr(model, "init_weights") and getattr(cfg, "init_weights", True):
        model.init_weights()

    # Thống kê tham số mô hình
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    log_info("=" * 60)
    log_info("MODEL PARAMETERS:")
    log_info(f"Total Parameters     : {total_params:,} ({total_params / 1e6:.2f} M)")
    log_info(f"Trainable Parameters : {trainable_params:,} ({trainable_params / 1e6:.2f} M)")
    log_info(f"Frozen Parameters    : {frozen_params:,} ({frozen_params / 1e6:.2f} M)")
    log_info("=" * 60)

    # 6. Build Dataloaders
    log_info("Building dataloaders...")
    data_cfg = cfg.data
    train_loader = build_dataloader(
        dataset=data_cfg.train.dataset,
        batch_size=data_cfg.train.get("batch_size", 16),
        shuffle=data_cfg.train.get("shuffle", True),
        num_workers=data_cfg.train.get("num_workers", 4),
        pin_memory=data_cfg.train.get("pin_memory", True),
        drop_last=data_cfg.train.get("drop_last", True),
        collate_fn=data_cfg.train.get("collate_fn", None),
    )

    val_loader = None
    if "val" in data_cfg:
        val_loader = build_dataloader(
            dataset=data_cfg.val.dataset,
            batch_size=data_cfg.val.get("batch_size", 16),
            shuffle=False,
            num_workers=data_cfg.val.get("num_workers", 4),
            pin_memory=data_cfg.val.get("pin_memory", True),
            drop_last=False,
            collate_fn=data_cfg.val.get("collate_fn", None),
        )

    # Thống kê số lượng dữ liệu
    num_train_data = len(train_loader.dataset) if hasattr(train_loader, "dataset") else len(train_loader)
    num_val_data = len(val_loader.dataset) if (val_loader and hasattr(val_loader, "dataset")) else (len(val_loader) if val_loader else 0)

    log_info("=" * 60)
    log_info("DATASET STATISTICS:")
    log_info(f"Num Train Data       : {num_train_data:,} samples ({len(train_loader)} batches)")
    if val_loader:
        log_info(f"Num Val Data         : {num_val_data:,} samples ({len(val_loader)} batches)")
    else:
        log_info("Num Val Data         : 0 samples (No validation set configured)")
    log_info("=" * 60)

    # 7. Build Optimizer
    log_info("Building optimizer...")
    optimizer = build_from_cfg(cfg.optimizer, OPTIMIZERS, default_args={"params": model.parameters()})

    # 8. Build Learning Rate Scheduler (optional)
    lr_scheduler = None
    if "lr_scheduler" in cfg:
        log_info("Building learning rate scheduler...")
        lr_scheduler = build_from_cfg(cfg.lr_scheduler, SCHEDULERS, default_args={"optimizer": optimizer})

    # 9. Build Loss Function (optional, if computed separately outside the model)
    loss_fn = None
    if "loss" in cfg:
        log_info("Building loss function...")
        loss_fn = build_loss(cfg.loss)

    # 10. Build Evaluation Metrics (optional)
    metrics: Dict[str, Any] = {}
    if "metrics" in cfg:
        log_info("Building evaluation metrics...")
        for name, metric_cfg in cfg.metrics.items():
            metrics[name] = build_metric(metric_cfg)

    # 11. Build BaseTrainer Engine
    train_cfg = getattr(cfg, "train_cfg", {})
    max_epochs = train_cfg.get("max_epochs", 100)
    grad_clip = train_cfg.get("grad_clip", None)
    use_amp = train_cfg.get("use_amp", False)
    val_interval = train_cfg.get("val_interval", 1)
    log_interval = train_cfg.get("log_interval", 20)
    accumulate = args.accumulate if args.accumulate is not None else train_cfg.get("accumulate", 1)

    use_tqdm = args.tqdm or getattr(cfg, "tqdm", False)
    use_accelerate = args.accelerate or getattr(cfg, "accelerate", False)

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        metrics=metrics,
        max_epochs=max_epochs,
        work_dir=work_dir,
        device=device,
        grad_clip=grad_clip,
        use_amp=use_amp,
        val_interval=val_interval,
        log_interval=log_interval,
        accumulate=accumulate,
        logger=logger,
        use_tqdm=use_tqdm,
        use_accelerate=use_accelerate,
    )
    trainer.cfg_dict = cfg.to_dict()

    # 12. Register TensorBoard & WandB hooks if requested
    use_tensorboard = args.tensorboard or getattr(cfg, "tensorboard", False)
    if use_tensorboard and trainer.is_main_process:
        from BaseModel.core.hooks import TensorboardHook
        trainer.register_hook(TensorboardHook())

    use_wandb = args.wandb or getattr(cfg, "wandb", False)
    if use_wandb and trainer.is_main_process:
        from BaseModel.core.hooks import WandbHook
        project_name = getattr(cfg, "project_name", "BaseModel_CV")
        trainer.register_hook(WandbHook(project=project_name, config=cfg.to_dict()))

    # 13. Register Custom Hooks if configured in YAML
    if "hooks" in cfg:
        # If user explicitly specifies a LoggerHook, remove default LoggerHook first to prevent duplicates
        has_custom_logger = any(h.get("type") == "LoggerHook" for h in cfg.hooks)
        if has_custom_logger:
            from BaseModel.core.hooks import LoggerHook
            trainer.hooks = [h for h in trainer.hooks if not isinstance(h, LoggerHook)]

        for hook_cfg in cfg.hooks:
            hook_inst = build_from_cfg(hook_cfg, HOOKS)
            trainer.register_hook(hook_inst)

    # 14. Resume from checkpoint if specified
    resume_path = args.resume or getattr(cfg, "resume_from", None)
    finetune_path = getattr(args, "finetune", None) or getattr(cfg, "finetune_from", None)
    if resume_path:
        trainer.resume(resume_path)
    elif finetune_path:
        trainer.finetune(finetune_path)

    # 15. Start Training or Eval Only
    if args.eval_only:
        log_info("Executing evaluation only...")
        val_metrics = trainer.val_epoch()
        log_info(f"Final Evaluation Metrics: {val_metrics}")
    else:
        trainer.train()


if __name__ == "__main__":
    main()
