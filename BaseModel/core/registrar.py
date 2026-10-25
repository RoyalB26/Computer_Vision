"""Global Registrars for all components in the Computer Vision framework."""
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from .registry import Registry

# Global Registries
MODELS = Registry("models")
BACKBONES = Registry("backbones")
NECKS = Registry("necks")
HEADS = Registry("heads")
LAYERS = Registry("layers")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
LOSSES = Registry("losses")
METRICS = Registry("metrics")
HOOKS = Registry("hooks")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
COLLATE_FUNCTIONS = Registry("collate_functions")


def register_torch_optimizers():
    """Register standard PyTorch optimizers into OPTIMIZERS registry."""
    torch_optimizers = [
        optim.SGD,
        optim.Adam,
        optim.AdamW,
        optim.RMSprop,
        optim.Adagrad,
        optim.Adadelta,
    ]
    for opt in torch_optimizers:
        if opt.__name__ not in OPTIMIZERS:
            OPTIMIZERS.register_module(module=opt)


def register_torch_schedulers():
    """Register standard PyTorch learning rate schedulers into SCHEDULERS registry."""
    torch_schedulers = [
        lr_scheduler.StepLR,
        lr_scheduler.MultiStepLR,
        lr_scheduler.ExponentialLR,
        lr_scheduler.CosineAnnealingLR,
        lr_scheduler.ReduceLROnPlateau,
        lr_scheduler.OneCycleLR,
        lr_scheduler.CosineAnnealingWarmRestarts,
    ]
    for sched in torch_schedulers:
        if sched.__name__ not in SCHEDULERS:
            SCHEDULERS.register_module(module=sched)


# Automatically register standard torch modules
register_torch_optimizers()
register_torch_schedulers()
