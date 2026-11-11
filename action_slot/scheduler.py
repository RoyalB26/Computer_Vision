"""Learning rate scheduler for Action-Slot / TACO.

Faithfully implements the original LambdaLR schedule from D:/v7/Action-slot/scripts/train_taco.py:
  - Epochs 0-10: Increases LR by 1.1x per epoch: lr = base_lr * (1.1 ** epoch)
  - Epochs >= 11: Decays LR by 0.7^(epoch // 3): lr = base_lr * (0.7 ** (epoch // 3))
"""
import math
from torch.optim.lr_scheduler import LambdaLR
from BaseModel.core.registrar import SCHEDULERS


def taco_lambda_lr(epoch: int) -> float:
    """Computes multiplier for learning rate according to TACO / Action-Slot schedule."""
    if epoch < 11:
        return math.pow(1.1, epoch)
    else:
        return math.pow(0.7, int(epoch / 3))


@SCHEDULERS.register_module(name="TACOLRScheduler")
@SCHEDULERS.register_module(name="ActionSlotLRScheduler")
@SCHEDULERS.register_module(name="TACOWarmupDecayLR")
class TACOLRScheduler(LambdaLR):
    """TACO multi-phase learning rate scheduler.

    Args:
        optimizer: Wrapped PyTorch optimizer.
        last_epoch: The index of last epoch. Default: -1.
        verbose: If True, prints a message to stdout for each update. Default: False.
    """

    def __init__(self, optimizer, last_epoch: int = -1, **kwargs):
        super().__init__(optimizer, lr_lambda=taco_lambda_lr, last_epoch=last_epoch)
