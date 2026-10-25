"""Base Loss Class and Loss Aggregator."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from ..core.registrar import LOSSES
from ..core.registry import build_from_cfg


class BaseLoss(nn.Module, ABC):
    """Abstract Base Class for loss functions.

    Args:
        loss_weight: Multiplier weight for this loss component in composite loss calculations.
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    @abstractmethod
    def forward(self, preds: Any, targets: Any, **kwargs) -> torch.Tensor:
        """Compute the unweighted loss tensor."""
        pass


@LOSSES.register_module()
class LossAggregator(nn.Module):
    """Composite loss manager combining multiple weighted loss functions.

    Implements the Composite Pattern for multi-task / multi-loss objectives.

    Args:
        losses: Dictionary mapping loss names to loss modules or config dictionaries.
            Example:
                {
                    'cls_loss': {'type': 'CrossEntropyLoss', 'loss_weight': 1.0},
                    'reg_loss': {'type': 'SmoothL1Loss', 'loss_weight': 0.5}
                }
    """

    def __init__(self, losses: Dict[str, Union[nn.Module, Dict[str, Any]]]):
        super().__init__()
        self.loss_modules = nn.ModuleDict()

        for name, loss_item in losses.items():
            if isinstance(loss_item, dict):
                loss_mod = build_from_cfg(loss_item, LOSSES)
            elif isinstance(loss_item, nn.Module):
                loss_mod = loss_item
            else:
                raise TypeError(f"Loss item must be dict or nn.Module, got {type(loss_item)}")
            self.loss_modules[name] = loss_mod

    def forward(
        self,
        preds: Union[torch.Tensor, Dict[str, Any]],
        targets: Union[torch.Tensor, Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute all component losses and aggregate their weighted sum."""
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device if len(list(self.parameters())) > 0 else "cpu")

        for name, loss_mod in self.loss_modules.items():
            # If preds is a dict, match key if exists; otherwise pass entire preds
            p = preds[name] if isinstance(preds, dict) and name in preds else preds
            t = targets[name] if isinstance(targets, dict) and name in targets else targets

            loss_val = loss_mod(p, t, **kwargs)
            weight = getattr(loss_mod, "loss_weight", 1.0)
            weighted_val = loss_val * weight

            loss_dict[name] = weighted_val
            total_loss = total_loss + weighted_val

        loss_dict["loss"] = total_loss
        return loss_dict
