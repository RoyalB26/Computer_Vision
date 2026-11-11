"""Base Hook interface for training lifecycle events.

Implements the Observer / Hook pattern.
"""
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ..trainer import BaseTrainer


class BaseHook:
    """Base class for all training hooks.

    Hooks can intercept various stages of the training and evaluation lifecycle.
    """

    def before_train(self, trainer: "BaseTrainer") -> None:
        """Called before training starts."""
        pass

    def after_train(self, trainer: "BaseTrainer") -> None:
        """Called after training finishes."""
        pass

    def before_epoch(self, trainer: "BaseTrainer") -> None:
        """Called before each training epoch starts."""
        pass

    def after_epoch(self, trainer: "BaseTrainer") -> None:
        """Called after each training epoch finishes."""
        pass

    def after_train_epoch(self, trainer: "BaseTrainer") -> None:
        """Called immediately after training loop of an epoch finishes (before validation)."""
        pass

    def before_train_iter(self, trainer: "BaseTrainer", batch_idx: int, data_batch: Dict[str, Any]) -> None:
        """Called before each training iteration."""
        pass

    def after_train_iter(self, trainer: "BaseTrainer", batch_idx: int, outputs: Dict[str, Any]) -> None:
        """Called after each training iteration."""
        pass

    def before_val(self, trainer: "BaseTrainer") -> None:
        """Called before validation starts."""
        pass

    def after_val(self, trainer: "BaseTrainer", val_metrics: Dict[str, float]) -> None:
        """Called after validation completes."""
        pass

    def before_val_iter(self, trainer: "BaseTrainer", batch_idx: int, data_batch: Dict[str, Any]) -> None:
        """Called before each validation iteration."""
        pass

    def after_val_iter(self, trainer: "BaseTrainer", batch_idx: int, outputs: Dict[str, Any]) -> None:
        """Called after each validation iteration."""
        pass
