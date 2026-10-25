"""Base Metric Interface for Computer Vision evaluation."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Union
import torch


class BaseMetric(ABC):
    """Abstract Base Metric class.

    Metrics accumulate predictions and ground truth across multiple batches,
    and compute final values at the end of an epoch.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset accumulated stats at the start of an evaluation epoch."""
        pass

    @abstractmethod
    def update(self, preds: Any, targets: Any) -> None:
        """Update metric accumulator with batch predictions and ground truths."""
        pass

    @abstractmethod
    def compute(self) -> Union[float, Dict[str, float]]:
        """Compute the final metric score(s)."""
        pass
