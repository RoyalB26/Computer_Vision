"""Base Model Interface for Computer Vision Architectures.

Provides unified forward contracts, polymorphic training/eval/inference pathways,
and weight management routines.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract Base Model for all Computer Vision architectures.

    This class enforces a clean, polymorphic contract between the model and the training engine.
    Every CV task (Classification, Detection, Segmentation, Pose, etc.) derives from this class.

    It supports three modes:
    1. 'train': Computes forward pass and returns a dictionary of losses.
    2. 'val': Computes forward pass, calculates evaluation metrics & validation loss.
    3. 'predict': Computes inference output with test-time post-processing.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        inputs: Any,
        targets: Optional[Any] = None,
        mode: str = "train",
        **kwargs,
    ) -> Union[Dict[str, Any], torch.Tensor, Any]:
        """Unified entry point for forward passes across different execution modes."""
        if mode == "train":
            return self.forward_train(inputs, targets, **kwargs)
        elif mode == "val":
            return self.forward_val(inputs, targets, **kwargs)
        elif mode in ["predict", "test"]:
            return self.predict(inputs, **kwargs)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'train', 'val', or 'predict'.")

    @abstractmethod
    def forward_train(self, inputs: Any, targets: Optional[Any] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass during training.

        Returns:
            Dictionary containing at least 'loss' or individual losses e.g. {'loss_cls': ..., 'loss_reg': ...}.
        """
        pass

    def forward_val(self, inputs: Any, targets: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """Forward pass during validation.

        Default implementation computes predictions and, if targets are provided, computes validation loss.
        """
        preds = self.predict(inputs, **kwargs)
        output = {"preds": preds}

        if targets is not None:
            # If model defines loss_fn or can compute loss during validation
            try:
                train_out = self.forward_train(inputs, targets, **kwargs)
                if isinstance(train_out, dict):
                    for k, v in train_out.items():
                        if "loss" in k.lower():
                            output[k] = v
                elif isinstance(train_out, torch.Tensor):
                    output["loss"] = train_out
            except Exception:
                pass

        return output

    @abstractmethod
    def predict(self, inputs: Any, **kwargs) -> Any:
        """Inference forward pass with post-processing (e.g. softmax/argmax, NMS)."""
        pass

    def init_weights(self) -> None:
        """Initialize weights of the model modules."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze(self, sub_module_name: Optional[str] = None) -> None:
        """Freeze parameters of the whole model or a specific submodule (e.g. 'backbone')."""
        target = getattr(self, sub_module_name) if sub_module_name else self
        for param in target.parameters():
            param.requires_grad = False
        target.eval()

    def unfreeze(self, sub_module_name: Optional[str] = None) -> None:
        """Unfreeze parameters of the whole model or a specific submodule."""
        target = getattr(self, sub_module_name) if sub_module_name else self
        for param in target.parameters():
            param.requires_grad = True
        target.train()

    def export_to_single_file(self, output_path: str) -> str:
        """Export this model and all its constituent layers into a standalone single Python file."""
        from .exporter import export_model_to_single_file
        return export_model_to_single_file(output_path=output_path, target_models=[self.__class__.__name__])
