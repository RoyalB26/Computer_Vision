"""Full MobileNetV3 Small Architecture with weight loading support."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .backbone import MobileNetV3SmallBackbone
from .classifier import MobileNetV3Classifier
from .modules import Conv2dNormActivation, InvertedResidual, SqueezeExcitation


def _extract_state_dict(weights: Union[str, Path, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Helper to load state_dict from a file path or dict."""
    if isinstance(weights, dict):
        if "state_dict" in weights:
            return weights["state_dict"]
        elif "model" in weights:
            m = weights["model"]
            return m.state_dict() if hasattr(m, "state_dict") else m
        return weights

    path = Path(weights)
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        elif "model" in ckpt:
            m = ckpt["model"]
            return m.state_dict() if hasattr(m, "state_dict") else m
        return ckpt
    elif hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()
    raise ValueError(f"Unable to parse state_dict from {weights}")


class MobileNetV3Small(nn.Module):
    """MobileNetV3 Small Model Architecture.

    Composed of two decoupled, standalone modules:
      - backbone: MobileNetV3SmallBackbone (features: 0 to 12)
      - classifier: MobileNetV3Classifier (avgpool + linear layers)
    """

    def __init__(
        self,
        num_classes: int = 1000,
        dropout: float = 0.2,
        pretrained: bool = False,
        weights: Optional[Union[str, Path, Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = MobileNetV3SmallBackbone()
        self.classifier = MobileNetV3Classifier(
            in_features=576,
            mid_features=1024,
            num_classes=num_classes,
            dropout=dropout,
        )

        if pretrained or weights:
            self.load_weights(weights=weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.classifier(feat)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract high-level feature maps before classification."""
        return self.backbone(x)

    def load_weights(
        self,
        weights: Optional[Union[str, Path, Dict[str, Any]]] = None,
        strict: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """Load pretrained weights.
        
        If weights is None, automatically downloads and loads official torchvision MobileNet_V3_Small weights.
        """
        if weights is None:
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
            tv_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            raw_sd = tv_model.state_dict()
        else:
            raw_sd = _extract_state_dict(weights)

        # Map standard torchvision keys:
        # 'features.X' -> 'backbone.features.X'
        # 'classifier.X' -> 'classifier.classifier.X'
        new_sd = {}
        for k, v in raw_sd.items():
            if k.startswith("features."):
                new_key = "backbone." + k
            elif k.startswith("classifier."):
                new_key = "classifier." + k
            else:
                new_key = k
            new_sd[new_key] = v

        missing, unexpected = self.load_state_dict(new_sd, strict=strict)
        print(f"MobileNetV3Small loaded {len(new_sd)} weights successfully! (missing: {len(missing)}, unexpected: {len(unexpected)})")
        return missing, unexpected
