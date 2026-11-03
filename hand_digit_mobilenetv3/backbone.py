"""Backbone MobileNetV3 for pure CNN feature extraction."""
from typing import Dict, Optional
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models.feature_extraction import create_feature_extractor

from BaseModel.core.registrar import BACKBONES


@BACKBONES.register_module(name="MobileNetV3Backbone")
class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3-Small Feature Extractor:
      - c0:  stride 2  (16 channels) - Chi tiết biên cạnh độ phân giải cao cho Deformable Head
      - c1:  stride 4  (16 channels)
      - c3:  stride 8  (24 channels)
      - c5:  stride 16 (40 channels)
      - c12: stride 32 (576 channels)
    """

    def __init__(
        self,
        pretrained: bool = True,
        return_nodes: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_backbone = mobilenet_v3_small(weights=weights)

        if return_nodes is None:
            return_nodes = {
                'features.0': 'c0',   # stride 2
                'features.1': 'c1',   # stride 4
                'features.3': 'c3',   # stride 8
                'features.5': 'c5',   # stride 16
                'features.12': 'c12', # stride 32
            }

        self.return_nodes = return_nodes
        self.extractor = create_feature_extractor(base_backbone, return_nodes=return_nodes)
        self.out_channels: Dict[str, int] = {
            'c0': 16,
            'c1': 16,
            'c3': 24,
            'c5': 40,
            'c12': 576,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.extractor(x)
