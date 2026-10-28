"""Feature extraction backbone for MobileNetV3 Small."""
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from .modules import Conv2dNormActivation, InvertedResidual


class MobileNetV3SmallBackbone(nn.Module):
    """MobileNetV3 Small Feature Extractor Backbone.

    Exact configuration matching torchvision / backbone_architecture/mobilenet_v3_small.txt:
      features.0 : Conv2d(3 -> 16, k=3, s=2), Hardswish
      features.1 : InvertedResidual(16 -> exp 16 -> 16, k=3, s=2, SE(8), RE)
      features.2 : InvertedResidual(16 -> exp 72 -> 24, k=3, s=2, no SE, RE)
      features.3 : InvertedResidual(24 -> exp 88 -> 24, k=3, s=1, no SE, RE)
      features.4 : InvertedResidual(24 -> exp 96 -> 40, k=5, s=2, SE(24), HS)
      features.5 : InvertedResidual(40 -> exp 240 -> 40, k=5, s=1, SE(64), HS)
      features.6 : InvertedResidual(40 -> exp 240 -> 40, k=5, s=1, SE(64), HS)
      features.7 : InvertedResidual(40 -> exp 120 -> 48, k=5, s=1, SE(32), HS)
      features.8 : InvertedResidual(48 -> exp 144 -> 48, k=5, s=1, SE(40), HS)
      features.9 : InvertedResidual(48 -> exp 288 -> 96, k=5, s=2, SE(72), HS)
      features.10: InvertedResidual(96 -> exp 576 -> 96, k=5, s=1, SE(144), HS)
      features.11: InvertedResidual(96 -> exp 576 -> 96, k=5, s=1, SE(144), HS)
      features.12: Conv2d(96 -> 576, k=1, s=1), Hardswish
    """

    def __init__(self):
        super().__init__()
        hs = nn.Hardswish
        re = nn.ReLU

        self.features = nn.Sequential(
            # 0: Conv 3 -> 16, stride 2
            Conv2dNormActivation(3, 16, kernel_size=3, stride=2, activation_layer=hs),
            # 1: bneck k=3, in=16, exp=16, out=16, SE, RE, stride=2
            InvertedResidual(16, 16, 16, kernel_size=3, stride=2, use_se=True, act_layer=re, se_channels=8),
            # 2: bneck k=3, in=16, exp=72, out=24, no SE, RE, stride=2
            InvertedResidual(16, 72, 24, kernel_size=3, stride=2, use_se=False, act_layer=re),
            # 3: bneck k=3, in=24, exp=88, out=24, no SE, RE, stride=1
            InvertedResidual(24, 88, 24, kernel_size=3, stride=1, use_se=False, act_layer=re),
            # 4: bneck k=5, in=24, exp=96, out=40, SE, HS, stride=2
            InvertedResidual(24, 96, 40, kernel_size=5, stride=2, use_se=True, act_layer=hs, se_channels=24),
            # 5: bneck k=5, in=40, exp=240, out=40, SE, HS, stride=1
            InvertedResidual(40, 240, 40, kernel_size=5, stride=1, use_se=True, act_layer=hs, se_channels=64),
            # 6: bneck k=5, in=40, exp=240, out=40, SE, HS, stride=1
            InvertedResidual(40, 240, 40, kernel_size=5, stride=1, use_se=True, act_layer=hs, se_channels=64),
            # 7: bneck k=5, in=40, exp=120, out=48, SE, HS, stride=1
            InvertedResidual(40, 120, 48, kernel_size=5, stride=1, use_se=True, act_layer=hs, se_channels=32),
            # 8: bneck k=5, in=48, exp=144, out=48, SE, HS, stride=1
            InvertedResidual(48, 144, 48, kernel_size=5, stride=1, use_se=True, act_layer=hs, se_channels=40),
            # 9: bneck k=5, in=48, exp=288, out=96, SE, HS, stride=2
            InvertedResidual(48, 288, 96, kernel_size=5, stride=2, use_se=True, act_layer=hs, se_channels=72),
            # 10: bneck k=5, in=96, exp=576, out=96, SE, HS, stride=1
            InvertedResidual(96, 576, 96, kernel_size=5, stride=1, use_se=True, act_layer=hs, se_channels=144),
            # 11: bneck k=5, in=96, exp=576, out=96, SE, HS, stride=1
            InvertedResidual(96, 576, 96, kernel_size=5, stride=1, use_se=True, act_layer=hs, se_channels=144),
            # 12: Conv 96 -> 576, k=1, stride 1
            Conv2dNormActivation(96, 576, kernel_size=1, stride=1, activation_layer=hs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through features to get the final feature map."""
        return self.features(x)

    def forward_multiscale(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale feature maps commonly used for detection heads.
        
        Returns features at layer 1 (stride 4), layer 3 (stride 8), layer 8 (stride 16), and layer 12 (stride 32).
        """
        feats = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 1:
                feats["stride_4"] = x    # [B, 16, H/4, W/4]
            elif i == 3:
                feats["stride_8"] = x    # [B, 24, H/8, W/8]
            elif i == 8:
                feats["stride_16"] = x   # [B, 48, H/16, W/16]
            elif i == 12:
                feats["stride_32"] = x   # [B, 576, H/32, W/32]
        return feats
