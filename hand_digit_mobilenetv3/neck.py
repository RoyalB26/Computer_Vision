"""Standard Feature Pyramid Network (FPN) for MobileNetV3."""
from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from BaseModel.core.registrar import NECKS


def build_conv_bn_relu(in_c: int, out_c: int, kernel_size: int = 3, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


@NECKS.register_module(name="MobileNetFPN")
class MobileNetFPN(nn.Module):
    """
    FPN tổng hợp đặc trưng đa tầng từ MobileNetV3:
      - c0:  [B, 16, H//2, W//2]   (stride 2)
      - c1:  [B, 16, H//4, W//4]   (stride 4)
      - c3:  [B, 24, H//8, W//8]   (stride 8)
      - c5:  [B, 40, H//16, W//16] (stride 16)
      - c12: [B, 576, H//32, W//32] (stride 32)
      
    Đầu ra:
      - map_pred: [B, out_dim, H//8, W//8] (feature map chính tại Stride 8 để làm detection query)
      - feat_deform: [B, out_dim, H//2, W//2] (feature map chi tiết cao tại Stride 2 cho Deformable Sampling)
    """

    def __init__(
        self,
        in_channels: Dict[str, int],
        out_dim: int = 64,
        return_tuple: bool = True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.return_tuple = return_tuple

        # Lateral 1x1 convolutions
        self.lat_c12 = nn.Conv2d(in_channels.get('c12', 576), out_dim, kernel_size=1)
        self.lat_c5 = nn.Conv2d(in_channels.get('c5', 40), out_dim, kernel_size=1)
        self.lat_c3 = nn.Conv2d(in_channels.get('c3', 24), out_dim, kernel_size=1)
        self.lat_c1 = nn.Conv2d(in_channels.get('c1', 16), out_dim, kernel_size=1)
        self.lat_c0 = nn.Conv2d(in_channels.get('c0', 16), out_dim, kernel_size=1)

        # Smooth Convolutions
        self.smooth_pred = build_conv_bn_relu(out_dim, out_dim, kernel_size=3, padding=1)
        self.smooth_deform = build_conv_bn_relu(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(
        self, features: Dict[str, torch.Tensor]
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        c12 = features['c12']
        c5 = features['c5']
        c3 = features['c3']
        c1 = features.get('c1', c3)
        c0 = features.get('c0', c1)

        # Top-down pathway
        p5 = self.lat_c12(c12)
        p4 = self.lat_c5(c5) + F.interpolate(p5, size=c5.shape[-2:], mode="nearest")
        p3 = self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat_c1(c1) + F.interpolate(p3, size=c1.shape[-2:], mode="nearest")
        p1 = self.lat_c0(c0) + F.interpolate(p2, size=c0.shape[-2:], mode="nearest")

        map_pred = self.smooth_pred(p3)      # [B, out_dim, H//8, W//8] (Stride 8)
        feat_deform = self.smooth_deform(p1) # [B, out_dim, H//2, W//2] (Stride 2)

        if self.return_tuple:
            return map_pred, feat_deform
        return map_pred
