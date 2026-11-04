"""Standard Feature Pyramid Network (FPN) for MobileNetV3."""
from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from BaseModel.core.registrar import NECKS


def build_conv_bn_relu(
    in_c: int,
    out_c: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


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


@NECKS.register_module(name=["VPANet"])
class MobileNetPANet(nn.Module):
    """
    Path Aggregation Network (PANet) cho MobileNetV3 / NormalBackbone:
    Kết hợp 2 luồng truyền đặc trưng:
      1. Top-Down Pathway (FPN):
         Truyền thông tin ngữ nghĩa cấp cao từ tầng sâu về tầng nông.
         C12 (stride 32) -> P5
         C5  (stride 16) -> P4
         C3  (stride 8)  -> P3
         C1  (stride 4)  -> P2
         C0  (stride 2)  -> P1

      2. Bottom-Up Pathway (Path Augmentation):
         Truyền đặc trưng chi tiết không gian, biên cạnh từ tầng nông ngược lên tầng sâu.
         N1 = P1 (Stride 2)
         N2 = Conv3x3_s2(N1) + P2 -> Conv (Stride 4)
         N3 = Conv3x3_s2(N2) + P3 -> Conv (Stride 8 - Detection Feature Map)
         N4 = Conv3x3_s2(N3) + P4 -> Conv (Stride 16)
         N5 = Conv3x3_s2(N4) + P5 -> Conv (Stride 32)

    Đầu ra chuẩn tương thích hoàn toàn với HandDigitMobileNetV3Detector:
      - map_pred:    [B, out_dim, H//8, W//8] (Feature map chính Stride 8 từ N3)
      - feat_deform: [B, out_dim, H//2, W//2] (Feature map độ phân giải cao Stride 2 từ N1)
    """

    def __init__(
        self,
        in_channels: Dict[str, int],
        out_dim: int = 64,
        return_tuple: bool = True,
        fusion_type: str = "concat",  # "concat" hoặc "add"
    ):
        super().__init__()
        self.out_dim = out_dim
        self.return_tuple = return_tuple
        self.fusion_type = fusion_type.lower()

        # 1. Lateral 1x1 convolutions cho nhánh Top-Down
        self.lat_c12 = nn.Conv2d(in_channels.get('c12', 576), out_dim, kernel_size=1)
        self.lat_c5 = nn.Conv2d(in_channels.get('c5', 40), out_dim, kernel_size=1)
        self.lat_c3 = nn.Conv2d(in_channels.get('c3', 24), out_dim, kernel_size=1)
        self.lat_c1 = nn.Conv2d(in_channels.get('c1', 16), out_dim, kernel_size=1)
        self.lat_c0 = nn.Conv2d(in_channels.get('c0', 16), out_dim, kernel_size=1)

        # 2. Smooth Convolutions cho nhánh Top-Down (FPN)
        self.smooth_p4 = build_conv_bn_relu(out_dim, out_dim, kernel_size=3, padding=1)
        self.smooth_p3 = build_conv_bn_relu(out_dim, out_dim, kernel_size=3, padding=1)
        self.smooth_p2 = build_conv_bn_relu(out_dim, out_dim, kernel_size=3, padding=1)
        self.smooth_p1 = build_conv_bn_relu(out_dim, out_dim, kernel_size=3, padding=1)

        # 3. Bottom-Up Downsampling Convolutions (Stride 2)
        self.down_conv1 = build_conv_bn_relu(out_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.down_conv2 = build_conv_bn_relu(out_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.down_conv3 = build_conv_bn_relu(out_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.down_conv4 = build_conv_bn_relu(out_dim, out_dim, kernel_size=3, stride=2, padding=1)

        # 4. Bottom-Up Fusion Convolutions
        bu_in_channels = out_dim * 2 if self.fusion_type == "concat" else out_dim
        self.bu_conv2 = build_conv_bn_relu(bu_in_channels, out_dim, kernel_size=3, padding=1)
        self.bu_conv3 = build_conv_bn_relu(bu_in_channels, out_dim, kernel_size=3, padding=1)
        self.bu_conv4 = build_conv_bn_relu(bu_in_channels, out_dim, kernel_size=3, padding=1)
        self.bu_conv5 = build_conv_bn_relu(bu_in_channels, out_dim, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, features: Dict[str, torch.Tensor], return_all: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        c12 = features['c12']
        c5 = features['c5']
        c3 = features['c3']
        c1 = features.get('c1', c3)
        c0 = features.get('c0', c1)

        # --- 1. Top-Down Pathway (FPN) ---
        p5 = self.lat_c12(c12)
        p4 = self.smooth_p4(self.lat_c5(c5) + F.interpolate(p5, size=c5.shape[-2:], mode="nearest"))
        p3 = self.smooth_p3(self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest"))
        p2 = self.smooth_p2(self.lat_c1(c1) + F.interpolate(p3, size=c1.shape[-2:], mode="nearest"))
        p1 = self.smooth_p1(self.lat_c0(c0) + F.interpolate(p2, size=c0.shape[-2:], mode="nearest"))

        # --- 2. Bottom-Up Pathway (PANet Path Augmentation) ---
        n1 = p1  # Stride 2 [B, out_dim, H//2, W//2]

        d1 = self.down_conv1(n1)
        fuse2 = torch.cat([d1, p2], dim=1) if self.fusion_type == "concat" else (d1 + p2)
        n2 = self.bu_conv2(fuse2)  # Stride 4 [B, out_dim, H//4, W//4]

        d2 = self.down_conv2(n2)
        fuse3 = torch.cat([d2, p3], dim=1) if self.fusion_type == "concat" else (d2 + p3)
        n3 = self.bu_conv3(fuse3)  # Stride 8 [B, out_dim, H//8, W//8]

        d3 = self.down_conv3(n3)
        fuse4 = torch.cat([d3, p4], dim=1) if self.fusion_type == "concat" else (d3 + p4)
        n4 = self.bu_conv4(fuse4)  # Stride 16 [B, out_dim, H//16, W//16]

        d4 = self.down_conv4(n4)
        fuse5 = torch.cat([d4, p5], dim=1) if self.fusion_type == "concat" else (d4 + p5)
        n5 = self.bu_conv5(fuse5)  # Stride 32 [B, out_dim, H//32, W//32]

        if return_all:
            return {
                'n1': n1,
                'n2': n2,
                'n3': n3,
                'n4': n4,
                'n5': n5,
            }

        map_pred = n3      # [B, out_dim, H//8, W//8] (Feature map chính Stride 8)
        feat_deform = n1   # [B, out_dim, H//2, W//2] (Feature map độ phân giải cao Stride 2)

        if self.return_tuple:
            return map_pred, feat_deform
        return map_pred


# Alias
PANet = MobileNetPANet

