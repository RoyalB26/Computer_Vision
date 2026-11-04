"""Backbone MobileNetV3 for pure CNN feature extraction."""
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch 
import torch.nn as nn
from torch import Tensor
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
        self.norm = nn.Sequential(
            StaticSquaredDeviationNorm(),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.extractor(x)

class StaticSquaredDeviationNorm(nn.Module):
    def __init__(self, per_channel: bool = True):
        """
        Norm thuần túy (không học):
        Tính trực tiếp trên từng sample độc lập, không lưu running stats,
        kết quả không phụ thuộc vào trạng thái train/eval.
        
        Args:
            per_channel (bool): 
                - False (mặc định - theo code gốc): tính mean trên toàn bộ sample (C, H, W).
                - True: tính mean riêng cho từng channel của mỗi sample (H, W).
        """
        super().__init__()
        self.per_channel = per_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.per_channel:
            # Gom các chiều không gian (H, W, ...), giữ lại chiều Batch và Channel
            # dims = (2, 3, ...) -> mean shape: (B, C, 1, 1, ...)
            dims = tuple(range(2, x.ndim))
        else:
            # Gom tất cả các chiều trừ chiều Batch (dim 0) giống code gốc
            # dims = (1, 2, 3, ...) -> mean shape: (B, 1, 1, 1, ...)
            dims = tuple(range(1, x.ndim))
        
        # 1. Tính mean nội tại của từng sample
        mean = x.mean(dim=dims, keepdim=True)
        
        # 2. Áp dụng công thức biến đổi thuần túy
        out = torch.exp(5 * torch.abs(x - mean))
        
        return out


class Conv2dNormActivation(nn.Sequential):
    """Convolution + Normalization + Activation block matching torchvision."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
        bias: Optional[bool] = None,
        inplace: Optional[bool] = True,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels, eps=0.001, momentum=0.01))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))

        super().__init__(*layers)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for MobileNetV3."""

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Hardsigmoid,
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale


class InvertedResidual(nn.Module):
    """Inverted Residual block (MobileNetV3 style) with configurable activation."""

    def __init__(
        self,
        in_channels: int,
        exp_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        act_layer: Callable[..., nn.Module] = nn.ReLU,
        se_channels: Optional[int] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
    ):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers: List[nn.Module] = []

        # 1. Expand (pointwise conv)
        if exp_channels != in_channels:
            layers.append(
                Conv2dNormActivation(
                    in_channels,
                    exp_channels,
                    kernel_size=1,
                    stride=1,
                    norm_layer=norm_layer,
                    activation_layer=act_layer,
                )
            )

        # 2. Depthwise conv
        layers.append(
            Conv2dNormActivation(
                exp_channels,
                exp_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=exp_channels,
                norm_layer=norm_layer,
                activation_layer=act_layer,
            )
        )

        # 3. Squeeze-and-Excitation
        if use_se:
            sq_channels = se_channels if se_channels is not None else exp_channels // 4
            layers.append(
                SqueezeExcitation(
                    input_channels=exp_channels,
                    squeeze_channels=sq_channels,
                )
            )

        # 4. Project (pointwise linear conv without activation)
        layers.append(
            Conv2dNormActivation(
                exp_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.block(x)
        if self.use_res_connect:
            return x + res
        return res


class NormalBackbone(nn.Module):
    """
    MobileNetV3-Small Feature Extractor sử dụng hàm kích hoạt ReLU thay thế cho Hardswish:
      - features.0 : Conv2d(3 -> 16, k=3, s=2), ReLU                -> c0  (stride 2, 16 channels)
      - features.1 : InvertedResidual(16->16, k=3, s=2, SE(8), ReLU) -> c1  (stride 4, 16 channels)
      - features.2 : InvertedResidual(16->24, k=3, s=2, no SE, ReLU)       (stride 8, 24 channels)
      - features.3 : InvertedResidual(24->24, k=3, s=1, no SE, ReLU) -> c3  (stride 8, 24 channels)
      - features.4 : InvertedResidual(24->40, k=5, s=2, SE(24), ReLU)      (stride 16, 40 channels)
      - features.5 : InvertedResidual(40->40, k=5, s=1, SE(64), ReLU)-> c5  (stride 16, 40 channels)
      - features.6 : InvertedResidual(40->40, k=5, s=1, SE(64), ReLU)      (stride 16, 40 channels)
      - features.7 : InvertedResidual(40->48, k=5, s=1, SE(32), ReLU)      (stride 16, 48 channels)
      - features.8 : InvertedResidual(48->48, k=5, s=1, SE(40), ReLU)      (stride 16, 48 channels)
      - features.9 : InvertedResidual(48->96, k=5, s=2, SE(72), ReLU)      (stride 32, 96 channels)
      - features.10: InvertedResidual(96->96, k=5, s=1, SE(144), ReLU)     (stride 32, 96 channels)
      - features.11: InvertedResidual(96->96, k=5, s=1, SE(144), ReLU)     (stride 32, 96 channels)
      - features.12: Conv2d(96 -> 576, k=1, s=1), ReLU              -> c12 (stride 32, 576 channels)
    """

    def __init__(
        self,
        pretrained: bool = False,
        use_norm: bool = True,
        act_layer: Callable[..., nn.Module] = nn.ReLU,
        return_nodes: Optional[Dict[Union[int, str], str]] = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        if return_nodes is None:
            self.return_nodes = {
                0: 'c0',   # stride 2
                1: 'c1',   # stride 4
                3: 'c3',   # stride 8
                5: 'c5',   # stride 16
                12: 'c12', # stride 32
            }
        else:
            self.return_nodes = {
                int(k.split('.')[-1]) if isinstance(k, str) else k: v
                for k, v in return_nodes.items()
            }

        self.out_channels: Dict[str, int] = {
            'c0': 16,
            'c1': 16,
            'c3': 24,
            'c5': 40,
            'c12': 576,
        }

        if use_norm:
            self.norm = nn.Sequential(
                StaticSquaredDeviationNorm(per_channel=True),
                nn.BatchNorm2d(3),
                nn.ReLU(),
            )
        else:
            self.norm = None

        self.features = nn.Sequential(
            # 0: Conv 3 -> 16, stride 2 (c0)
            Conv2dNormActivation(3, 16, kernel_size=3, stride=2, activation_layer=act_layer),
            # 1: bneck k=3, in=16, exp=16, out=16, SE, RE, stride=2 (c1)
            InvertedResidual(16, 16, 16, kernel_size=3, stride=2, use_se=True, act_layer=act_layer, se_channels=8),
            # 2: bneck k=3, in=16, exp=72, out=24, no SE, RE, stride=2
            InvertedResidual(16, 72, 24, kernel_size=3, stride=2, use_se=False, act_layer=act_layer),
            # 3: bneck k=3, in=24, exp=88, out=24, no SE, RE, stride=1 (c3)
            InvertedResidual(24, 88, 24, kernel_size=3, stride=1, use_se=False, act_layer=act_layer),
            # 4: bneck k=5, in=24, exp=96, out=40, SE, RE, stride=2
            InvertedResidual(24, 96, 40, kernel_size=5, stride=2, use_se=True, act_layer=act_layer, se_channels=24),
            # 5: bneck k=5, in=40, exp=240, out=40, SE, RE, stride=1 (c5)
            InvertedResidual(40, 240, 40, kernel_size=5, stride=1, use_se=True, act_layer=act_layer, se_channels=64),
            # 6: bneck k=5, in=40, exp=240, out=40, SE, RE, stride=1
            InvertedResidual(40, 240, 40, kernel_size=5, stride=1, use_se=True, act_layer=act_layer, se_channels=64),
            # 7: bneck k=5, in=40, exp=120, out=48, SE, RE, stride=1
            InvertedResidual(40, 120, 48, kernel_size=5, stride=1, use_se=True, act_layer=act_layer, se_channels=32),
            # 8: bneck k=5, in=48, exp=144, out=48, SE, RE, stride=1
            InvertedResidual(48, 144, 48, kernel_size=5, stride=1, use_se=True, act_layer=act_layer, se_channels=40),
            # 9: bneck k=5, in=48, exp=288, out=96, SE, RE, stride=2
            InvertedResidual(48, 288, 96, kernel_size=5, stride=2, use_se=True, act_layer=act_layer, se_channels=72),
            # 10: bneck k=5, in=96, exp=576, out=96, SE, RE, stride=1
            InvertedResidual(96, 576, 96, kernel_size=5, stride=1, use_se=True, act_layer=act_layer, se_channels=144),
            # 11: bneck k=5, in=96, exp=576, out=96, SE, RE, stride=1
            InvertedResidual(96, 576, 96, kernel_size=5, stride=1, use_se=True, act_layer=act_layer, se_channels=144),
            # 12: Conv 96 -> 576, k=1, stride 1 (c12)
            Conv2dNormActivation(96, 576, kernel_size=1, stride=1, activation_layer=act_layer),
        )

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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.norm is not None:
            x = self.norm(x)

        feats: Dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.return_nodes:
                feats[self.return_nodes[idx]] = x
        return feats



@BACKBONES.register_module(name=["YOLOv11SmallBackbone", "YOLO11SmallBackbone", "YOLOv11sBackbone"], force=True)
class YOLOv11SmallBackbone(nn.Module):
    """
    YOLOv11-Small (YOLO11s) Feature Extractor:
      - c0:  stride 2  (32 channels)  - High-resolution spatial features
      - c1:  stride 4  (64 channels)  - Fine-grained object boundaries
      - c3:  stride 8  (128 channels) - Primary detection feature level (P3)
      - c5:  stride 16 (256 channels) - Medium object feature level (P4)
      - c10: stride 32 (512 channels) - Semantic context feature level (P5 / SPPF / C2PSA)
    """

    def __init__(
        self,
        pretrained: Union[bool, str] = True,
        weights_path: Optional[str] = "yolo11s.pt",
    ):
        super().__init__()
        from ultralytics import YOLO

        # Tải cấu trúc backbone chuẩn từ YOLO11s
        yolo_model = YOLO(weights_path if (pretrained and weights_path) else "yolo11s.pt")
        # 11 tầng đầu tiên cấu thành toàn bộ Backbone của YOLOv11
        self.stages = nn.ModuleList(list(yolo_model.model.model[:11]))

        # Mở khóa toàn bộ tham số (unfreeze) để cho phép huấn luyện (trainable)
        for param in self.stages.parameters():
            param.requires_grad = True

        if not pretrained:
            self._init_weights()

        self.out_channels: Dict[str, int] = {
            "c0": 32,   # Stage 0: Conv stride 2 (32 ch)
            "c1": 64,   # Stage 1: Conv stride 2 (64 ch)
            "c3": 128,  # Stage 3: Conv stride 2 (128 ch)
            "c5": 256,  # Stage 5: Conv stride 2 (256 ch)
            "c12": 512, # Stage 10: C2PSA stride 2 (512 ch) - tương thích khóa c12/P5
        }

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = {}
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx == 0:
                feats["c0"] = x   # Stride 2  [B, 32, H/2, W/2]
            elif idx == 1:
                feats["c1"] = x   # Stride 4  [B, 64, H/4, W/4]
            elif idx == 3:
                feats["c3"] = x   # Stride 8  [B, 128, H/8, W/8]
            elif idx == 5:
                feats["c5"] = x   # Stride 16 [B, 256, H/16, W/16]
            elif idx == 10:
                feats["c12"] = x  # Stride 32 [B, 512, H/32, W/32]
        return feats



@BACKBONES.register_module(name=["YOLOv11NanoBackbone", "YOLO11NanoBackbone", "YOLOv11nBackbone"], force=True)
class YOLOv11NanoBackbone(nn.Module):
    """
    YOLOv11-Nano (YOLO11n) Feature Extractor:
      - c0:  stride 2  (16 channels)  - High-resolution spatial features
      - c1:  stride 4  (32 channels)  - Fine-grained object boundaries
      - c3:  stride 8  (64 channels)  - Primary detection feature level (P3)
      - c5:  stride 16 (128 channels) - Medium object feature level (P4)
      - c12: stride 32 (256 channels) - Semantic context feature level (P5 / SPPF / C2PSA)
    """

    def __init__(
        self,
        pretrained: Union[bool, str] = True,
        weights_path: Optional[str] = "yolo11n.pt",
    ):
        super().__init__()
        from ultralytics import YOLO

        yolo_model = YOLO(weights_path if (pretrained and weights_path) else "yolo11n.pt")
        self.stages = nn.ModuleList(list(yolo_model.model.model[:11]))

        # Mở khóa toàn bộ tham số để cho phép huấn luyện (trainable)
        for param in self.stages.parameters():
            param.requires_grad = True

        if not pretrained:
            self._init_weights()

        self.out_channels: Dict[str, int] = {
            "c0": 16,   # Stage 0: Conv stride 2 (16 ch)
            "c1": 32,   # Stage 1: Conv stride 2 (32 ch)
            "c3": 64,   # Stage 3: Conv stride 2 (64 ch)
            "c5": 128,  # Stage 5: Conv stride 2 (128 ch)
            "c12": 256, # Stage 10: C2PSA stride 2 (256 ch)
        }

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = {}
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx == 0:
                feats["c0"] = x   # Stride 2  [B, 16, H/2, W/2]
            elif idx == 1:
                feats["c1"] = x   # Stride 4  [B, 32, H/4, W/4]
            elif idx == 3:
                feats["c3"] = x   # Stride 8  [B, 64, H/8, W/8]
            elif idx == 5:
                feats["c5"] = x   # Stride 16 [B, 128, H/16, W/16]
            elif idx == 10:
                feats["c12"] = x  # Stride 32 [B, 256, H/32, W/32]
        return feats
