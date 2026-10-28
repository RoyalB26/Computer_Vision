"""Modular building blocks for MobileNetV3."""
from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            layers.append(activation_layer(inplace=True))

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
    """Inverted Residual block (MobileNetV3 style)."""

    def __init__(
        self,
        in_channels: int,
        exp_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        act_layer: Callable[..., nn.Module],
        se_channels: Optional[int] = None,
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
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.block(x)
        if self.use_res_connect:
            return x + res
        return res
