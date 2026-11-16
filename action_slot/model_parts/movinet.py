"""Pure-PyTorch MoViNet-A0 backbone implementation (matching Atze00/MoViNet-pytorch).

Provides non-causal (3D) and causal (2+1D streaming) MoViNet-A0 feature extraction
without external dependencies on ``fvcore`` or ``einops``.
Output feature map for input ``(B, 3, T, 224, 224)`` has shape ``(B, 480, T, 7, 7)``.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.utils import _pair, _triple

MOVINET_A0_WEIGHTS_URL = (
    "https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA0_statedict_v3"
)
MOVINET_A0_STREAM_WEIGHTS_URL = (
    "https://github.com/Atze00/MoViNet-pytorch/raw/main/weights/modelA0_stream_statedict_v3"
)


@dataclass(frozen=True)
class ConvLayerConfig:
    """Configuration for a standalone 3D convolution block."""

    input_channels: int
    out_channels: int
    kernel_size: Tuple[int, int, int]
    stride: Tuple[int, int, int]
    padding: Tuple[int, int, int]


@dataclass(frozen=True)
class SEBlockConfig:
    """Configuration for a MoViNet inverted bottleneck block with Squeeze-Excitation."""

    input_channels: int
    out_channels: int
    expanded_channels: int
    kernel_size: Tuple[int, int, int]
    stride: Tuple[int, int, int]
    padding: Tuple[int, int, int]
    padding_avg: Tuple[int, int, int]


MOVINET_A0_CONV1_CFG = ConvLayerConfig(
    input_channels=3,
    out_channels=8,
    kernel_size=(1, 3, 3),
    stride=(1, 2, 2),
    padding=(0, 1, 1),
)

MOVINET_A0_BLOCKS_CFG: Tuple[Tuple[SEBlockConfig, ...], ...] = (
    # Block 2 (b0) -> spatial stride 4 (56x56)
    (
        SEBlockConfig(8, 8, 24, (1, 5, 5), (1, 2, 2), (0, 2, 2), (0, 1, 1)),
    ),
    # Block 3 (b1) -> spatial stride 8 (28x28)
    (
        SEBlockConfig(8, 32, 80, (3, 3, 3), (1, 2, 2), (1, 0, 0), (0, 0, 0)),
        SEBlockConfig(32, 32, 80, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)),
        SEBlockConfig(32, 32, 80, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)),
    ),
    # Block 4 (b2) -> spatial stride 16 (14x14)
    (
        SEBlockConfig(32, 56, 184, (5, 3, 3), (1, 2, 2), (2, 0, 0), (0, 0, 0)),
        SEBlockConfig(56, 56, 112, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)),
        SEBlockConfig(56, 56, 184, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)),
    ),
    # Block 5 (b3) -> spatial stride 16 (14x14)
    (
        SEBlockConfig(56, 56, 184, (5, 3, 3), (1, 1, 1), (2, 1, 1), (0, 1, 1)),
        SEBlockConfig(56, 56, 184, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)),
        SEBlockConfig(56, 56, 184, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)),
        SEBlockConfig(56, 56, 184, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)),
    ),
    # Block 6 (b4) -> spatial stride 32 (7x7)
    (
        SEBlockConfig(56, 104, 384, (5, 3, 3), (1, 2, 2), (2, 1, 1), (0, 1, 1)),
        SEBlockConfig(104, 104, 280, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1)),
        SEBlockConfig(104, 104, 280, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1)),
        SEBlockConfig(104, 104, 344, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1)),
    ),
)

MOVINET_A0_CONV7_CFG = ConvLayerConfig(
    input_channels=104,
    out_channels=480,
    kernel_size=(1, 1, 1),
    stride=(1, 1, 1),
    padding=(0, 0, 0),
)


class Hardsigmoid(nn.Module):
    """TF-compatible Hardsigmoid activation used in MoViNet streaming SE layers."""

    def forward(self, x: Tensor) -> Tensor:
        return (0.2 * x + 0.5).clamp(min=0.0, max=1.0)


class Swish(nn.Module):
    """Swish (SiLU) activation matching state_dict/module structure of MoViNet."""

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class CausalModule(nn.Module):
    """Base module holding temporal stream buffer for causal MoViNet variants."""

    def __init__(self) -> None:
        super().__init__()
        self.activation: Optional[Tensor] = None

    def reset_activation(self) -> None:
        self.activation = None


class TemporalCGAvgPool3D(CausalModule):
    """Cumulative Global Average Pooling across time for causal streaming SE blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.n_cumulated_values = 0
        self.register_forward_hook(self._detach_activation)

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape
        device = x.device
        cumulative_sum = torch.cumsum(x, dim=2)
        if self.activation is None:
            self.activation = cumulative_sum[:, :, -1:].clone()
        else:
            cumulative_sum = cumulative_sum + self.activation
            self.activation = cumulative_sum[:, :, -1:].clone()
        divisor = torch.arange(
            1, input_shape[2] + 1, device=device, dtype=x.dtype
        )[None, None, :, None, None].expand_as(x)
        x = cumulative_sum / (self.n_cumulated_values + divisor)
        self.n_cumulated_values += input_shape[2]
        return x

    @staticmethod
    def _detach_activation(module: "TemporalCGAvgPool3D", _input: Any, _output: Any) -> None:
        if module.activation is not None:
            module.activation = module.activation.detach()

    def reset_activation(self) -> None:
        super().reset_activation()
        self.n_cumulated_values = 0


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Round channel count to a multiple of ``divisor`` matching TF MobileNet rules."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def same_padding(
    x: Tensor,
    in_height: int,
    in_width: int,
    stride_h: int,
    stride_w: int,
    filter_height: int,
    filter_width: int,
) -> Tensor:
    """Apply TensorFlow-style asymmetric SAME padding on H and W dimensions."""
    if in_height % stride_h == 0:
        pad_along_height = max(filter_height - stride_h, 0)
    else:
        pad_along_height = max(filter_height - (in_height % stride_h), 0)
    if in_width % stride_w == 0:
        pad_along_width = max(filter_width - stride_w, 0)
    else:
        pad_along_width = max(filter_width - (in_width % stride_w), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


class tfAvgPool3D(nn.Module):
    """TF-like spatial 3x3 AvgPool3d (stride 2) handling odd and even spatial sizes."""

    def __init__(self) -> None:
        super().__init__()
        self.avgf = nn.AvgPool3d((1, 3, 3), stride=(1, 2, 2))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != x.shape[-2]:
            raise RuntimeError("tfAvgPool3D only supports equal height and width")
        odd_size = x.shape[-1] % 2 != 0
        if odd_size:
            return F.avg_pool3d(
                x, (1, 3, 3), stride=(1, 2, 2), count_include_pad=False, padding=(0, 1, 1)
            )
        x = F.pad(x, (0, 1, 0, 1))
        x = self.avgf(x)
        x = x.clone()
        x[..., -1] = x[..., -1] * (9.0 / 6.0)
        x[..., -1, :] = x[..., -1, :] * (9.0 / 6.0)
        return x


class Conv2dBNActivation(nn.Sequential):
    """2D Conv + BatchNorm + Activation block using MoViNet state_dict key names."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        *,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        if norm_layer is None:
            norm_layer = nn.Identity
        if activation_layer is None:
            activation_layer = nn.Identity
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_planes

        layers = OrderedDict(
            {
                "conv2d": nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    **kwargs,
                ),
                "norm": norm_layer(out_planes, eps=0.001),
                "act": activation_layer(),
            }
        )
        super().__init__(layers)


class Conv3DBNActivation(nn.Sequential):
    """3D Conv + BatchNorm + Activation block using MoViNet state_dict key names."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        *,
        kernel_size: Union[int, Tuple[int, int, int]],
        padding: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        if norm_layer is None:
            norm_layer = nn.Identity
        if activation_layer is None:
            activation_layer = nn.Identity
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_planes

        layers = OrderedDict(
            {
                "conv3d": nn.Conv3d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    **kwargs,
                ),
                "norm": norm_layer(out_planes, eps=0.001),
                "act": activation_layer(),
            }
        )
        super().__init__(layers)


class ConvBlock3D(CausalModule):
    """Unified 3D or (2+1)D causal convolution block matching Atze00/MoViNet-pytorch."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        *,
        kernel_size: Union[int, Tuple[int, int, int]],
        tf_like: bool,
        causal: bool,
        conv_type: str,
        padding: Union[int, Tuple[int, int, int]] = 0,
        stride: Union[int, Tuple[int, int, int]] = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        bias: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        self.conv_2: Optional[nn.Module] = None

        if tf_like:
            if kernel_size[0] % 2 == 0:
                raise ValueError("tf_like supports only odd kernels for temporal dimension")
            padding = ((kernel_size[0] - 1) // 2, 0, 0)
            if stride[0] != 1:
                raise ValueError("tf_like supports only stride == 1 for temporal dimension")
            if stride[1] > kernel_size[1] or stride[2] > kernel_size[2]:
                raise ValueError("tf_like supports only stride <= kernel_size")

        if causal:
            padding = (0, padding[1], padding[2])
        if conv_type not in ("2plus1d", "3d"):
            raise ValueError("conv_type must be '2plus1d' or '3d'")

        if conv_type == "2plus1d":
            self.conv_1 = Conv2dBNActivation(
                in_planes,
                out_planes,
                kernel_size=(kernel_size[1], kernel_size[2]),
                padding=(padding[1], padding[2]),
                stride=(stride[1], stride[2]),
                activation_layer=activation_layer,
                norm_layer=norm_layer,
                bias=bias,
                **kwargs,
            )
            if kernel_size[0] > 1:
                self.conv_2 = Conv2dBNActivation(
                    in_planes,
                    out_planes,
                    kernel_size=(kernel_size[0], 1),
                    padding=(padding[0], 0),
                    stride=(stride[0], 1),
                    activation_layer=activation_layer,
                    norm_layer=norm_layer,
                    bias=bias,
                    **kwargs,
                )
        else:
            self.conv_1 = Conv3DBNActivation(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                padding=padding,
                activation_layer=activation_layer,
                norm_layer=norm_layer,
                stride=stride,
                bias=bias,
                **kwargs,
            )

        self.padding = padding
        self.kernel_size = kernel_size
        self.dim_pad = self.kernel_size[0] - 1
        self.stride = stride
        self.causal = causal
        self.conv_type = conv_type
        self.tf_like = tf_like

    def _cat_stream_buffer(self, x: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.activation is None or self.activation.shape[0] != x.shape[0] or self.activation.shape[3:] != x.shape[3:]:
            self.activation = torch.zeros(
                x.shape[0], x.shape[1], self.dim_pad, *x.shape[3:],
                device=device,
                dtype=dtype,
            )
        x = torch.cat((self.activation.to(device=device, dtype=dtype), x), dim=2)
        self.activation = x[:, :, -self.dim_pad:, ...].clone().detach()
        return x

    def _forward(self, x: Tensor) -> Tensor:
        device = x.device
        dtype = x.dtype
        if self.dim_pad > 0 and self.conv_2 is None and self.causal:
            x = self._cat_stream_buffer(x, device, dtype)

        t_buf = x.shape[2]
        if self.conv_type == "2plus1d":
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        x = self.conv_1(x)

        if self.conv_type == "2plus1d":
            bt, c_out, h_out, w_out = x.shape
            x = x.reshape(bt // t_buf, t_buf, c_out, h_out, w_out).permute(0, 2, 1, 3, 4)

            if self.conv_2 is not None:
                if self.dim_pad > 0 and self.causal:
                    x = self._cat_stream_buffer(x, device, dtype)
                b2, c2, t2, h2, w2 = x.shape
                x = x.reshape(b2, c2, t2, h2 * w2)
                x = self.conv_2(x)
                b3, c3, t3, hw3 = x.shape
                x = x.reshape(b3, c3, t3, hw3 // w2, w2)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if self.tf_like:
            x = same_padding(
                x,
                x.shape[-2],
                x.shape[-1],
                self.stride[-2],
                self.stride[-1],
                self.kernel_size[-2],
                self.kernel_size[-1],
            )
        return self._forward(x)


class SqueezeExcitation(nn.Module):
    """3D (or causal streaming) Squeeze-and-Excitation block."""

    def __init__(
        self,
        input_channels: int,
        activation_2: Callable[..., nn.Module],
        activation_1: Callable[..., nn.Module],
        conv_type: str,
        causal: bool,
        squeeze_factor: int = 4,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.causal = causal
        se_multiplier = 2 if causal else 1
        squeeze_channels = _make_divisible(input_channels // squeeze_factor * se_multiplier, 8)
        self.temporal_cumualtive_GAvg3D = TemporalCGAvgPool3D()
        self.fc1 = ConvBlock3D(
            input_channels * se_multiplier,
            squeeze_channels,
            kernel_size=(1, 1, 1),
            padding=0,
            tf_like=False,
            causal=causal,
            conv_type=conv_type,
            bias=bias,
        )
        self.activation_1 = activation_1()
        self.activation_2 = activation_2()
        self.fc2 = ConvBlock3D(
            squeeze_channels,
            input_channels,
            kernel_size=(1, 1, 1),
            padding=0,
            tf_like=False,
            causal=causal,
            conv_type=conv_type,
            bias=bias,
        )

    def _scale(self, x: Tensor) -> Tensor:
        if self.causal:
            x_space = torch.mean(x, dim=(3, 4), keepdim=True)
            scale = self.temporal_cumualtive_GAvg3D(x_space)
            scale = torch.cat((scale, x_space), dim=1)
        else:
            scale = F.adaptive_avg_pool3d(x, 1)
        scale = self.fc1(scale)
        scale = self.activation_1(scale)
        scale = self.fc2(scale)
        return self.activation_2(scale)

    def forward(self, x: Tensor) -> Tensor:
        return self._scale(x) * x


class BasicBneck(nn.Module):
    """MoViNet inverted residual bottleneck with depthwise conv, SE, and ReZero scaling."""

    def __init__(
        self,
        cfg: SEBlockConfig,
        causal: bool,
        tf_like: bool,
        conv_type: str,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if cfg.stride[0] != 1 or not (1 <= cfg.stride[1] <= 2) or not (1 <= cfg.stride[2] <= 2):
            raise ValueError(f"Illegal stride value: {cfg.stride}")

        self.res: Optional[nn.Module] = None
        self.expand: Optional[nn.Module] = None

        if cfg.expanded_channels != cfg.out_channels:
            self.expand = ConvBlock3D(
                in_planes=cfg.input_channels,
                out_planes=cfg.expanded_channels,
                kernel_size=(1, 1, 1),
                padding=(0, 0, 0),
                causal=causal,
                conv_type=conv_type,
                tf_like=tf_like,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )

        self.deep = ConvBlock3D(
            in_planes=cfg.expanded_channels,
            out_planes=cfg.expanded_channels,
            kernel_size=cfg.kernel_size,
            padding=cfg.padding,
            stride=cfg.stride,
            groups=cfg.expanded_channels,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.se = SqueezeExcitation(
            cfg.expanded_channels,
            causal=causal,
            activation_1=activation_layer if activation_layer is not None else Swish,
            activation_2=nn.Sigmoid if conv_type == "3d" else Hardsigmoid,
            conv_type=conv_type,
        )

        self.project = ConvBlock3D(
            cfg.expanded_channels,
            cfg.out_channels,
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0),
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=nn.Identity,
        )

        if not (cfg.stride == (1, 1, 1) and cfg.input_channels == cfg.out_channels):
            res_layers = []
            if cfg.stride != (1, 1, 1):
                if tf_like:
                    res_layers.append(tfAvgPool3D())
                else:
                    res_layers.append(
                        nn.AvgPool3d((1, 3, 3), stride=cfg.stride, padding=cfg.padding_avg)
                    )
            res_layers.append(
                ConvBlock3D(
                    in_planes=cfg.input_channels,
                    out_planes=cfg.out_channels,
                    kernel_size=(1, 1, 1),
                    padding=(0, 0, 0),
                    norm_layer=norm_layer,
                    activation_layer=nn.Identity,
                    causal=causal,
                    conv_type=conv_type,
                    tf_like=tf_like,
                )
            )
            self.res = nn.Sequential(*res_layers)

        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.res(x) if self.res is not None else x
        out = self.expand(x) if self.expand is not None else x
        out = self.deep(out)
        out = self.se(out)
        out = self.project(out)
        return residual + self.alpha * out


class MoViNetA0Features(nn.Module):
    """MoViNet-A0 backbone stages (`conv1`, `blocks`, `conv7`) matching official Kinetics-600 keys."""

    def __init__(
        self,
        causal: bool = False,
        conv_type: Optional[str] = None,
        tf_like: bool = True,
    ) -> None:
        super().__init__()
        if conv_type is None:
            conv_type = "2plus1d" if causal else "3d"

        self.causal = causal
        self.conv_type = conv_type
        self.tf_like = tf_like

        norm_layer = nn.BatchNorm3d if conv_type == "3d" else nn.BatchNorm2d
        activation_layer = Swish if conv_type == "3d" else nn.Hardswish

        self.conv1 = ConvBlock3D(
            in_planes=MOVINET_A0_CONV1_CFG.input_channels,
            out_planes=MOVINET_A0_CONV1_CFG.out_channels,
            kernel_size=MOVINET_A0_CONV1_CFG.kernel_size,
            stride=MOVINET_A0_CONV1_CFG.stride,
            padding=MOVINET_A0_CONV1_CFG.padding,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        blocks_dict: "OrderedDict[str, nn.Module]" = OrderedDict()
        self.stage_block_names: Tuple[Tuple[str, ...], ...] = tuple(
            tuple(f"b{i}_l{j}" for j in range(len(block_group)))
            for i, block_group in enumerate(MOVINET_A0_BLOCKS_CFG)
        )
        for i, block_group in enumerate(MOVINET_A0_BLOCKS_CFG):
            for j, block_cfg in enumerate(block_group):
                blocks_dict[f"b{i}_l{j}"] = BasicBneck(
                    block_cfg,
                    causal=causal,
                    conv_type=conv_type,
                    tf_like=tf_like,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
        self.blocks = nn.Sequential(blocks_dict)

        self.conv7 = ConvBlock3D(
            in_planes=MOVINET_A0_CONV7_CFG.input_channels,
            out_planes=MOVINET_A0_CONV7_CFG.out_channels,
            kernel_size=MOVINET_A0_CONV7_CFG.kernel_size,
            stride=MOVINET_A0_CONV7_CFG.stride,
            padding=MOVINET_A0_CONV7_CFG.padding,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.apply(self._weight_init)

    @staticmethod
    def _weight_init(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv3d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _reset_causal_module(m: nn.Module) -> None:
        if isinstance(m, CausalModule):
            m.reset_activation()

    def clean_activation_buffers(self) -> None:
        self.apply(self._reset_causal_module)

    def forward(self, x: Tensor) -> Tensor:
        if self.causal:
            self.clean_activation_buffers()
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv7(x)
        return x
