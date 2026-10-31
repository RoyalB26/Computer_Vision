"""Modular building blocks and submodules for assembling Computer Vision architectures."""
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.registrar import (
    BACKBONES,
    HEADS,
    LAYERS,
    MODELS,
    NECKS,
)
from ..core.registry import build_from_cfg
from .base_model import BaseModel


# =========================================================================
# 1. CORE LAYERS: ConvModule, ResidualBlock, DepthwiseSeparableConv
# =========================================================================

@LAYERS.register_module()
class ConvModule(nn.Module):
    """Convolution + Normalization + Activation block.

    Args:
        in_channels: Số channel đầu vào.
        out_channels: Số channel đầu ra.
        kernel_size: Kích thước kernel.
        stride: Bước trượt (stride).
        padding: Padding (None để tự tính 'same').
        dilation: Dilation rate.
        groups: Số nhóm convolution.
        norm_cfg: Cấu hình chuẩn hóa (e.g. 'bn', 'gn', None).
        act_cfg: Cấu hình hàm kích hoạt ('relu', 'leaky_relu', 'silu', 'gelu', None).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        norm_cfg: Optional[str] = "bn",
        act_cfg: Optional[str] = "relu",
    ):
        super().__init__()
        if padding is None:
            padding = ((kernel_size - 1) * dilation) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=(norm_cfg is None),
        )

        # Normalization
        if norm_cfg == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_cfg == "gn":
            self.norm = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        else:
            self.norm = nn.Identity()

        # Activation
        if act_cfg == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act_cfg == "leaky_relu":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif act_cfg == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act_cfg == "gelu":
            self.act = nn.GELU()
        elif act_cfg == "Hardswish":
            self.act = nn.Hardswish()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


@LAYERS.register_module()
class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution block (MobileNet style)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = ConvModule(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels)
        self.pointwise = ConvModule(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


@LAYERS.register_module()
class ResidualBlock(nn.Module):
    """Residual Block with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvModule(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = ConvModule(out_channels, out_channels, kernel_size=3, stride=1, act_cfg=None)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvModule(in_channels, out_channels, kernel_size=1, stride=stride, act_cfg=None)
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv2(self.conv1(x))
        return self.relu(out + residual)


# =========================================================================
# 2. BACKBONES: SimpleCNNBackbone, ResNetBackbone, MobileNetBackbone
# =========================================================================

@BACKBONES.register_module()
class SimpleCNNBackbone(nn.Module):
    """Configurable multi-stage CNN backbone."""

    def __init__(
        self,
        in_channels: int = 3,
        stage_channels: Tuple[int, ...] = (32, 64, 128),
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        curr_c = in_channels
        self.out_channels = []

        for out_c in stage_channels:
            stage = nn.Sequential(
                ConvModule(curr_c, out_c, kernel_size=3, stride=2),
                ResidualBlock(out_c, out_c),
            )
            self.stages.append(stage)
            curr_c = out_c
            self.out_channels.append(out_c)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features  # Trả về multi-scale feature maps


@BACKBONES.register_module()
class ResNetBackbone(nn.Module):
    """Pretrained or Scratch ResNet Backbone via torchvision."""

    def __init__(self, depth: int = 18, pretrained: bool = False, frozen_stages: int = -1):
        super().__init__()
        import torchvision.models as tvm

        weights_dict = {
            18: tvm.ResNet18_Weights.DEFAULT if pretrained else None,
            34: tvm.ResNet34_Weights.DEFAULT if pretrained else None,
            50: tvm.ResNet50_Weights.DEFAULT if pretrained else None,
        }
        fn_dict = {18: tvm.resnet18, 34: tvm.resnet34, 50: tvm.resnet50}

        model = fn_dict[depth](weights=weights_dict[depth])

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1  # /4
        self.layer2 = model.layer2  # /8
        self.layer3 = model.layer3  # /16
        self.layer4 = model.layer4  # /32

        ch_map = {18: [64, 128, 256, 512], 34: [64, 128, 256, 512], 50: [256, 512, 1024, 2048]}
        self.out_channels = ch_map[depth]

        if frozen_stages >= 0:
            for p in self.conv1.parameters():
                p.requires_grad = False
            for p in self.bn1.parameters():
                p.requires_grad = False

        for i in range(1, frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]


# =========================================================================
# 3. NECKS: Feature Pyramid Network (FPN), SimpleNeck
# =========================================================================

@NECKS.register_module()
class FPN(nn.Module):
    """Feature Pyramid Network (FPN) fusion for multi-scale features."""

    def __init__(self, in_channels: List[int], out_channels: int = 128):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels
        ])
        self.out_channels = out_channels

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        # Lateral connections
        laterals = [lat(inputs[i]) for i, lat in enumerate(self.lateral_convs)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode="nearest")

        # Smooth convolutions
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        return outs


@NECKS.register_module()
class GlobalAvgPoolNeck(nn.Module):
    """Global Average Pooling neck for classification networks."""

    def __init__(self, in_channels: Optional[int] = None):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        feat = inputs[-1] if isinstance(inputs, (list, tuple)) else inputs
        return torch.flatten(self.pool(feat), 1)


# =========================================================================
# 4. HEADS: LinearClsHead, AnchorFreeDetHead
# =========================================================================

@HEADS.register_module()
class LinearClsHead(nn.Module):
    """Classification Head with Linear layer."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        return self.fc(self.drop(feat))


@HEADS.register_module()
class AnchorFreeDetHead(nn.Module):
    """Single-stage anchor-free detection head predicting classes, boxes, and objectness."""

    def __init__(self, in_channels: int, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.cls_conv = ConvModule(in_channels, in_channels, kernel_size=1)
        self.cls_out = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        self.box_conv = ConvModule(in_channels, in_channels, kernel_size=3)
        self.box_out = nn.Conv2d(in_channels, 4, kernel_size=1)

        self.obj_conv = ConvModule(in_channels, in_channels, kernel_size=3)
        self.obj_out = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_logits = self.cls_out(self.cls_conv(feat))
        pred_boxes = self.box_out(self.box_conv(feat))
        obj_logits = self.obj_out(self.obj_conv(feat))
        return cls_logits, pred_boxes, obj_logits


# =========================================================================
# 5. MODULAR COMPOSABLE MODEL: ModularModel
# =========================================================================

@MODELS.register_module()
class ModularModel(BaseModel):
    """Mô hình dạng lắp ghép mô-đun (Plug-and-Play Modular Model).

    Cho phép hoán đổi bất kỳ Backbone, Neck, Head nào trực tiếp từ file YAML
    mà không cần viết lại class mô hình!

    Cấu trúc:
        Input Image -> [Backbone] -> [Neck (optional)] -> [Head] -> Outputs
    """

    def __init__(
        self,
        backbone: Union[Dict[str, Any], nn.Module],
        neck: Optional[Union[Dict[str, Any], nn.Module]] = None,
        head: Optional[Union[Dict[str, Any], nn.Module]] = None,
        loss: Optional[Union[Dict[str, Any], nn.Module]] = None,
    ):
        super().__init__()
        # 1. Build Backbone
        self.backbone = build_from_cfg(backbone, BACKBONES) if isinstance(backbone, dict) else backbone

        # 2. Build Neck (optional)
        self.neck = None
        if neck is not None:
            self.neck = build_from_cfg(neck, NECKS) if isinstance(neck, dict) else neck

        # 3. Build Head
        self.head = None
        if head is not None:
            self.head = build_from_cfg(head, HEADS) if isinstance(head, dict) else head

        # 4. Built-in Loss (optional)
        self.loss_fn = None
        if loss is not None:
            from ..core.registrar import LOSSES
            self.loss_fn = build_from_cfg(loss, LOSSES) if isinstance(loss, dict) else loss

    def extract_features(self, x: torch.Tensor) -> Any:
        """Trích xuất đặc trưng qua Backbone và Neck."""
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats

    def forward_train(self, inputs: torch.Tensor, targets: Optional[Any] = None, **kwargs) -> Dict[str, torch.Tensor]:
        feats = self.extract_features(inputs)

        # Case 1: Head là AnchorFreeDetHead (Object Detection)
        if isinstance(self.head, AnchorFreeDetHead):
            feat_map = feats[-1] if isinstance(feats, (list, tuple)) else feats
            cls_logits, pred_boxes, obj_logits = self.head(feat_map)

            # Tính toán Detection Loss
            device = inputs.device
            batch_size, _, gh, gw = cls_logits.shape
            total_cls_loss, total_box_loss, total_obj_loss = (
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
            )

            for b in range(batch_size):
                img_cls = cls_logits[b]
                img_box = pred_boxes[b]
                img_obj = obj_logits[b, 0]

                target_dict = targets[b] if targets is not None and b < len(targets) else None
                gt_boxes = target_dict.get("boxes", torch.empty((0, 4), device=device)) if target_dict else torch.empty((0, 4), device=device)
                gt_labels = target_dict.get("labels", torch.empty((0,), device=device, dtype=torch.long)) if target_dict else torch.empty((0,), device=device, dtype=torch.long)

                obj_target = torch.zeros((gh, gw), device=device)
                if len(gt_boxes) > 0:
                    gx = (gt_boxes[:, 0].clamp(0.0, 0.999) * gw).long()
                    gy = (gt_boxes[:, 1].clamp(0.0, 0.999) * gh).long()
                    obj_target[gy, gx] = 1.0

                    p_cls = img_cls[:, gy, gx].permute(1, 0)
                    p_box = img_box[:, gy, gx].permute(1, 0)

                    valid_lbls = gt_labels.clamp(0, self.head.num_classes - 1)
                    total_cls_loss += F.cross_entropy(p_cls, valid_lbls)
                    total_box_loss += F.smooth_l1_loss(p_box, gt_boxes)

                total_obj_loss += F.binary_cross_entropy_with_logits(
                    img_obj, obj_target, pos_weight=torch.tensor([5.0], device=device)
                )

            total_cls_loss /= max(batch_size, 1)
            total_box_loss /= max(batch_size, 1)
            total_obj_loss /= max(batch_size, 1)
            return {
                "loss": total_cls_loss + 2.0 * total_box_loss + total_obj_loss,
                "loss_cls": total_cls_loss,
                "loss_box": total_box_loss,
                "loss_obj": total_obj_loss,
            }

        # Case 2: Head thông thường (Classification)
        out = self.head(feats)
        if self.loss_fn is not None and targets is not None:
            loss = self.loss_fn(out, targets)
            return {"loss": loss, "loss_cls": loss}
        elif targets is not None:
            loss = F.cross_entropy(out, targets)
            return {"loss": loss, "loss_cls": loss}

        return {"preds": out}

    def predict(self, inputs: torch.Tensor, conf_threshold: float = 0.3, **kwargs) -> Any:
        feats = self.extract_features(inputs)

        # Detection inference
        if isinstance(self.head, AnchorFreeDetHead):
            feat_map = feats[-1] if isinstance(feats, (list, tuple)) else feats
            cls_logits, pred_boxes, obj_logits = self.head(feat_map)
            batch_size = cls_logits.shape[0]

            cls_probs = cls_logits.softmax(dim=1)
            obj_scores = obj_logits.sigmoid()

            results = []
            for b in range(batch_size):
                scores = cls_probs[b] * obj_scores[b]
                max_scores, labels = scores.max(dim=0)
                mask = max_scores >= conf_threshold

                results.append({
                    "boxes": pred_boxes[b, :, mask].permute(1, 0),
                    "scores": max_scores[mask],
                    "labels": labels[mask],
                })
            return results

        # Classification inference
        return self.head(feats).softmax(dim=-1)
