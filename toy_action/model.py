"""Model module for Toy Action Video Recognition.

Implements:
1. DeformableMambaActionModel: Spatiotemporal model using Backbone (P4+P3) ->
   DeformableFeatureHead (10x10 output map) -> Flatten 100 spatial tokens + 2D PosEmb ->
   Mamba temporal modeling -> MambaActionHead (48-class multi-label action prediction).
2. ToyActionNet: Lightweight Conv2D-GRU baseline model.
3. MultiLabelMetric: Standard multi-label evaluation metric.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from BaseModel.core import METRICS, MODELS
from BaseModel.model import BaseModel
from BaseModel.utils.metrics.base_metric import BaseMetric

from .dataset import IDX_TO_CLASS, NUM_CLASSES
from .loss import DeformableActionLoss
from .model_parts.backbones import MoViNetA0Backbone, X3DBackbone, YoloV11Backbone
from .model_parts.heads import (
    DeformableFeatureHead,
    MambaActionHead,
    build_2d_sincos_position_embedding,
)
from .model_parts.Mamba.model import Mamba, MambaConfig


class _ScaleGrad(torch.autograd.Function):
    """Identity in forward pass; scales backward gradients by ``scale``."""

    @staticmethod
    def forward(ctx, x: Tensor, scale: float) -> Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output * ctx.scale, None


@MODELS.register_module(name="DeformableMambaActionModel")
@MODELS.register_module(name="ToyDeformableMambaModel")
class DeformableMambaActionModel(BaseModel):
    """Deformable Spatial Attention + Mamba model for Toy Action benchmark.

    Pipeline:
        1. Backbone: YOLOv11 (multi-scale P4 + P3), X3D-M, or MoViNet-A0.
        2. Projection: 3D 1x1 convolutions projecting map_pred and feat_deform to ``dim``.
        3. DeformableFeatureHead: Samples deformable features onto a ``10x10`` grid.
        4. Spatial Flattening + PosEmb: Reshapes ``10x10`` feature map to ``100`` points
           per frame and adds 2D sin-cos positional embeddings.
        5. Mamba: Scans each of the ``100`` spatial points across ``T`` frames.
        6. MambaActionHead: Attentive + max + mean spatial pooling -> 48 action logits.
    """

    def __init__(
        self,
        num_actor_class: int = 48,
        num_ego_class: int = 0,
        dim: int = 256,
        ego_c: int = 128,
        grid_size: Tuple[int, int] = (10, 10),
        num_heads: int = 4,
        num_points: int = 4,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.num_actor_class = int(num_actor_class)
        self.num_ego_class = int(num_ego_class)
        self.dim = int(kwargs.get("slot_dim", dim))
        self.ego_c = int(ego_c)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.num_tokens = self.grid_size[0] * self.grid_size[1]  # 10x10 -> 100
        self.threshold = float(kwargs.get("threshold", 0.5))

        finetune_last_n_stages = kwargs.get("finetune_last_n_stages", 2)
        freeze_stages = kwargs.get("freeze_stages", None)
        freeze_all = kwargs.get("freeze_all", False)
        unfreeze_all = kwargs.get("unfreeze_all", False)
        freeze_bn = kwargs.get("freeze_bn", True)
        pretrained_backbone = kwargs.get("pretrained_backbone", True)
        pretrained_backbone_path = kwargs.get("pretrained_backbone_path", None)
        backbone_type = str(kwargs.get("backbone_type", "yolo")).lower()

        # 1. Video Backbone
        self._multiscale_backbone = False
        if backbone_type in ("yolo", "yolov11", "yolo11"):
            self.backbone = YoloV11Backbone(
                pretrained=pretrained_backbone,
                pretrained_path=pretrained_backbone_path,
                finetune_last_n_stages=finetune_last_n_stages,
                freeze_stages=freeze_stages,
                freeze_all=freeze_all,
                unfreeze_all=unfreeze_all,
                freeze_bn=freeze_bn,
            )
            self._multiscale_backbone = True
        elif backbone_type in ("movinet", "movinet_a0", "movinet-a0", "movineta0"):
            self.backbone = MoViNetA0Backbone(
                pretrained=pretrained_backbone,
                pretrained_path=pretrained_backbone_path,
                causal=kwargs.get("causal", False),
                finetune_last_n_stages=finetune_last_n_stages,
                freeze_stages=freeze_stages,
                freeze_all=freeze_all,
                unfreeze_all=unfreeze_all,
                freeze_bn=freeze_bn,
            )
        else:
            self.backbone = X3DBackbone(
                pretrained=pretrained_backbone,
                pretrained_path=pretrained_backbone_path,
                finetune_last_n_stages=finetune_last_n_stages,
                freeze_stages=freeze_stages,
                freeze_all=freeze_all,
                unfreeze_all=unfreeze_all,
                freeze_bn=freeze_bn,
            )
        in_c = self.backbone.out_channels

        # 2. Optional Ego Feature Branch
        if self.num_ego_class > 0:
            self.conv3d_ego = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm3d(in_c),
                nn.Conv3d(in_c, self.ego_c, (1, 1, 1), stride=1),
            )
            self.pool_ego = nn.AdaptiveAvgPool3d(output_size=1)
        else:
            self.conv3d_ego = None
            self.pool_ego = None

        # 3. Spatiotemporal Projections for Base Grid Map and Deformable Sampling Map
        self.conv3d = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm3d(in_c),
            nn.Conv3d(in_c, self.dim, (1, 1, 1), stride=1),
            nn.ReLU(),
        )
        if self._multiscale_backbone:
            self.conv3d_deform = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm3d(in_c),
                nn.Conv3d(in_c, self.dim, (1, 1, 1), stride=1),
                nn.ReLU(),
            )
        else:
            self.conv3d_deform = None

        # 4. Deformable Spatial Head (Visdrone DeformableSlotDetectionHead without slots)
        self.deformable_head = DeformableFeatureHead(
            dim=self.dim,
            grid_size=self.grid_size,
            num_points=int(num_points),
            num_heads=int(num_heads),
        )

        # 5. Mamba Temporal Modeling over 100 flattened spatial points
        mamba_cfg = MambaConfig(
            d_model=self.dim,
            n_layer=int(kwargs.get("mamba_layers", 2)),
            d_state=int(kwargs.get("mamba_d_state", 16)),
            num_slot=self.num_tokens,
        )
        self.mamba = Mamba(mamba_cfg)
        self.temporal_pool = str(kwargs.get("temporal_pool", "last_mean")).lower()
        self.mamba_grad_scale = float(kwargs.get("mamba_grad_scale", 1.0))
        self.backbone_grad_scale = float(kwargs.get("backbone_grad_scale", 0.5))

        # 6. Action Classification Head
        self.head = MambaActionHead(
            in_channel=self.dim,
            num_actor_classes=self.num_actor_class,
            grid_size=self.grid_size,
            num_ego_classes=self.num_ego_class,
            ego_channel=self.ego_c,
            dropout=float(dropout),
        )

        # 7. Loss Function
        self.loss_fn = DeformableActionLoss(
            num_actor_class=self.num_actor_class,
            bce_pos_weight=float(kwargs.get("bce_pos_weight", 5.0)),
            ego_loss_weight=float(kwargs.get("ego_loss_weight", 0.2)),
        )

    def init_weights(self) -> None:
        """Preserve pretrained backbone weights and submodule-specific initializations."""
        return None

    def _normalize_input(self, inputs: torch.Tensor) -> torch.Tensor:
        """Ensure input tensor is of shape (B, C=3, T=16, H, W)."""
        if inputs.dim() != 5:
            raise ValueError(f"Expected 5D tensor, got {inputs.shape}")

        if inputs.shape[2] == 3 and inputs.shape[1] == 16:
            inputs = inputs.permute(0, 2, 1, 3, 4)

        if inputs.dtype == torch.uint8:
            inputs = inputs.float().div_(255.0)

        return inputs.contiguous()

    def _get_token_pos_emb(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Cached 2D sin-cos position embedding for the flattened 100 grid tokens: [1, 1, 100, D]."""
        h_g, w_g = self.grid_size
        cache_key = (h_g, w_g, self.dim, device, dtype)
        if getattr(self, "_mamba_pos_cache_key", None) != cache_key:
            pos_2d = build_2d_sincos_position_embedding(h_g, w_g, self.dim, device=device, dtype=dtype)
            self._mamba_pos_cache = (
                pos_2d.flatten(2).permute(0, 2, 1).unsqueeze(1).contiguous()
            )  # [1, 1, 100, D]
            self._mamba_pos_cache_key = cache_key
        return self._mamba_pos_cache

    def extract_features(
        self,
        videos: torch.Tensor,
    ) -> Tuple[Tensor, Optional[Tensor], Dict[str, Tensor]]:
        """Run Backbone -> DeformableFeatureHead (10x10) -> Flatten 100 + PosEmb -> Mamba -> Temporal Pool."""
        videos = self._normalize_input(videos)
        batch_size = videos.shape[0]

        # 1. Backbone extraction
        if self._multiscale_backbone:
            feats_map, feats_high = self.backbone(videos, return_multiscale=True)
        else:
            feats_map = self.backbone(videos)
            feats_high = None

        if self.training and feats_map.requires_grad and self.backbone_grad_scale != 1.0:
            b_scale = self.backbone_grad_scale
            feats_map.register_hook(lambda grad: grad * b_scale)
            if feats_high is not None and feats_high.requires_grad:
                feats_high.register_hook(lambda grad: grad * b_scale)

        # Optional ego branch
        ego_x = None
        if self.conv3d_ego is not None:
            ego_feat = self.conv3d_ego(feats_map)
            ego_feat = self.pool_ego(ego_feat)
            ego_x = torch.reshape(ego_feat, (batch_size, self.ego_c))

        # 2. Spatiotemporal projections -> reshape per-frame [B*T, D, H, W]
        x_map = self.conv3d(feats_map)  # [B, D, T, H_m, W_m]
        _, _, seq_len, h_m, w_m = x_map.shape
        map_pred = x_map.permute(0, 2, 1, 3, 4).reshape(batch_size * seq_len, self.dim, h_m, w_m)

        if self.conv3d_deform is not None and feats_high is not None:
            x_def = self.conv3d_deform(feats_high)
            _, _, _, h_d, w_d = x_def.shape
            feat_deform = x_def.permute(0, 2, 1, 3, 4).reshape(batch_size * seq_len, self.dim, h_d, w_d)
        else:
            feat_deform = map_pred

        # 3. DeformableFeatureHead -> 10x10 output feature image [B*T, D, 10, 10]
        deform_map, sampling_info = self.deformable_head(map_pred, feat_deform)

        # 4. Flatten 10x10 -> 100 points + Position Embedding -> [B, T, 100, D]
        tokens = (
            deform_map.flatten(2)
            .permute(0, 2, 1)
            .contiguous()
            .view(batch_size, seq_len, self.num_tokens, self.dim)
        )
        mamba_in = tokens + self._get_token_pos_emb(device=tokens.device, dtype=tokens.dtype)

        # 5. Mamba temporal modeling over the 100 points
        if self.training and mamba_in.requires_grad and self.mamba_grad_scale != 1.0:
            m_scaled_in = _ScaleGrad.apply(mamba_in, 1.0 / self.mamba_grad_scale)
            mamba_out = _ScaleGrad.apply(self.mamba(m_scaled_in), self.mamba_grad_scale)
        else:
            mamba_out = self.mamba(mamba_in)

        seq_tokens = mamba_in + mamba_out  # [B, T, 100, D]

        # 6. Temporal aggregation -> [B, 100, D]
        if self.temporal_pool == "last":
            spatial_tokens = seq_tokens[:, -1, :, :]
        elif self.temporal_pool == "mean":
            spatial_tokens = seq_tokens.mean(dim=1)
        else:
            spatial_tokens = 0.5 * (seq_tokens[:, -1, :, :] + seq_tokens.mean(dim=1))

        aux_info = {
            "sampling_locations": sampling_info["sampling_locations"].view(
                batch_size, seq_len, *self.grid_size, self.deformable_head.num_heads, self.deformable_head.num_points, 2
            ),
            "attn_weights": sampling_info["attn_weights"].view(
                batch_size, seq_len, *self.grid_size, self.deformable_head.num_heads, self.deformable_head.num_points
            ),
        }
        return spatial_tokens, ego_x, aux_info

    def forward_train(
        self,
        inputs: torch.Tensor,
        targets: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses and attach predictions for hooks."""
        spatial_tokens, ego_x, _ = self.extract_features(inputs)
        pred_ego, pred_actor, spatial_attn = self.head(spatial_tokens, ego_x=ego_x)

        pred_dict = {
            "actor": pred_actor,
            "ego": pred_ego,
            "spatial_attn": spatial_attn,
        }
        loss_dict = self.loss_fn(pred_dict, targets, validate=False)

        target_tensor = (
            targets
            if isinstance(targets, torch.Tensor)
            else (targets.get("actor", targets.get("target", None)) if isinstance(targets, dict) else None)
        )
        loss_dict["logits"] = pred_actor
        loss_dict["preds"] = pred_actor
        loss_dict["pred_actor"] = pred_actor
        if target_tensor is not None:
            loss_dict["targets"] = target_tensor
            loss_dict["target_actor"] = target_tensor
        return loss_dict

    def forward_val(
        self,
        inputs: torch.Tensor,
        targets: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Validation forward pass."""
        spatial_tokens, ego_x, _ = self.extract_features(inputs)
        pred_ego, pred_actor, _ = self.head(spatial_tokens, ego_x=ego_x)

        output: Dict[str, Any] = {}
        if targets is not None:
            loss_dict = self.loss_fn({"actor": pred_actor, "ego": pred_ego}, targets, validate=True)
            output.update(loss_dict)

        probs = torch.sigmoid(pred_actor)
        preds_binary = (probs > self.threshold).float()

        target_tensor = (
            targets
            if isinstance(targets, torch.Tensor)
            else (targets.get("actor", targets.get("target", None)) if isinstance(targets, dict) else None)
        )
        output["logits"] = pred_actor
        output["preds"] = pred_actor
        output["probs"] = probs
        output["actor"] = pred_actor
        output["pred_actor"] = pred_actor
        if target_tensor is not None:
            output["exact_match"] = (preds_binary == target_tensor).all(dim=-1).float().mean()
            output["hamming_acc"] = (preds_binary == target_tensor).float().mean()
        return output

    def predict(
        self,
        inputs: torch.Tensor,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Inference forward pass returning predictions, 10x10 spatial attention, and sampling locations."""
        th = threshold if threshold is not None else self.threshold
        spatial_tokens, ego_x, aux_info = self.extract_features(inputs)
        _, pred_actor, spatial_attn = self.head(spatial_tokens, ego_x=ego_x)

        probs = torch.sigmoid(pred_actor)
        preds_binary = (probs > th).float()

        preds_np = preds_binary.detach().cpu().numpy()
        decoded_labels = []
        for row in preds_np:
            tokens = [IDX_TO_CLASS[i] for i, val in enumerate(row) if val >= 0.5]
            decoded_labels.append(", ".join(sorted(tokens)))

        return {
            "probs": probs,
            "preds": preds_binary,
            "logits": pred_actor,
            "spatial_attn": spatial_attn,
            "sampling_locations": aux_info["sampling_locations"],
            "sampling_attn": aux_info["attn_weights"],
            "label_strings": decoded_labels,
        }


# ==============================================================================
# Baseline Model: ToyActionNet (Conv2D + GRU)
# ==============================================================================

class ConvBlock(nn.Module):
    """Residual convolutional block with BatchNorm and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + res)
        return out


@MODELS.register_module(name="ToyActionNet")
class ToyActionNet(BaseModel):
    """Spatio-Temporal Video Recognition Model for Toy Action symbols."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = NUM_CLASSES,
        feat_dim: int = 256,
        rnn_hidden: int = 128,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.rnn_hidden = rnn_hidden
        self.threshold = threshold

        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, feat_dim, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.temporal_rnn = nn.GRU(
            input_size=feat_dim,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 4, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def _normalize_input(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 5:
            raise ValueError(f"Expected 5D tensor, got {inputs.shape}")
        if inputs.shape[1] == self.in_channels:
            return inputs.permute(0, 2, 1, 3, 4).contiguous()
        return inputs

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._normalize_input(inputs)
        b, t, c, h, w = x.shape

        x_flat = x.view(b * t, c, h, w)
        spatial_feats = self.backbone(x_flat).view(b * t, self.feat_dim)

        seq_feats = spatial_feats.view(b, t, self.feat_dim)
        rnn_out, _ = self.temporal_rnn(seq_feats)

        avg_pool = torch.mean(rnn_out, dim=1)
        max_pool, _ = torch.max(rnn_out, dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=-1)

        logits = self.classifier(pooled)
        return logits

    def forward_train(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        if targets is None:
            raise ValueError("Targets must be provided for forward_train.")

        logits = self.extract_features(inputs)
        loss = self.criterion(logits, targets)

        return {
            "loss": loss,
            "loss_bce": loss,
            "logits": logits,
            "preds": logits,
            "pred_actor": logits,
            "targets": targets,
            "target_actor": targets,
        }

    def forward_val(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        logits = self.extract_features(inputs)
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        output: Dict[str, Any] = {
            "preds": logits,
            "probs": probs,
            "logits": logits,
            "actor": logits,
            "pred_actor": logits,
        }

        if targets is not None:
            loss = self.criterion(logits, targets)
            output["loss"] = loss
            output["loss_bce"] = loss
            output["exact_match"] = (preds == targets).all(dim=-1).float().mean()
            output["hamming_acc"] = (preds == targets).float().mean()

        return output

    def predict(self, inputs: torch.Tensor, threshold: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        th = threshold if threshold is not None else self.threshold
        logits = self.extract_features(inputs)
        probs = torch.sigmoid(logits)
        preds = (probs > th).float()

        preds_np = preds.detach().cpu().numpy()
        decoded_labels = []
        for row in preds_np:
            tokens = [IDX_TO_CLASS[i] for i, val in enumerate(row) if val >= 0.5]
            decoded_labels.append(", ".join(sorted(tokens)))

        return {
            "probs": probs,
            "preds": preds,
            "label_strings": decoded_labels,
        }


# ==============================================================================
# MultiLabelMetric
# ==============================================================================

@METRICS.register_module()
class MultiLabelMetric(BaseMetric):
    """Multi-Label Evaluation Metric for exact match, hamming accuracy, and micro F1."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.total_samples = 0
        self.exact_matches = 0
        self.total_elements = 0
        self.correct_elements = 0
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, preds: Any, targets: Any) -> None:
        if isinstance(preds, dict):
            preds = preds.get("preds", preds.get("probs", preds.get("logits", None)))
        if isinstance(targets, dict):
            targets = targets.get("actor", targets.get("target", targets.get("targets", None)))

        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if isinstance(targets, (list, tuple)):
            targets = targets[0]

        if preds is None or targets is None:
            return

        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        if preds.min() < 0.0 or preds.max() > 1.0 or preds.shape != targets.shape:
            probs = torch.sigmoid(preds)
            binary_preds = (probs > self.threshold).float()
        else:
            binary_preds = (preds > self.threshold).float()

        targets = targets.float()
        batch_size = targets.size(0)
        self.total_samples += batch_size

        is_exact = (binary_preds == targets).all(dim=-1)
        self.exact_matches += int(is_exact.sum().item())

        self.total_elements += int(targets.numel())
        self.correct_elements += int((binary_preds == targets).sum().item())

        self.tp += float((binary_preds * targets).sum().item())
        self.fp += float((binary_preds * (1.0 - targets)).sum().item())
        self.fn += float(((1.0 - binary_preds) * targets).sum().item())

    def compute(self) -> Dict[str, float]:
        if self.total_samples == 0:
            return {"exact_match": 0.0, "hamming_acc": 0.0, "f1_score": 0.0}

        exact_match_pct = (self.exact_matches / self.total_samples) * 100.0
        hamming_acc_pct = (self.correct_elements / self.total_elements) * 100.0 if self.total_elements > 0 else 0.0

        denom = 2.0 * self.tp + self.fp + self.fn
        f1_pct = ((2.0 * self.tp) / denom * 100.0) if denom > 0 else 0.0

        return {
            "exact_match": round(exact_match_pct, 2),
            "hamming_acc": round(hamming_acc_pct, 2),
            "f1_score": round(f1_pct, 2),
        }
