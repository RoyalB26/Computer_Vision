"""Model module for Toy Action Video Recognition.

Implements:
1. ActionSlotModel: Advanced Spatio-Temporal Slot Attention model with X3D backbone,
   adapted from action_slot for the 48-class toy_action benchmark.
2. ToyActionNet: Lightweight Conv2D-GRU baseline model.
3. MultiLabelMetric: Standard multi-label evaluation metric.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from BaseModel.core import METRICS, MODELS
from BaseModel.model import BaseModel
from BaseModel.utils.metrics.base_metric import BaseMetric

from .dataset import ALL_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES, SYMBOL_VARIANTS, decode_target
from .loss import ActionSlotLoss
from .model_parts.backbones import MoViNetA0Backbone, X3DBackbone, YoloV11Backbone
from .model_parts.heads import (
    Allocated_Head,
    FrameObjectCountHead,
    ObjectActionHead,
    ReduceAllocatedHead,
    VideoObjectClassHead,
)
from .model_parts.slot_attention import SlotAttention
from .model_parts.Mamba.model import Mamba, MambaConfig


class _ScaleGrad(torch.autograd.Function):
    """Identity in forward pass; scales backward gradients by `scale`."""

    @staticmethod
    def forward(ctx, x: Tensor, scale: float) -> Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output * ctx.scale, None



def build_2d_sincos_position_embedding(H: int, W: int, embed_dim: int, device=None, dtype=torch.float32) -> Tensor:
    """
    Tạo 2D Sin-Cos Positional Encoding hoàn toàn bằng PyTorch trên device thích hợp.
    Output shape: [1, embed_dim, H, W]
    """
    assert embed_dim % 4 == 0, f"embed_dim ({embed_dim}) phải chia hết cho 4."
    dim_per_axis = embed_dim // 2
    # Tọa độ lưới
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    grid_x = grid_x.reshape(-1)  # [H * W]
    grid_y = grid_y.reshape(-1)  # [H * W]

    def get_1d_sincos(grid: Tensor, dim: int) -> Tensor:
        omega = torch.arange(dim // 2, device=device, dtype=dtype)
        omega /= (dim / 2.0)
        omega = 1.0 / (10000.0 ** omega)  # [dim // 2]
        out = torch.einsum('n, d -> nd', grid, omega)  # [H * W, dim // 2]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)  # [H * W, dim]

    emb_x = get_1d_sincos(grid_x, dim_per_axis)  # [H * W, dim_per_axis]
    emb_y = get_1d_sincos(grid_y, dim_per_axis)  # [H * W, dim_per_axis]
    pos_emb = torch.cat([emb_x, emb_y], dim=-1)  # [H * W, embed_dim]
    # Đổi về format chuẩn ảnh: [1, embed_dim, H, W]
    pos_emb = pos_emb.view(H, W, embed_dim).permute(2, 0, 1).unsqueeze(0)
    return pos_emb

@MODELS.register_module(name="ActionSlotModel")
@MODELS.register_module(name="ToyActionSlotModel")
class ActionSlotModel(BaseModel):
    """Spatiotemporal Action-Slot model adapted for Toy Action benchmark.

    Architecture:
        1. Backbone: X3D-M video backbone with stage-freezing (e.g. finetune last 2 stages).
        2. Spatiotemporal Feature Projection: 3D Conv projecting backbone features to slot_dim.
        3. Slot Attention: 3D iterative clustering into semantic object/action slots.
        4. Classification Head: ReduceAllocatedHead predicting [single, group+, background]
           per slot -> 24 slots x 2 = 48 action classes!
        5. ActionSlotLoss: Multi-task loss supervising slot action and attention masks.

    Args:
        num_actor_class: Total number of action classes (default: 48).
        num_slots: Number of semantic action slots (default: 24, corresponding to
                   6 symbols x 4 actions).
        slot_dim: Feature dimension for slots (default: 256).
        bg_slot: Whether to include an extra background slot (default: True).
        resolution3d: Spatiotemporal feature grid shape (default: (16, 7, 7)).
        dropout: Dropout rate (default: 0.5).
        finetune_last_n_stages: Number of final backbone stages to train (default: 2).
    """

    def __init__(
        self,
        num_actor_class: int = 48,
        num_slots: int = 6,
        num_ego_class: int = 0,
        slot_dim: int = 256,
        ego_c: int = 128,
        bg_slot: bool = True,
        resolution3d: Tuple[int, int, int] = (16, 7, 7),
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.num_actor_class = num_actor_class
        self.num_slots = num_slots
        self.num_ego_class = num_ego_class
        self.slot_dim = slot_dim
        self.ego_c = ego_c
        self.bg_slot = bg_slot
        self.resolution3d = resolution3d
        self.threshold = kwargs.get("threshold", 0.5)

        finetune_last_n_stages = kwargs.get("finetune_last_n_stages", 2)
        freeze_stages = kwargs.get("freeze_stages", None)
        freeze_all = kwargs.get("freeze_all", False)
        unfreeze_all = kwargs.get("unfreeze_all", False)
        freeze_bn = kwargs.get("freeze_bn", True)
        pretrained_backbone = kwargs.get("pretrained_backbone", True)
        pretrained_backbone_path = kwargs.get("pretrained_backbone_path", None)
        backbone_type = kwargs.get("backbone_type", "x3d").lower()

        # 1. Video Backbone with stage freezing (supports 'x3d', 'yolov11', or 'movinet-a0')
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

        # 2. Ego Feature Branch (optional, default 0 for toy_action)
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

        # 3. Spatiotemporal Feature Projection 
        self.conv3d = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm3d(in_c),
            nn.Conv3d(in_c, self.slot_dim, (1, 1, 1), stride=1),
            nn.ReLU(),
        )


        # 4. Slot Attention
        actual_slots = self.num_slots + 1 if bg_slot else self.num_slots
        self.slot_attention = SlotAttention(
            num_slots=actual_slots,
            dim=self.slot_dim,
            eps=1e-8,
            input_dim=self.slot_dim,
            resolution=list(resolution3d),
            num_actor_class=num_actor_class,
            allocated_slot=kwargs.get("allocated_slot", True),
            slot_dropout=kwargs.get("slot_dropout", 0.0),
        )

        # 5. Classification Head & Per-Frame Object Count Head
        self.actor_loss_type = str(kwargs.get("actor_loss_type", "bce")).lower()
        if self.actor_loss_type == "bce" and self.num_slots < num_actor_class:
            self.head = ObjectActionHead(
                in_channel=slot_dim,
                num_actor_classes=num_actor_class,
                num_slots=self.num_slots,
                num_ego_classes=num_ego_class,
                ego_channel=ego_c,
            )
        else:
            self.head = ReduceAllocatedHead(
                in_channel=slot_dim,
                num_actor_classes=num_actor_class,
                num_slots=self.num_slots,
                num_ego_classes=num_ego_class,
                ego_channel=ego_c,
            )

        self.use_obj_count_loss = bool(kwargs.get("use_obj_count_loss", True))
        num_obj_classes = int(kwargs.get("num_obj_classes", len(SYMBOL_VARIANTS)))
        if self.use_obj_count_loss:
            self.count_head = FrameObjectCountHead(
                in_channel=slot_dim,
                num_slots=self.num_slots,
                hidden_dim=int(kwargs.get("count_hidden_dim", 64)),
            )
            self.obj_cls_head = None
        else:
            self.count_head = None
            self.obj_cls_head = VideoObjectClassHead(
                in_channel=slot_dim,
                num_slots=actual_slots,
                num_obj_classes=num_obj_classes,
                hidden_dim=int(kwargs.get("obj_cls_hidden_dim", 256)),
                dropout=float(kwargs.get("obj_cls_dropout", 0.2)),
            )

        self.drop = nn.Dropout(p=dropout)

        # 6. Loss Function
        self.loss_fn = ActionSlotLoss(
            num_actor_class=num_actor_class,
            num_slots=self.num_slots,
            attention_res=(resolution3d[1], resolution3d[2]),
            bce_pos_weight=float(kwargs.get("bce_pos_weight", 5.0)),
            actor_loss_type=self.actor_loss_type,
            use_obj_count_loss=self.use_obj_count_loss,
            obj_count_weight=float(kwargs.get("obj_count_weight", 1.0)),
            obj_cls_weight=float(kwargs.get("obj_cls_weight", 1.0)),
            obj_cls_pos_weight=float(kwargs.get("obj_cls_pos_weight", 3.0)),
            num_obj_classes=num_obj_classes,
            action_attn_weight=float(kwargs.get("action_attn_weight", 1.0)),
            bg_attn_weight=float(kwargs.get("bg_attn_weight", 0.5)),
            bg_mask=True,
            bg_slot=bg_slot,
        )

        # Mamba operates strictly over the S object slots (excluding bg_slot)
        mambaCfg = MambaConfig(
            d_model=self.slot_dim,
            n_layer=int(kwargs.get("mamba_layers", 1)),
            d_state=int(kwargs.get("mamba_d_state", 16)),
            num_slot=self.num_slots,
        )
        self.mamba = Mamba(mambaCfg)
        self.temporal_pool = str(kwargs.get("temporal_pool", "last_mean")).lower()
        self.mamba_grad_scale = float(kwargs.get("mamba_grad_scale", 10.0))
        self.backbone_grad_scale = float(kwargs.get("backbone_grad_scale", 2.0))

    def init_weights(self) -> None:
        """Preserve pretrained backbone weights and submodule-specific initializations.

        BaseModel.scripts.train invokes `model.init_weights()` after `build_model()`,
        whose default implementation in `BaseModel` overwrites all Conv2d/BatchNorm2d/Linear
        parameters (wiping pretrained backbones while leaving pretrained BN running stats
        intact, causing activation explosion in frozen BN layers, and clobbering Mamba/Slot
        initializations). Each submodule already initializes itself during `__init__`.
        """
        return None

    def _normalize_input(self, inputs: torch.Tensor) -> torch.Tensor:
        """Ensure input tensor is of shape (B, C=3, T=16, H=224, W=224)."""
        if inputs.dim() != 5:
            raise ValueError(f"Expected 5D tensor, got {inputs.shape}")

        if inputs.shape[2] == 3 and inputs.shape[1] == 16:
            # (B, T, C, H, W) -> (B, C, T, H, W)
            inputs = inputs.permute(0, 2, 1, 3, 4)

        if inputs.dtype == torch.uint8:
            inputs = inputs.float().div_(255.0)

        return inputs.contiguous()

    def _extract_all_features(
        self,
        videos: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """Extract object action slots, ego features, attention masks, and auxiliary predictions."""
        videos = self._normalize_input(videos)
        batch_size, _, seq_len, _, _ = videos.shape

        # 1. Backbone forward: (B, C_feat, T=16, H=7, W=7)
        feats = self.backbone(videos)

        if self.training and feats.requires_grad and self.backbone_grad_scale != 1.0:
            b_scale = self.backbone_grad_scale
            feats.register_hook(lambda grad: grad * b_scale)

        # 2. Ego feature branch (if configured)
        ego_x = None
        if self.conv3d_ego is not None:
            ego_feat = self.conv3d_ego(feats)
            ego_feat = self.pool_ego(ego_feat)
            ego_x = torch.reshape(ego_feat, (batch_size, self.ego_c))
            ego_x = self.drop(ego_x)

        new_seq_len = feats.shape[2]
        new_h, new_w = feats.shape[3], feats.shape[4]

        # 3. Project to slot dimension: (B, T, H, W, slot_dim)
        x = self.conv3d(feats)
        x = x + build_2d_sincos_position_embedding(
            new_h, new_w, self.slot_dim, device=x.device, dtype=x.dtype
        ).unsqueeze(2)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, T, H, W, slot_dim)

        # 4. Per-frame Slot Attention -> (B, T, actual_slots, slot_dim)
        slots_seq_all, attn_masks = self.slot_attention(x, return_temporal_slots=True)
        obj_slots = slots_seq_all[:, :, :self.num_slots, :]  # (B, T, S, slot_dim)

        # 4b. Auxiliary head: either per-frame object counts (from obj_slots) or video-level object classes (from all slots)
        pred_counts = self.count_head(obj_slots) if self.count_head is not None else None
        pred_obj_cls = self.obj_cls_head(slots_seq_all) if self.obj_cls_head is not None else None

        # 5. Mamba temporal modeling over object slots: (B, T, S, slot_dim)
        if self.training and obj_slots.requires_grad and self.mamba_grad_scale != 1.0:
            mamba_in = _ScaleGrad.apply(obj_slots, 1.0 / self.mamba_grad_scale)
            mamba_out = _ScaleGrad.apply(self.mamba(mamba_in), self.mamba_grad_scale)
        else:
            mamba_out = self.mamba(obj_slots)

        slots_seq = obj_slots + mamba_out  # (B, T, S, slot_dim)

        # Temporal aggregation -> (B, S, slot_dim)
        if self.temporal_pool == "last":
            slots = slots_seq[:, -1, :, :]
        elif self.temporal_pool == "mean":
            slots = slots_seq.mean(dim=1)
        else:
            slots = 0.5 * (slots_seq[:, -1, :, :] + slots_seq.mean(dim=1))

        # 6. Reshape attention masks: (B, T, actual_slots, H, W)
        b, actual_slots, _ = attn_masks.shape
        attn_masks = attn_masks.view(b, actual_slots, new_seq_len, new_h, new_w) # (B, actual_slots, T, H, W)
        attn_masks = attn_masks.permute(0, 2, 1, 3, 4)  # (B, T, actual_slots, H, W)

        if seq_len > new_seq_len:
            attn_flat = attn_masks.permute(0, 2, 1, 3, 4).reshape(b * actual_slots, 1, new_seq_len, new_h, new_w)
            attn_up = F.interpolate(attn_flat, size=(seq_len, new_h, new_w), mode="trilinear", align_corners=False)
            attn_masks = attn_up.view(b, actual_slots, seq_len, new_h, new_w).permute(0, 2, 1, 3, 4)

        aux_preds = {
            "obj_counts": pred_counts,
            "obj_cls": pred_obj_cls,
        }
        return slots, ego_x, attn_masks, aux_preds

    def extract_features(
        self,
        videos: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Extract slot representations and attention masks."""
        slots, ego_x, attn_masks, _ = self._extract_all_features(videos)
        return slots, ego_x, attn_masks

    def _process_output(self, pred_actor: Tensor) -> Tensor:
        """Map slot predictions to (B, 48) multi-label action logits."""
        if pred_actor.dim() == 2 and pred_actor.shape[-1] == self.num_actor_class:
            return pred_actor

        B = pred_actor.shape[0]
        fg_logits = pred_actor[:, :, :2] - pred_actor[:, :, 2:3]  # (B, 24, 2)

        # (B, 24, 2) -> (B, 6 symbols, 4 actions, 2 variants)
        # Permute to (B, 6, 2 variants [single, group+], 4 actions) -> (B, 48)
        logits_48 = fg_logits.view(B, 6, 4, 2).permute(0, 1, 3, 2).reshape(B, 48)
        return logits_48

    def forward_train(
        self,
        inputs: torch.Tensor,
        targets: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses and attach predictions for hooks."""
        slots, ego_x, attn_masks, aux_preds = self._extract_all_features(inputs)
        pred_ego, pred_actor = self.head(slots, ego_x=ego_x)
        pred_counts = aux_preds.get("obj_counts", None)
        pred_obj_cls = aux_preds.get("obj_cls", None)

        pred_dict = {
            "actor": pred_actor,
            "ego": pred_ego,
            "attn": attn_masks,
            "obj_counts": pred_counts,
            "obj_cls": pred_obj_cls,
        }

        # If targets is provided as raw tensor (B, 48) or dict
        loss_dict = self.loss_fn(pred_dict, targets, validate=False)

        # Calibrate logits to 48 classes for metrics and hooks
        logits_48 = self._process_output(pred_actor)
        target_tensor = (
            targets
            if isinstance(targets, torch.Tensor)
            else (targets.get("actor", targets.get("target", None)) if isinstance(targets, dict) else None)
        )

        loss_dict["logits"] = logits_48
        loss_dict["preds"] = logits_48
        loss_dict["pred_actor"] = logits_48
        if pred_counts is not None:
            loss_dict["pred_obj_counts"] = pred_counts
        if pred_obj_cls is not None:
            loss_dict["pred_obj_cls"] = pred_obj_cls
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
        slots, ego_x, attn_masks, aux_preds = self._extract_all_features(inputs)
        pred_ego, pred_actor = self.head(slots, ego_x=ego_x)
        pred_counts = aux_preds.get("obj_counts", None)
        pred_obj_cls = aux_preds.get("obj_cls", None)

        pred_dict = {
            "actor": pred_actor,
            "ego": pred_ego,
            "attn": attn_masks,
            "obj_counts": pred_counts,
            "obj_cls": pred_obj_cls,
        }

        output: Dict[str, Any] = {}
        if targets is not None:
            loss_dict = self.loss_fn(pred_dict, targets, validate=True)
            output.update(loss_dict)

        logits_48 = self._process_output(pred_actor)
        probs_48 = torch.sigmoid(logits_48)
        preds_binary = (probs_48 > self.threshold).float()

        target_tensor = (
            targets
            if isinstance(targets, torch.Tensor)
            else (targets.get("actor", targets.get("target", None)) if isinstance(targets, dict) else None)
        )

        output["logits"] = logits_48
        output["preds"] = logits_48
        output["probs"] = probs_48
        output["actor"] = logits_48
        output["pred_actor"] = logits_48
        if pred_counts is not None:
            output["obj_counts"] = pred_counts.clamp(min=0.0)
        if pred_obj_cls is not None:
            output["obj_cls"] = pred_obj_cls
            output["obj_cls_probs"] = torch.sigmoid(pred_obj_cls)

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
        """Inference forward pass with decoded text labels."""
        th = threshold if threshold is not None else self.threshold
        slots, ego_x, attn_masks, aux_preds = self._extract_all_features(inputs)
        _, pred_actor = self.head(slots, ego_x=ego_x)
        pred_counts = aux_preds.get("obj_counts", None)
        pred_obj_cls = aux_preds.get("obj_cls", None)

        logits_48 = self._process_output(pred_actor)
        probs_48 = torch.sigmoid(logits_48)
        preds_binary = (probs_48 > th).float()

        preds_np = preds_binary.detach().cpu().numpy()
        decoded_labels = []
        for row in preds_np:
            tokens = [IDX_TO_CLASS[i] for i, val in enumerate(row) if val >= 0.5]
            decoded_labels.append(", ".join(sorted(tokens)))

        return {
            "probs": probs_48,
            "preds": preds_binary,
            "logits": logits_48,
            "attn": attn_masks,
            "obj_counts": pred_counts.clamp(min=0.0) if pred_counts is not None else None,
            "obj_cls_probs": torch.sigmoid(pred_obj_cls) if pred_obj_cls is not None else None,
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
