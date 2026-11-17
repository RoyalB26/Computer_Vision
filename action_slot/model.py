"""COCO / Hand Digit Object Detection Model predicting bounding boxes and class labels with FPN."""
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
from torch import Tensor
from model_parts.backbones import X3DBackbone, YoloV11Backbone
from model_parts.heads import (
    Allocated_Head,
    ReduceAllocatedHead,
    VideoObjectClassHead,
    ObjectActionHead,
    SpatialBroadcastFeatureDecoder,
)
from model_parts.slot_attention import SlotAttention
from model_parts.Mamba.model import Mamba, MambaConfig
from loss import ActionSlotLoss
from typing import Dict, List, Optional, Union

from BaseModel.core import MODELS
from BaseModel.model import BaseModel

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


class PanoramicRayPositionEncoding(nn.Module):
    """Camera-ray & panoramic spherical positional encoding with ground-plane depth proxy.

    Combines:
      1. Multi-frequency spherical sin-cos bands over azimuth `theta` and elevation `phi`.
      2. A learnable projection of 7D camera-ray & road-horizon geometry:
         `[ray_x, ray_y, ray_z, theta_norm, phi_norm, ground_inv_depth, ground_log_depth]`.
    Output shape: `(1, embed_dim, H, W)`.
    """

    def __init__(
        self,
        embed_dim: int,
        hfov_deg: float = 180.0,
        vfov_deg: float = 60.0,
        horizon_ratio: float = 0.45,
        default_hw: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        assert embed_dim % 4 == 0, f"embed_dim ({embed_dim}) phải chia hết cho 4."
        self.embed_dim = embed_dim
        self.hfov_rad = math.radians(float(hfov_deg))
        self.vfov_rad = math.radians(float(vfov_deg))
        self.horizon_ratio = float(horizon_ratio)

        hidden_dim = max(64, embed_dim // 2)
        num_groups = math.gcd(embed_dim, 16)
        self.ray_mlp = nn.Sequential(
            nn.Conv2d(7, hidden_dim, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim),
        )
        self.ray_scale = nn.Parameter(torch.full((1, embed_dim, 1, 1), 0.25))

        if default_hw is not None:
            sincos, geom = self._build_spherical_grid(default_hw[0], default_hw[1], device=None, dtype=torch.float32)
            self.register_buffer("_cached_sincos", sincos, persistent=False)
            self.register_buffer("_cached_geom", geom, persistent=False)
        else:
            self._cached_sincos = None
            self._cached_geom = None

    def _build_spherical_grid(
        self, H: int, W: int, device=None, dtype: torch.dtype = torch.float32
    ) -> Tuple[Tensor, Tensor]:
        """Construct spherical sin-cos embedding `(1, D, H, W)` and 7D ray geometry `(1, 7, H, W)`."""
        u = ((torch.arange(W, device=device, dtype=dtype) + 0.5) / float(W)) * 2.0 - 1.0  # [-1, 1]
        v = ((torch.arange(H, device=device, dtype=dtype) + 0.5) / float(H)) * 2.0 - 1.0  # [-1, 1]
        v_01 = (torch.arange(H, device=device, dtype=dtype) + 0.5) / float(H)  # [0, 1]

        theta_1d = u * (0.5 * self.hfov_rad)
        phi_1d = v * (0.5 * self.vfov_rad)
        phi, theta = torch.meshgrid(phi_1d, theta_1d, indexing="ij")
        v_grid, u_grid = torch.meshgrid(v_01, u, indexing="ij")
        v_signed, _ = torch.meshgrid(v, u, indexing="ij")

        # Multi-frequency spherical sin-cos encoding
        dim_per_axis = self.embed_dim // 2
        num_bands = dim_per_axis // 2
        freq_theta = torch.logspace(
            0.0, math.log2(max(float(W), 16.0)), steps=num_bands, base=2.0, device=device, dtype=dtype
        )
        freq_phi = torch.logspace(
            0.0, math.log2(max(float(H), 16.0)), steps=num_bands, base=2.0, device=device, dtype=dtype
        )
        ang_theta = theta.reshape(-1, 1) * freq_theta.unsqueeze(0)
        ang_phi = phi.reshape(-1, 1) * freq_phi.unsqueeze(0)
        emb_theta = torch.cat([torch.sin(ang_theta), torch.cos(ang_theta)], dim=-1)
        emb_phi = torch.cat([torch.sin(ang_phi), torch.cos(ang_phi)], dim=-1)
        sincos = torch.cat([emb_theta, emb_phi], dim=-1).view(H, W, self.embed_dim).permute(2, 0, 1).unsqueeze(0)

        # 3D camera unit ray + flat-ground perspective depth proxies relative to horizon
        cos_phi = torch.cos(phi)
        ray_x = torch.sin(theta) * cos_phi
        ray_y = torch.sin(phi)
        ray_z = torch.cos(theta) * cos_phi

        dy_below = F.relu(v_grid - self.horizon_ratio)
        ground_inv_depth = dy_below / max(1.0 - self.horizon_ratio, 1e-4)
        ground_log_depth = torch.log1p(1.0 / (dy_below + 0.05)) / math.log1p(20.0)

        geom = torch.stack(
            [ray_x, ray_y, ray_z, u_grid, v_signed, ground_inv_depth, ground_log_depth],
            dim=0,
        ).unsqueeze(0)
        return sincos, geom

    def forward(self, H: int, W: int, device=None, dtype: torch.dtype = torch.float32) -> Tensor:
        if (
            self._cached_sincos is not None
            and self._cached_geom is not None
            and self._cached_sincos.shape[2:] == (H, W)
        ):
            sincos = self._cached_sincos.to(device=device, dtype=dtype)
            geom = self._cached_geom.to(device=device, dtype=self.ray_mlp[0].weight.dtype)
        else:
            sincos, geom = self._build_spherical_grid(H, W, device=device, dtype=torch.float32)
            sincos = sincos.to(dtype=dtype)
            geom = geom.to(dtype=self.ray_mlp[0].weight.dtype)

        ray_emb = self.ray_mlp(geom).to(dtype=dtype)
        return sincos + self.ray_scale.to(dtype=dtype) * ray_emb


class TemporalDifferenceBlock(nn.Module):
    """Bidirectional temporal-difference motion encoder for 5D volumes `(B, C, T, H, W)`.

    Computes backward (`x_t - x_{t-1}`) and forward (`x_{t+1} - x_t`) feature differences,
    mixes local spatial displacements and short-range temporal consistency via separable
    (2+1)D depthwise-pointwise convolutions, and fuses back via a scaled residual branch.
    """

    def __init__(self, channels: int, spatial_kernel: int = 3, temporal_kernel: int = 3):
        super().__init__()
        self.channels = channels
        sp_pad = spatial_kernel // 2
        t_pad = temporal_kernel // 2
        num_groups = math.gcd(channels, 16)

        self.diff_proj = nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False)
        self.spatial_dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, spatial_kernel, spatial_kernel),
            padding=(0, sp_pad, sp_pad),
            groups=channels,
            bias=False,
        )
        self.temporal_dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(temporal_kernel, 1, 1),
            padding=(t_pad, 0, 0),
            groups=channels,
            bias=False,
        )
        self.pw_fuse = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
        )
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1, 1), 0.25))

    @staticmethod
    def _compute_bidirectional_diff(x: Tensor) -> Tensor:
        """Compute `[x_t - x_{t-1}, x_{t+1} - x_t]` with zero-padded temporal boundaries."""
        if x.shape[2] <= 1:
            zeros = torch.zeros_like(x)
            return torch.cat([zeros, zeros], dim=1)

        delta = x[:, :, 1:] - x[:, :, :-1]
        bwd = F.pad(delta, (0, 0, 0, 0, 1, 0))
        fwd = F.pad(delta, (0, 0, 0, 0, 0, 1))
        return torch.cat([bwd, fwd], dim=1)

    def forward(self, x: Tensor) -> Tensor:
        diffs = self._compute_bidirectional_diff(x)
        motion = self.diff_proj(diffs)
        motion = self.spatial_dw(motion)
        motion = self.temporal_dw(motion)
        motion = self.pw_fuse(motion)
        return x + self.gamma * motion

@MODELS.register_module(name="ActionSlotModel")
class ActionSlotModel(BaseModel):
    """Spatiotemporal Action-Slot model.

    Modularly connects:
        1. Backbone (X3D, VideoSwin, etc.)
        2. Spatiotemporal Projection & Ego Branch
        3. Slot Attention (Iterative Object/Action feature clustering)
        4. Multi-task Classification Head (Linear, Allocated, Deformable)
    """

    def __init__(
        self,
        num_ego_class: int = 4,
        num_actor_class: int = 64,
        num_slots: int = 64,
        slot_dim: int = 256,
        ego_c: int = 128,
        bg_slot: bool = False,
        resolution3d: tuple = (16, 8, 24),
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.num_ego_class = num_ego_class
        self.num_actor_class = num_actor_class
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.ego_c = ego_c

        finetune_last_n_stages = kwargs.get('finetune_last_n_stages', 2)
        freeze_stages = kwargs.get('freeze_stages', None)
        freeze_all = kwargs.get('freeze_all', False)
        unfreeze_all = kwargs.get('unfreeze_all', False)
        freeze_bn = kwargs.get('freeze_bn', True)
        pretrained_backbone = kwargs.get('pretrained_backbone', True)
        pretrained_backbone_path = kwargs.get('pretrained_backbone_path', None)

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

        self.use_temporal_diff = bool(kwargs.get("use_temporal_diff", True))
        td_spatial_kernel = int(kwargs.get("td_spatial_kernel", 3))
        td_temporal_kernel = int(kwargs.get("td_temporal_kernel", 3))

        # 2. Ego Feature Branch
        if self.num_ego_class > 0:
            self.conv3d_ego = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm3d(in_c),
                nn.Conv3d(in_c, self.ego_c, (1, 1, 1), stride=1),
            )
            self.ego_temporal_diff = (
                TemporalDifferenceBlock(
                    channels=self.ego_c,
                    spatial_kernel=td_spatial_kernel,
                    temporal_kernel=td_temporal_kernel,
                )
                if self.use_temporal_diff
                else None
            )
            self.pool_ego = nn.AdaptiveAvgPool3d(output_size=1)
        else:
            self.conv3d_ego = None
            self.ego_temporal_diff = None
            self.pool_ego = None

        # 3. Spatiotemporal Feature Projection
        self.conv3d = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm3d(in_c),
            nn.Conv3d(in_c, self.slot_dim, (1, 1, 1), stride=1),
            nn.ReLU(),
        )
        self.temporal_diff = (
            TemporalDifferenceBlock(
                channels=self.slot_dim,
                spatial_kernel=td_spatial_kernel,
                temporal_kernel=td_temporal_kernel,
            )
            if self.use_temporal_diff
            else None
        )
        self.use_ray_pe = bool(kwargs.get("use_ray_pe", True))
        self.ray_pos_enc = (
            PanoramicRayPositionEncoding(
                embed_dim=self.slot_dim,
                hfov_deg=float(kwargs.get("ray_hfov_deg", 180.0)),
                vfov_deg=float(kwargs.get("ray_vfov_deg", 60.0)),
                horizon_ratio=float(kwargs.get("ray_horizon_ratio", 0.45)),
                default_hw=(int(resolution3d[1]), int(resolution3d[2])),
            )
            if self.use_ray_pe
            else None
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
            allocated_slot=kwargs.get('allocated_slot', True),
            slot_dropout=kwargs.get('slot_dropout', 0.0),
        )

        # self.obj_cls_head = VideoObjectClassHead(
        #     in_channel=slot_dim,
        #     num_slots=actual_slots,
        #     num_obj_classes=num_actor_class,
        #     hidden_dim=int(kwargs.get("obj_cls_hidden_dim", 256)),
        #     dropout=float(kwargs.get("obj_cls_dropout", 0.2)),
        # )

        # 5. Modular Classification Head
        self.head = ObjectActionHead(
        in_channel= slot_dim, 
        num_ego_classes= num_ego_class,
        num_actor_classes=num_actor_class,
        ego_channel= ego_c,
        num_slots= self.num_slots,
        actions_per_slot= [24,24,16]
        )
        self.drop = nn.Dropout(p=dropout)

        # DINOSAUR-style Spatial Broadcast Feature Decoder
        self.use_feat_recon = bool(kwargs.get("use_feat_recon", True))
        feat_recon_weight = float(kwargs.get("feat_recon_weight", 0.15))
        if self.use_feat_recon:
            self.feat_decoder = SpatialBroadcastFeatureDecoder(
                slot_dim=self.slot_dim,
                hidden_dim=int(kwargs.get("feat_recon_hidden_dim", 128)),
                out_dim=self.slot_dim,
            )
        else:
            self.feat_decoder = None

        bce_pos_weight = kwargs.get('bce_pos_weight', 5.0)

        self.loss_fn = ActionSlotLoss(
            num_actor_class=num_actor_class,
            attention_res=(resolution3d[1], resolution3d[2]),
            bce_pos_weight=bce_pos_weight,
            num_slots=self.num_slots,
            feat_recon_weight=feat_recon_weight if self.use_feat_recon else 0.0,
            action_attn_weight=float(kwargs.get("action_attn_weight", 1.0)),
            bg_attn_weight=float(kwargs.get("bg_attn_weight", 0.5)),
            ego_loss_weight=float(kwargs.get("ego_loss_weight", 0.2)),
            use_class_pos_weight=bool(kwargs.get("use_class_pos_weight", True)),
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
        self.mamba_grad_scale = float(kwargs.get("mamba_grad_scale", 1.0))
        self.backbone_grad_scale = float(kwargs.get("backbone_grad_scale", 1.0))
        self.ego_backbone_grad_scale = float(kwargs.get("ego_backbone_grad_scale", 0.1))

    def init_weights(self) -> None:
        """Preserve pretrained backbone weights and submodule-specific initializations.

        BaseModel.scripts.train invokes `model.init_weights()` after `build_model()`,
        whose default implementation in `BaseModel` overwrites Conv2d/BatchNorm2d/Linear
        parameters (wiping pretrained backbones while leaving pretrained BN running stats
        intact, causing activation explosion in frozen BN layers). Each submodule already
        initializes itself during `__init__`.
        """
        return None


    def _extract_features_with_aux(
        self,
        videos: Union[List[torch.Tensor], torch.Tensor],
        compute_recon: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        if isinstance(videos, list):
            videos = torch.stack(videos, dim=0)
        if videos.dtype == torch.uint8:
            # Chuyển uint8 [0, 255] -> float32 chuẩn hóa (mean=0.45, std=0.225) trên GPU để giảm 4x băng thông CPU->GPU
            videos = videos.float().mul_(1.0 / (255.0 * 0.225)).add_(-0.45 / 0.225)
        videos = videos.permute(0, 2, 1, 3, 4)
        seq_len = videos.shape[2]
        batch_size = videos.shape[0]

        # 1. Backbone feature extraction: (B, C, T_feat, H_feat, W_feat)
        feats = self.backbone(videos)

        # 2. Ego feature branch (attenuates gradient pull back into backbone)
        ego_x = None
        if self.conv3d_ego is not None:
            ego_in = feats
            if self.training and feats.requires_grad:
                if self.ego_backbone_grad_scale == 0.0:
                    ego_in = feats.detach()
                elif self.ego_backbone_grad_scale != 1.0:
                    ego_in = _ScaleGrad.apply(feats, self.ego_backbone_grad_scale)
            ego_feat = self.conv3d_ego(ego_in)
            if self.ego_temporal_diff is not None:
                ego_feat = self.ego_temporal_diff(ego_feat)
            ego_feat = self.pool_ego(ego_feat)
            ego_x = torch.reshape(ego_feat, (batch_size, self.ego_c))

        new_seq_len = feats.shape[2]
        new_h, new_w = feats.shape[3], feats.shape[4]

        # 3. Project to slot dimension (decouple slot->backbone gradient scale from ego branch)
        slot_in = feats
        if self.training and feats.requires_grad:
            if self.backbone_grad_scale == 0.0:
                slot_in = feats.detach()
            elif self.backbone_grad_scale != 1.0:
                slot_in = _ScaleGrad.apply(feats, self.backbone_grad_scale)
        x_vis = self.conv3d(slot_in)
        if self.temporal_diff is not None:
            x_vis = self.temporal_diff(x_vis)
        x_vis_5d = x_vis.permute(0, 2, 3, 4, 1).contiguous()
        if self.ray_pos_enc is not None:
            pos_emb = self.ray_pos_enc(new_h, new_w, device=x_vis.device, dtype=x_vis.dtype)
        else:
            pos_emb = build_2d_sincos_position_embedding(
                new_h, new_w, self.slot_dim, device=x_vis.device, dtype=x_vis.dtype
            )
        x = x_vis + pos_emb.unsqueeze(2)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, T, H, W, slot_dim)

        # 4. Per-frame Slot Attention -> (B, T, actual_slots, slot_dim)
        slots_seq_all, attn_masks = self.slot_attention(x, return_temporal_slots=True)
        obj_slots = slots_seq_all[:, :, :self.num_slots, :]  # (B, T, S, slot_dim)

        # 4b. DINOSAUR-style Spatial Broadcast feature reconstruction from all slots
        recon_feats = None
        target_feats = None
        if compute_recon and self.feat_decoder is not None:
            recon_feats, _ = self.feat_decoder(slots_seq_all, new_h, new_w)
            target_feats = F.layer_norm(x_vis_5d.detach(), (self.slot_dim,))

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
            attn_masks = attn_up.view(b, actual_slots, seq_len, new_h, new_w).permute(0, 2, 1, 3, 4).clamp(0.0, 1.0)

        if ego_x is not None:
            ego_x = self.drop(ego_x)

        aux_preds = {
            "recon_feats": recon_feats,
            "target_feats": target_feats,
        }
        return slots, ego_x, attn_masks, aux_preds

    def extract_features(
        self,
        videos: Union[List[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        slots, ego_x, attn_masks, _ = self._extract_features_with_aux(videos, compute_recon=False)
        return slots, ego_x, attn_masks



    def _process_output(
        self,
        pred: Tensor
    ) -> Tensor:
  
        """
        pred: shape (B, 32, 3) - raw logits
        return: shape (B, 64) - logits đã hiệu chỉnh theo background, đúng thứ tự gốc
        """
        B = pred.shape[0]
        
        # 1. Hiệu chỉnh logits 2 lớp đầu dựa trên lớp background (index 2)
        fg_logits = pred[:, :, :2] - pred[:, :, 2:3]  # shape: (B, 32, 2)
        
        # 2. Tách thành 3 cụm (12, 12, 8) và đảo ngược transpose + reshape
        p1 = fg_logits[:, 0:12, :].transpose(1, 2).reshape(B, 24)   # khôi phục index 0..23
        p2 = fg_logits[:, 12:24, :].transpose(1, 2).reshape(B, 24)  # khôi phục index 24..47
        p3 = fg_logits[:, 24:32, :].transpose(1, 2).reshape(B, 16)  # khôi phục index 48..63
        
        # 3. Ghép lại thành (B, 64)
        logits_64 = torch.cat([p1, p2, p3], dim=1)
        return logits_64

    def _compute_losses(
        self, 
        pred, 
        label, 
        validate=False
    ) -> Dict[str, torch.Tensor]:

        # print(label)
        
        label = {k: [d[k] for d in label] for k in label[0]} if label else {}
        label['ego'] = torch.stack(label['ego'], dim = 0)
        label['actor'] = torch.stack(label['actor'], dim = 0)
        label['bg_seg'] = torch.stack(label['bg_seg'], dim = 0)

        loss = self.loss_fn(pred, label, validate)

        # Attach predictions and ground truth for metric evaluation (e.g., train actor mAP)
        if isinstance(loss, dict):
            if "actor" in pred:
                loss["pred_actor"] = pred['actor'].detach()
            if "actor" in label and isinstance(label["actor"], torch.Tensor):
                loss["target_actor"] = label["actor"].detach()
        
        return loss

    def _post_process(self, pred: Optional[Dict[str, Any]] = None):
        if pred is None:
            return None
        return pred

    def forward_train(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # print(inputs[0]['videos'].shape)
        # print(inputs["bg_seg"].shape)
        # print(targets['ego'].shape)
        # print(targets['actor'].shape)
        """Tính toán Detection Loss trong quá trình huấn luyện với FPN và Deformable Sampling Loss."""
        slots, ego_x, attn_masks, aux_preds = self._extract_features_with_aux(
            inputs, compute_recon=(self.feat_decoder is not None)
        )

        pred_ego, pred_actor = self.head(slots, ego_x=ego_x)

        pred = {
            "actor": pred_actor,
            "ego": pred_ego,
            "attn": attn_masks,
            "recon_feats": aux_preds.get("recon_feats"),
            "target_feats": aux_preds.get("target_feats"),
            # "cls_pred": cls_pred
        }

        return self._compute_losses(pred, targets)
    

    def predict(
        self,
        inputs: torch.Tensor,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        """Dự đoán bounding box, class label và confidence score khi suy luận / đánh giá."""
        
        slots, ego_x, attn_masks = self.extract_features(inputs)
        pred_ego, pred_actor = self.head(slots, ego_x=ego_x)

        pred = {
            "actor": pred_actor,
            "ego": pred_ego,
            "attn": attn_masks,
        }

        return self._post_process(pred)

    def forward_val(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Validation forward pass: tính validation losses và trích xuất preds phục vụ tính mAP."""
        slots, ego_x, attn_masks, aux_preds = self._extract_features_with_aux(
            inputs, compute_recon=(self.feat_decoder is not None and targets is not None)
        )
        pred_ego, pred_actor = self.head(slots, ego_x=ego_x)

        pred = {
            "actor": pred_actor,
            "ego": pred_ego,
            "attn": attn_masks,
            "recon_feats": aux_preds.get("recon_feats"),
            "target_feats": aux_preds.get("target_feats"),
        }

        output = {}
        if targets is not None:
            loss_dict = self._compute_losses(pred, targets)
            output.update(loss_dict)

        preds = self._post_process(pred)
        output["preds"] = preds
        return output

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ):
        """Standard nn.Module forward dispatching to forward_train or forward_val."""
        if self.training:
            return self.forward_train(inputs, targets=targets, **kwargs)
        else:
            return self.forward_val(inputs, targets=targets, **kwargs)
