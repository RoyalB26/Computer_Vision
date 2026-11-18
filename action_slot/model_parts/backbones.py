"""Modular Video Backbones for Toy Action models."""

from abc import ABC, abstractmethod
import hashlib
import logging
import os
import ssl
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolo.yolov11 import YOLOv11, YOLOv11Backbone, YOLOv11Neck, _extract_state_dict
from .movinet import (
    MOVINET_A0_STREAM_WEIGHTS_URL,
    MOVINET_A0_WEIGHTS_URL,
    MoViNetA0Features,
)
_LOGGER = logging.getLogger("BaseModel.Train")
_X3D_M_HTTP_URL = "http://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_M.pyth"
_X3D_M_SHA256 = "eb9583d5f5a988a5c5f06f75760b070a5c9715794bd3b0536b78bd30595dcf67"


def _is_ssl_cert_error(exc: BaseException) -> bool:
    """Return True when an exception chain originates from an SSL certificate verification error."""
    stack = [exc]
    visited = set()
    while stack:
        curr = stack.pop()
        if curr is None or id(curr) in visited:
            continue
        visited.add(id(curr))
        if isinstance(curr, ssl.SSLError):
            return True
        msg = str(curr)
        if "CERTIFICATE_VERIFY_FAILED" in msg or "certificate has expired" in msg.lower():
            return True
        for attr in ("reason", "__cause__", "__context__"):
            nxt = getattr(curr, attr, None)
            if isinstance(nxt, BaseException):
                stack.append(nxt)
    return False


def _build_x3d_m_hub(pretrained: bool = False) -> nn.Module:
    """Instantiate PyTorchVideo X3D-M via installed package or torch.hub fallback."""
    try:
        from pytorchvideo.models.hub import x3d_m
        return x3d_m(pretrained=pretrained)
    except ImportError:
        return torch.hub.load("facebookresearch/pytorchvideo:main", "x3d_m", pretrained=pretrained)


def _load_x3d_weights(model: nn.Module, ckpt_path: str) -> nn.Module:
    """Load an X3D-M checkpoint safely into an instantiated X3D-M module."""
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model


def _resolve_local_x3d_checkpoint(pretrained_path: Optional[str]) -> Optional[str]:
    """Locate a pre-downloaded X3D_M.pyth file if explicitly configured or already cached."""
    if pretrained_path is not None:
        if os.path.isfile(pretrained_path):
            return pretrained_path
        raise FileNotFoundError(f"Specified X3D-M checkpoint does not exist: {pretrained_path}")

    env_path = os.environ.get("X3D_M_WEIGHTS_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    cache_path = os.path.join(torch.hub.get_dir(), "checkpoints", "X3D_M.pyth")
    for candidate in (
        cache_path,
        "./X3D_M.pyth",
        "./checkpoints/X3D_M.pyth",
        os.path.expanduser("~/.cache/torch/hub/checkpoints/X3D_M.pyth"),
    ):
        if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            return candidate
    return None


def _download_x3d_verified_http(dest_path: str) -> str:
    """Download X3D_M.pyth over HTTP and verify its exact SHA-256 digest before caching."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = f"{dest_path}.{os.getpid()}.tmp"
    sha256 = hashlib.sha256()
    try:
        with urllib.request.urlopen(_X3D_M_HTTP_URL, timeout=60) as resp, open(tmp_path, "wb") as out_f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                sha256.update(chunk)
                out_f.write(chunk)
        digest = sha256.hexdigest()
        if digest != _X3D_M_SHA256:
            raise RuntimeError(
                f"SHA-256 mismatch for downloaded X3D_M.pyth: got {digest}, expected {_X3D_M_SHA256}"
            )
        os.replace(tmp_path, dest_path)
        return dest_path
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def load_x3d_m_backbone(pretrained: bool = True, pretrained_path: Optional[str] = None) -> nn.Module:
    """Load X3D-M model with local cache lookup, HTTPS primary download, and SHA-256-verified HTTP fallback."""
    if not pretrained:
        return _build_x3d_m_hub(pretrained=False)

    local_ckpt = _resolve_local_x3d_checkpoint(pretrained_path)
    if local_ckpt is not None:
        return _load_x3d_weights(_build_x3d_m_hub(pretrained=False), local_ckpt)

    try:
        return _build_x3d_m_hub(pretrained=True)
    except Exception as exc:
        if not _is_ssl_cert_error(exc):
            raise
        cache_path = os.path.join(torch.hub.get_dir(), "checkpoints", "X3D_M.pyth")
        _LOGGER.warning(
            "HTTPS download for X3D_M.pyth failed due to upstream SSL certificate verification (%s). "
            "Falling back to SHA-256-verified HTTP download -> %s",
            exc,
            cache_path,
        )
        _download_x3d_verified_http(cache_path)
        return _load_x3d_weights(_build_x3d_m_hub(pretrained=False), cache_path)


def _freeze_batchnorm_stats(module: nn.Module) -> None:
    """Keep BatchNorm running stats in eval mode inside ``module``."""
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


class BaseBackbone(nn.Module, ABC):
    """Abstract base class for all spatiotemporal feature extractors."""

    @abstractmethod
    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Extract spatiotemporal feature volume.

        Returns:
            torch.Tensor: Feature map of shape (B, C_out, T_feat, H_feat, W_feat)
        """
        pass

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Output feature dimension of the backbone."""
        pass


class X3DBackbone(BaseBackbone):
    """Pretrained X3D-M backbone with flexible stage freezing.

    Blocks in X3D-M:
        - Block 0: Stem (stride 2)
        - Block 1: ResStage1 (stride 2)
        - Block 2: ResStage2 (stride 2)
        - Block 3: ResStage3 (stride 2)
        - Block 4: ResStage4 (stride 2)
    Total 5 blocks. Output channels = 192.
    """

    def __init__(
        self,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        freeze_stages: Optional[int] = None,
        finetune_last_n_stages: Optional[int] = 2,
        freeze_all: bool = False,
        unfreeze_all: bool = False,
        freeze_bn: bool = True,
    ):
        super().__init__()
        x3d_hub = load_x3d_m_backbone(pretrained=pretrained, pretrained_path=pretrained_path)
        # All residual stages except the final 1x1x1 classification head
        self.blocks = x3d_hub.blocks[:-1]
        self._out_channels = 192
        self.freeze_bn = freeze_bn

        # Determine number of stages to freeze
        if freeze_all:
            self.freeze_stages = len(self.blocks)
        elif unfreeze_all:
            self.freeze_stages = 0
        elif finetune_last_n_stages is not None:
            # e.g., 5 blocks total, finetune_last_n_stages=2 -> freeze first 3 blocks (Stem, Stage1, Stage2)
            self.freeze_stages = max(0, len(self.blocks) - finetune_last_n_stages)
        elif freeze_stages is not None:
            self.freeze_stages = freeze_stages
        else:
            self.freeze_stages = 3  # Default: freeze stem + first 2 stages (fine-tune last 2 stages)

        # Configure parameter freezing
        if self.freeze_stages >= len(self.blocks):
            for param in self.blocks.parameters():
                param.requires_grad = False
        elif self.freeze_stages <= 0:
            for param in self.blocks.parameters():
                param.requires_grad = True
        else:
            # Freeze the first self.freeze_stages blocks
            for block in self.blocks[:self.freeze_stages]:
                for param in block.parameters():
                    param.requires_grad = False
            # Unfreeze the remaining blocks (e.g. last 2 stages)
            for block in self.blocks[self.freeze_stages:]:
                for param in block.parameters():
                    param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen blocks in eval mode, and freeze BatchNorm running stats across all blocks when freeze_bn=True
        if mode and self.freeze_stages > 0:
            for block in self.blocks[:self.freeze_stages]:
                block.eval()
        if mode and self.freeze_bn:
            _freeze_batchnorm_stats(self)
        return self

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(x, list):
            # list of T * (B, C, H, W) -> (B, C, T, H, W)
            x = torch.stack(x, dim=0).permute((1, 2, 0, 3, 4))

        for block in self.blocks:
            x = block(x)
        return x

class YoloV11Backbone(BaseBackbone):
    """Modular YOLOv11 Backbone + Neck feature extractor for video recognition.

    Extracts high-level multi-scale features through the YOLOv11 backbone,
    fuses them through the PANet Neck, pushes P5 semantics up into P4 spatial
    resolution (stride 16, 256 channels), and optionally blends P3 detail.
    For video inputs with temporal dimension (B, C=3, T=16, H=224, W=224),
    frames are processed through 2D convolutions and re-assembled into
    a spatiotemporal volume (B, C=256, T=16, H=14, W=14).

    Args:
        pretrained: Whether to load pretrained weights.
        pretrained_path: Path to pretrained weights (.pt). Default: 'D:/Study/Projects/Computer_Vision/yolo11n.pt'.
        freeze_stages: Explicit number of initial stages to freeze (0 to 19).
        finetune_last_n_stages: Number of final stages to keep trainable (default: 2).
        freeze_all: If True, freeze all backbone and neck parameters.
        unfreeze_all: If True, keep all parameters trainable.
        freeze_bn: Keep BatchNorm running stats in eval mode for frozen stages.
    """

    DEFAULT_PRETRAINED_PATH = "D:/Study/Projects/Computer_Vision/yolo11n.pt"

    def __init__(
        self,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        freeze_stages: Optional[int] = None,
        finetune_last_n_stages: Optional[int] = 2,
        freeze_all: bool = False,
        unfreeze_all: bool = False,
        freeze_bn: bool = True,
    ):
        super().__init__()
        self.backbone = YOLOv11Backbone()
        self.neck = YOLOv11Neck()
        self.p3_proj = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.p4_proj = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.p4_fuse = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256, bias=False),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.SiLU(inplace=True),
        )
        self._out_channels = 256  # Fused P4 output channel dimension (14x14 for 224x224)
        self.freeze_bn = freeze_bn

        # Group modular components into 19 sequential stages:
        # Stages 0..10: Backbone stage0..stage10
        # Stages 11..18: Neck up11, c3k2_13, up14, c3k2_16, conv17, c3k2_19, conv20, c3k2_22
        self.stages = nn.ModuleList([
            self.backbone.stage0,
            self.backbone.stage1,
            self.backbone.stage2,
            self.backbone.stage3,
            self.backbone.stage4,
            self.backbone.stage5,
            self.backbone.stage6,
            self.backbone.stage7,
            self.backbone.stage8,
            self.backbone.stage9,
            self.backbone.stage10,
            self.neck.up11,
            self.neck.c3k2_13,
            self.neck.up14,
            self.neck.c3k2_16,
            self.neck.conv17,
            self.neck.c3k2_19,
            self.neck.conv20,
            self.neck.c3k2_22,
        ])

        # Load pretrained weights
        if pretrained:
            resolved_path = self._resolve_pretrained_path(pretrained_path)
            if resolved_path:
                self.load_pretrained(resolved_path)
            else:
                _LOGGER.warning(
                    "[YoloV11Backbone] Pretrained weights enabled, but no valid checkpoint found at '%s'.",
                    pretrained_path or self.DEFAULT_PRETRAINED_PATH,
                )

        # Determine stage freezing
        total_stages = len(self.stages)
        if freeze_all:
            self.freeze_stages = total_stages
        elif unfreeze_all:
            self.freeze_stages = 0
        elif finetune_last_n_stages is not None:
            self.freeze_stages = max(0, total_stages - finetune_last_n_stages)
        elif freeze_stages is not None:
            self.freeze_stages = freeze_stages
        else:
            self.freeze_stages = max(0, total_stages - 2)

        # Apply parameter requires_grad configuration
        if self.freeze_stages >= total_stages:
            for p in self.parameters():
                p.requires_grad = False
        elif self.freeze_stages <= 0:
            for p in self.parameters():
                p.requires_grad = True
        else:
            for stage in self.stages[:self.freeze_stages]:
                for p in stage.parameters():
                    p.requires_grad = False
            for stage in self.stages[self.freeze_stages:]:
                for p in stage.parameters():
                    p.requires_grad = True

        # Keep multi-scale P3/P4 projections & P4 fusion layer aligned with freeze_all flag
        for proj in (self.p3_proj, self.p4_proj, self.p4_fuse):
            for p in proj.parameters():
                p.requires_grad = not freeze_all

    def _resolve_pretrained_path(self, pretrained_path: Optional[str]) -> Optional[str]:
        """Check user-provided path, local candidate paths, or default path."""
        candidates = []
        if pretrained_path is not None:
            candidates.append(pretrained_path)
        candidates.extend([
            self.DEFAULT_PRETRAINED_PATH,
            "./yolo11n.pt",
            "yolo11n.pt",
            os.path.join(os.getcwd(), "yolo11n.pt"),
        ])
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def load_pretrained(self, weights_path: Union[str, Path]) -> Tuple[List[str], List[str]]:
        """Load pretrained weights into both YOLOv11 Backbone and Neck."""
        weights_file = Path(weights_path)
        if not weights_file.exists():
            _LOGGER.warning("[YoloV11Backbone] Pretrained file %s not found. Skipping.", weights_file)
            return [], []

        raw_sd = _extract_state_dict(weights_file)
        prefix_map = {
            "model.0.": "backbone.stage0.",
            "model.1.": "backbone.stage1.",
            "model.2.": "backbone.stage2.",
            "model.3.": "backbone.stage3.",
            "model.4.": "backbone.stage4.",
            "model.5.": "backbone.stage5.",
            "model.6.": "backbone.stage6.",
            "model.7.": "backbone.stage7.",
            "model.8.": "backbone.stage8.",
            "model.9.": "backbone.stage9.",
            "model.10.": "backbone.stage10.",
            "model.13.": "neck.c3k2_13.",
            "model.16.": "neck.c3k2_16.",
            "model.17.": "neck.conv17.",
            "model.19.": "neck.c3k2_19.",
            "model.20.": "neck.conv20.",
            "model.22.": "neck.c3k2_22.",
        }

        bb_sd = {}
        neck_sd = {}
        for k, v in raw_sd.items():
            for old_p, new_p in prefix_map.items():
                if k.startswith(old_p):
                    mapped_k = k.replace(old_p, new_p, 1)
                    if mapped_k.startswith("backbone."):
                        bb_sd[mapped_k.replace("backbone.", "", 1)] = v
                    elif mapped_k.startswith("neck."):
                        neck_sd[mapped_k.replace("neck.", "", 1)] = v
                    break

        missing_b, unexp_b = self.backbone.load_state_dict(bb_sd, strict=False)
        missing_n, unexp_n = self.neck.load_state_dict(neck_sd, strict=False)

        total_loaded = len(bb_sd) + len(neck_sd)
        total_missing = len(missing_b) + len(missing_n)
        _LOGGER.info(
            "[YoloV11Backbone] Loaded %d pretrained tensors from %s (missing: %d, unexpected: %d)",
            total_loaded,
            weights_file,
            total_missing,
            len(unexp_b) + len(unexp_n),
        )
        return missing_b + missing_n, unexp_b + unexp_n

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen stages in eval mode, and freeze BatchNorm running stats when freeze_bn=True
        if mode and self.freeze_stages > 0:
            for stage in self.stages[:self.freeze_stages]:
                stage.eval()
        if mode and self.freeze_bn:
            _freeze_batchnorm_stats(self)
        return self

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def _fuse_neck_features(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> torch.Tensor:
        """Fuse neck features at P4 spatial resolution (14x14), pushing P5 semantics up to P4."""
        h_out, w_out = p4.shape[-2], p4.shape[-1]
        p3_down = F.adaptive_avg_pool2d(p3, (h_out, w_out)) + F.adaptive_max_pool2d(p3, (h_out, w_out))
        p4_feat = self.p4_proj(p4)
        p5_up = F.interpolate(p5, size=(h_out, w_out), mode="bilinear", align_corners=False)
        fused = p4_feat + p5_up + 0.5 * self.p3_proj(p3_down)
        return p4_feat + self.p4_fuse(fused)

    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Forward pass extracting P4-resolution fused feature map (pushing P5 semantics into P4).

        Args:
            x: Input video tensor of shape (B, C=3, T=16, H=224, W=224)
               or (B, T=16, C=3, H=224, W=224), or list of T frames.

        Returns:
            torch.Tensor: Spatiotemporal feature volume of shape (B, C=256, T=16, H=14, W=14).
        """
        if isinstance(x, list):
            # List of T * (B, C, H, W) -> (B, C, T, H, W)
            x = torch.stack(x, dim=0).permute(1, 2, 0, 3, 4)

        if x.dim() == 4:
            # Single image or already flat batch of frames: (B, C, H, W)
            feat4, feat6, feat10 = self.backbone(x)
            p3, p4, p5 = self.neck(feat4, feat6, feat10)
            return self._fuse_neck_features(p3, p4, p5)

        if x.dim() != 5:
            raise ValueError(f"Expected 4D or 5D tensor, got {x.dim()}D tensor with shape {x.shape}")

        # Check tensor layout: ensure (B, C, T, H, W)
        if x.shape[1] != 3 and x.shape[2] == 3:
            # (B, T, C, H, W) -> (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)

        b, c, t, h, w = x.shape
        # Reshape to 2D frames for YOLOv11: (B, C, T, H, W) -> (B*T, C, H, W)
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        feat4, feat6, feat10 = self.backbone(x_2d)
        p3, p4, p5 = self.neck(feat4, feat6, feat10)
        p4_fused = self._fuse_neck_features(p3, p4, p5)
        # p4_fused shape: (B*T, 256, H_feat=14, W_feat=14)

        # Reshape back to 5D spatiotemporal volume: (B, C=256, T=16, H=14, W=14)
        _, c_out, h_out, w_out = p4_fused.shape
        p4_5d = p4_fused.reshape(b, t, c_out, h_out, w_out).permute(0, 2, 1, 3, 4).contiguous()
        return p4_5d


class MoViNetA0Backbone(BaseBackbone):
    """Pretrained MoViNet-A0 backbone with flexible stage freezing.

    Stages in MoViNet-A0 (7 sequential stages):
        - Stage 0: ``conv1`` stem (stride 2 -> 112x112, 8 ch)
        - Stage 1: ``b0`` / Block 2 (1 layer, stride 4 -> 56x56, 8 ch)
        - Stage 2: ``b1`` / Block 3 (3 layers, stride 8 -> 28x28, 32 ch)
        - Stage 3: ``b2`` / Block 4 (3 layers, stride 16 -> 14x14, 56 ch)
        - Stage 4: ``b3`` / Block 5 (4 layers, stride 16 -> 14x14, 56 ch)
        - Stage 5: ``b4`` / Block 6 (4 layers, stride 32 -> 7x7, 104 ch)
        - Stage 6: ``conv7`` projection (480 ch)
    Total output shape for ``(B, 3, 16, 224, 224)`` is ``(B, 480, 16, 7, 7)``.
    """

    def __init__(
        self,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        causal: bool = False,
        freeze_stages: Optional[int] = None,
        finetune_last_n_stages: Optional[int] = 2,
        freeze_all: bool = False,
        unfreeze_all: bool = False,
        freeze_bn: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.freeze_bn = freeze_bn
        self._out_channels = 480

        self.features = MoViNetA0Features(
            causal=causal,
            conv_type="2plus1d" if causal else "3d",
            tf_like=True,
        )

        # Group layers into 7 sequential stage groups without duplicating modules in state_dict
        self._stage_groups: List[List[nn.Module]] = [
            [self.features.conv1],
            *[
                [getattr(self.features.blocks, name) for name in block_names]
                for block_names in self.features.stage_block_names
            ],
            [self.features.conv7],
        ]

        if pretrained or pretrained_path is not None:
            self.load_pretrained(pretrained_path=pretrained_path)

        total_stages = len(self._stage_groups)
        if freeze_all:
            self.freeze_stages = total_stages
        elif unfreeze_all:
            self.freeze_stages = 0
        elif finetune_last_n_stages is not None:
            self.freeze_stages = max(0, total_stages - finetune_last_n_stages)
        elif freeze_stages is not None:
            self.freeze_stages = freeze_stages
        else:
            self.freeze_stages = max(0, total_stages - 2)

        for stage_idx, stage_modules in enumerate(self._stage_groups):
            trainable = stage_idx >= self.freeze_stages
            for mod in stage_modules:
                for param in mod.parameters():
                    param.requires_grad = trainable

    def _resolve_local_checkpoint(self, pretrained_path: Optional[str]) -> Optional[str]:
        """Locate an explicit or locally cached MoViNet-A0 checkpoint file."""
        if pretrained_path is not None:
            if os.path.isfile(pretrained_path):
                return pretrained_path
            raise FileNotFoundError(f"Specified MoViNet-A0 checkpoint does not exist: {pretrained_path}")

        filename = "modelA0_stream_statedict_v3" if self.causal else "modelA0_statedict_v3"
        env_var = "MOVINET_A0_STREAM_WEIGHTS_PATH" if self.causal else "MOVINET_A0_WEIGHTS_PATH"
        env_path = os.environ.get(env_var)
        if env_path and os.path.isfile(env_path):
            return env_path

        cache_path = os.path.join(torch.hub.get_dir(), "checkpoints", filename)
        for candidate in (
            cache_path,
            f"./{filename}",
            f"./checkpoints/{filename}",
            os.path.expanduser(f"~/.cache/torch/hub/checkpoints/{filename}"),
        ):
            if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                return candidate
        return None

    @staticmethod
    def _filter_feature_state_dict(raw_ckpt: Any) -> Dict[str, torch.Tensor]:
        """Extract and align MoViNet backbone state_dict keys (conv1, blocks, conv7)."""
        if isinstance(raw_ckpt, dict):
            if "state_dict" in raw_ckpt:
                sd = raw_ckpt["state_dict"]
            elif "model_state" in raw_ckpt:
                sd = raw_ckpt["model_state"]
            elif "model" in raw_ckpt:
                m = raw_ckpt["model"]
                sd = m.state_dict() if hasattr(m, "state_dict") else m
            else:
                sd = raw_ckpt
        elif hasattr(raw_ckpt, "state_dict"):
            sd = raw_ckpt.state_dict()
        else:
            raise ValueError(f"Unsupported checkpoint object type: {type(raw_ckpt)}")

        filtered: Dict[str, torch.Tensor] = {}
        for key, val in sd.items():
            clean_key = key
            for prefix in ("module.", "backbone.", "features."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            if clean_key.startswith(("conv1.", "blocks.", "conv7.")):
                filtered[clean_key] = val
        return filtered

    def load_pretrained(self, pretrained_path: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """Load Kinetics-600 pretrained weights into MoViNet-A0 backbone stages."""
        local_ckpt = self._resolve_local_checkpoint(pretrained_path)
        if local_ckpt is not None:
            try:
                raw_ckpt = torch.load(local_ckpt, map_location="cpu", weights_only=True)
            except TypeError:
                raw_ckpt = torch.load(local_ckpt, map_location="cpu")
            source_desc = local_ckpt
        else:
            url = MOVINET_A0_STREAM_WEIGHTS_URL if self.causal else MOVINET_A0_WEIGHTS_URL
            filename = "modelA0_stream_statedict_v3" if self.causal else "modelA0_statedict_v3"
            raw_ckpt = torch.hub.load_state_dict_from_url(
                url,
                map_location="cpu",
                file_name=filename,
                progress=True,
            )
            source_desc = url

        feature_sd = self._filter_feature_state_dict(raw_ckpt)
        missing, unexpected = self.features.load_state_dict(feature_sd, strict=False)
        _LOGGER.info(
            "[MoViNetA0Backbone] Loaded %d pretrained tensors from %s (missing: %d, unexpected: %d)",
            len(feature_sd),
            source_desc,
            len(missing),
            len(unexpected),
        )
        if missing:
            _LOGGER.warning("[MoViNetA0Backbone] Missing keys while loading pretrained weights: %s", missing)
        return list(missing), list(unexpected)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_stages > 0:
            for stage_modules in self._stage_groups[:self.freeze_stages]:
                for mod in stage_modules:
                    mod.eval()
        if mode and self.freeze_bn:
            _freeze_batchnorm_stats(self)
        return self

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(x, list):
            x = torch.stack(x, dim=0).permute(1, 2, 0, 3, 4)

        if x.dim() != 5:
            raise ValueError(f"Expected 5D video tensor, got {x.dim()}D tensor with shape {x.shape}")

        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4)

        return self.features(x)


class MobileNetV3Backbone(BaseBackbone):
    """MobileNetV3 Spatiotemporal Feature Extractor with optional P4 Neck Fusion.

    Extracts multi-scale features through MobileNetV3 (Small or Nano variant).
    When ``fuse_neck=True`` (default), fuses multi-scale features at P4 spatial
    resolution (14x14 for 224x224 input, stride 16), pushing P5 high-level semantics
    up into P4, and blending P3 spatial detail.

    Args:
        variant: "nano" (width_mult=0.5) or "small" (width_mult=1.0).
        width_mult: Custom channel multiplier if variant is None. Defaults to 0.5 (Nano).
        fuse_neck: Whether to fuse neck features at P4 spatial resolution (pushing P5 semantics up to P4). Default: True.
        out_channels: Projected channel dimension after P4 neck fusion (default: 256 when fuse_neck=True).
        pretrained: Whether to load ImageNet-1k pretrained weights.
        pretrained_path: Optional explicit path to pretrained checkpoint (.pth / .pt).
        freeze_stages: Number of stages to freeze (0 to 5).
        finetune_last_n_stages: Number of final stages to keep trainable (default: 2).
        freeze_all: If True, freeze all backbone parameters.
        unfreeze_all: If True, keep all parameters trainable.
        freeze_bn: Keep BatchNorm running stats in eval mode for frozen stages.
    """

    def __init__(
        self,
        variant: str = "nano",
        width_mult: Optional[float] = None,
        fuse_neck: bool = True,
        out_channels: Optional[int] = None,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        freeze_stages: Optional[int] = None,
        finetune_last_n_stages: Optional[int] = 2,
        freeze_all: bool = False,
        unfreeze_all: bool = False,
        freeze_bn: bool = True,
    ):
        super().__init__()
        import torchvision.models as tv_models

        # Resolve width multiplier
        if width_mult is not None:
            self.width_mult = float(width_mult)
        elif variant.lower() in ("nano", "mobilenetv3_nano", "mobilenet_v3_nano", "0.5", "0.50"):
            self.width_mult = 0.5
        else:
            self.width_mult = 1.0

        self.freeze_bn = freeze_bn
        self.fuse_neck = bool(fuse_neck)

        # Instantiate torchvision MobileNetV3 Small features
        tv_model = tv_models.mobilenet_v3_small(width_mult=self.width_mult)
        self.features = tv_model.features

        # Determine raw stage output channels:
        # P3: features[3] (stride 8 -> 28x28 for 224x224): 16 (nano) / 24 (small)
        # P4: features[8] (stride 16 -> 14x14 for 224x224): 24 (nano) / 48 (small)
        # P5: features[12] (stride 32 -> 7x7 for 224x224): 288 (nano) / 576 (small)
        p3_mod = self.features[3]
        p4_mod = self.features[8]
        p5_mod = self.features[12]
        self.p3_raw_channels = list(p3_mod.block.children())[-1][0].out_channels
        self.p4_raw_channels = list(p4_mod.block.children())[-1][0].out_channels
        self.p5_raw_channels = p5_mod[0].out_channels

        if self.fuse_neck:
            target_out = int(out_channels) if out_channels is not None else 256
            self._out_channels = target_out
            self.p3_proj = nn.Conv2d(self.p3_raw_channels, target_out, kernel_size=1, bias=False)
            self.p4_proj = nn.Conv2d(self.p4_raw_channels, target_out, kernel_size=1, bias=False)
            self.p5_proj = nn.Conv2d(self.p5_raw_channels, target_out, kernel_size=1, bias=False)
            num_gn_groups = 16 if target_out % 16 == 0 else (8 if target_out % 8 == 0 else 1)
            self.p4_fuse = nn.Sequential(
                nn.Conv2d(target_out, target_out, kernel_size=3, padding=1, groups=target_out, bias=False),
                nn.Conv2d(target_out, target_out, kernel_size=1, bias=False),
                nn.GroupNorm(num_gn_groups, target_out),
                nn.SiLU(inplace=True),
            )
        else:
            self._out_channels = self.p5_raw_channels
            self.p3_proj = None
            self.p4_proj = None
            self.p5_proj = None
            self.p4_fuse = None

        # Group 13 feature modules into 5 natural sequential stages:
        # Stage 0: features[0] (Conv2d 3->16*w, stride 2) -> 112x112
        # Stage 1: features[1] (bneck, stride 2) -> 56x56
        # Stage 2: features[2:4] (bneck, stride 2) -> 28x28 (P3)
        # Stage 3: features[4:9] (bneck, stride 2) -> 14x14 (P4)
        # Stage 4: features[9:13] (bneck, stride 2) -> 7x7 (P5)
        self._stage_groups: List[List[nn.Module]] = [
            [self.features[0]],
            [self.features[1]],
            list(self.features[2:4]),
            list(self.features[4:9]),
            list(self.features[9:13]),
        ]

        # Load pretrained weights
        if pretrained or pretrained_path is not None:
            self.load_pretrained(pretrained_path=pretrained_path)

        # Stage freezing logic
        total_stages = len(self._stage_groups)
        if freeze_all:
            self.freeze_stages = total_stages
        elif unfreeze_all:
            self.freeze_stages = 0
        elif finetune_last_n_stages is not None:
            self.freeze_stages = max(0, total_stages - finetune_last_n_stages)
        elif freeze_stages is not None:
            self.freeze_stages = freeze_stages
        else:
            self.freeze_stages = max(0, total_stages - 2)

        for idx, stage in enumerate(self._stage_groups):
            req_grad = (idx >= self.freeze_stages)
            for mod in stage:
                for p in mod.parameters():
                    p.requires_grad = req_grad

        # Keep P4 fusion and projections aligned with freeze_all flag
        if self.fuse_neck:
            for proj in (self.p3_proj, self.p4_proj, self.p5_proj, self.p4_fuse):
                for p in proj.parameters():
                    p.requires_grad = not freeze_all

    def load_pretrained(self, pretrained_path: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """Load pretrained weights from torchvision or local file with tensor channel slicing if width_mult != 1.0."""
        import torchvision.models as tv_models
        from torchvision.models import MobileNet_V3_Small_Weights

        if pretrained_path is not None and os.path.isfile(pretrained_path):
            try:
                ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            except Exception:
                ckpt = torch.load(pretrained_path, map_location="cpu")
            if isinstance(ckpt, dict):
                raw_sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
            else:
                raw_sd = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt
            source_desc = pretrained_path
        else:
            tv_ref = tv_models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            raw_sd = tv_ref.features.state_dict()
            source_desc = "torchvision MobileNet_V3_Small_Weights.DEFAULT"

        # Filter and adapt state_dict to self.features
        target_sd = self.features.state_dict()
        adapted_sd = {}
        sliced_count = 0
        direct_count = 0

        for k, v in raw_sd.items():
            # Strip potential prefixes
            clean_k = k
            for pfx in ("features.", "backbone.", "module."):
                if clean_k.startswith(pfx):
                    clean_k = clean_k[len(pfx):]

            if clean_k in target_sd:
                tgt_shape = target_sd[clean_k].shape
                if v.shape == tgt_shape:
                    adapted_sd[clean_k] = v
                    direct_count += 1
                elif len(v.shape) == len(tgt_shape):
                    # Slice tensor channels for width_mult < 1.0 (e.g. Nano 0.50x)
                    slices = tuple(slice(0, s) for s in tgt_shape)
                    adapted_sd[clean_k] = v[slices]
                    sliced_count += 1

        missing, unexpected = self.features.load_state_dict(adapted_sd, strict=False)
        _LOGGER.info(
            "[MobileNetV3Backbone] Loaded weights from %s (direct: %d, sliced: %d, missing: %d, unexpected: %d)",
            source_desc,
            direct_count,
            sliced_count,
            len(missing),
            len(unexpected),
        )
        return list(missing), list(unexpected)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_stages > 0:
            for stage_modules in self._stage_groups[:self.freeze_stages]:
                for mod in stage_modules:
                    mod.eval()
        if mode and self.freeze_bn:
            _freeze_batchnorm_stats(self)
        return self

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def _extract_stages(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-scale feature maps P3, P4, P5 sequentially through MobileNetV3."""
        out = x
        for layer in self.features[:4]:
            out = layer(out)
        p3 = out

        for layer in self.features[4:9]:
            out = layer(out)
        p4 = out

        for layer in self.features[9:13]:
            out = layer(out)
        p5 = out
        return p3, p4, p5

    def _fuse_neck_features(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> torch.Tensor:
        """Fuse neck features at P4 spatial resolution (14x14), pushing P5 semantics up to P4."""
        h_out, w_out = p4.shape[-2], p4.shape[-1]
        p3_down = F.adaptive_avg_pool2d(p3, (h_out, w_out)) + F.adaptive_max_pool2d(p3, (h_out, w_out))
        p4_feat = self.p4_proj(p4)
        p5_feat = self.p5_proj(p5)
        p5_up = F.interpolate(p5_feat, size=(h_out, w_out), mode="bilinear", align_corners=False)
        fused = p4_feat + p5_up + 0.5 * self.p3_proj(p3_down)
        return p4_feat + self.p4_fuse(fused)

    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Forward pass extracting P4-resolution fused feature map (pushing P5 semantics into P4).

        Args:
            x: Input video tensor of shape (B, C=3, T=16, H, W)
               or (B, T=16, C=3, H, W), or list of T frames.

        Returns:
            torch.Tensor: Spatiotemporal feature volume of shape (B, C_out, T=16, H_feat, W_feat).
        """
        if isinstance(x, list):
            x = torch.stack(x, dim=0).permute(1, 2, 0, 3, 4)

        if x.dim() == 4:
            if self.fuse_neck:
                p3, p4, p5 = self._extract_stages(x)
                return self._fuse_neck_features(p3, p4, p5)
            return self.features(x)

        if x.dim() != 5:
            raise ValueError(f"Expected 4D or 5D tensor, got {x.dim()}D tensor with shape {x.shape}")

        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4)

        b, c, t, h, w = x.shape
        # (B, C, T, H, W) -> (B*T, C, H, W)
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        if self.fuse_neck:
            p3, p4, p5 = self._extract_stages(x_2d)
            feat_2d = self._fuse_neck_features(p3, p4, p5)
        else:
            feat_2d = self.features(x_2d)

        _, c_out, h_out, w_out = feat_2d.shape
        # Reshape back to 5D: (B, C_out, T, H_feat, W_feat)
        feat_5d = feat_2d.reshape(b, t, c_out, h_out, w_out).permute(0, 2, 1, 3, 4).contiguous()
        return feat_5d


class MobileNetV3NanoBackbone(MobileNetV3Backbone):
    """MobileNetV3 Nano Backbone with P4 Neck Fusion (width_mult=0.5, 256 out_channels)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("variant", "nano")
        kwargs.setdefault("width_mult", 0.5)
        super().__init__(**kwargs)


class MobileNetV3SmallBackbone(MobileNetV3Backbone):
    """MobileNetV3 Small Backbone with P4 Neck Fusion (width_mult=1.0, 256 out_channels)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("variant", "small")
        kwargs.setdefault("width_mult", 1.0)
        super().__init__(**kwargs)

