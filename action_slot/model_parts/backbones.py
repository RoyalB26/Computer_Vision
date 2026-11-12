"""Modular Video Backbones with registration."""

from abc import ABC, abstractmethod
import hashlib
import logging
import os
import ssl
from typing import List, Optional, Union
import urllib.request
import torch
import torch.nn as nn

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
    for candidate in (cache_path, "./X3D_M.pyth", "./checkpoints/X3D_M.pyth", "/content/data/X3D_M.pyth"):
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
    """Pretrained X3D-M backbone from PyTorchVideo hub."""

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
        # Blocks: [0: Stem, 1: ResStage1, 2: ResStage2, 3: ResStage3, 4: ResStage4]
        self.blocks = x3d_hub.blocks[:-1]
        self._out_channels = 192
        self.freeze_bn = freeze_bn

        # Determine number of stages to freeze
        if freeze_all:
            self.freeze_stages = len(self.blocks)
        elif unfreeze_all:
            self.freeze_stages = 0
        elif finetune_last_n_stages is not None:
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
        # Keep frozen blocks (especially BatchNorm running stats) in eval mode
        if mode and self.freeze_bn and self.freeze_stages > 0:
            for block in self.blocks[:self.freeze_stages]:
                block.eval()

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(x, list):
            # Shape transition: list of T * (B, C, H, W) -> (B, C, T, H, W)
            x = torch.stack(x, dim=0).permute((1, 2, 0, 3, 4))

        for block in self.blocks:
            x = block(x)
        return x
