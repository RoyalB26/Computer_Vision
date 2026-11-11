"""Modular Video Backbones with registration."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import torch
import torch.nn as nn


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
        freeze_stages: Optional[int] = None,
        finetune_last_n_stages: Optional[int] = 2,
        freeze_all: bool = False,
        unfreeze_all: bool = False,
        freeze_bn: bool = True,
    ):
        super().__init__()
        try:
            from pytorchvideo.models.hub import x3d_m
            x3d_hub = x3d_m(pretrained=pretrained)
        except Exception:
            x3d_hub = torch.hub.load('facebookresearch/pytorchvideo:main', 'x3d_m', pretrained=pretrained)
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
