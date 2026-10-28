"""Backbone architecture for YOLOv11."""
from typing import List, Tuple
import torch
import torch.nn as nn
from .modules import Conv, C3k2, SPPF, C2PSA


class YOLOv11Backbone(nn.Module):
    """YOLOv11 Backbone feature extractor.
    
    Layers:
      0: Conv(3 -> 16, k=3, s=2)
      1: Conv(16 -> 32, k=3, s=2)
      2: C3k2(32 -> 64, n=1, c3k=False, e=0.25)
      3: Conv(64 -> 64, k=3, s=2)
      4: C3k2(64 -> 128, n=1, c3k=False, e=0.25)   -> P3 backbone feature (to neck)
      5: Conv(128 -> 128, k=3, s=2)
      6: C3k2(128 -> 128, n=1, c3k=True, e=0.5)    -> P4 backbone feature (to neck)
      7: Conv(128 -> 256, k=3, s=2)
      8: C3k2(256 -> 256, n=1, c3k=True, e=0.5)
      9: SPPF(256 -> 256, k=5)
      10: C2PSA(256 -> 256, n=1, e=0.5)           -> P5 backbone feature (to neck)
    """

    def __init__(self):
        super().__init__()
        self.stage0 = Conv(3, 16, k=3, s=2)                          # 0
        self.stage1 = Conv(16, 32, k=3, s=2)                         # 1
        self.stage2 = C3k2(32, 64, n=1, c3k=False, e=0.25)           # 2
        self.stage3 = Conv(64, 64, k=3, s=2)                         # 3
        self.stage4 = C3k2(64, 128, n=1, c3k=False, e=0.25)          # 4 (feat4)
        self.stage5 = Conv(128, 128, k=3, s=2)                       # 5
        self.stage6 = C3k2(128, 128, n=1, c3k=True, e=0.5)           # 6 (feat6)
        self.stage7 = Conv(128, 256, k=3, s=2)                       # 7
        self.stage8 = C3k2(256, 256, n=1, c3k=True, e=0.5)           # 8
        self.stage9 = SPPF(256, 256, k=5)                            # 9
        self.stage10 = C2PSA(256, 256, n=1, e=0.5)                   # 10 (feat10)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor [B, 3, H, W]
        Returns:
            Tuple of feature maps (feat4, feat6, feat10)
        """
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feat4 = self.stage4(x)     # [B, 128, H/8, W/8]

        x = self.stage5(feat4)
        feat6 = self.stage6(x)     # [B, 128, H/16, W/16]

        x = self.stage7(feat6)
        x = self.stage8(x)
        x = self.stage9(x)
        feat10 = self.stage10(x)   # [B, 256, H/32, W/32]

        return feat4, feat6, feat10
