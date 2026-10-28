"""PANet / FPN Neck architecture for YOLOv11."""
from typing import List, Tuple
import torch
import torch.nn as nn
from .modules import Conv, C3k2


class YOLOv11Neck(nn.Module):
    """YOLOv11 PANet Neck (Top-Down and Bottom-Up feature fusion).
    
    Layers:
      11: Upsample(scale_factor=2)
      12: Concat(11, 6) -> 256 + 128 = 384
      13: C3k2(384 -> 128, n=1, c3k=False, e=0.5)
      14: Upsample(scale_factor=2)
      15: Concat(14, 4) -> 128 + 128 = 256
      16: C3k2(256 -> 64, n=1, c3k=False, e=0.5)    -> P3 (stride 8)
      17: Conv(64 -> 64, k=3, s=2)
      18: Concat(17, 13) -> 64 + 128 = 192
      19: C3k2(192 -> 128, n=1, c3k=False, e=0.5)   -> P4 (stride 16)
      20: Conv(128 -> 128, k=3, s=2)
      21: Concat(20, 10) -> 128 + 256 = 384
      22: C3k2(384 -> 256, n=1, c3k=True, e=0.5)    -> P5 (stride 32)
    """

    def __init__(self):
        super().__init__()
        # Top-down pathway
        self.up11 = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.c3k2_13 = C3k2(384, 128, n=1, c3k=False, e=0.5)

        self.up14 = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.c3k2_16 = C3k2(256, 64, n=1, c3k=False, e=0.5)

        # Bottom-up pathway
        self.conv17 = Conv(64, 64, k=3, s=2)
        self.c3k2_19 = C3k2(192, 128, n=1, c3k=False, e=0.5)

        self.conv20 = Conv(128, 128, k=3, s=2)
        self.c3k2_22 = C3k2(384, 256, n=1, c3k=True, e=0.5)

    def forward(
        self, feat4: torch.Tensor, feat6: torch.Tensor, feat10: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            feat4: Feature from layer 4 [B, 128, H/8, W/8]
            feat6: Feature from layer 6 [B, 128, H/16, W/16]
            feat10: Feature from layer 10 [B, 256, H/32, W/32]
        Returns:
            Tuple of fused features (P3, P4, P5)
        """
        # Top-down
        x11 = self.up11(feat10)
        x12 = torch.cat([x11, feat6], dim=1)
        x13 = self.c3k2_13(x12)

        x14 = self.up14(x13)
        x15 = torch.cat([x14, feat4], dim=1)
        p3 = self.c3k2_16(x15)              # P3: [B, 64, H/8, W/8]

        # Bottom-up
        x17 = self.conv17(p3)
        x18 = torch.cat([x17, x13], dim=1)
        p4 = self.c3k2_19(x18)              # P4: [B, 128, H/16, W/16]

        x20 = self.conv20(p4)
        x21 = torch.cat([x20, feat10], dim=1)
        p5 = self.c3k2_22(x21)              # P5: [B, 256, H/32, W/32]

        return p3, p4, p5
