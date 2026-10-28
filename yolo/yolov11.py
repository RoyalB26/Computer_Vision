"""YOLOv11 Full Architecture with Modularity and Weight Loading support."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .backbone import YOLOv11Backbone
from .neck import YOLOv11Neck
from .head import Detect
from .modules import Conv, C3k2, SPPF, C2PSA


def _extract_state_dict(weights_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Helper to extract clean state_dict from a .pt checkpoint or raw state_dict file."""
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            model = ckpt["model"]
            if hasattr(model, "state_dict"):
                return model.state_dict()
            elif isinstance(model, dict):
                return model
        elif "state_dict" in ckpt:
            return ckpt["state_dict"]
        return ckpt
    elif hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()
    raise ValueError(f"Unable to parse state_dict from {weights_path}")


class YOLOv11(nn.Module):
    """Modular YOLOv11 Object Detection Architecture.

    Composed of three distinct modules:
      - Backbone: YOLOv11Backbone (extracts feat4, feat6, feat10)
      - Neck: YOLOv11Neck (PANet fusing features into P3, P4, P5)
      - Head: Detect (predicts boxes and class scores)
    """

    def __init__(self, nc: int = 80, weights: Optional[Union[str, Path]] = None):
        super().__init__()
        self.nc = nc
        self.backbone = YOLOv11Backbone()
        self.neck = YOLOv11Neck()
        self.head = Detect(nc=nc, ch=(64, 128, 256))

        if weights:
            self.load_weights(weights)

    def forward(
        self, x: torch.Tensor
    ) -> Union[Dict[str, Any], Tuple[torch.Tensor, Dict[str, Any]], torch.Tensor]:
        feat4, feat6, feat10 = self.backbone(x)
        p3, p4, p5 = self.neck(feat4, feat6, feat10)
        return self.head([p3, p4, p5])

    def load_weights(self, weights_path: Union[str, Path], strict: bool = True):
        """Load pretrained weights from an Ultralytics .pt checkpoint or standard state_dict."""
        raw_sd = _extract_state_dict(weights_path)

        # Mapping dictionary from Ultralytics sequential layer names (model.X.xxx)
        # to our modular class structure (backbone.*, neck.*, head.*)
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
            "model.23.": "head.",
        }

        new_sd = {}
        for k, v in raw_sd.items():
            matched = False
            for old_p, new_p in prefix_map.items():
                if k.startswith(old_p):
                    new_key = k.replace(old_p, new_p, 1)
                    new_sd[new_key] = v
                    matched = True
                    break
            if not matched:
                new_sd[k] = v

        missing, unexpected = self.load_state_dict(new_sd, strict=strict)
        print(f"Loaded weights from {weights_path}: {len(new_sd)} keys mapped (missing: {len(missing)}, unexpected: {len(unexpected)})")
        return missing, unexpected


class YOLOv11Sequential(nn.Module):
    """Sequential YOLOv11 mapping exactly to the 24 layers of the official specification."""

    def __init__(self, nc: int = 80, weights: Optional[Union[str, Path]] = None):
        super().__init__()
        self.nc = nc

        self.model = nn.ModuleList([
            Conv(3, 16, k=3, s=2),                           # 0
            Conv(16, 32, k=3, s=2),                          # 1
            C3k2(32, 64, n=1, c3k=False, e=0.25),            # 2
            Conv(64, 64, k=3, s=2),                          # 3
            C3k2(64, 128, n=1, c3k=False, e=0.25),           # 4
            Conv(128, 128, k=3, s=2),                        # 5
            C3k2(128, 128, n=1, c3k=True, e=0.5),            # 6
            Conv(128, 256, k=3, s=2),                        # 7
            C3k2(256, 256, n=1, c3k=True, e=0.5),            # 8
            SPPF(256, 256, k=5),                             # 9
            C2PSA(256, 256, n=1, e=0.5),                     # 10
            nn.Upsample(scale_factor=2.0, mode="nearest"),   # 11
            nn.Identity(),                                   # 12
            C3k2(384, 128, n=1, c3k=False, e=0.5),           # 13
            nn.Upsample(scale_factor=2.0, mode="nearest"),   # 14
            nn.Identity(),                                   # 15
            C3k2(256, 64, n=1, c3k=False, e=0.5),            # 16
            Conv(64, 64, k=3, s=2),                          # 17
            nn.Identity(),                                   # 18
            C3k2(192, 128, n=1, c3k=False, e=0.5),           # 19
            Conv(128, 128, k=3, s=2),                        # 20
            nn.Identity(),                                   # 21
            C3k2(384, 256, n=1, c3k=True, e=0.5),            # 22
            Detect(nc=nc, ch=(64, 128, 256)),                # 23
        ])

        if weights:
            self.load_weights(weights)

    def forward(self, x: torch.Tensor):
        y: List[Optional[torch.Tensor]] = [None] * 24

        y[0] = self.model[0](x)
        y[1] = self.model[1](y[0])
        y[2] = self.model[2](y[1])
        y[3] = self.model[3](y[2])
        y[4] = self.model[4](y[3])
        y[5] = self.model[5](y[4])
        y[6] = self.model[6](y[5])
        y[7] = self.model[7](y[6])
        y[8] = self.model[8](y[7])
        y[9] = self.model[9](y[8])
        y[10] = self.model[10](y[9])

        y[11] = self.model[11](y[10])
        y[12] = torch.cat([y[11], y[6]], dim=1)
        y[13] = self.model[13](y[12])

        y[14] = self.model[14](y[13])
        y[15] = torch.cat([y[14], y[4]], dim=1)
        y[16] = self.model[16](y[15])

        y[17] = self.model[17](y[16])
        y[18] = torch.cat([y[17], y[13]], dim=1)
        y[19] = self.model[19](y[18])

        y[20] = self.model[20](y[19])
        y[21] = torch.cat([y[20], y[10]], dim=1)
        y[22] = self.model[22](y[21])

        return self.model[23]([y[16], y[19], y[22]])

    def load_weights(self, weights_path: Union[str, Path], strict: bool = True):
        raw_sd = _extract_state_dict(weights_path)
        missing, unexpected = self.load_state_dict(raw_sd, strict=strict)
        print(f"Loaded weights from {weights_path}: (missing: {len(missing)}, unexpected: {len(unexpected)})")
        return missing, unexpected
