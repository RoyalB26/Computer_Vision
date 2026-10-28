"""Detection Head for YOLOv11."""
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
from .modules import Conv, DWConv, DFL


def make_anchors(
    feats: Union[List[torch.Tensor], torch.Tensor],
    strides: Union[torch.Tensor, List[float], List[int]],
    grid_cell_offset: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate anchor points and stride tensors from feature maps."""
    anchor_points, stride_tensor = [], []
    dtype = feats[0].dtype
    for i in range(len(feats)):
        stride = strides[i]
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(w, device=feats[0].device).type_as(feats[0]) + grid_cell_offset
        sy = torch.arange(h, device=feats[0].device).type_as(feats[0]) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(feats[0].new_full((h * w, 1), stride, dtype=dtype))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = True, dim: int = -1) -> torch.Tensor:
    """Transform distance (ltrb) to box (xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)
    return torch.cat((x1y1, x2y2), dim)


class Detect(nn.Module):
    """YOLOv11 Detection Head."""

    dynamic: bool = False
    export: bool = False

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        ch: Tuple[int, ...] = (64, 128, 256),
    ):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection levels (scales)
        self.reg_max = reg_max  # DFL channels per coordinate (default 16)
        self.no = nc + self.reg_max * 4  # outputs per anchor
        self.stride = torch.tensor([8.0, 16.0, 32.0])  # feature strides

        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))

        # Box regression branch
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )

        # Classification branch with depthwise convolutions
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.shape = None
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)

    def forward(
        self, x: List[torch.Tensor]
    ) -> Union[Dict[str, Any], Tuple[torch.Tensor, Dict[str, Any]], torch.Tensor]:
        """Forward pass.
        
        Args:
            x: List of feature maps [P3, P4, P5].
        Returns:
            In train mode: dict(boxes, scores, feats)
            In eval mode: (decoded_bboxes_and_scores, dict(boxes, scores, feats))
        """
        bs = x[0].shape[0]
        boxes = torch.cat([self.cv2[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([self.cv3[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        preds = {"boxes": boxes, "scores": scores, "feats": x}

        if self.training:
            return preds

        y = self._inference(preds)
        return y if self.export else (y, preds)

    def _inference(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities."""
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), dim=1)

    def _get_decode_boxes(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode bounding boxes in xywh format."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)
