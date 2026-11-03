"""MobileNetV3 Object Detector with DeformableHead for Hand Digit Detection."""
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from BaseModel.core.registrar import MODELS
from BaseModel.model import BaseModel

from .utils import cxcywh_to_xyxy
from .backbone import MobileNetV3Backbone
from .neck import MobileNetFPN
from .head import MobileNetDetectionHead, DeformableHead
from .loss import MobileNetDetectionLoss

@torch.no_grad() # Bỏ nếu bạn cần truyền ngược đạo hàm (backward) qua bước này
def expand_islands_batch_gpu(x: torch.Tensor, connectivity: int = 8, coast_thickness: int = 1):
    """
    Tìm toàn bộ ốc đảo trên kênh đầu tiên x[:, 0] của cả Batch [B, C, H, W]
    và phóng to (interpolate) từng ốc đảo lấn ra biển thêm `coast_thickness` pixel.
    """
    B, C, H, W = x.shape
    device = x.device

    # 1. Xác định mặt đất trên channel 0 cho toàn bộ batch: [B, 1, H, W]
    land_mask = (x[:, 0:1] > 0)
    if not land_mask.any():
        return x

    # 2. Khởi tạo nhãn duy nhất cho mỗi pixel trên toàn bộ Batch (1 -> B*H*W)
    # Dùng float32 nếu B*H*W < 16 triệu, hoặc float64 nếu ảnh/batch cực lớn
    labels = torch.arange(1, B * H * W + 1, device=device, dtype=torch.float32).view(B, 1, H, W)
    labels = labels * land_mask.float()
    land_float = land_mask.float()

    # 3. Lan truyền nhãn song song trên toàn bộ Batch (Connected Components)
    # Tối ưu: Kiểm tra điều kiện dừng mỗi 5 bước để tránh đồng bộ CPU-GPU liên tục
    while True:
        prev_labels = labels
        for _ in range(5):
            if connectivity == 8:
                labels = F.max_pool2d(labels, kernel_size=3, stride=1, padding=1) * land_float
            else:
                h_pool = F.max_pool2d(labels, kernel_size=(1, 3), stride=1, padding=(0, 1))
                v_pool = F.max_pool2d(labels, kernel_size=(3, 1), stride=1, padding=(1, 0))
                labels = torch.maximum(h_pool, v_pool) * land_float
        
        if torch.equal(labels, prev_labels):
            break

    # 4. Trích xuất Bounding Box của MỌI ốc đảo trong 1 lần duy nhất bằng scatter_reduce
    labels_flat = labels.view(-1).long()
    valid_mask = labels_flat > 0
    valid_labels = labels_flat[valid_mask]

    # Nén các nhãn rời rạc về dạng chỉ số liên tiếp: 0, 1, ..., K-1 (K là tổng số đảo trong cả batch)
    unique_labels, inverse_indices = torch.unique(valid_labels, return_inverse=True)
    num_islands = unique_labels.numel()

    # Tạo lưới tọa độ (Batch_idx, Y, X) cho các pixel thuộc đảo
    grid_b, grid_y, grid_x = torch.meshgrid(
        torch.arange(B, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    b_valid = grid_b.reshape(-1)[valid_mask]
    y_valid = grid_y.reshape(-1)[valid_mask]
    x_valid = grid_x.reshape(-1)[valid_mask]

    # Tìm min/max tọa độ Y, X cho toàn bộ K ốc đảo cùng lúc trên GPU
    i_ymin = torch.full((num_islands,), H, device=device, dtype=torch.long)
    i_ymax = torch.full((num_islands,), 0, device=device, dtype=torch.long)
    i_xmin = torch.full((num_islands,), W, device=device, dtype=torch.long)
    i_xmax = torch.full((num_islands,), 0, device=device, dtype=torch.long)
    batch_ids = torch.zeros((num_islands,), device=device, dtype=torch.long)

    i_ymin.scatter_reduce_(0, inverse_indices, y_valid, reduce="amin", include_self=False)
    i_ymax.scatter_reduce_(0, inverse_indices, y_valid, reduce="amax", include_self=False)
    i_xmin.scatter_reduce_(0, inverse_indices, x_valid, reduce="amin", include_self=False)
    i_xmax.scatter_reduce_(0, inverse_indices, x_valid, reduce="amax", include_self=False)
    batch_ids.scatter_(0, inverse_indices, b_valid)

    # Cộng 1 vào cận trên để dùng cho slice Python
    i_ymax += 1
    i_xmax += 1

    # 5. Tính thẳng BBox mục tiêu (đảo + bờ biển) mà KHÔNG CẦN Dilation
    t_ymin = (i_ymin - coast_thickness).clamp(min=0)
    t_ymax = (i_ymax + coast_thickness).clamp(max=H)
    t_xmin = (i_xmin - coast_thickness).clamp(min=0)
    t_xmax = (i_xmax + coast_thickness).clamp(max=W)

    # Chuyển mảng tọa độ (kích thước rất nhỏ: K x 9) về CPU 1 lần duy nhất để chạy vòng lặp cắt ghép
    coords = torch.stack([
        batch_ids, i_ymin, i_ymax, i_xmin, i_xmax, t_ymin, t_ymax, t_xmin, t_xmax
    ], dim=1).tolist()

    # Clone một bản sao để đọc dữ liệu gốc tránh trường hợp các đảo gần nhau ghi đè lên nhau
    x_out = x.clone()

    # 6. Phóng to và gán đè từng đảo
    for b, iy1, iy2, ix1, ix2, ty1, ty2, tx1, tx2 in coords:
        target_h = ty2 - ty1
        target_w = tx2 - tx1
        
        # Nếu đảo đã chạm sát mép ảnh và không mở rộng thêm được thì bỏ qua
        if (iy2 - iy1 == target_h) and (ix2 - ix1 == target_w):
            continue

        island_only = x[b:b+1, :, iy1:iy2, ix1:ix2]
        expanded = F.interpolate(
            island_only,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False
        )
        x_out[b, :, ty1:ty2, tx1:tx2] = expanded.squeeze(0)

    return x_out

def fast_expand_islands(x: torch.Tensor, coast_thickness: int = 1):
    k = 2 * coast_thickness + 1
    # Nở theo chiều ngang rồi nở theo chiều dọc (nhanh gấp K lần so với 2D pool chuẩn)
    out = F.max_pool2d(x, kernel_size=(1, k), stride=1, padding=(0, coast_thickness))
    out = F.max_pool2d(out, kernel_size=(k, 1), stride=1, padding=(coast_thickness, 0))
    return out

def diffuse_islands_into_sea(x: torch.Tensor, coast_thickness: int = 1):
    """
    Toàn bộ các đảo trong Batch [B, C, H, W] tự động lan tỏa mượt mà (như bilinear)
    ra vùng biển xung quanh mà không cần tìm từng đảo.
    """
    k_size = 2 * coast_thickness + 1
    
    # 1. Xác định mặt đất dựa trên channel 0: shape [B, 1, H, W]
    land_mask = (x.mean(dim=1, keepdim=True) > 0).float()
    
    # 2. Chỉ giữ lại giá trị thuộc về đảo, biển gán bằng 0
    island_values = x * land_mask
    
    # 3. Tính tổng giá trị đảo và tổng số pixel đảo trong vùng lân cận k_size x k_size
    # Dùng avg_pool2d nhân lại diện tích kernel để lấy tổng cực nhanh trên GPU
    area = k_size ** 2
    sum_values = F.avg_pool2d(island_values, kernel_size=k_size, stride=1, padding=coast_thickness) * area
    count_land = F.avg_pool2d(land_mask, kernel_size=k_size, stride=1, padding=coast_thickness) * area
    
    # 4. Giá trị nội suy tại vùng biển = Trung bình cộng các pixel đảo lân cận
    # (1e-6 để tránh chia cho 0 ở vùng biển xa không có đảo nào gần)
    interpolated_sea = sum_values / (count_land + 1e-6)
    
    # 5. Giữ nguyên pixel đảo gốc, chỉ đắp giá trị mới vào vùng biển sát đảo
    near_coast_mask = (count_land > 0).float()
    out = x * land_mask + interpolated_sea * (1.0 - land_mask) * near_coast_mask
    
    return out


class LocalStaticSquaredDeviationNorm(nn.Module):

    def __init__(
        self,
        kernel_size: int = 51,
        per_channel: bool = True,
        padding_mode: str = "reflect",
    ):
        """Norm cục bộ (không học):

        Tính mean trong vùng lân cận kích thước `kernel_size x kernel_size`
        (khoảng 50 pixel) xung quanh từng điểm ảnh, sau đó áp dụng công thức:
        out = exp(2 * |x - local_mean|).

        Args:
            kernel_size (int): Kích thước cửa sổ lân cận. Nên là số lẻ
              (mặc định 51 tương ứng bán kính ~25px xung quanh pixel tâm).
            per_channel (bool):
                - True: tính local mean độc lập cho từng channel (chuẩn cho xử
                  lý ảnh).
                - False: tính local mean gộp trên cả các channel lân cận.
            padding_mode (str): Phương pháp bù biên ('reflect', 'replicate', hoặc
              'constant'). 'reflect' giúp vùng biên không bị tối/sai lệch mean.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.per_channel = per_channel
        self.padding_mode = padding_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x (torch.Tensor): Đầu vào dạng 4D [B, C, H, W].
        """
        assert (
            x.ndim == 4
        ), f"Yêu cầu đầu vào 4D [B, C, H, W], nhận được ndim={x.ndim}"

        pad_size = self.kernel_size // 2

        # 1. Đệm biên để kích thước sau avg_pool2d bằng đúng kích thước ban đầu (H, W)
        x_padded = F.pad(
            x,
            (pad_size, pad_size, pad_size, pad_size),
            mode=self.padding_mode,
        )

        # 2. Tính local mean theo từng pixel
        if self.per_channel:
            # avg_pool2d mặc định trượt trên từng channel độc lập
            local_mean = F.avg_pool2d(
                x_padded,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0,
            )
        else:
            # Nếu muốn gộp toàn bộ channels: trung bình theo C trước rồi broadcast
            x_channel_mean = x_padded.mean(dim=1, keepdim=True)
            local_mean = F.avg_pool2d(
                x_channel_mean,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0,
            ).expand_as(x)

        # 3. Áp dụng công thức biến đổi thuần túy
        out = torch.exp(5 * torch.abs(x - local_mean))

        return out


class ConvKernelDownsampleModule(nn.Module):
    """Module GPU thực hiện giảm kích thước ảnh bằng chuỗi n lớp Conv2d:

    - Kernel size: 3x3, tất cả giá trị đều = 1.
    - Stride = 2, Padding = 1.
    - Normalize: chia cho 9.0 (tổng trọng số kernel 3x3) để giữ khoảng [0, 1].
    - Tự động lặp lại n bước Conv trên GPU đến khi ảnh đạt sát target_size,
      sau đó nội suy bilinear về chính xác target_size.
    """

    def __init__(
        self,
        in_channels: int = 3,
        target_size: Union[int, tuple, list] = (64, 64),
        normalize_kernel: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = (int(target_size[0]), int(target_size[1]))

        self.normalize_kernel = normalize_kernel

        # Kernel 3x3 toàn số 1, áp dụng depthwise (groups=in_channels)
        weight = torch.ones((in_channels, 1, 3, 3), dtype=torch.float32)
        if self.normalize_kernel:
            weight = weight / 9.0

        # Đăng ký weight làm buffer (không tính gradient, tự động chuyển theo model sang GPU/CPU)
        self.register_buffer("conv_weight", weight)

        self.norm = LocalStaticSquaredDeviationNorm(kernel_size= 151,per_channel= False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        
        B,C,H,W = x.shape
        
        x = self.norm.forward(x)
        x = F.layer_norm(x, normalized_shape= [H,W])
        x = F.relu(x)
        x= diffuse_islands_into_sea(x, coast_thickness= 15)
        cnt= 1
        target_h, target_w = self.target_size
        while x.shape[2] >= target_h * 2 and x.shape[3] >= target_w * 2:
            x = F.conv2d(
                x,
                self.conv_weight,
                bias=None,
                stride=2,
                padding=1,
                groups=self.in_channels,
            )
            b,c,h,w= x.shape
            if cnt % 2 == 0:
                x = F.layer_norm(x, normalized_shape= [h,w])
            x = F.relu(x)
            cnt+= 1
        # Căn chỉnh về đúng target_size mong muốn nếu còn chênh lệch
        
        if (x.shape[2], x.shape[3]) != (target_h, target_w):
            x = F.interpolate(
                x,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        x = F.layer_norm(x, normalized_shape= [target_h,target_w])

        x = F.relu(x)
        return x


@MODELS.register_module(name="HandDigitMobileNetV3Detector")
class HandDigitMobileNetV3Detector(BaseModel):
    """
    Detector thuần CNN dùng MobileNetV3 backbone kết hợp Deformable Detection Head:
      - Backbone: MobileNetV3-Small (pretrained ImageNet)
      - Neck: Feature Pyramid Network (FPN) đưa về map_pred (Stride 8) & feat_deform (Stride 2)
      - Head: DeformableHead (hoặc MobileNetDetectionHead)
      - Loss: Dense Gaussian Focal Loss + Distance-IoU (DIoU) Loss + Deformable Regularization
    """

    def __init__(
        self,
        num_classes: int = 10,
        dim: int = 64,
        head_type: str = "DeformableHead",  # "DeformableHead" hoặc "MobileNetDetectionHead"
        num_heads: int = 4,
        num_points: int = 4,
        backbone: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
        target_size: int = 160,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.head_type = head_type

        self.downsampler = ConvKernelDownsampleModule(
            in_channels=3,
            target_size=target_size,
            normalize_kernel=False,
        )

        # 1. MobileNetV3 Backbone
        if backbone is not None:
            self.backbone = backbone
            in_channels = getattr(
                backbone, "out_channels", {'c0': 16, 'c1': 16, 'c3': 24, 'c5': 40, 'c12': 576}
            )
        else:
            self.backbone = MobileNetV3Backbone(pretrained=True)
            in_channels = self.backbone.out_channels

        # 2. Feature Pyramid Network (FPN)
        if neck is not None:
            self.neck = neck
        else:
            self.neck = MobileNetFPN(in_channels=in_channels, out_dim=dim, return_tuple=True)

        # 3. Detection Head (DeformableHead mặc định, hoặc Standard Head)
        if head is not None:
            self.head = head
        else:
            if head_type.lower() in ("deformable", "deformablehead", "deformabledetectionhead"):
                self.head = DeformableHead(
                    num_classes=num_classes, dim=dim, num_heads=num_heads, num_points=num_points
                )
            else:
                self.head = MobileNetDetectionHead(num_classes=num_classes, dim=dim, head_conv=dim)

        # 4. Loss Function (Focal Loss + DIoU Loss)
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = MobileNetDetectionLoss(num_classes=num_classes)

    def extract_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Trích xuất đặc trưng qua Backbone và Neck FPN."""
        feats = self.backbone(x)
        fpn_out = self.neck(feats)
        if isinstance(fpn_out, tuple):
            map_pred, feat_deform = fpn_out
            return map_pred, feat_deform
        return fpn_out, None

    def forward_train(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        map_pred, feat_deform = self.extract_features(inputs)
        head_out = self.head(map_pred, feat_deform)

        if len(head_out) == 3:
            cls_logits, pred_boxes, sampling_info = head_out
        else:
            cls_logits, pred_boxes = head_out
            sampling_info = None

        return self.loss_fn(
            cls_logits=cls_logits,
            pred_boxes=pred_boxes,
            targets=targets,
            device=inputs.device,
            sampling_info=sampling_info,
        )

    def _post_process(
        self,
        cls_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        conf_threshold: float = 0.10,
        nms_threshold: float = 0.5,
        max_det: int = 100,
        max_candidates: int = 1000,
    ) -> List[Dict[str, torch.Tensor]]:
        """Lọc peak cực đại cục bộ 3x3 và batched NMS."""
        batch_size, num_classes, grid_h, grid_w = cls_logits.shape
        cls_scores = cls_logits.sigmoid()

        # CenterNet Local Peak Max-Pooling (3x3)
        hmax = F.max_pool2d(cls_scores, kernel_size=3, stride=1, padding=1)
        cls_scores = cls_scores * (cls_scores == hmax).float()

        device = cls_logits.device
        results = []

        for b in range(batch_size):
            b_scores_map = cls_scores[b]  # [C, H, W]
            c_idx, y_idx, x_idx = torch.where(b_scores_map >= conf_threshold)

            if len(c_idx) > 0:
                b_scores = b_scores_map[c_idx, y_idx, x_idx]
                b_labels = c_idx

                if len(b_scores) > max_candidates:
                    topk_idx = torch.topk(b_scores, max_candidates).indices
                    b_scores = b_scores[topk_idx]
                    b_labels = b_labels[topk_idx]
                    y_idx = y_idx[topk_idx]
                    x_idx = x_idx[topk_idx]

                sample_boxes = pred_boxes[b, :, y_idx, x_idx].permute(1, 0)

                if nms_threshold is not None and len(sample_boxes) > 0:
                    xyxy = cxcywh_to_xyxy(sample_boxes)
                    keep = torchvision.ops.batched_nms(xyxy, b_scores, b_labels, nms_threshold)
                    keep = keep[:max_det]
                    sample_boxes = sample_boxes[keep]
                    b_scores = b_scores[keep]
                    b_labels = b_labels[keep]

                results.append({
                    "boxes": sample_boxes,
                    "scores": b_scores,
                    "labels": b_labels,
                })
            else:
                results.append({
                    "boxes": torch.empty((0, 4), device=device),
                    "scores": torch.empty((0,), device=device),
                    "labels": torch.empty((0,), dtype=torch.long, device=device),
                })
        return results

    def predict(
        self,
        inputs: torch.Tensor,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        max_det: int = 100,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        map_pred, feat_deform = self.extract_features(inputs)
        head_out = self.head(map_pred, feat_deform)
        cls_logits, pred_boxes = head_out[0], head_out[1]

        return self._post_process(
            cls_logits, pred_boxes,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            max_det=max_det,
        )

    def forward_val(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        max_det: int = 300,
        **kwargs,
    ) -> Dict[str, Any]:
        """Validation forward pass: tính validation losses và sinh preds phục vụ tính mAP."""
        map_pred, feat_deform = self.extract_features(inputs)
        head_out = self.head(map_pred, feat_deform)

        if len(head_out) == 3:
            cls_logits, pred_boxes, sampling_info = head_out
        else:
            cls_logits, pred_boxes = head_out
            sampling_info = None

        output = {}
        if targets is not None:
            loss_dict = self.loss_fn(
                cls_logits=cls_logits,
                pred_boxes=pred_boxes,
                targets=targets,
                device=inputs.device,
                sampling_info=sampling_info,
            )
            output.update(loss_dict)

            # Chuẩn hóa target boxes sang định dạng [cx, cy, w, h] cho MeanAveragePrecision
            for t in targets:
                if "boxes" in t and len(t["boxes"]) > 0:
                    if not t.get("_is_cxcywh", False):
                        b = t["boxes"]
                        t["boxes"] = torch.stack([
                            b[:, 0] + b[:, 2] / 2.0,
                            b[:, 1] + b[:, 3] / 2.0,
                            b[:, 2],
                            b[:, 3]
                        ], dim=-1)
                        t["_is_cxcywh"] = True

        preds = self._post_process(
            cls_logits, pred_boxes,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            max_det=max_det,
        )
        output["preds"] = preds
        return output

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ):
        if self.training:
            return self.forward_train(inputs, targets=targets, **kwargs)
        else:
            return self.forward_val(inputs, targets=targets, **kwargs)

    def analyze_detection_components(self, inputs: torch.Tensor) -> Dict[str, float]:
        """Thống kê các chỉ số đặc trưng phục vụ ExcelTracker."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            map_pred, feat_deform = self.extract_features(inputs)
            head_out = self.head(map_pred, feat_deform)
            cls_logits, pred_boxes = head_out[0], head_out[1]
            sampling_info = head_out[2] if len(head_out) == 3 else None

            cls_scores = cls_logits.sigmoid()

            results = {
                "Cls_Logits_Mean": round(cls_logits.mean().item(), 4),
                "Cls_Logits_Max": round(cls_logits.max().item(), 4),
                "Cls_Logits_Min": round(cls_logits.min().item(), 4),
                "Cls_Prob_Mean": round(cls_scores.mean().item(), 4),
                "Cls_Prob_Max": round(cls_scores.max().item(), 4),
                "Box_CX_Mean": round(pred_boxes[:, 0].mean().item(), 4),
                "Box_CY_Mean": round(pred_boxes[:, 1].mean().item(), 4),
                "Box_W_Mean": round(pred_boxes[:, 2].mean().item(), 4),
                "Box_H_Mean": round(pred_boxes[:, 3].mean().item(), 4),
                "MapPred_Energy": round(map_pred.norm(dim=1).mean().item(), 4),
            }

            if sampling_info is not None:
                all_locs = sampling_info["sampling_locations"]
                all_offsets = sampling_info["offsets"]
                offset_norm = torch.norm(all_offsets, p=2, dim=-1).mean()
                bound_violation = (
                    F.relu(-all_locs[..., 0]) + F.relu(all_locs[..., 0] - 1.0) +
                    F.relu(-all_locs[..., 1]) + F.relu(all_locs[..., 1] - 1.0)
                ).gt(0).float().mean()
                results["Deform_Offset_Norm"] = round(offset_norm.item(), 4)
                results["Deform_Out_Bound_Ratio"] = round(bound_violation.item(), 4)

        if was_training:
            self.train()
        return results
