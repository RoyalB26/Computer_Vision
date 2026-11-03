from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Union
import torch
from scipy.ndimage import distance_transform_edt

from collections import deque

def find_islands_and_coastlines_gpu(x: torch.Tensor, connectivity: int = 8, coast_thickness: int = 1):
    """
    Chạy 100% trên GPU mà không chuyển dữ liệu về CPU.
    Sử dụng Label Propagation và Multi-step GPU Max-Pooling.
    
    Args:
        x (torch.Tensor): Tensor 2D (H, W) trên CUDA/GPU.
        connectivity (int): 4 hoặc 8 hướng liên thông.
        coast_thickness (int): Bán kính pixel vùng biển bao quanh.
    """
    assert x.ndim == 2, "Tensor phải có dạng (H, W)"
    device = x.device
    H, W = x.shape
    
    # 1. Mặt đất và gán ID ban đầu cho từng pixel
    land_mask = x > 0
    if not land_mask.any():
        return []

    # Mỗi pixel đất ban đầu có 1 ID duy nhất (tọa độ 1D: 1 -> H*W)
    labels = torch.arange(1, H * W + 1, device=device, dtype=torch.int32).view(H, W)
    labels = torch.where(land_mask, labels, torch.zeros_like(labels))
    
    # 2. GPU Connected Components bằng Label Propagation
    # Cho các nhãn lớn nhất lan truyền sang hàng xóm cho đến khi hội tụ
    kernel_size = 3
    padding = 1
    # Biến đổi labels về dạng float 4D để dùng MaxPool2d
    labels_float = labels.float().unsqueeze(0).unsqueeze(0)
    land_float = land_mask.float().unsqueeze(0).unsqueeze(0)

    prev_labels = None
    while True:
        # Lan truyền nhãn theo 4 hoặc 8 hướng
        if connectivity == 8:
            propagated = F.max_pool2d(labels_float, kernel_size=kernel_size, stride=1, padding=padding)
        else:
            # Liên thông 4 hướng: Max theo chiều ngang và dọc riêng
            h_pool = F.max_pool2d(labels_float, kernel_size=(1, 3), stride=1, padding=(0, 1))
            v_pool = F.max_pool2d(labels_float, kernel_size=(3, 1), stride=1, padding=(1, 0))
            propagated = torch.maximum(h_pool, v_pool)
            
        # Chỉ giữ nhãn trên mặt đất
        labels_float = propagated * land_float
        
        # Kiểm tra hội tụ (khi không còn nhãn nào thay đổi)
        if prev_labels is not None and torch.equal(labels_float, prev_labels):
            break
        prev_labels = labels_float.clone()

    unique_labels = torch.unique(labels_float.squeeze())
    unique_labels = unique_labels[unique_labels > 0]  # Bỏ nền biển (nhãn 0)

    islands_data = []

    # 3. Mở rộng dải biển bằng Fast GPU Max-Pooling
    # Bước nhảy: dùng lặp pool 3x3 (mỗi lần nở 1 pixel)
    for label_val in unique_labels:
        island_mask = (labels_float == label_val)  # Shape (1, 1, H, W)
        
        # Dilation lặp N lần trên GPU cực nhanh
        dilated = island_mask.float()
        remaining = coast_thickness
        while remaining > 0:
            step = min(remaining, 15)  # Nhảy tối đa 15 pixel mỗi kernel để tối ưu CUDA core
            k = 2 * step + 1
            dilated = F.max_pool2d(dilated, kernel_size=k, stride=1, padding=step)
            remaining -= step
            
        dilated = dilated.squeeze(0).squeeze(0) > 0
        island_2d = island_mask.squeeze(0).squeeze(0)
        
        # Biển bao quanh = Vùng giãn nở VÀ Là biển gốc
        sea_boundary = dilated & (~land_mask)
        
        # Bounding box
        combined = island_2d | sea_boundary
        pos = combined.nonzero(as_tuple=False)
        ymin, ymax = pos[:, 0].min().item(), pos[:, 0].max().item() + 1
        xmin, xmax = pos[:, 1].min().item(), pos[:, 1].max().item() + 1
        
        islands_data.append({
            'island_mask': island_2d,
            'sea_mask': sea_boundary,
            'bbox': (ymin, ymax, xmin, xmax)
        })

    return islands_data

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

class StaticSquaredDeviationNorm:
    def __init__(self, per_channel: bool = True):
        """
        Norm thuần túy (không học):
        Tính trực tiếp trên từng sample độc lập, không lưu running stats,
        kết quả không phụ thuộc vào trạng thái train/eval.
        
        Args:
            per_channel (bool): 
                - False (mặc định - theo code gốc): tính mean trên toàn bộ sample (C, H, W).
                - True: tính mean riêng cho từng channel của mỗi sample (H, W).
        """
        self.per_channel = per_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.per_channel:
            # Gom các chiều không gian (H, W, ...), giữ lại chiều Batch và Channel
            # dims = (2, 3, ...) -> mean shape: (B, C, 1, 1, ...)
            dims = tuple(range(2, x.ndim))
        else:
            # Gom tất cả các chiều trừ chiều Batch (dim 0) giống code gốc
            # dims = (1, 2, 3, ...) -> mean shape: (B, 1, 1, 1, ...)
            dims = tuple(range(1, x.ndim))
        
        # 1. Tính mean nội tại của từng sample
        mean = x.mean(dim=dims, keepdim=True)
        
        # 2. Áp dụng công thức biến đổi thuần túy
        out = torch.exp(5 * torch.abs(x - mean))
        
        return out

def plot_first_batch(x: torch.Tensor):
    """
    Nhận vào tensor x có kích thước [B, 3, H, W],
    lấy mẫu đầu tiên và hiển thị bằng matplotlib.
    """
    # 1. Lấy phần tử đầu tiên của batch -> kích thước: [3, H, W]
    img = x[0].detach().cpu()
    
    # 2. Đổi trục từ [C, H, W] sang [H, W, C] cho matplotlib
    img = img.permute(1, 2, 0).numpy()
    
    # 3. Chuẩn hóa về [0, 1] nếu giá trị vượt quá khoảng này (ví dụ sau normalization)
    if img.min() < 0 or img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min())
        
    # 4. Hiển thị ảnh
    plt.figure()
    plt.imshow(img)
    plt.axis('off')  # Tắt trục tọa độ
    plt.show()

def get_min_nonzero_bbox(x: torch.Tensor):
    """
    Nhận vào tensor x có kích thước (H, W).
    Tìm vùng bao (bounding box) nhỏ nhất chứa tất cả các phần tử > 0.

    Trả về:
        tuple (ymin, ymax, xmin, xmax), cropped_tensor
        Nếu không có phần tử nào > 0, trả về (None, None)
    """
    # Lấy tọa độ (row, col) của các phần tử > 0
    # nonzero() trả về tensor shape [N, 2] với cột 0 là H (y), cột 1 là W (x)
    pos = (x > 0).nonzero(as_tuple=False)

    # Trường hợp không có phần tử nào > 0
    if pos.shape[0] == 0:
        return None, None

    # Tọa độ nhỏ nhất và lớn nhất (bao gồm cả biên)
    ymin = pos[:, 0].min().item()
    ymax = pos[:, 0].max().item()
    xmin = pos[:, 1].min().item()
    xmax = pos[:, 1].max().item()

    # Cắt vùng bao (lưu ý slice trong Python cần + 1 ở cận trên để lấy cả biên)
    cropped = x[ymin:ymax + 1, xmin:xmax + 1]

    # Trả về tọa độ dạng slice và tensor sau khi cắt
    return (ymin, ymax + 1, xmin, xmax + 1), cropped

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
    land_mask = (x[:, :].mean(dim=1) > 0).float()
    
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
        # torch.set_printoptions(precision=2, sci_mode=False)
        # plot_first_batch(x)
        # patch, coord = find_28x28_patch_with_val_less_than_one(x[0, 0])
        
        # print(x)
        
        
        x = self.norm.forward(x)
        # print(x[0, 0, coord[0]:coord[0]+28, coord[1]:coord[1]+28])
        x = F.layer_norm(x, normalized_shape= [H,W])
        x = F.relu(x)
        # a = get_min_nonzero_bbox(x[0,0,150:300,400:550])
        x= diffuse_islands_into_sea(x, coast_thickness= 15)
            # Hiển thị minh họa: 2 = Đảo, 1 = Vùng biển bao quanh, 0 = Biển xa
            # visual = torch.zeros_like(sea_map, dtype=torch.float32)
            # visual[island['sea_mask']] = 1
            # visual[island['island_mask']] = 2
            
            # print("Vùng quan sát đã crop:")
            # print(visual[ymin:ymax, xmin:xmax])
        plot_first_batch(x)
        # print(x[:,:,150:300,400:550])
        # exit(0)
        # Thực hiện các bước conv với kernel=3, stride=2, padding=1 hoàn toàn trên GPU
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
        x = x * 100
        x = 1 - x
        # print(x[:,:,89:178, 133:183])
        plot_first_batch(x)
        # print(x[0,0])
        
        # print(x[0,0])
        # # print(x[0, 0, coord[0]:coord[0]+28, coord[1]:coord[1]+28])
        
        # exit(0)
        return x

def load_image_to_bchw_tensor(image_path: str | Path) -> torch.Tensor:
    """Đọc ảnh từ đường dẫn và chuyển đổi sang PyTorch Tensor dạng [B, C, H, W].

    Args:
        image_path (str | Path): Đường dẫn tới file ảnh.

    Returns:
        torch.Tensor: Tensor có shape (1, C, H, W), kiểu float32, dải giá trị
        [0.0, 1.0].
    """
    # 1. Đọc ảnh và chuyển sang RGB (đảm bảo C=3)
    img = Image.open(image_path).convert("RGB")
    img = img.resize((64, 64), Image.Resampling.BILINEAR)
    # 2. Chuyển ảnh PIL (H, W, C) [0, 255] thành Tensor (C, H, W) [0.0, 1.0]
    transform = transforms.ToTensor()
    tensor_chw = transform(img)

    # 3. Thêm chiều batch B=1 ở đầu -> [1, C, H, W]
    tensor_bchw = tensor_chw.unsqueeze(0)

    return tensor_bchw


# --- Ví dụ sử dụng ---
tensor = load_image_to_bchw_tensor(r"D:\apps\Datasets\COCO\dataset\images\000000016567.jpg")
downsampler = ConvKernelDownsampleModule(
    in_channels=3,
    target_size=64,
    normalize_kernel=False,
)
# tensor = downsampler(tensor)
print(tensor.shape)  # torch.Size([1, 3, H, W])
print(tensor.dtype)  # torch.float32
plot_first_batch(tensor)
# sea_map = torch.zeros((10, 10))

# # Đảo 1
# sea_map[2:4, 2:4] = 5.0

# # Đảo 2
# sea_map[7:9, 6:9] = 9.0

