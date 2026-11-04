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
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from collections import deque

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
def find_islands_and_coastlines(
    x: torch.Tensor,
    connectivity: int = 8,
    coast_thickness: int = 1,
    min_box_size: int = 20,
):
    """Tìm tất cả các ốc đảo và vùng biển bao quanh, gộp các kết quả giao nhau,

    đồng thời loại bỏ các box có chiều cao (h) hoặc chiều rộng (w) < min_box_size.

    Args:
        x (torch.Tensor): Tensor đầu vào kích thước (H, W).
        connectivity (int): 4 hoặc 8.
        coast_thickness (int): Độ dày (pixel) của vùng biển bao quanh.
        min_box_size (int): Kích thước tối thiểu của h và w (mặc định 20).
    """
    assert x.ndim == 2, "Tensor đầu vào phải có shape (H, W)"
    H, W = x.shape
    device = x.device

    # 1. Xác định mặt đất và biển
    land_mask = (x > 0.1).cpu()
    visited = torch.zeros_like(land_mask, dtype=torch.bool)

    if connectivity == 8:
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
    else:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    raw_islands = []
    k_size = 2 * coast_thickness + 1
    kernel = torch.ones((1, 1, k_size, k_size), dtype=torch.float32)

    # 2. Tìm từng đảo ban đầu
    for r in range(H):
        for c in range(W):
            if land_mask[r, c] and not visited[r, c]:
                current_island = torch.zeros((H, W), dtype=torch.bool)
                queue = deque([(r, c)])
                visited[r, c] = True
                current_island[r, c] = True

                while queue:
                    curr_r, curr_c = queue.popleft()
                    for dr, dc in directions:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if land_mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                current_island[nr, nc] = True
                                queue.append((nr, nc))

                # Dilation để tìm vùng biển
                island_float = current_island.float().unsqueeze(0).unsqueeze(0)
                dilated = (
                    F.conv2d(island_float, kernel, padding=coast_thickness) > 0
                ).squeeze(0, 1)
                sea_boundary = dilated & (~land_mask)

                combined = current_island | sea_boundary
                pos = combined.nonzero(as_tuple=False)
                ymin, ymax = pos[:, 0].min().item(), pos[:, 0].max().item() + 1
                xmin, xmax = pos[:, 1].min().item(), pos[:, 1].max().item() + 1

                raw_islands.append({
                    "island_mask": current_island,
                    "sea_mask": sea_boundary,
                    "combined": combined,
                    "bbox": (ymin, ymax, xmin, xmax),
                })

    n = len(raw_islands)
    if n == 0:
        return []

    # 3. Gộp các đảo giao nhau bằng Union-Find
    uf = UnionFind(n)

    for i in range(n):
        y1_min, y1_max, x1_min, x1_max = raw_islands[i]["bbox"]
        for j in range(i + 1, n):
            y2_min, y2_max, x2_min, x2_max = raw_islands[j]["bbox"]

            if not (
                y1_max > y2_min
                and y2_max > y1_min
                and x1_max > x2_min
                and x2_max > x1_min
            ):
                continue

            if (raw_islands[i]["combined"] & raw_islands[j]["combined"]).any():
                uf.union(i, j)

    # 4. Gom nhóm và lọc kích thước h, w >= min_box_size
    groups = {}
    for idx in range(n):
        root = uf.find(idx)
        groups.setdefault(root, []).append(idx)

    merged_results = []
    for member_indices in groups.values():
        merged_island = torch.zeros((H, W), dtype=torch.bool)
        merged_sea = torch.zeros((H, W), dtype=torch.bool)

        for idx in member_indices:
            merged_island |= raw_islands[idx]["island_mask"]
            merged_sea |= raw_islands[idx]["sea_mask"]

        merged_sea = merged_sea & (~merged_island)

        # Tính toán bounding box
        combined = merged_island | merged_sea
        pos = combined.nonzero(as_tuple=False)
        ymin, ymax = pos[:, 0].min().item(), pos[:, 0].max().item() + 1
        xmin, xmax = pos[:, 1].min().item(), pos[:, 1].max().item() + 1

        box_h = ymax - ymin
        box_w = xmax - xmin

        # Lọc bỏ nếu h < 20 hoặc w < 20
        if box_h < min_box_size or box_w < min_box_size:
            continue

        merged_results.append({
            "island_mask": merged_island.to(device),
            "sea_mask": merged_sea.to(device),
            "bbox": (ymin, ymax, xmin, xmax),
        })

    return merged_results

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

from scipy.ndimage import binary_dilation, label
import torch
import numpy as np
def find_islands_and_coastlines_optimized(
    x: torch.Tensor,
    connectivity: int = 8,
    coast_thickness: int = 1,
    min_box_size: int = 20,
):
    """Tìm tất cả các ốc đảo và vùng biển bao quanh, tự động gộp các cụm giao nhau

    và lọc bỏ box có h < min_box_size hoặc w < min_box_size.

    Độ phức tạp: O(H * W) - Không phụ thuộc số lượng đảo M^2.
    """
    assert x.ndim == 2, "Tensor đầu vào phải có shape (H, W)"
    H, W = x.shape
    device = x.device

    # 1. Chuyển sang NumPy bool array để xử lý bằng SciPy C-engine
    land = (x > 0.1).cpu().numpy()

    # Cấu trúc liên thông (4 hoặc 8 hướng)
    if connectivity == 4:
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    else:
        struct = np.ones((3, 3), dtype=bool)

    # 2. DILATION TOÀN CỤC:
    # Bất kỳ 2 đảo nào cách nhau <= 2 * coast_thickness sẽ tự động chạm vào nhau
    # kernel hình vuông kích thước (2*coast_thickness + 1) tương đương iterations=coast_thickness với struct 8-hướng
    dilated_land = binary_dilation(
        land, structure=np.ones((3, 3), dtype=bool), iterations=coast_thickness
    )

    # 3. LABELING (Connected Components) trên vùng đã giãn nở:
    # Mỗi cụm liên thông ở đây đại diện cho 1 nhóm đảo đã được gộp hoàn chỉnh!
    # labeled_clusters, num_clusters = label(dilated_land, structure=struct)

    labeled_clusters= dilated_land
    num_clusters = len(dilated_land)
    if num_clusters == 0:
        return []

    # 4. Tìm Bounding Box cho từng cụm bằng vectorization (O(H * W))
    # Dùng np.where một lần duy nhất thay vì quét lặp
    results = []

    # Lấy tọa độ các pixel thuộc các cụm
    cluster_ids = labeled_clusters[labeled_clusters > 0]
    r_coords, c_coords = np.nonzero(labeled_clusters)

    # Gom tọa độ min/max nhanh bằng cách sort theo label hoặc bincount
    # Với số lượng cluster thông thường, duyệt trực tiếp slices từ find_objects là nhanh nhất:
    from scipy.ndimage import find_objects

    slices = find_objects(labeled_clusters)

    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue

        r_slice, c_slice = sl
        ymin, ymax = r_slice.start, r_slice.stop
        xmin, xmax = c_slice.start, c_slice.stop

        box_h = ymax - ymin
        box_w = xmax - xmin

        # Lọc bỏ nếu h < min_box_size hoặc w < min_box_size
        if box_h < min_box_size or box_w < min_box_size:
            continue

        # Cụm hiện tại
        cluster_mask = labeled_clusters == i

        # Đất liền thuộc cụm này
        island_mask_np = cluster_mask & land

        # Vùng biển = (vùng giãn nở của cụm) VÀ (không phải đất liền)
        sea_mask_np = cluster_mask & (~land)

        results.append({
            "island_mask": torch.from_numpy(island_mask_np).to(device),
            "sea_mask": torch.from_numpy(sea_mask_np).to(device),
            "bbox": (ymin, ymax, xmin, xmax),
        })

    return results

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

        # self.norm = LocalStaticSquaredDeviationNorm(kernel_size= 301, per_channel= False)
        self.norm = StaticSquaredDeviationNorm(per_channel= True)

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
        # x= expand_islands_batch_gpu(x, coast_thickness= 5)
            # Hiển thị minh họa: 2 = Đảo, 1 = Vùng biển bao quanh, 0 = Biển xa
            # visual = torch.zeros_like(sea_map, dtype=torch.float32)
            # visual[island['sea_mask']] = 1
            # visual[island['island_mask']] = 2
            
            # print("Vùng quan sát đã crop:")
            # print(visual[ymin:ymax, xmin:xmax])
        # sea_map= x[0,0]

        plot_first_batch(x)
        exit(0)
        # results= find_islands_and_coastlines_optimized(sea_map)
        # print(f"Số lượng ốc đảo tìm thấy: {len(results)}\n")
    
        # for idx, island in enumerate(results):
        #     ymin, ymax, xmin, xmax = island['bbox']
        #     print(f"--- Ốc đảo {idx + 1} ---")
        #     print(f"Bounding Box (H, W): [{ymin}:{ymax}, {xmin}:{xmax}]")
            
        #     # Hiển thị minh họa: 2 = Đảo, 1 = Vùng biển bao quanh, 0 = Biển xa
        #     # visual = torch.zeros_like(sea_map, dtype=torch.int)
        #     # visual[island['sea_mask']] = 1
        #     # visual[island['island_mask']] = 2
            
        #     print("Vùng quan sát đã crop:")
        #     # print(sea_map[ymin:ymax, xmin:xmax])
        #     m = x[:,:,ymin:ymax, xmin:xmax]
        #     # m = 1 - m
        #     plot_first_batch(m)
        #     print()
        # exit(0)
        
        # print(x[:,:,150:300,400:550])
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
            # if cnt % 2 == 0:
            #     x = F.layer_norm(x, normalized_shape= [h,w])
            # x = F.relu(x)
            # cnt+= 1
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
        x = x / torch.amax(x, dim=(2, 3), keepdim=True).clamp(min=1e-8)
        # plot_first_batch(x)
        return x

def load_image_with_clahe_to_bchw(
    image_path: str | Path,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    target_size: tuple[int, int] | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Đọc ảnh, cân bằng histogram độ tương phản cục bộ (CLAHE) trên kênh L (LAB)

    và xuất ra BCHW Tensor.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path.resolve()}")

    # Đọc bằng PIL để giữ chuẩn RGB
    with Image.open(path) as raw_img:
        img_rgb = np.array(raw_img.convert("RGB"))

    # Áp dụng CLAHE trên không gian màu LAB (tránh làm lệch màu)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_equalized = clahe.apply(l)

    lab_equalized = cv2.merge((l_equalized, a, b))
    rgb_equalized = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)

    img = Image.fromarray(rgb_equalized)

    if target_size is not None:
        img = img.resize(target_size, Image.Resampling.BILINEAR)

    tensor_chw = transforms.ToTensor()(img)
    return tensor_chw.unsqueeze(0).to(device=device, dtype=torch.float32)


def process_frame(
    frame_bgr: np.ndarray,
    downsampler: nn.Module,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    target_size: tuple[int, int] | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Xử lý từng frame hình ảnh/video qua CLAHE và ConvKernelDownsampleModule.

    Trả về frame BGR để hiển thị hoặc ghi video.
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 1. Cân bằng histogram CLAHE trên kênh L
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_equalized = clahe.apply(l)
    lab_equalized = cv2.merge((l_equalized, a, b))
    rgb_equalized = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)

    # 2. Resize nếu có chỉ định target_size
    if target_size is not None:
        rgb_equalized = cv2.resize(rgb_equalized, target_size, interpolation=cv2.INTER_LINEAR)

    # 3. Chuẩn hóa về [0, 1] và chuyển sang PyTorch Tensor [1, C, H, W]
    tensor_chw = torch.from_numpy(rgb_equalized).permute(2, 0, 1).float() / 255.0
    tensor_bchw = tensor_chw.unsqueeze(0).to(device=device, dtype=torch.float32)

    # 4. Đưa qua downsampler module
    with torch.no_grad():
        out_tensor = downsampler(tensor_bchw)

    # 5. Chuyển tensor kết quả trở lại dạng NumPy image (BGR) để hiển thị / lưu video
    out_img = out_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    if out_img.min() < 0 or out_img.max() > 1:
        out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min() + 1e-8)
    out_img = (out_img * 255.0).clip(0, 255).astype(np.uint8)

    # Chuyển RGB sang BGR cho OpenCV
    out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    return out_bgr


def process_video(
    video_source: Union[str, int, Path],
    downsampler: nn.Module,
    output_path: Optional[Union[str, Path]] = None,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    target_size: Optional[tuple[int, int]] = None,
    device: str | torch.device = "cpu",
    show: bool = True,
):
    """Xử lý luồng video (từ file video hoặc webcam) qua downsampler module.

    Args:
        video_source: Đường dẫn tới file video (.mp4, .avi, ...) hoặc webcam index (0, 1).
        downsampler: Module torch xử lý frame (e.g. ConvKernelDownsampleModule).
        output_path: Đường dẫn file video xuất kết quả (tùy chọn).
        clip_limit: Tham số clipLimit cho CLAHE.
        tile_grid_size: Kích thước grid cho CLAHE.
        target_size: Kích thước resize frame (w, h) trước khi xử lý (tùy chọn).
        device: Thiết bị chạy ('cpu' hoặc 'cuda').
        show: Cho phép hiển thị trực tiếp bằng cv2.imshow hay không.
    """
    cap = cv2.VideoCapture(str(video_source) if isinstance(video_source, Path) else video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video source: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Bắt đầu xử lý video: {video_source}")
    print(f"[INFO] FPS: {fps:.2f}, Tổng số frame: {total_frames if total_frames > 0 else 'N/A'}")

    writer = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(
                frame_bgr=frame,
                downsampler=downsampler,
                clip_limit=clip_limit,
                tile_grid_size=tile_grid_size,
                target_size=target_size,
                device=device,
            )

            # Khởi tạo VideoWriter nếu cần lưu file
            if output_path is not None and writer is None:
                out_h, out_w = processed_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

            if writer is not None:
                writer.write(processed_frame)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"[INFO] Đã xử lý {frame_idx} frames...")

            if show:
                try:
                    cv2.imshow("Video Processing - Result", processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        print("[INFO] Người dùng dừng xử lý video.")
                        break
                except cv2.error as e:
                    print(f"[WARN] OpenCV GUI không khả dụng ({e}). Tắt hiển thị `show=False`.")
                    show = False

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[INFO] Video kết quả đã được lưu tại: {output_path}")
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    print(f"[INFO] Hoàn thành xử lý {frame_idx} frames.")


# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device}")

    downsampler = ConvKernelDownsampleModule(
        in_channels=3,
        target_size=(64, 64),
        normalize_kernel=False,
    ).to(device)

    # # 1. Xử lý ảnh mẫu nếu file tồn tại
    sample_img_path = r"D:\apps\Datasets\VisDrone\dataset\images\0000277_00001_d_0000539.jpg"
    if Path(sample_img_path).exists():
        tensor = load_image_with_clahe_to_bchw(sample_img_path, device=device)
        tensor = downsampler(tensor)
        print("Tensor shape sau khi xử lý ảnh:", tensor.shape)
        print("Tensor dtype:", tensor.dtype)
    else:
        print(f"[INFO] File ảnh mẫu không tồn tại: {sample_img_path}")

    # 2. Xử lý video (file video hoặc webcam):
    # Ví dụ với webcam:
    # process_video(video_source=0, downsampler=downsampler, device=device, show=True)
    #
    # Ví dụ với file video:
    # process_video(
    #     video_source=r"D:\apps\Datasets\TACO\ap_Town02\variant_scenario\t5\187\187.mp4",
    #     downsampler=downsampler,
    #     output_path="output_processed.mp4",
    #     device=device,
    #     show=True,
    # )

