import torch
from torch import nn
from torch.functional import F
from torch import Tensor
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
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

import numpy as np
import torch
import torch.nn.functional as F


def filter_color_bg(
    images: torch.Tensor | np.ndarray,
    target_res: tuple[int, int] = (8, 24),
    exp_factor: float = 5.0,
) -> torch.Tensor:
    """Filter flat background vs foreground on a batch of images using color distance, LayerNorm, and ReLU.

    Formula:
        score = mean(exp(exp_factor * |x - global_mean|), dim=C)
        relu_out = ReLU(LayerNorm_spatial(score))

    Args:
        images: Batch of images of shape (B, C, H, W) or single image (C, H, W)
                (numpy array or torch Tensor), normalized in [0, 1], ImageNet-normalized, or raw [0, 255].
        target_res: Target spatial resolution (H_feat, W_feat).
        exp_factor: Factor for exponentiation (default 5.0).

    Returns:
        flat_mask: Tensor of shape (B, H_feat, W_feat) in {0.0, 1.0},
                   where 1.0 indicates flat background (relu_out == 0)
                   and 0.0 indicates salient/foreground objects (relu_out > 0).
    """
    # 0. Convert numpy array to torch.Tensor if needed
    if isinstance(images, np.ndarray):
        img = torch.from_numpy(images)
    elif isinstance(images, torch.Tensor):
        img = images
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(images)}")

    if img.dim() == 3:
        img = img.unsqueeze(0)  # (1, C, H, W)
    elif img.dim() != 4:
        raise ValueError(f"Expected 4D batch (B, C, H, W) or 3D image (C, H, W), got {img.shape}")

    H_feat, W_feat = target_res

    # 1. Normalize image về [0, 1]
    if img.dtype == torch.uint8 or img.max() > 1.0:
        v_float = img.float().div(255.0)
    else:
        v_float = img.float()
        if v_float.min() < -0.5:
            v_float = (v_float * 0.225 + 0.45).clamp(0.0, 1.0)

    # 2. Downsample ảnh về kích thước (B, C, H_feat, W_feat)
    x = F.interpolate(v_float, size=(H_feat, W_feat), mode="area")  # (B, C, H_feat, W_feat)

    # 3. Tính màu trung bình độc lập cho từng ảnh trong batch theo không gian (H, W)
    global_mean = x.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

    # 4. Tính khoảng cách màu theo kênh C và khuếch đại bằng hàm mũ
    color_diff = torch.abs(x - global_mean)  # (B, C, H_feat, W_feat)
    exp_diff = torch.exp(float(exp_factor) * color_diff)
    score_map = exp_diff.mean(dim=1)  # (B, H_feat, W_feat)

    # 5. LayerNorm độc lập trên từng ảnh theo không gian (H_feat, W_feat), sau đó qua ReLU
    mean_v = score_map.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1)
    std_v = score_map.std(dim=(1, 2), keepdim=True)    # (B, 1, 1)
    norm_v = (score_map - mean_v) / (std_v + 1e-6)     # (B, H_feat, W_feat)
    relu_v = torch.relu(norm_v)

    # Vùng có relu_v == 0 (tức norm_v <= 0) là vùng nền đồng nhất
    flat_mask = (relu_v == 0).float()  # (B, H_feat, W_feat)

    return flat_mask

def plot_first_batch(x: torch.Tensor):
    """
    Nhận vào tensor x có kích thước [B, 3, H, W],
    lấy mẫu đầu tiên và hiển thị bằng matplotlib.
    """
    # 1. Lấy phần tử đầu tiên của batch -> kích thước: [3, H, W]
    img = x.detach().cpu()
    
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
        C, H, W = x.shape

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
        out = F.layer_norm(out, normalized_shape= [H,W])
        # out = F.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, dim: int= 32, num_class: int = 10):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        num_slots: int= num_class + 1
        self.obj_queries = nn.Parameter(torch.randn(1, num_slots, dim) * 0.05)

        self.FC1 = nn.Linear(dim, dim)
        self.FC2 = nn.Linear(dim, dim)
        self.LN = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.eps = 1e-8
        self.scale = dim**-0.5

    def forward(self, inputs: Tensor):
        inputs = inputs + self.FC2(F.relu(self.FC1(self.LN(inputs))))

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
        attn_ori = dots.softmax(dim=1) + self.eps
        attn_norm = attn_ori / attn_ori.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        slots = torch.einsum("bjd,bij->bid", v, attn_norm)
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots

    def loss(self):
        B, C,H,W= x.shape
        target_res = (H,W)
        mask= filter_color_bg(x, target_res= (H,W))
        fg_mask1 = (1.0 - mask)
        x = x * fg_mask1.view(1, 1, *target_res)


        x = x + self.FC2(F.relu(self.FC1(self.LN(x))))
        pass
    
file_path = r"D:\apps\Datasets\VisDrone\dataset\images\0000063_08000_d_0000009.jpg"
target_res = (960, 960)  # (H, W)
def show_bchw_tensor(tensor_bchw: torch.Tensor):
    """Hàm phụ trợ chuyển Tensor (1, 3, H, W) sang ảnh RGB PIL và hiển thị."""
    img_np = tensor_bchw.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    else:
        img_np = img_np.clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np, mode="RGB").show()


def show_mask_tensor(mask_hw: torch.Tensor):
    """Hàm phụ trợ chuyển Mask (H, W) sang ảnh Grayscale PIL và hiển thị."""
    mask_np = (mask_hw.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(mask_np, mode="L").show()