import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import torch
from torchvision.ops import roi_align

def create_crop_mask_single_image(
    image: torch.Tensor, 
    boxes_norm: torch.Tensor, 
    combine: bool = False
) -> torch.Tensor:
    """
    Tạo binary mask cho 1 ảnh với N bounding boxes.
    
    Args:
        image: Tensor shape (C, H, W)
        boxes_norm: Tensor shape (N, 4) định dạng [x1, y1, w, h] chuẩn hóa trong [0, 1]
        combine: 
            - Nếu False: Trả về mask shape (N, H, W) - mỗi box một lớp mask riêng.
            - Nếu True: Trả về mask shape (H, W) - gộp chung tất cả N box lại.
            
    Returns:
        mask: Tensor bool shape (N, H, W) hoặc (H, W)
    """
    C, H, W = image.shape
    N = boxes_norm.shape[0]
    device = image.device

    # Nếu không có box nào (N = 0)
    if N == 0:
        if combine:
            return torch.zeros((H, W), dtype=torch.bool, device=device)
        return torch.zeros((0, H, W), dtype=torch.bool, device=device)

    # 1. Tách tọa độ từ (x1, y1, w, h) sang (x1, y1, x2, y2)
    x1, y1, bw, bh = boxes_norm.unbind(dim=-1)
    x2 = x1 + bw
    y2 = y1 + bh

    # Reshape thành (N, 1, 1) để broadcast qua không gian (H, W)
    x1 = x1.view(N, 1, 1)
    y1 = y1.view(N, 1, 1)
    x2 = x2.view(N, 1, 1)
    y2 = y2.view(N, 1, 1)

    # 2. Tạo lưới tọa độ tâm pixel chuẩn hóa [0, 1] (pixel-aligned: +0.5)
    y_coords = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    x_coords = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
    
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")  # (H, W)
    grid_y = grid_y.unsqueeze(0)  # (1, H, W)
    grid_x = grid_x.unsqueeze(0)  # (1, H, W)

    # 3. So sánh broadcasting song song trên GPU -> Shape (N, H, W)
    mask = (grid_x >= x1) & (grid_x < x2) & (grid_y >= y1) & (grid_y < y2)

    # 4. Gộp mask nếu cần
    if combine:
        mask = mask.any(dim=0)  # Shape (H, W)

    return mask

def group_boxes_by_class(
    gt_boxes: torch.Tensor, 
    gt_labels: torch.Tensor, 
    num_classes: int
) -> List[torch.Tensor]:
    """
    Gom nhóm boxes theo từng class label.

    Args:
        gt_boxes (torch.Tensor): Tensor chứa tọa độ boxes, shape (N, 4).
        gt_labels (torch.Tensor): Tensor chứa label của từng box, shape (N,).
        num_classes (int): Tổng số lượng classes (từ 0 đến num_classes - 1).

    Returns:
        List[torch.Tensor]: Danh sách có độ dài `num_classes`. 
                            Phần tử thứ c là tensor shape (n_c, 4) chứa các box thuộc class c.
    """
    device = gt_boxes.device
    
    # Nếu không có box nào trong ảnh, trả về list các tensor rỗng shape (0, 4)
    if gt_boxes.numel() == 0 or gt_labels.numel() == 0:
        return [torch.empty((0, 4), device=device, dtype=gt_boxes.dtype) for _ in range(num_classes)]

    # Gom nhóm theo từng class index
    grouped_boxes = []
    for c in range(num_classes):
        mask = (gt_labels == c)
        c_boxes = gt_boxes[mask]  # Shape: (n_c, 4)
        grouped_boxes.append(c_boxes)

    return grouped_boxes

class DummyDetector(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

    def slot_orthogonality_loss(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Mô phỏng hàm tính loss trực giao giữa các slot vectors:
        slots shape: (batch_size, num_slots, slot_dim)
        """
        # Chuẩn hóa slots
        slots_norm = F.normalize(slots, p=2, dim=-1)
        # Tính ma trận tương quan cosin giữa các slots: (B, num_slots, num_slots)
        sim_matrix = torch.bmm(slots_norm, slots_norm.transpose(1, 2))
        
        # Identity matrix để loại bỏ đường chéo chính
        eye = torch.eye(sim_matrix.shape[1], device=slots.device).unsqueeze(0)
        off_diag = sim_matrix - eye
        
        return torch.mean(off_diag ** 2)

    def compute_loss(
        self,
        cls_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        obj_logits: torch.Tensor,
        slots: torch.Tensor,
        feat_map: torch.Tensor,
        attn_map: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tính toán Detection Loss: Classification + Bounding Box Regression + Objectness + Orthogonality.
        feat_map: B, d, H, W
        attn_map: B, num_classes + 1, H, W
        """
        device = cls_logits.device
        batch_size, _, grid_h, grid_w = cls_logits.shape
        B,d,H,W= feat_map.shape
        feat_scale = H * W

        
        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        pos_weight = torch.tensor([5.0], device=device)
        total_map_loss = torch.tensor(0.0, device= device)
        total_slot_loss = torch.tensor(0.0, device= device)
        total_bg_loss = torch.tensor(0.0, device= device)

        for b in range(batch_size):
            img_cls = cls_logits[b]       # (num_classes, H, W)
            img_box = pred_boxes[b]       # (4, H, W)
            img_obj = obj_logits[b, 0]    # (H, W)

            target_dict = targets[b] if targets is not None and b < len(targets) else None
            gt_boxes = target_dict.get("boxes", torch.empty((0, 4), device=device)) if target_dict else torch.empty((0, 4), device=device)
            gt_labels = target_dict.get("labels", torch.empty((0,), device=device, dtype=torch.long)) if target_dict else torch.empty((0,), device=device, dtype=torch.long)

            num_gt = len(gt_boxes)
            obj_target = torch.zeros((grid_h, grid_w), device=device)

            img_w, img_h = 320.0, 320.0
            scale = torch.tensor([img_w, img_h, img_w, img_h], device=device)

            # Chuẩn hóa về [0, 1]
            gt_boxes = gt_boxes / scale
            num_classes = 5
            boxes_per_class = group_boxes_by_class(gt_boxes, gt_labels, num_classes=num_classes)

            for s in range(num_classes):
                attn_s_mask = create_crop_mask_single_image(attn_map[b, s].unsqueeze(0), boxes_per_class[s].to(device))
                N, H, W = attn_s_mask.shape
                attn_s_map = attn_map[b, s].unsqueeze(0).expand(N, -1, -1)
                attn_s_map = attn_s_map * attn_s_mask.to(attn_s_map.dtype)
                print(f"s = {s}, N= {N}, ATTN SHAPE: ",attn_s_map.shape, "attn_s_mask: ", attn_s_mask.shape)
                if N > 0:
                    norm = torch.sqrt(torch.sum(attn_s_map ** 2, dim=(-2, -1)) + 1e-8)
                    total_slot_loss += norm.numel()**2 * torch.var(norm, unbiased=False) * 0.01
            print("")
            if num_gt > 0:
                gt_boxes = gt_boxes.to(device)
                gt_labels = gt_labels.to(device)
                
                # Gán nhãn vào cell gần tâm bounding box nhất
                gt_cx = gt_boxes[:, 0].clamp(0.0, 0.999)
                gt_cy = gt_boxes[:, 1].clamp(0.0, 0.999)

                grid_x = (gt_cx * grid_w).long()
                grid_y = (gt_cy * grid_h).long()

                # print("grid x", grid_x)
                # print("grid y", grid_y)

                obj_target[grid_y, grid_x] = 1.0

                p_cls = img_cls[:, grid_y, grid_x].permute(1, 0)
                p_box = img_box[:, grid_y, grid_x].permute(1, 0)

                valid_labels = gt_labels.clamp(0, self.num_classes - 1)
                mask= create_crop_mask_single_image(feat_map[b], gt_boxes, combine= True).unsqueeze(0)

                feat_obj = feat_map[b][mask.expand(d, -1, -1)]

                print("FEAT_OBJ SHAPE: ", feat_obj.shape)

                attn_bg_map = attn_map[b, num_classes + 1].unsqueeze(0)
                print(mask.shape)
                print(attn_map[b, num_classes + 1].shape)
                attn_bg_map = attn_bg_map[~mask]

                total_bg_loss+= (torch.sqrt(torch.sum(attn_map[b, num_classes + 1] ** 2, dim=(-2, -1)) + 1e-8) - torch.sqrt(torch.sum(attn_bg_map ** 2, dim=(-1)) + 1e-8))
                total_map_loss += (torch.sqrt(torch.sum(feat_map[b] ** 2, dim=(-3, -2, -1)) + 1e-8) - torch.sqrt(torch.sum(feat_obj ** 2, dim=(-1)) + 1e-8)) * 0.01

                total_cls_loss += F.cross_entropy(p_cls, valid_labels)
                total_box_loss += F.smooth_l1_loss(p_box, gt_boxes)

            total_obj_loss += F.binary_cross_entropy_with_logits(
                img_obj, obj_target, pos_weight=pos_weight
            )

        total_map_loss /= max(batch_size, 1)
        total_cls_loss /= max(batch_size, 1)
        total_box_loss /= max(batch_size, 1)
        total_obj_loss /= max(batch_size, 1)
        total_slot_loss /= max(batch_size, 1)
        total_bg_loss /= max(batch_size, 1)
        ortho_loss = self.slot_orthogonality_loss(slots)

        total_loss = total_cls_loss + 2.0 * total_box_loss + total_obj_loss + ortho_loss

        return {
            "loss": total_loss,
            "loss_cls": total_cls_loss,
            "loss_box": total_box_loss,
            "loss_obj": total_obj_loss,
            "ortho_loss": ortho_loss,
            "total_map_loss": total_map_loss,
            "total_slot_loss": total_slot_loss,
            "total_bg_loss": total_bg_loss,
        }


# ==========================================
# 1. KHỞI TẠO DỮ LIỆU GIẢ (DUMMY DATA)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thiết lập kích thước
batch_size = 2
num_classes = 5
grid_h, grid_w = 320, 320      # Độ phân giải feature map / output grid
feat_channels = 16
num_slots = 6
slot_dim = 64

# Predictions (từ model)
cls_logits = torch.randn(batch_size, num_classes, grid_h, grid_w, device=device, requires_grad=True)
pred_boxes = torch.rand(batch_size, 4, grid_h, grid_w, device=device, requires_grad=True)
obj_logits = torch.randn(batch_size, 1, grid_h, grid_w, device=device, requires_grad=True)
slots = torch.randn(batch_size, num_slots, slot_dim, device=device, requires_grad=True)
feat_map = torch.randn(batch_size, feat_channels, grid_h, grid_w, device=device)
attn_map = torch.randn(batch_size, num_slots + 1, grid_h, grid_w, device=device)
# print(feat_map)

# Targets (Ground truth cho từng ảnh trong batch)
targets = [
    {
        # Ảnh 1: có 2 vật thể [cx, cy, w, h] được chuẩn hoá về [0, 1]
        "boxes": torch.tensor([
            [125, 135, 60, 60],
            [70, 65, 15, 25],
        ], dtype=torch.float32, device=device),
        "labels": torch.tensor([1, 1], dtype=torch.long, device=device),
    },
    {
        # Ảnh 2: có 1 vật thể
        "boxes": torch.tensor([
            [50, 50, 40, 40],
        ], dtype=torch.float32, device=device),
        "labels": torch.tensor([1], dtype=torch.long, device=device),
    }
]



# ==========================================
# 2. CHẠY THỬ VÀ KIỂM TRA BACKWARD
# ==========================================
model = DummyDetector(num_classes=num_classes).to(device)

losses = model.compute_loss(
    cls_logits=cls_logits,
    pred_boxes=pred_boxes,
    obj_logits=obj_logits,
    slots=slots,
    feat_map=feat_map,
    attn_map=attn_map,
    targets=targets,
)

print("--- Kết quả tính Loss ---")
for k, v in losses.items():
    print(f"{k:12s}: {v.item():.4f}")

# Kiểm tra tính toán gradient ngược (backward pass)
losses["loss"].backward()
print("\nGradient check:")
print(f"cls_logits grad shape: {cls_logits.grad.shape}")
print(f"pred_boxes grad shape: {pred_boxes.grad.shape}")
print(f"obj_logits grad shape: {obj_logits.grad.shape}")
print(f"slots grad shape     : {slots.grad.shape}")