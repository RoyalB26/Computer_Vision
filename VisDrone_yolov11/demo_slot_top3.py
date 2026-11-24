#!/usr/bin/env python3
"""Demo tối giản, độc lập cho 3 nâng cấp hàng đầu của Slot Attention trên dữ liệu giả lập:
  1. ASA Presence Gate          : Dự đoán các lớp đang hoạt động từ GAP(P4+P5), khóa (gate) các slot vắng mặt.
  2. MUFASA Coarse-to-Fine (T=2): Bước 1 đọc P4+P5 -> slots.detach() (Vi phân ẩn 1 bước) -> Bước 2 đọc P3+P4+P5.
  3. Sparse attn_norm + FG Focus: Cắt bỏ phần đuôi nền rất nhỏ khi tổng hợp slot + gán trọng số loss FG > BG.

Chạy thử:
  python VisDrone_yolov11/demo_slot_top3.py
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedSlotBlock(nn.Module):
    """Một bước Slot Attention đơn với tùy chọn Sparse attn_norm (#3) và ASA logit gating (#1)."""

    def __init__(self, dim: int = 32, num_classes: int = 10, sparse_thresh: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.num_slots = num_classes + 1  # K lớp tiền cảnh (FG) + 1 slot nền (BG)
        self.sparse_thresh = sparse_thresh
        self.eps = eps
        self.scale = dim ** -0.5

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(
        self,
        tokens: torch.Tensor,
        slots: torch.Tensor,
        presence_gate: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Chạy 1 bước Slot Attention.

        Tham số:
            tokens: (B, N, D)
            slots:  (B, K+1, D)
            presence_gate: tùy chọn (B, K) trong khoảng (0, 1) từ ASA Presence Gate.
        """
        x = self.norm_input(tokens)               # (B, N, D)
        k = self.to_k(x)                          # (B, N, D)
        v = self.to_v(x)                          # (B, N, D)
        q = self.to_q(self.norm_slots(slots))     # (B, K+1, D)

        dots = torch.einsum("bsd,bnd->bsn", q, k) * self.scale  # (B, K+1, N)

        # [Nâng cấp #1 - ASA Presence Gate] Thêm log-gate bias vào các slot FG để các lớp vắng mặt ngừng cạnh tranh.
        # QUAN TRỌNG: detach presence_gate ở đây để các pixel nền không đẩy gate của các lớp đang hoạt động về 0!
        if presence_gate is not None:
            gate_bias = torch.log(presence_gate.detach().clamp(min=1e-4)).unsqueeze(-1)  # (B, K, 1), giá trị <= 0
            dots = torch.cat(
                [dots[:, : self.num_classes, :] + gate_bias, dots[:, self.num_classes :, :]],
                dim=1,
            )  # (B, K+1, N) = cat[(B, K, N), (B, 1, N)]

        attn_ori = dots.softmax(dim=1)  # (B, K+1, N), tổng theo chiều slot (dim=1) == 1

        # [Nâng cấp #3 - Sparse attn_norm] Đưa các xác suất đuôi rất nhỏ trên slot FG về 0
        # để hàng nghìn pixel nền không làm loãng prototype của các lớp hiếm
        fg_attn = attn_ori[:, : self.num_classes, :]                  # (B, K, N)
        bg_attn = attn_ori[:, self.num_classes :, :]                  # (B, 1, N)
        sparse_mask = (fg_attn >= self.sparse_thresh).float()         # (B, K, N)
        pruned_pct = float((1.0 - sparse_mask.mean()).item() * 100.0) # scalar float

        fg_sparse = fg_attn * sparse_mask                             # (B, K, N)
        attn_for_agg = torch.cat([fg_sparse, bg_attn], dim=1) + self.eps  # (B, K+1, N)
        attn_norm = attn_for_agg / attn_for_agg.sum(dim=-1, keepdim=True).clamp(min=self.eps)  # (B, K+1, N) / (B, K+1, 1) -> (B, K+1, N)

        slots_out = torch.einsum("bnd,bsn->bsd", v, attn_norm)        # (B, K+1, D)
        slots_out = slots_out + self.fc2(F.silu(self.fc1(self.norm_pre_ff(slots_out))))  # (B, K+1, D) -> fc1: (B, K+1, 2*D) -> fc2: (B, K+1, D)
        return slots_out, attn_ori, dots, pruned_pct                  # (B, K+1, D), (B, K+1, N), (B, K+1, N), float


class Top3SlotFiLMRefiner(nn.Module):
    """Bộ tinh chỉnh Post-Neck Slot+FiLM gọn nhẹ kết hợp ASA + MUFASA (T=2, Implicit Diff) + Sparse SA."""

    def __init__(
        self,
        in_channels: Tuple[int, int, int] = (64, 128, 256),
        num_classes: int = 10,
        slot_dim: int = 32,
        sparse_thresh: float = 0.05,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.slot_dim = slot_dim

        # Phép chiếu ngang 1x1 (Lateral): (P3, P4, P5) -> slot_dim
        self.proj_in = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, slot_dim, 1, bias=False), nn.BatchNorm2d(slot_dim), nn.SiLU(inplace=True))
            for ch in in_channels
        ])
        self.level_embed = nn.Parameter(torch.randn(3, slot_dim) * 0.02)            # (3, D)
        self.obj_queries = nn.Parameter(torch.randn(1, num_classes + 1, slot_dim) * 0.05)  # (1, K+1, D)

        # [Nâng cấp #1 - ASA Presence Gate] Dự đoán các lớp có mặt trong ảnh từ GAP(P4) + GAP(P5)
        self.presence_head = nn.Sequential(
            nn.Linear(in_channels[1] + in_channels[2], slot_dim * 2),               # (B, C4+C5) -> (B, 2*D)
            nn.SiLU(inplace=True),
            nn.Linear(slot_dim * 2, num_classes),                                   # (B, 2*D) -> (B, K)
        )

        # Khối Slot Attention dùng chung cho Bước 1 (Thô P4+P5) và Bước 2 (Tinh chỉnh P3+P4+P5)
        self.slot_block = EnhancedSlotBlock(dim=slot_dim, num_classes=num_classes, sparse_thresh=sparse_thresh)

        # Các bộ điều biến FiLM khởi tạo bằng 0 cho P3, P4, P5
        self.modulators = nn.ModuleList([nn.Conv2d(slot_dim, 2 * ch, 1, bias=True) for ch in in_channels])
        for m in self.modulators:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        # feats: [(B, C3, H3, W3), (B, C4, H4, W4), (B, C5, H5, W5)]
        p3, p4, p5 = feats
        bs = p3.shape[0]

        # 1. Chiếu các đặc trưng đa tỉ lệ từ neck thành token
        tokens_per_scale = []
        resolutions = []
        counts = []
        for i, f in enumerate(feats):                                               # f: (B, C_i, H_i, W_i)
            h, w = f.shape[2], f.shape[3]
            resolutions.append((h, w))
            counts.append(h * w)                                                    # N_i = H_i * W_i
            tok = self.proj_in[i](f).flatten(2).transpose(1, 2) + self.level_embed[i]  # (B, D, H_i, W_i) -> (B, D, N_i) -> (B, N_i, D) + (D,)
            tokens_per_scale.append(tok)

        tokens_coarse = torch.cat(tokens_per_scale[1:], dim=1)  # (B, N4+N5, D) — chỉ P4 + P5 (ngữ nghĩa cấp cao)
        tokens_all = torch.cat(tokens_per_scale, dim=1)         # (B, N3+N4+N5, D) = (B, N, D) — P3 + P4 + P5

        # 2. [Nâng cấp #1 - ASA Presence Gate] Dự đoán xác suất xuất hiện của từng lớp trong (0, 1)
        gap_p4 = p4.mean(dim=(2, 3))                                                # (B, C4)
        gap_p5 = p5.mean(dim=(2, 3))                                                # (B, C5)
        presence_logits = self.presence_head(torch.cat([gap_p4, gap_p5], dim=-1))   # cat: (B, C4+C5) -> (B, K)
        presence_gate = torch.sigmoid(presence_logits)                              # (B, K)

        # 3. [Nâng cấp #2 - MUFASA Coarse-to-Fine T=2 + 1-Step Implicit Diff]
        init_slots = self.obj_queries.expand(bs, -1, -1)                            # (B, K+1, D)
        # Bước 1 (Thô - Coarse): tổng hợp các prototype thích ứng theo ảnh từ ngữ nghĩa sạch của P4+P5
        slots_coarse, _, _, _ = self.slot_block(tokens_coarse, init_slots, presence_gate=presence_gate)  # (B, K+1, D)
        # Vi phân ẩn 1 bước (1-Step Implicit Differentiation): detach Bước 1 để backprop chỉ đi qua Bước 2 (tiết kiệm VRAM & ổn định hơn)
        slots_step2_in = slots_coarse.detach()                                      # (B, K+1, D)
        # Bước 2 (Tinh chỉnh - Fine): tinh chỉnh trên toàn bộ token P3+P4+P5
        slots_fine, attn_ori, dots, pruned_pct = self.slot_block(tokens_all, slots_step2_in, presence_gate=presence_gate)
        # slots_fine: (B, K+1, D), attn_ori: (B, K+1, N), dots: (B, K+1, N)

        # 4. Ghi ngược có cổng (Gated Write-back / Broadcast) + FiLM giới hạn biên độ khởi tạo 0
        fg_attn_gated = attn_ori[:, : self.num_classes, :] * presence_gate.unsqueeze(-1)  # (B, K, N) * (B, K, 1) -> (B, K, N)
        slot_broadcast = torch.einsum("bsd,bsn->bdn", slots_fine[:, : self.num_classes, :], fg_attn_gated)  # (B, K, D) x (B, K, N) -> (B, D, N)

        cls_feats = []
        offset = 0
        for i, (f, (h, w), n_i) in enumerate(zip(feats, resolutions, counts)):
            s_map = slot_broadcast[:, :, offset : offset + n_i].view(bs, self.slot_dim, h, w)  # (B, D, N_i) -> (B, D, H_i, W_i)
            offset += n_i
            dg, db = self.modulators[i](s_map).chunk(2, dim=1)                      # modulators[i]: (B, 2*C_i, H_i, W_i) -> dg, db: (B, C_i, H_i, W_i)
            cls_feats.append(f * (1.0 + 0.5 * torch.tanh(dg)) + 0.5 * torch.tanh(db))          # (B, C_i, H_i, W_i)

        info = {
            "slots_coarse": slots_coarse,       # (B, K+1, D)
            "slots_fine": slots_fine,           # (B, K+1, D)
            "attn_ori": attn_ori,               # (B, K+1, N)
            "dots": dots,                       # (B, K+1, N)
            "presence_logits": presence_logits, # (B, K)
            "presence_gate": presence_gate,     # (B, K)
            "pruned_pct": pruned_pct,           # float
            "resolutions": resolutions,         # List[(H_i, W_i)]
        }
        return cls_feats, info


def compute_demo_loss(
    info: Dict[str, torch.Tensor],
    bg_mask: torch.Tensor,      # (B, N) 1=BG (nền), 0=FG (tiền cảnh)
    cls_mask: torch.Tensor,     # (B, K, N) 1=pixel thuộc lớp k
    active_gt: torch.Tensor,    # (B, K) 1=lớp có mặt trong ảnh, 0=vắng mặt
) -> Dict[str, torch.Tensor]:
    """Loss Slot tập trung tiền cảnh (#3) + Loss BCE ASA Presence (#1)."""
    dots = info["dots"]         # (B, K+1, N)
    attn_ori = info["attn_ori"] # (B, K+1, N)
    k = active_gt.shape[1]      # K

    # 1. Loss BCE cho ASA Presence Gate (giám sát trực tiếp các lớp xuất hiện trong ảnh)
    loss_presence = F.binary_cross_entropy_with_logits(info["presence_logits"], active_gt)  # (B, K) vs (B, K) -> scalar ()

    # 2. Tách BG/FG tập trung tiền cảnh (fg_weight=0.7 > bg_weight=0.3 để giảm rò rỉ nền - BG Leak)
    bg_logit = dots[:, -1, :] - torch.logsumexp(dots[:, :k, :], dim=1)      # (B, N) - (B, N) -> (B, N)
    loss_bg = F.binary_cross_entropy_with_logits(bg_logit, bg_mask)         # (B, N) vs (B, N) -> scalar ()
    loss_fg = F.binary_cross_entropy_with_logits(-bg_logit, 1.0 - bg_mask)  # (B, N) vs (B, N) -> scalar ()

    # 3. Dice cân bằng theo lớp + Positive CE trên các lớp đang hoạt động
    p_cls = attn_ori[:, :k, :].clamp(1e-4, 1.0 - 1e-4)                      # (B, K, N)
    inter = (p_cls * cls_mask).sum(dim=-1)                                  # (B, K)
    card = p_cls.sum(dim=-1) + cls_mask.sum(dim=-1)                         # (B, K)
    dice = 1.0 - (2.0 * inter + 1.0) / (card + 1.0)                         # (B, K)
    active = active_gt > 0.5                                                # (B, K) bool
    loss_dice = dice[active].mean()                                         # (M_active,) -> scalar ()

    neg_log_p = -F.log_softmax(dots, dim=1)[:, :k, :]                       # (B, K+1, N) -> slice (B, K, N)
    pos_cnt = cls_mask.sum(dim=-1)                                          # (B, K)
    valid = active & (pos_cnt > 0)                                          # (B, K) bool
    loss_pos = ((neg_log_p * cls_mask).sum(dim=-1) / pos_cnt.clamp(min=1.0))[valid].mean()  # (B, K) -> (M_valid,) -> scalar ()

    loss_total = 0.5 * loss_presence + 0.3 * loss_bg + 0.7 * loss_fg + 0.8 * (loss_dice + 0.5 * loss_pos)  # scalar ()
    return {
        "total": loss_total,
        "presence": loss_presence,
        "fg": loss_fg,
        "bg": loss_bg,
        "dice": loss_dice,
        "pos_ce": loss_pos,
    }


def make_synthetic_batch(
    bs: int = 2,
    num_classes: int = 10,
    device: torch.device = torch.device("cpu"),
):
    """Tạo các đặc trưng giả lập P3/P4/P5 (40x40, 20x20, 10x10) với các mẫu lớp phân biệt."""
    torch.manual_seed(42)
    resolutions = [(40, 40), (20, 20), (10, 10)]
    channels = (64, 128, 256)

    # Ảnh 0 có các lớp [0, 3, 8]; Ảnh 1 có các lớp [4, 9] (7-8 lớp vắng mặt mỗi ảnh!)
    sample_objects = [
        [(0, 0.15, 0.20, 0.30, 0.40), (3, 0.50, 0.50, 0.75, 0.75), (8, 0.10, 0.65, 0.40, 0.90)],
        [(4, 0.20, 0.20, 0.55, 0.50), (9, 0.60, 0.55, 0.85, 0.85)],
    ]

    # Tạo các đặc trưng mẫu giả lập theo lớp để feature mang tín hiệu ngữ nghĩa
    class_sigs = [torch.randn(num_classes, ch, device=device) * 1.5 for ch in channels]               # [(K, C3), (K, C4), (K, C5)]
    feats = [torch.randn(bs, ch, h, w, device=device) * 0.2 for ch, (h, w) in zip(channels, resolutions)]  # [(B, C_i, H_i, W_i)]

    bg_scales, cls_scales = [], []
    active_gt = torch.zeros(bs, num_classes, device=device)                 # (B, K)

    for s_idx, (h, w) in enumerate(resolutions):
        cls_m = torch.zeros(bs, num_classes, h, w, device=device)           # (B, K, H_i, W_i)
        for b in range(bs):
            for cid, x1n, y1n, x2n, y2n in sample_objects[b]:
                active_gt[b, cid] = 1.0
                x1, x2 = int(x1n * w), max(int(x1n * w) + 1, int(x2n * w))
                y1, y2 = int(y1n * h), max(int(y1n * h) + 1, int(y2n * h))
                cls_m[b, cid, y1:y2, x1:x2] = 1.0
                # Bơm đặc trưng của lớp vào bản đồ đặc trưng bên trong hộp GT
                feats[s_idx][b, :, y1:y2, x1:x2] += class_sigs[s_idx][cid].view(-1, 1, 1)  # (C_i, dy, dx) + (C_i, 1, 1)
        bg_m = (cls_m.sum(dim=1) == 0).float().view(bs, -1)                 # (B, H_i, W_i) -> (B, N_i)
        bg_scales.append(bg_m)
        cls_scales.append(cls_m.view(bs, num_classes, -1))                  # (B, K, N_i)

    # Trả về: feats [(B, C_i, H_i, W_i)], bg_mask (B, N), cls_mask (B, K, N), active_gt (B, K)
    return feats, torch.cat(bg_scales, dim=1), torch.cat(cls_scales, dim=2), active_gt


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Demo Top-3 Slot Attention Upgrades (ASA + MUFASA T=2 Implicit Diff + Sparse SA) on {device} ===")

    feats, bg_mask, cls_mask, active_gt = make_synthetic_batch(bs=2, num_classes=10, device=device)
    model = Top3SlotFiLMRefiner(in_channels=(64, 128, 256), num_classes=10, slot_dim=32, sparse_thresh=0.05).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Refiner Params: {n_params:,} | Coarse tokens (P4+P5): 500 | Fine tokens (P3+P4+P5): 2,100")
    print(f"Active GT classes -> img0: {active_gt[0].nonzero().squeeze(-1).tolist()} | img1: {active_gt[1].nonzero().squeeze(-1).tolist()}\n")

    for step in range(41):
        opt.zero_grad()
        cls_feats, info = model(feats)                                      # cls_feats: [(B, C_i, H_i, W_i)]
        losses = compute_demo_loss(info, bg_mask, cls_mask, active_gt)
        losses["total"].backward()
        opt.step()

        if step in (0, 10, 20, 40):
            with torch.no_grad():
                gate = info["presence_gate"]                                # (B, K)
                p_active = gate[active_gt > 0.5].mean().item()              # scalar float
                p_absent = gate[active_gt < 0.5].mean().item()              # scalar float
                # Tỉ lệ % BG Leak trên các pixel FG
                fg_m = bg_mask < 0.5                                        # (B, N) bool
                pred_slot = info["attn_ori"].argmax(dim=1)                  # (B, N)
                bg_leak = (pred_slot[fg_m] == 10).float().mean().item() * 100.0  # scalar float
                print(
                    f"Step {step:2d} | total={losses['total']:.3f} | pres={losses['presence']:.3f} "
                    f"| dice={losses['dice']:.3f} | posCE={losses['pos_ce']:.3f} "
                    f"| ASA gate(active/absent)={p_active:.2f}/{p_absent:.2f} "
                    f"| Sparse pruned={info['pruned_pct']:.1f}% | BG_leak={bg_leak:.1f}%"
                )

    print("\nOutput modulated cls_feats shapes:", [tuple(f.shape) for f in cls_feats])


if __name__ == "__main__":
    main()
