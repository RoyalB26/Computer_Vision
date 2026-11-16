"""Prototype & Reference Implementations for Improved Slot Attention Convergence.

Demonstrates concrete techniques tailored for `toy_action` and `action_slot` (TACO):
1. Class-specific learnable slot prototypes + separate background slot + BO-QSA query trick.
2. Multi-iteration GRU refinement (num_iterations=3) instead of single-step cross-attention.
3. SAVi-style temporal carry-over across frames t = 0..T-1 via a slot transition predictor.
4. Separation of Key (spatial PE) and Value (pure visual semantics) + ISA relative coords.
5. L2-normalized Q/K with learnable temperature scaling for sharper object boundaries.
6. Preserved unnormalized `slot_mass` (object area/count cue before weighted averaging).
7. DINOSAUR-style Spatial Broadcast Feature Decoder (reconstructs backbone features, not pixels).
8. Spatial convergence losses: Foreground Coverage (Dice+BCE), Overlap Penalty, and Entropy.
9. Higher-resolution (14x14) multi-scale YOLOv11 neck fusion example.

Run smoke test:
    python toy_action/tmp.py
"""

from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Higher-Resolution (14x14) Neck Feature Fusion (for small objects like o/O)
# ==============================================================================

def fuse_neck_14x14(
    p3: torch.Tensor,
    p4: torch.Tensor,
    p5: torch.Tensor,
    p3_proj: nn.Module,
    p4_proj: nn.Module,
) -> torch.Tensor:
    """Fuse YOLOv11 P3 (28x28), P4 (14x14), P5 (7x7) at 14x14 instead of 7x7.

    Keeping 14x14 resolution (stride 16 -> 16x16 px per token on 224x224 input)
    preserves fine interior vs border details (e.g., hollow 'o' vs filled 'O')
    and helps separate nearby objects.
    """
    target_hw = p4.shape[-2:]  # (14, 14)
    p3_down = F.adaptive_avg_pool2d(p3, target_hw) + F.adaptive_max_pool2d(p3, target_hw)
    p5_up = F.interpolate(p5, size=target_hw, mode="bilinear", align_corners=False)
    return p5_up + 0.5 * (p3_proj(p3_down) + p4_proj(p4))


# ==============================================================================
# 2. Positional Helpers & Invariant Slot-Centric Coordinates (ISA)
# ==============================================================================

def make_2d_coords(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return normalized 2D coordinates in [-1, 1] of shape (1, H*W, 2)."""
    ys = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([gx, gy], dim=-1).reshape(1, h * w, 2)
    return coords


def build_sincos_2d_pe(h: int, w: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return 2D sin-cos positional embedding of shape (1, H*W, dim)."""
    assert dim % 4 == 0
    half = dim // 2
    ys = torch.arange(h, device=device, dtype=dtype)
    xs = torch.arange(w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    def _1d(pos: torch.Tensor, d: int) -> torch.Tensor:
        omega = torch.arange(d // 2, device=device, dtype=dtype) / (d / 2.0)
        omega = 1.0 / (10000.0 ** omega)
        ang = torch.einsum("n,d->nd", pos.reshape(-1), omega)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    return torch.cat([_1d(gx, half), _1d(gy, half)], dim=-1).unsqueeze(0)


# ==============================================================================
# 3. Improved Video Slot Attention (SAVi + GRU + BO-QSA + ISA + Slot Mass)
# ==============================================================================

class ImprovedVideoSlotAttention(nn.Module):
    """Spatiotemporal Slot Attention engineered for clean object binding.

    Key upgrades over single-step 3D SlotAttention:
      - Separate learnable prototype per allocated object class + dedicated bg_slot.
      - Multi-iteration competitive refinement with GRUCell (`num_iterations`).
      - BO-QSA straight-through query gradient stabilization (`use_boqsa`).
      - Key gets spatial PE (and optional ISA slot-relative coords); Value stays pure semantic.
      - Cosine L2-normalized Q/K attention with learnable temperature (`logit_scale`).
      - SAVi temporal carry-over: frame t slots initialize frame t+1 via `transition`.
      - Returns `slot_mass` (B, T, S, 1) before per-frame averaging to retain object count cues.
    """

    def __init__(
        self,
        num_obj_slots: int = 6,
        dim: int = 256,
        num_iterations: int = 3,
        bg_slot: bool = True,
        use_boqsa: bool = True,
        use_isa_rel_pe: bool = True,
        use_savi_carryover: bool = True,
        init_temp: float = 0.25,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_obj_slots = num_obj_slots
        self.bg_slot = bg_slot
        self.total_slots = num_obj_slots + 1 if bg_slot else num_obj_slots
        self.dim = dim
        self.num_iterations = num_iterations
        self.use_boqsa = use_boqsa
        self.use_isa_rel_pe = use_isa_rel_pe
        self.use_savi_carryover = use_savi_carryover
        self.eps = eps

        # Class-specific learnable slot queries (allocated slots, not permutation-symmetric)
        self.obj_queries = nn.Parameter(torch.randn(1, num_obj_slots, dim) * 0.05)
        if bg_slot:
            self.bg_query = nn.Parameter(torch.randn(1, 1, dim) * 0.05)

        # Input projections: Key gets PE, Value stays semantic-only
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Learnable relative slot-centric PE (ISA): maps (dx, dy) -> dim bias on key
        if use_isa_rel_pe:
            self.rel_pos_mlp = nn.Sequential(
                nn.Linear(4, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, dim),
            )

        # Cosine attention temperature: exp(logit_scale) starts around 1 / init_temp
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))

        # Iterative GRU + FFN slot updater
        self.gru = nn.GRUCell(dim, dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        # SAVi temporal transition from frame t-1 slots -> frame t prior queries
        if use_savi_carryover:
            self.transition = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

        # Fuse unnormalized mass (area/count signal) back into object slot features
        self.mass_proj = nn.Sequential(
            nn.Linear(2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
        )

    def _initial_slots(self, batch_size: int) -> torch.Tensor:
        if self.bg_slot:
            q = torch.cat([self.obj_queries, self.bg_query], dim=1)
        else:
            q = self.obj_queries
        return q.expand(batch_size, -1, -1).contiguous()

    def _step_frame(
        self,
        slots_init: torch.Tensor,
        feat_frame: torch.Tensor,
        abs_pe: torch.Tensor,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run `num_iterations` competitive Slot Attention on one frame.

        Args:
            slots_init: (B, S_total, D)
            feat_frame: (B, HW, D) visual features WITHOUT positional embedding
            abs_pe: (1, HW, D) 2D sin-cos PE
            coords: (1, HW, 2) normalized grid coordinates in [-1, 1]
        Returns:
            slots: (B, S_total, D)
            attn_compete: (B, S_total, HW) competitive slot probabilities per pixel (sum_S == 1)
            slot_mass: (B, S_total, 2) [mean_pixel_prob, max_pixel_prob] per slot
        """
        b, hw, d = feat_frame.shape
        feat_norm = self.norm_input(feat_frame)
        v = self.to_v(feat_norm)                       # Pure visual semantics (no PE)
        k_base = self.to_k(feat_norm + 0.5 * abs_pe)   # Key anchored with spatial PE

        slots = slots_init
        scale = self.logit_scale.exp().clamp(max=50.0)
        slot_centers = None
        attn_compete = None

        for it in range(self.num_iterations):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))

            if self.use_isa_rel_pe and slot_centers is not None:
                # Relative coordinates from each slot's previous center of mass (ISA)
                # slot_centers: (B, S, 2), coords: (1, HW, 2) -> rel: (B, S, HW, 2)
                rel = coords.unsqueeze(1) - slot_centers.unsqueeze(2)
                rel_feat = torch.cat([rel, rel.abs()], dim=-1)  # (B, S, HW, 4)
                k_rel = self.rel_pos_mlp(rel_feat)              # (B, S, HW, D)
                k_all = F.normalize(k_base.unsqueeze(1) + k_rel, dim=-1)
                q_norm = F.normalize(q, dim=-1)
                logits = torch.einsum("bsd,bshd->bsh", q_norm, k_all) * scale
            else:
                q_norm = F.normalize(q, dim=-1)
                k_norm = F.normalize(k_base, dim=-1)
                logits = torch.einsum("bsd,bhd->bsh", q_norm, k_norm) * scale

            # Competitive softmax over slots at each pixel
            attn_compete = logits.softmax(dim=1) + self.eps         # (B, S, HW)
            attn_sum = attn_compete.sum(dim=-1, keepdim=True)       # (B, S, 1)
            attn_weights = attn_compete / attn_sum                  # (B, S, HW), sum_HW == 1

            # Update slot centers for next ISA iteration
            slot_centers = torch.einsum("bsh,bhd->bsd", attn_weights, coords.expand(b, -1, -1))

            # Weighted aggregate of pure semantic values
            updates = torch.einsum("bsh,bhd->bsd", attn_weights, v)

            slots = self.gru(
                updates.reshape(b * self.total_slots, d),
                slots_prev.reshape(b * self.total_slots, d),
            ).view(b, self.total_slots, d)
            slots = slots + self.ffn(self.norm_ffn(slots))

        # BO-QSA straight-through query gradient shortcut:
        # Gives learnable queries an unattenuated gradient path alongside GRU unrolling.
        if self.use_boqsa:
            slots = slots + (slots_init - slots_init.detach())

        mean_cov = attn_compete.mean(dim=-1, keepdim=True)              # Fraction of frame owned
        peak_cov = attn_compete.max(dim=-1, keepdim=True).values        # Peak confidence in frame
        slot_mass = torch.cat([mean_cov, peak_cov], dim=-1)             # (B, S_total, 2)
        return slots, attn_compete, slot_mass

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass over a video volume.

        Args:
            inputs: Pure backbone feature volume (B, T, H, W, D) without external PE baked in.
        Returns:
            dict with:
              - slots_seq: (B, T, S_total, D) refined slots (with mass cue added)
              - obj_slots: (B, T, num_obj_slots, D) foreground object slots
              - attn_masks: (B, T, S_total, H, W) per-pixel slot probabilities
              - slot_mass: (B, T, S_total, 2) unnormalized coverage/peak cues
        """
        b, t, h, w, d = inputs.shape
        hw = h * w
        coords = make_2d_coords(h, w, inputs.device, inputs.dtype)
        abs_pe = build_sincos_2d_pe(h, w, d, inputs.device, inputs.dtype)

        slots_prior = self._initial_slots(b)
        learned_anchor = slots_prior

        slots_list = []
        attn_list = []
        mass_list = []

        for frame_idx in range(t):
            feat_t = inputs[:, frame_idx].reshape(b, hw, d)
            slots_t, attn_t, mass_t = self._step_frame(slots_prior, feat_t, abs_pe, coords)

            # Inject unnormalized coverage/count cue so weighted-average slots retain object count
            slots_with_mass = slots_t + self.mass_proj(mass_t)
            slots_list.append(slots_with_mass)
            attn_list.append(attn_t.view(b, self.total_slots, h, w))
            mass_list.append(mass_t)

            if self.use_savi_carryover:
                # Blend predicted temporal prior with class prototype anchor so slots keep class identity
                slots_prior = 0.5 * learned_anchor + 0.5 * (slots_t + self.transition(slots_t))

        slots_seq = torch.stack(slots_list, dim=1)     # (B, T, S_total, D)
        attn_masks = torch.stack(attn_list, dim=1)     # (B, T, S_total, H, W)
        slot_mass = torch.stack(mass_list, dim=1)      # (B, T, S_total, 2)

        return {
            "slots_seq": slots_seq,
            "obj_slots": slots_seq[:, :, :self.num_obj_slots, :],
            "attn_masks": attn_masks,
            "slot_mass": slot_mass,
        }


# ==============================================================================
# 4. DINOSAUR-Style Spatial Broadcast Feature Decoder
# ==============================================================================

class SpatialBroadcastFeatureDecoder(nn.Module):
    """Lightweight feature-space decoder (DINOSAUR-style).

    Reconstructs input feature map (B, T, H, W, D) from slots (B, T, S_total, D).
    Because every slot uses the exact same shared MLP decoder over broadcast 2D
    coordinates, slots are forced to factorize into distinct spatial entities.
    """

    def __init__(self, slot_dim: int = 256, hidden_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim + 1),  # out_dim features + 1 alpha logit
        )

    def forward(self, slots_seq: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode all slots into reconstructed features (B, T, H, W, D) and alpha masks."""
        b, t, s, d = slots_seq.shape
        coords = make_2d_coords(h, w, slots_seq.device, slots_seq.dtype).view(1, 1, 1, h * w, 2)
        coords_exp = coords.expand(b, t, s, -1, -1)
        slots_exp = slots_seq.unsqueeze(3).expand(-1, -1, -1, h * w, -1)

        dec_in = torch.cat([slots_exp, coords_exp], dim=-1)  # (B, T, S, HW, D+2)
        dec_out = self.mlp(dec_in)

        feats_pred = dec_out[..., :-1]                       # (B, T, S, HW, out_dim)
        alpha_logits = dec_out[..., -1]                      # (B, T, S, HW)
        alpha_masks = alpha_logits.softmax(dim=2)            # Compete across slots

        recon = (feats_pred * alpha_masks.unsqueeze(-1)).sum(dim=2)  # (B, T, HW, out_dim)
        return recon.view(b, t, h, w, -1), alpha_masks.view(b, t, s, h, w)


# ==============================================================================
# 5. Spatial Convergence Losses for Slot Attention
# ==============================================================================

class SlotConvergenceLoss(nn.Module):
    """Combines spatial losses that force crisp, non-overlapping object binding."""

    def __init__(
        self,
        num_obj_slots: int = 6,
        coverage_weight: float = 1.0,
        overlap_weight: float = 0.5,
        entropy_weight: float = 0.1,
        recon_weight: float = 0.5,
    ):
        super().__init__()
        self.num_obj_slots = num_obj_slots
        self.coverage_weight = coverage_weight
        self.overlap_weight = overlap_weight
        self.entropy_weight = entropy_weight
        self.recon_weight = recon_weight

    @staticmethod
    def _soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pred_f = pred.reshape(pred.shape[0], -1)
        tgt_f = target.reshape(target.shape[0], -1)
        inter = (pred_f * tgt_f).sum(dim=-1)
        union = pred_f.sum(dim=-1) + tgt_f.sum(dim=-1)
        dice = (2.0 * inter + eps) / (union + eps)
        return (1.0 - dice).mean()

    def forward(
        self,
        attn_masks: torch.Tensor,
        bg_seg: torch.Tensor,
        target_feats: Optional[torch.Tensor] = None,
        recon_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute spatial convergence losses.

        Args:
            attn_masks: (B, T, S_total, H, W)
            bg_seg: (B, T, H_img, W_img) with 1.0=background, 0.0=foreground
            target_feats: optional (B, T, H, W, D) backbone features
            recon_feats: optional (B, T, H, W, D) decoded features
        """
        b, t, _, h, w = attn_masks.shape
        fg_attn = attn_masks[:, :, :self.num_obj_slots]  # (B, T, S, H, W)

        # Downsample bg_seg to attention spatial resolution
        bg_ds = F.interpolate(
            bg_seg.float().reshape(b * t, 1, bg_seg.shape[-2], bg_seg.shape[-1]),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).view(b, t, h, w).clamp(0.0, 1.0)
        fg_target = 1.0 - bg_ds

        # 1. Foreground Coverage: union of object slots must match foreground (1 - bg_seg)
        fg_union = fg_attn.sum(dim=2).clamp(1e-6, 1.0 - 1e-6)
        cov_bce = F.binary_cross_entropy(fg_union, fg_target)
        cov_dice = self._soft_dice_loss(fg_union, fg_target)
        loss_coverage = self.coverage_weight * (cov_bce + cov_dice)

        # 2. Mutual Exclusivity / Overlap Penalty among foreground slots
        # Penalizes pairwise dot product sum_{i != j} A_i * A_j at every pixel
        sum_sq = fg_attn.sum(dim=2).pow(2)
        sq_sum = fg_attn.pow(2).sum(dim=2)
        overlap = 0.5 * (sum_sq - sq_sum)  # (B, T, H, W)
        loss_overlap = self.overlap_weight * overlap.mean()

        # 3. Foreground Entropy Loss: encourages one-hot slot selection on object pixels
        fg_probs = fg_attn / fg_attn.sum(dim=2, keepdim=True).clamp(min=1e-6)
        pixel_entropy = -(fg_probs * fg_probs.clamp(min=1e-6).log()).sum(dim=2)
        fg_weight = (fg_target > 0.1).float()
        loss_entropy = self.entropy_weight * (
            (pixel_entropy * fg_weight).sum() / fg_weight.sum().clamp(min=1.0)
        )

        losses = {
            "loss_fg_coverage": loss_coverage,
            "loss_slot_overlap": loss_overlap,
            "loss_fg_entropy": loss_entropy,
        }

        # 4. Feature Reconstruction (DINOSAUR)
        if target_feats is not None and recon_feats is not None and self.recon_weight > 0:
            losses["loss_feat_recon"] = self.recon_weight * F.mse_loss(
                recon_feats, target_feats.detach()
            )

        return losses


# ==============================================================================
# 6. Smoke Test
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, H, W, D = 2, 16, 14, 14, 256
    S = 6

    slot_attn = ImprovedVideoSlotAttention(
        num_obj_slots=S,
        dim=D,
        num_iterations=3,
        bg_slot=True,
        use_boqsa=True,
        use_isa_rel_pe=True,
        use_savi_carryover=True,
    ).to(device)
    decoder = SpatialBroadcastFeatureDecoder(slot_dim=D, hidden_dim=128, out_dim=D).to(device)
    criterion = SlotConvergenceLoss(num_obj_slots=S).to(device)

    feats = torch.randn(B, T, H, W, D, device=device, requires_grad=True)
    bg_seg = torch.ones(B, T, 224, 224, device=device)
    bg_seg[:, :, 60:110, 70:120] = 0.0  # Dummy foreground object region

    out = slot_attn(feats)
    recon_feats, alpha_masks = decoder(out["slots_seq"], H, W)
    losses = criterion(out["attn_masks"], bg_seg, target_feats=feats, recon_feats=recon_feats)
    total_loss = sum(losses.values())
    total_loss.backward()

    print("slots_seq :", tuple(out["slots_seq"].shape))
    print("obj_slots :", tuple(out["obj_slots"].shape))
    print("attn_masks:", tuple(out["attn_masks"].shape))
    print("slot_mass :", tuple(out["slot_mass"].shape))
    print("recon     :", tuple(recon_feats.shape))
    print("alpha     :", tuple(alpha_masks.shape))
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    assert torch.isfinite(total_loss) and feats.grad is not None
    print("Smoke test PASSED!")
