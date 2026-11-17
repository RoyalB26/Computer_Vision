"""Utility functions for road segmentation and color-difference foreground filtering."""
import torch
import torch.nn.functional as F


def filter_road_color_difference(
    videos: torch.Tensor,
    bg_seg: torch.Tensor,
    target_res: tuple = (8, 24),
    exp_factor: float = 5.0,
) -> torch.Tensor:
    """Filter road foreground vs flat background using color distance, LayerNorm, and ReLU.

    Formula:
        score = mean(exp(5 * |x - local_mean|))
        relu_out = ReLU(LayerNorm(score))
    
    The regions where relu_out == 0 within the road (bg_seg <= 0.5) correspond to the
    flat road background, while relu_out > 0 isolates dynamic vehicles / pedestrians.

    Args:
        videos: RGB video tensor of shape (B, T, 3, H, W) normalized in [0, 1] or raw.
        bg_seg: Background mask tensor of shape (B, T_bg, 1, H_bg, W_bg) or (B, T_bg, H_bg, W_bg),
                where bg_seg >= 0.5 is sky/buildings and bg_seg <= 0.5 is road + foreground.
        target_res: Target spatial resolution (H_feat, W_feat) to match slot attention.
        exp_factor: Factor for exponentiation (default 5.0).

    Returns:
        road_flat_mask: Tensor of shape (B, T, H_feat, W_feat) in {0.0, 1.0},
                        where 1.0 indicates flat road background (relu_out == 0)
                        and 0.0 elsewhere (sky/buildings or detected dynamic objects).
    """
    if videos.dim() != 5:
        raise ValueError(f"Expected 5D tensor (B, T, 3, H, W) for videos, got {videos.shape}")

    B, T, C, H, W = videos.shape
    H_feat, W_feat = target_res

    # Normalize videos if uint8
    if videos.dtype == torch.uint8:
        v_float = videos.float().div(255.0)
    else:
        v_float = videos.float()
        # If standardized with mean/std (e.g. mean=0.45, std=0.225), bring back to [0, 1]
        if v_float.min() < -0.5:
            v_float = (v_float * 0.225 + 0.45).clamp(0.0, 1.0)

    # 1. Temporal & spatial alignment of bg_seg to (B, T, H_feat, W_feat)
    if bg_seg.dim() == 5:
        bg_seg_t = bg_seg.squeeze(2)  # (B, T_bg, H_bg, W_bg)
    else:
        bg_seg_t = bg_seg

    bg_5d = bg_seg_t.unsqueeze(1).float()  # (B, 1, T_bg, H_bg, W_bg)
    bg_aligned = F.interpolate(
        bg_5d, size=(T, H_feat, W_feat), mode="nearest"
    ).squeeze(1)  # (B, T, H_feat, W_feat)

    black_mask = (bg_aligned <= 0.5)  # (B, T, H_feat, W_feat)

    # 2. Downsample videos to (H_feat, W_feat)
    vid_flat = v_float.reshape(B * T, C, H, W)
    vid_down = F.interpolate(vid_flat, size=(H_feat, W_feat), mode="area").reshape(
        B, T, C, H_feat, W_feat
    )
    x = vid_down.permute(0, 1, 3, 4, 2)  # (B, T, H_feat, W_feat, C)

    # 3. Compute baseline road color mean per sample
    road_means = []
    for b in range(B):
        mask_b = black_mask[b]
        if mask_b.any():
            m = x[b][mask_b].mean(dim=0, keepdim=True)
        else:
            m = x[b].mean(dim=(0, 1, 2), keepdim=True)
        road_means.append(m)
    road_mean = torch.stack(road_means, dim=0).view(B, 1, 1, 1, C)

    # 4. Color distance and exponential amplification
    color_diff = torch.abs(x - road_mean)
    exp_diff = torch.exp(float(exp_factor) * color_diff)
    score_map = exp_diff.mean(dim=-1)  # (B, T, H_feat, W_feat)

    # 5. LayerNorm over road region distribution, followed by ReLU
    road_flat_mask = torch.zeros_like(score_map)
    for b in range(B):
        mask_b = black_mask[b]
        if mask_b.any():
            vals = score_map[b][mask_b]
            mean_v = vals.mean()
            std_v = vals.std()
            norm_v = (vals - mean_v) / (std_v + 1e-6)
            relu_v = torch.relu(norm_v)
            # The user requested: focus on regions where relu == 0 (flat road background)
            flat_indicator = (relu_v == 0).float()
            road_flat_mask[b][mask_b] = flat_indicator

    return road_flat_mask
