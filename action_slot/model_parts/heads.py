"""Action and Ego Classification Heads."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Allocated_Head(nn.Module):
	def __init__(self, in_channel: int, num_ego_classes: int, num_actor_classes: int, ego_channel: int = 0, **kwargs):
		super(Allocated_Head, self).__init__()
		self.num_ego_classes = num_ego_classes
		self.num_actor_classes = num_actor_classes
		if self.num_ego_classes != 0:
			if ego_channel == 0:
				self.fc_ego = nn.Sequential(
		                nn.ReLU(inplace=False),
		                nn.Linear(in_channel, num_ego_classes),
		                )
			else:
				self.fc_ego = nn.Sequential(
		                nn.ReLU(inplace=False),
		                nn.Linear(ego_channel, num_ego_classes),
		                )
				
		self.fc_actor = nn.ModuleList()
		for i in range(num_actor_classes):
			self.fc_actor.append(nn.Sequential(
	                nn.ReLU(inplace=False),
	                nn.Linear(in_channel, 1),
	                )
				)

	def forward(self, x, ego_x=None):

		b, n, _ = x.shape
		y_actor = []
		y_ego = None
		for i in range(self.num_actor_classes):
			y_actor.append(self.fc_actor[i](x[:, i, :]))
		y_actor = torch.stack(y_actor, dim=0)
		y_actor = y_actor.permute((1, 0, 2))
		y_actor = torch.reshape(y_actor, (b, n))
		# x = torch.reshape(x, (b, n))
		if self.num_ego_classes != 0:
			y_ego = self.fc_ego(ego_x)
			
		return y_ego, y_actor

class ReduceAllocatedHead(nn.Module):

	def __init__(self, in_channel: int, num_ego_classes: int, num_actor_classes: int, ego_channel: int = 0, num_slots: int = 0, **kwargs):
		super(ReduceAllocatedHead, self).__init__()
		self.num_ego_classes = num_ego_classes
		self.num_actor_classes = num_actor_classes
		self.num_slots = num_slots
		if self.num_ego_classes != 0:
			if ego_channel == 0:
				self.fc_ego = nn.Sequential(
		                nn.ReLU(inplace=False),
		                nn.Linear(in_channel, num_ego_classes),
		                )
			else:
				self.fc_ego = nn.Sequential(
		                nn.ReLU(inplace=False),
		                nn.Linear(ego_channel, num_ego_classes),
		                )
				
		self.fc_actor = nn.ModuleList()
		for i in range(num_slots):
			self.fc_actor.append(nn.Sequential(
	                nn.ReLU(inplace=False),
	                nn.Linear(in_channel, 3),
	                )
				)

	def forward(self, x, ego_x=None):

		b, n, _ = x.shape
		y_actor = []
		y_ego = None
		for i in range(self.num_slots):
			y_actor.append(self.fc_actor[i](x[:, i, :]))
		y_actor = torch.stack(y_actor, dim=0)
		y_actor = y_actor.permute((1, 0, 2))
		y_actor = torch.reshape(y_actor, (b, n-1, 3))
		# x = torch.reshape(x, (b, n))
		if self.num_ego_classes != 0:
			y_ego = self.fc_ego(ego_x)
			
		return y_ego, y_actor


class ObjectActionHead(nn.Module):
    """Object-slot action head predicting `num_actor_classes // num_slots` binary logits per object slot.

    For 6 object slots and 48 classes (4 actions x [single, group+] per symbol):
        Input (B, >=6, D) -> (B, 6, 8) -> (B, 48) logits matching ALL_CLASSES order.
    """

    def __init__(
        self,
        in_channel: int,
        num_actor_classes: int = 64,
        num_slots: int = 3,
        actions_per_slot: list[int] = [24, 24, 16],
        num_ego_classes: int = 0,
        ego_channel: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.num_ego_classes = num_ego_classes
        self.num_actor_classes = num_actor_classes
        self.num_slots = num_slots
        self.actions_per_slot = actions_per_slot

        if self.num_ego_classes > 0:
            c = ego_channel if ego_channel > 0 else in_channel
            self.fc_ego = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(c, num_ego_classes),
            )
        else:
            self.fc_ego = None

        self.fc_actor = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(in_channel, self.actions_per_slot[i]),
            )
            for i in range(num_slots)
        ])

    def forward(
        self,
        x: torch.Tensor,
        ego_x: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        b, n, _ = x.shape
        slots_to_use = min(n, self.num_slots)
        y_actor = [self.fc_actor[i](x[:, i, :]) for i in range(slots_to_use)]
        y_actor = torch.cat(y_actor, dim=1).reshape(b, self.num_actor_classes)

        y_ego = self.fc_ego(ego_x) if self.fc_ego is not None and ego_x is not None else None
        return y_ego, y_actor


class DualStreamActionHead(nn.Module):
    """Decoupled dual-stream head for TACO / multi-slot action recognition.

    Separates:
      - Stream A (Appearance / Variants): takes `vis_slots` `(B, S, D)` and predicts
        `[single, group+]` logits `(B, S, 2)` per object slot.
      - Stream B (Motion / Trajectory): weight-shared across slots with the same
        transition space (e.g. Car & Bike share 12 zone transitions; Ped uses 8),
        takes `motion_slots` `(B, S, D)` and predicts direction logits.
      - Multiplicative (Hadamard) stream interaction `vis_h * motion_h` combined with
        additive log-odds `var_logits + dir_logits` to produce `(B, num_actor_classes)`.
    """

    def __init__(
        self,
        in_channel: int,
        num_actor_classes: int = 64,
        num_slots: int = 3,
        actions_per_slot: Optional[List[int]] = None,
        num_variants: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        num_ego_classes: int = 0,
        ego_channel: int = 0,
        **kwargs,
    ):
        super().__init__()
        if actions_per_slot is None:
            if num_slots == 3 and num_actor_classes == 64:
                actions_per_slot = [24, 24, 16]
            elif num_actor_classes % num_slots == 0:
                actions_per_slot = [num_actor_classes // num_slots] * num_slots
            else:
                raise ValueError(
                    f"actions_per_slot must be provided when num_actor_classes ({num_actor_classes}) "
                    f"is not divisible by num_slots ({num_slots})."
                )

        if len(actions_per_slot) != num_slots or sum(actions_per_slot) != num_actor_classes:
            raise ValueError(
                f"Invalid actions_per_slot={actions_per_slot} for num_slots={num_slots}, "
                f"num_actor_classes={num_actor_classes}."
            )
        for a in actions_per_slot:
            if a % num_variants != 0:
                raise ValueError(
                    f"Each entry in actions_per_slot ({actions_per_slot}) must be divisible by num_variants ({num_variants})."
                )

        self.num_ego_classes = num_ego_classes
        self.num_actor_classes = num_actor_classes
        self.num_slots = num_slots
        self.actions_per_slot = list(actions_per_slot)
        self.num_variants = num_variants
        self.dirs_per_slot = [a // num_variants for a in self.actions_per_slot]

        if self.num_ego_classes > 0:
            c = ego_channel if ego_channel > 0 else in_channel
            self.fc_ego = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(c, num_ego_classes),
            )
        else:
            self.fc_ego = None

        # Stream A: Appearance & cardinality [single, group+]
        self.slot_embed = nn.Parameter(torch.randn(1, num_slots, in_channel) * 0.02)
        self.vis_norm = nn.LayerNorm(in_channel)
        self.vis_proj = nn.Sequential(
            nn.Linear(in_channel, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.var_head = nn.Linear(hidden_dim, num_variants)

        # Stream B: Motion direction heads shared across slots sharing the same num_dirs
        self.motion_norm = nn.LayerNorm(in_channel)
        self.motion_proj = nn.Sequential(
            nn.Linear(in_channel, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        unique_dirs = sorted(set(self.dirs_per_slot))
        self.dir_heads = nn.ModuleDict({
            str(n_dir): nn.Linear(hidden_dim, n_dir) for n_dir in unique_dirs
        })
        self.inter_heads = nn.ModuleDict({
            str(n_dir): nn.Linear(hidden_dim, num_variants * n_dir) for n_dir in unique_dirs
        })

    def forward_with_streams(
        self,
        vis_slots: torch.Tensor,
        motion_slots: Optional[torch.Tensor] = None,
        ego_x: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Compute final actor logits together with decoupled variant and direction logits."""
        if motion_slots is None:
            motion_slots = vis_slots

        b = vis_slots.shape[0]
        s = self.num_slots
        vis_in = vis_slots[:, :s, :] + self.slot_embed
        motion_in = motion_slots[:, :s, :]

        vis_h = self.vis_proj(self.vis_norm(vis_in))  # (B, S, H)
        motion_h = self.motion_proj(self.motion_norm(motion_in))  # (B, S, H)

        var_logits = self.var_head(vis_h)  # (B, S, num_variants)

        actor_chunks: List[torch.Tensor] = []
        dir_chunks: List[torch.Tensor] = []
        for i in range(s):
            n_dir = self.dirs_per_slot[i]
            key = str(n_dir)
            dir_i = self.dir_heads[key](motion_h[:, i, :])  # (B, n_dir)
            inter_i = self.inter_heads[key](vis_h[:, i, :] * motion_h[:, i, :]).view(
                b, self.num_variants, n_dir
            )
            logits_i = var_logits[:, i, :].unsqueeze(-1) + dir_i.unsqueeze(-2) + inter_i
            actor_chunks.append(logits_i.reshape(b, self.actions_per_slot[i]))
            dir_chunks.append(dir_i)

        y_actor = torch.cat(actor_chunks, dim=1)
        dir_logits = torch.cat(dir_chunks, dim=1)
        y_ego = self.fc_ego(ego_x) if self.fc_ego is not None and ego_x is not None else None

        return {
            "ego": y_ego,
            "actor": y_actor,
            "var_logits": var_logits,
            "dir_logits": dir_logits,
        }

    def forward(
        self,
        x: torch.Tensor,
        ego_x: Optional[torch.Tensor] = None,
        motion_slots: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        out = self.forward_with_streams(x, motion_slots=motion_slots, ego_x=ego_x)
        return out["ego"], out["actor"]


class VideoObjectClassHead(nn.Module):
    """Predict video-level object class presence (e.g. x, x+, ..., O, O+) from all slots."""

    def __init__(
        self,
        in_channel: int,
        num_slots: int = 7,
        num_obj_classes: int = 12,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.num_obj_classes = num_obj_classes
        in_features = num_slots * in_channel
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, num_obj_classes),
        )

    def forward(self, all_slots: torch.Tensor) -> torch.Tensor:
        """Predict multi-label object class logits from all slots across T frames.

        Args:
            all_slots: Tensor of shape (B, T, actual_slots, D).
        Returns:
            Logits of shape (B, num_obj_classes).
        """
        b = all_slots.shape[0]
        pooled = 0.5 * (all_slots.mean(dim=1) + all_slots.max(dim=1).values)
        flat = pooled.reshape(b, -1)
        return self.classifier(flat)


class SpatialBroadcastFeatureDecoder(nn.Module):
    """DINOSAUR-style spatial broadcast decoder in backbone feature space.

    Reconstructs input feature map `(B, T, H, W, out_dim)` from slot representations
    `(B, T, actual_slots, slot_dim)` via a shared per-pixel MLP over broadcasted
    slot vectors and normalized 2D coordinates, plus slot-wise alpha masks.
    """

    def __init__(self, slot_dim: int = 256, hidden_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.slot_dim = slot_dim
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim + 1),  # out_dim features + 1 alpha logit
        )

    def forward(self, slots_seq: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode slots into reconstructed feature maps `(B, T, H, W, out_dim)` and alpha masks `(B, T, S, H, W)`."""
        b, t, s, _ = slots_seq.shape
        ys = torch.linspace(-1.0, 1.0, steps=h, device=slots_seq.device, dtype=slots_seq.dtype)
        xs = torch.linspace(-1.0, 1.0, steps=w, device=slots_seq.device, dtype=slots_seq.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([gx, gy], dim=-1).reshape(1, 1, 1, h * w, 2)

        coords_exp = coords.expand(b, t, s, -1, -1)
        slots_exp = slots_seq.unsqueeze(3).expand(-1, -1, -1, h * w, -1)
        dec_in = torch.cat([slots_exp, coords_exp], dim=-1)  # (B, T, S, HW, slot_dim + 2)
        dec_out = self.mlp(dec_in)

        feats_pred = dec_out[..., :-1]
        alpha_masks = dec_out[..., -1].softmax(dim=2)  # Compete across all slots
        recon = (feats_pred * alpha_masks.unsqueeze(-1)).sum(dim=2)
        return recon.view(b, t, h, w, self.out_dim), alpha_masks.view(b, t, s, h, w)
