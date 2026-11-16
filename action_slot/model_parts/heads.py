"""Action and Ego Classification Heads."""

from typing import Optional, Tuple
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
