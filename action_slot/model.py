"""COCO / Hand Digit Object Detection Model predicting bounding boxes and class labels with FPN."""
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
from torch import Tensor
from model_parts.backbones import X3DBackbone
from model_parts.heads import Allocated_Head
from model_parts.slot_attention import SlotAttention
from model_parts.x3d_baseline import X3D_Baseline
from loss import ActionSlotLoss
from typing import Dict, List, Optional, Union

from BaseModel.core import MODELS
from BaseModel.model import BaseModel


@MODELS.register_module(name="ActionSlotModel")
class ActionSlotModel(BaseModel):
    """Spatiotemporal Action-Slot model.

    Modularly connects:
        1. Backbone (X3D, VideoSwin, etc.)
        2. Spatiotemporal Projection & Ego Branch
        3. Slot Attention (Iterative Object/Action feature clustering)
        4. Multi-task Classification Head (Linear, Allocated, Deformable)
    """

    def __init__(
        self,
        num_ego_class: int = 7,
        num_actor_class: int = 64,
        num_slots: int = 64,
        slot_dim: int = 256,
        ego_c: int = 64,
        bg_slot: bool = False,
        resolution3d: tuple = (16, 8, 24),
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.num_ego_class = num_ego_class
        self.num_actor_class = num_actor_class
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.ego_c = ego_c

        finetune_last_n_stages = kwargs.get('finetune_last_n_stages', 2)
        freeze_stages = kwargs.get('freeze_stages', None)
        freeze_all = kwargs.get('freeze_all', False)
        unfreeze_all = kwargs.get('unfreeze_all', False)
        freeze_bn = kwargs.get('freeze_bn', True)
        pretrained_backbone = kwargs.get('pretrained_backbone', True)

        self.backbone = X3DBackbone(
            pretrained=pretrained_backbone,
            finetune_last_n_stages=finetune_last_n_stages,
            freeze_stages=freeze_stages,
            freeze_all=freeze_all,
            unfreeze_all=unfreeze_all,
            freeze_bn=freeze_bn,
        )
        in_c = self.backbone.out_channels

        # 2. Ego Feature Branch
        if self.num_ego_class > 0:
            self.conv3d_ego = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm3d(in_c),
                nn.Conv3d(in_c, self.ego_c, (1, 1, 1), stride=1),
            )
            self.pool_ego = nn.AdaptiveAvgPool3d(output_size=1)
        else:
            self.conv3d_ego = None
            self.pool_ego = None

        # 3. Spatiotemporal Feature Projection
        self.conv3d = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm3d(in_c),
            nn.Conv3d(in_c, self.slot_dim, (1, 1, 1), stride=1),
            nn.ReLU(),
        )

        # 4. Slot Attention
        actual_slots = self.num_slots + 1 if bg_slot else self.num_slots
        self.slot_attention = SlotAttention(
            num_slots=actual_slots,
            dim=self.slot_dim,
            eps=1e-8,
            input_dim=self.slot_dim,
            resolution=list(resolution3d),
            num_actor_class=num_actor_class,
            slot_dropout=kwargs.get('slot_dropout', 0.0),
        )

        # 5. Modular Classification Head
        self.head = Allocated_Head(
        in_channel= slot_dim, 
        num_ego_classes= num_ego_class,
        num_actor_classes=num_actor_class,
        ego_channel= ego_c
        )

        self.drop = nn.Dropout(p=dropout)

        self.loss_fn = ActionSlotLoss(num_actor_class = num_actor_class, attention_res=(resolution3d[1],resolution3d[2]), bce_pos_weight = 10)

    def extract_features(
        self,
        videos: Union[List[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        videos = videos.permute(0, 2, 1, 3, 4)
        seq_len = videos.shape[2]
        batch_size = videos.shape[0]

        # 1. Backbone feature extraction: (B, C, T_feat, H_feat, W_feat)
        feats = self.backbone(videos)

        # 2. Ego feature branch
        ego_x = None
        if self.conv3d_ego is not None:
            ego_feat = self.conv3d_ego(feats)
            ego_feat = self.pool_ego(ego_feat)
            ego_x = torch.reshape(ego_feat, (batch_size, self.ego_c))

        new_seq_len = feats.shape[2]
        new_h, new_w = feats.shape[3], feats.shape[4]

        # 3. Project to slot dimension: (B, T_feat, H_feat, W_feat, slot_dim)
        x = self.conv3d(feats)
        x = x.permute((0, 2, 3, 4, 1))
        x = torch.reshape(x, (batch_size, new_seq_len, new_h, new_w, -1))

        # 4. Slot Attention
        slots, attn_masks = self.slot_attention(x)

        # 5. Attention map upsampling back to input frame count
        b, n, _ = attn_masks.shape
        attn_masks = attn_masks.view(b * n, 1, new_seq_len, new_h, new_w)
        if seq_len > new_seq_len:
            attn_masks = F.interpolate(attn_masks, size=(seq_len, new_h, new_w), mode='trilinear')
        attn_masks = torch.reshape(attn_masks, (b, n, seq_len, new_h, new_w))
        attn_masks = attn_masks.permute((0, 2, 1, 3, 4))  # (B, T, num_slots, H, W)

        # 6. Apply Dropout & Head
        slots = self.drop(slots)
        if ego_x is not None:
            ego_x = self.drop(ego_x)

        slots = slots[:, :self.num_actor_class]

        return slots, ego_x, attn_masks

    def _compute_losses(
        self, 
        pred, 
        label, 
        validate=False
    ) -> Dict[str, torch.Tensor]:

        # print(label)
        
        label = {k: [d[k] for d in label] for k in label[0]} if label else {}
        label['ego'] = torch.stack(label['ego'], dim = 0)
        label['actor'] = torch.stack(label['actor'], dim = 0)
        label['bg_seg'] = torch.stack(label['bg_seg'], dim = 0)

        loss = self.loss_fn(pred, label, validate)
        
        return loss

    def _post_process(self, pred: Optional[Dict[str, Any]] = None):
        if pred is None:
            return None
        return pred

    def forward_train(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # print(inputs[0]['videos'].shape)
        # print(inputs["bg_seg"].shape)
        # print(targets['ego'].shape)
        # print(targets['actor'].shape)
        
        """Tính toán Detection Loss trong quá trình huấn luyện với FPN và Deformable Sampling Loss."""
        slots, ego_x, attn_masks = self.extract_features(inputs)
        pred_ego, pred_actor = self.head(slots, ego_x=ego_x)

        pred = {
            "actor": pred_actor,
            "ego": pred_ego,
            "attn": attn_masks,
        }

        return self._compute_losses(pred, targets)
    

    def predict(
        self,
        inputs: torch.Tensor,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        """Dự đoán bounding box, class label và confidence score khi suy luận / đánh giá."""
        slots, ego_x, attn_masks = self.extract_features(inputs)
        pred_ego, pred_actor = self.head(slots, ego_x=ego_x)

        pred = {
            "actor": pred_actor,
            "ego": pred_ego,
            "attn": attn_masks,
        }

        return self._post_process(pred)

    def forward_val(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Validation forward pass: tính validation losses và trích xuất preds phục vụ tính mAP."""
        slots, ego_x, attn_masks = self.extract_features(inputs)
        pred_ego, pred_actor = self.head(slots, ego_x=ego_x)

        pred = {
            "actor": pred_actor,
            "ego": pred_ego,
            "attn": attn_masks,
        }

        output = {}
        if targets is not None:
            loss_dict = self._compute_losses(pred, targets)
            output.update(loss_dict)

        preds = self._post_process(pred)
        output["preds"] = preds
        return output

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ):
        """Standard nn.Module forward dispatching to forward_train or forward_val."""
        if self.training:
            return self.forward_train(inputs, targets=targets, **kwargs)
        else:
            return self.forward_val(inputs, targets=targets, **kwargs)
