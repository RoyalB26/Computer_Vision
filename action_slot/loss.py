import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def inter_and_union(pred, mask, num_class=1, start_class=0):
    pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
    mask = mask.data.cpu().numpy().astype(np.uint8)
    pred = np.asarray(pred, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=np.uint8).copy()
    inter = pred * (pred == mask)
    (area_inter, _) = np.histogram(inter, bins=num_class, range=(start_class, num_class))
    (area_pred, _) = np.histogram(pred, bins=num_class, range=(start_class, num_class))
    (area_mask, _) = np.histogram(mask, bins=num_class, range=(start_class, num_class))
    area_union = area_pred + area_mask - area_inter
    return (area_inter, area_union)



class ActionSlotLoss(nn.Module):
    def __init__(self, num_actor_class, attention_res=None, bce_pos_weight = 10, num_slots: int= 32, device = None):
        super(ActionSlotLoss, self).__init__()
        self.num_actor_class = num_actor_class
        self.num_slots = num_slots
        self.attention_res = attention_res
        self.bce_pos_weight = bce_pos_weight

        self.bg_mask = True
        self.bg_slot = True
        self.obj_mask = False
        self.action_attn_weight = 1
        self.bg_attn_weight = 0.5

        self.seq_len = 16
        self.mask_every_frame = 4
        self.bg_upsample = 4
        if self.attention_res: self.attention_res = (self.attention_res[0] * self.bg_upsample, self.attention_res[1] * self.bg_upsample)

        self.device = device

        self.ego_ce = nn.CrossEntropyLoss(reduction='mean')
        self.actor_loss_type =  self._parse_actor_loss()
        self.attn_loss_type = self._parse_attn_loss()

    def _parse_actor_loss(self):
        pos_weight = torch.ones([self.num_actor_class])*self.bce_pos_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

        weight_matrix = torch.tensor([
            [6.76, 6.34, 1.0],  [8.04, 6.30, 1.0],  [6.43, 16.30, 1.0], [6.52, 5.44, 1.0],
            [7.71, 6.34, 1.0],  [7.19, 5.53, 1.0],  [6.65, 7.18, 1.0],  [6.67, 21.65, 1.0],
            [9.00, 7.12, 1.0],  [7.56, 5.82, 1.0],  [6.29, 5.83, 1.0],  [6.66, 6.71, 1.0],

            [5.72, 13.46, 1.0], [7.31, 6.79, 1.0],  [6.42, 6.64, 1.0],  [8.49, 5.11, 1.0],
            [10.88, 7.17, 1.0], [10.70, 5.64, 1.0], [7.31, 7.00, 1.0],  [8.72, 5.56, 1.0],
            [11.18, 5.01, 1.0], [10.34, 7.10, 1.0], [6.78, 5.00, 1.0],  [10.24, 5.42, 1.0],
            
            [6.13, 5.88, 1.0],  [6.54, 5.15, 1.0],  [5.64, 6.53, 1.0],  [7.05, 7.35, 1.0],
            [8.69, 5.95, 1.0],  [6.82, 6.07, 1.0],  [6.43, 5.52, 1.0],  [6.68, 6.29, 1.0]
        ], device=self.device)

        self.register_buffer('weight_matrix', weight_matrix)  # Tự động chuyển device theo model
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        return 0

    def _parse_attn_loss(self):
        flag = 0
        if not self.bg_mask and self.action_attn_weight>0:
            flag = 2
        elif self.obj_mask:
            flag = 1
        elif self.bg_slot and self.bg_mask and self.action_attn_weight>0. and self.bg_attn_weight>0.:
            flag = 3
        elif self.bg_slot and self.bg_mask and self.bg_attn_weight>0. and not self.action_attn_weight > 0.:
            flag = 4
        if flag >0:
            self.obj_bce = nn.BCELoss()
        return flag
    
    def ego_loss(self,pred, label):
        if pred is None:
            return None
        return self.ego_ce(pred, label)

    def actor_loss(self, pred, label):
        if self.actor_loss_type == 1:
            bs, num_queries = pred.shape[:2]
            out_prob = pred.clone().detach().flatten(0, 1).softmax(-1)
            actor_gt_np = label.clone().detach()
            tgt_ids = torch.cat([v for v in actor_gt_np.detach()])
            C = -out_prob[:, tgt_ids].clone().detach()
            C = C.view(bs, num_queries, -1).cpu()
            sizes = [len(v) for v in actor_gt_np]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            indx = [(torch.as_tensor(i, dtype=torch.int64).detach(), torch.as_tensor(j, dtype=torch.int64).detach()) for i, j in indices]
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indx)]).detach()
            src_idx = torch.cat([src for (src, _) in indx]).detach()
            idx = (batch_idx, src_idx)
            target_classes_o = torch.cat([t[J] for t, (_, J) in zip(label, indx)]).to(pred.device)
            target_classes = torch.full(pred.shape[:2], self.num_actor_class,
                                dtype=torch.int64, device=out_prob.device)

            target_classes[idx] = target_classes_o
            actor_loss = self.bce(pred.transpose(1, 2), target_classes)
        else:
            actor_loss = self.bce(pred, label)
        return actor_loss

    def attn_loss(self, attn, label, validate):
        attn_loss = None
        bg_attn_loss = None
        action_attn = None
        bg_attn = None

        # prepare background mask
        if self.bg_attn_weight>0:
            bg_seg = []
            bg_seg: Tensor = label['bg_seg']
            b, l, _, h, w = bg_seg.shape
            bg_seg = torch.reshape(bg_seg, (b, l, h, w))
            
        # object mask supervision for action slot
        if self.attn_loss_type == 1:
            obj_mask_list = []
            obj_mask = label['obj_masks']
            for i in range(self.seq_len//self.mask_every_frame):
                obj_mask_list.append(obj_mask[i].to(attn.device, dtype=torch.float32))
            obj_mask_list = torch.stack(obj_mask_list, 0)
            obj_mask_list = torch.permute(obj_mask_list, (1, 0, 2, 3, 4)) #[batch, len, n, h, w]
            b, l, n, h, w = obj_mask_list.shape

            attn_loss = 0.0
            # 8, 16, 64, 8, 24
            b, l, n, h, w = attn.shape
            attn = torch.reshape(attn, (-1, 1 ,8, 24))
            attn = F.interpolate(attn, size=(32, 96))
            attn = torch.reshape(attn, (b, l , n, 32, 96))

            # sup_idx = [1,3,5,7,9,11,13,15]
            # attn = attn[:,sup_idx,:,:,:].reshape((b,8,n,32,96))
            attn = attn[:, ::self.mask_every_frame, :, :, :].reshape(b, -1, n, 32, 96)

            b, seq, n_obj, h, w = obj_mask_list.shape
            mask_detach = attn.detach().flatten(3,4)
            mask_detach = mask_detach.cpu().numpy()
            mask_gt_np = obj_mask_list.flatten(3,4)
            mask_gt_np = mask_gt_np.detach().cpu().numpy()
            scores = np.zeros((b, 4, n, n_obj))
            for i in range(b):
                for j in range(4):
                    cross_entropy_cur = np.matmul(np.log( mask_detach[i,j]), mask_gt_np[i,j].T) + np.matmul(np.log(1 - mask_detach[i,j]), (1 - mask_gt_np[i,j]).T)
                    scores[i,j] += cross_entropy_cur
            for i in range(b):
                for j in range(4):
                    matches = linear_sum_assignment(-scores[i,j])
                    id_slot, id_gt = matches 
                    attn_loss += self.obj_bce(attn[i,j,id_slot,:,:], obj_mask_list[i,j,id_gt,:,:])

        elif self.attn_loss_type == 2:
            b, l, n, h, w = attn.shape
            if self.bg_upsample != 1:
                attn = attn.reshape(-1, 1, h, w)
                attn = F.interpolate(attn, size=self.attention_res, mode='bilinear')
                _, _, h, w = attn.shape
                attn = attn.reshape(b, l, n, h, w)
            action_attn = attn[:, :, :self.num_actor_class, :, :]

            class_idx = label['actor'] == 0.0
            class_idx = class_idx.view(b, self.num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
            class_idx = torch.permute(class_idx, (0, 2, 1, 3, 4))

            attn_gt = torch.zeros([b, l, self.num_actor_class, h, w], dtype=torch.float32, device=action_attn.device)
            if class_idx.any():
                attn_loss = self.obj_bce(action_attn[class_idx].clamp(1e-6, 1.0 - 1e-6), attn_gt[class_idx])
            else:
                attn_loss = action_attn.sum() * 0.0

        # Action-slot, background mask + negative mask
        elif self.attn_loss_type == 3:
            b, l, n, h, w = attn.shape


            if self.bg_upsample != 1:
                attn = attn.reshape(-1, 1, h, w)
                attn = F.interpolate(attn, size=self.attention_res, mode='bilinear')
                _, _, h, w = attn.shape
                attn = attn.reshape(b, l, n, h, w)

            action_attn = attn[:, :, :self.num_slots, :, :]
            bg_attn = attn[:, ::self.mask_every_frame, -1, :, :].reshape(b, -1, h, w)

            mask_64 = (label['actor'] == 0.0)  # shape: (B, 64), dtype: torch.bool
            B = mask_64.shape[0]

            
            c1 = (mask_64[:, 0:24] == 1).any(dim=1)
            c2 = (mask_64[:, 24:48] == 1).any(dim=1)
            c3 = (mask_64[:, 48:64] == 1).any(dim=1)

            
            class_idx = torch.stack([c1, c2, c3], dim=-1).to(mask_64.dtype) # (B,3)
            class_idx = class_idx.view(b, self.num_slots, 1, 1, 1).repeat(1, 1, l, h, w)
            class_idx = torch.permute(class_idx, (0, 2, 1, 3, 4))

            attn_gt = torch.zeros([b, l, self.num_slots, h, w], dtype=torch.float32, device=action_attn.device)
            
            if class_idx.any():
                attn_loss = self.obj_bce(action_attn[class_idx].clamp(1e-6, 1.0 - 1e-6), attn_gt[class_idx])
            else:
                attn_loss = action_attn.sum() * 0.0
            bg_attn_loss = self.obj_bce(bg_attn.clamp(1e-6, 1.0 - 1e-6), bg_seg.clamp(0.0, 1.0))
        # background mask only
        elif self.attn_loss_type == 4:
            b, l, n, h, w = attn.shape

            if self.bg_upsample != 1:
                attn = attn.reshape(-1, 1, h, w)
                attn = F.interpolate(attn, size=self.attention_res, mode='bilinear')
                _, _, h, w = attn.shape
                attn = attn.reshape(b, l, n, h, w)

            bg_attn = attn[:, ::self.mask_every_frame, -1, :, :].reshape(b, l//self.mask_every_frame, h, w)
            bg_attn_loss = self.obj_bce(bg_attn.clamp(1e-6, 1.0 - 1e-6), bg_seg.clamp(0.0, 1.0))
            loss = {'bg_attn_loss': 0.5 * bg_attn_loss}
            return loss

        loss = {'attn_loss':attn_loss,'bg_attn_loss': 0.5 * bg_attn_loss}

        if validate:
            loss['action_inter'] = None
            loss['action_union'] = None
            loss['bg_inter'] = None
            loss['bg_union'] = None

            if action_attn is not None:
                action_attn_pred = action_attn[class_idx] > 0.5
                inter, union = inter_and_union(action_attn_pred.reshape(-1, h, w), attn_gt[class_idx].reshape(-1, h, w), 1, 0)
                loss['action_inter'] = inter
                loss['action_union'] = union
            if bg_attn is not None:
                bg_attn_pred = bg_attn > 0.5
                inter, union = inter_and_union(bg_attn_pred, bg_seg, 1, 1)
                loss['bg_inter'] = inter
                loss['bg_union'] = union

        return loss

    def actor_loss_reduce(self, pred, label):
        # logits: (B, 32, 3), targets: (B, 32)
        raw_loss = self.criterion(pred.reshape(-1, 3), label.reshape(-1)).view(-1, 32)  # shape: (B, 32)

        # Tra cứu trọng số tương ứng cho từng slot (0..31) và từng target (0, 1, 2)
        slot_indices = torch.arange(32, device=pred.device).unsqueeze(0)  # shape: (1, 32)
        sample_weights = self.weight_matrix[slot_indices, label]          # shape: (B, 32)

        # Nhân trọng số và chuẩn hóa (nên chia cho tổng trọng số để learning rate không bị biến động mạnh)
        loss = (raw_loss * sample_weights).sum() / sample_weights.sum()
        return loss

    def forward(self, pred, label, validate=False):

        # actor = label['actor']  # shape: (B, 64)
        # B = actor.shape[0]

        # # Gom thành (B, 32, 2)
        # part1 = actor[:, 0:24].reshape(B, 2, 12).transpose(1, 2)
        # part2 = actor[:, 24:48].reshape(B, 2, 12).transpose(1, 2)
        # part3 = actor[:, 48:64].reshape(B, 2, 8).transpose(1, 2)
        # fg = torch.cat([part1, part2, part3], dim=1)  # shape: (B, 32, 2)

        # # Chuyển thẳng thành index {0, 1, 2} kiểu torch.long:
        # # [1, 0] -> 0 | [0, 1] -> 1 | [0, 0] -> 2
        # targets = torch.where(
        #     fg.sum(dim=-1) == 0,
        #     2,
        #     fg.argmax(dim=-1)
        # ).long()  # shape: (B, 32), dtype: torch.long



        ego_loss = self.ego_loss(pred['ego'],label['ego'])
        actor_loss = self.actor_loss(pred['actor'],label['actor'])
        # attention_loss = self.attn_loss(pred['attn'],label,validate)

        loss = {"loss_ego": 0.5 * ego_loss, "loss_actor": actor_loss} 
        # loss |= attention_loss

        return loss
