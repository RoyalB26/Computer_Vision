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
    def __init__(self, num_actor_class, attention_res=None, bce_pos_weight = 10, device = None):
        super(ActionSlotLoss, self).__init__()
        self.num_actor_class = num_actor_class
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
            attn_loss = self.obj_bce(action_attn[class_idx], attn_gt[class_idx])

        # Action-slot, background mask + negative mask
        elif self.attn_loss_type == 3:
            b, l, n, h, w = attn.shape


            if self.bg_upsample != 1:
                attn = attn.reshape(-1, 1, h, w)
                attn = F.interpolate(attn, size=self.attention_res, mode='bilinear')
                _, _, h, w = attn.shape
                attn = attn.reshape(b, l, n, h, w)

            action_attn = attn[:, :, :self.num_actor_class, :, :]
            bg_attn = attn[:, ::self.mask_every_frame, -1, :, :].reshape(b, -1, h, w)

            class_idx = label['actor'] == 0.0
            class_idx = class_idx.view(b, self.num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
            class_idx = torch.permute(class_idx, (0, 2, 1, 3, 4))

            attn_gt = torch.zeros([b, l, self.num_actor_class, h, w], dtype=torch.float32, device=action_attn.device)

            attn_loss = self.obj_bce(action_attn[class_idx], attn_gt[class_idx])
            bg_attn_loss = self.obj_bce(bg_attn, bg_seg)

        # background mask only
        elif self.attn_loss_type == 4:
            b, l, n, h, w = attn.shape

            if self.bg_upsample != 1:
                attn = attn.reshape(-1, 1, h, w)
                attn = F.interpolate(attn, size=self.attention_res, mode='bilinear')
                _, _, h, w = attn.shape
                attn = attn.reshape(b, l, n, h, w)

            bg_attn = attn[:, ::self.mask_every_frame, -1, :, :].reshape(b, l//self.mask_every_frame, h, w)
            bg_attn_loss = self.obj_bce(bg_attn, bg_seg)

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

    def forward(self, pred, label, validate=False):

        ego_loss = self.ego_loss(pred['ego'],label['ego'])
        actor_loss = self.actor_loss(pred['actor'],label['actor'])
        attention_loss = self.attn_loss(pred['attn'],label,validate)

        loss = {"loss_ego": 0.5 * ego_loss, "loss_actor": actor_loss} 
        loss |= attention_loss
        return loss
