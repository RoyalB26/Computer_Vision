"""PCGrad: Projecting Conflicting Gradients for Multi-Task Learning.
Paper: Gradient Surgery for Multi-Task Learning (Wang et al., NeurIPS 2020)
"""
from typing import List, Optional, Sequence
import random
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .registrar import OPTIMIZERS


@OPTIMIZERS.register_module(name="PCGrad")
class PCGrad:
    """Projecting Conflicting Gradients (PCGrad) optimizer wrapper.
    
    When training on multi-task / multi-loss objectives, gradients from different
    losses can conflict (inner product < 0). PCGrad projects each gradient onto the
    normal plane of any conflicting gradient to prevent gradient cancellation.
    """
    def __init__(self, optimizer: Optimizer, reduction: str = "sum"):
        self.optimizer = optimizer
        self.reduction = reduction

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        return self.optimizer.step(closure=closure)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def pc_backward(self, objectives: Sequence[torch.Tensor], scaler: Optional[torch.cuda.amp.GradScaler] = None):
        """
        Calculate projected conflicting gradients across objectives and assign to parameter gradients.
        
        Args:
            objectives: A list or tuple of scalar loss tensors [L_1, L_2, ..., L_m]
            scaler: Optional PyTorch GradScaler for mixed precision (AMP)
        """
        # 1. Filter active objectives with requires_grad
        active_objs = [obj for obj in objectives if isinstance(obj, torch.Tensor) and obj.requires_grad]
        if not active_objs:
            return

        # 2. Gather trainable parameters
        params = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)

        if len(active_objs) == 1:
            if scaler is not None:
                scaler.scale(active_objs[0]).backward()
            else:
                active_objs[0].backward()
            return

        num_tasks = len(active_objs)
        task_grads = []

        # 3. Compute task-specific gradients
        for i, obj in enumerate(active_objs):
            self.optimizer.zero_grad(set_to_none=True)
            retain_graph = (i < num_tasks - 1)
            
            if scaler is not None:
                scaler.scale(obj).backward(retain_graph=retain_graph)
            else:
                obj.backward(retain_graph=retain_graph)

            # Flatten gradients into a single 1D vector for fast projection
            grad_pieces = []
            for p in params:
                if p.grad is not None:
                    g_val = p.grad.detach()
                    if scaler is not None:
                        # Unscale gradient if scaler is used
                        inv_scale = 1.0 / (scaler.get_scale() + 1e-12)
                        g_val = g_val * inv_scale
                    grad_pieces.append(g_val.flatten())
                else:
                    grad_pieces.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
            task_grads.append(torch.cat(grad_pieces))

        # 4. Gradient Surgery (PCGrad Projection)
        projected_grads = [g.clone() for g in task_grads]
        
        task_indices = list(range(num_tasks))
        random.shuffle(task_indices)

        for i in task_indices:
            other_indices = [j for j in task_indices if j != i]
            random.shuffle(other_indices)
            for j in other_indices:
                g_i = projected_grads[i]
                g_j = task_grads[j]
                
                inner_prod = torch.dot(g_i, g_j)
                if inner_prod < 0:
                    # Conflicting gradients: project g_i onto the normal plane of g_j
                    norm_sq = torch.dot(g_j, g_j) + 1e-12
                    projected_grads[i] = g_i - (inner_prod / norm_sq) * g_j

        # 5. Aggregate projected gradients across tasks
        if self.reduction == "sum":
            final_grad = torch.stack(projected_grads, dim=0).sum(dim=0)
        elif self.reduction == "mean":
            final_grad = torch.stack(projected_grads, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

        # 6. Unpack final gradient back into parameter .grad attributes
        self.optimizer.zero_grad(set_to_none=True)
        offset = 0
        for p in params:
            numel = p.numel()
            p.grad = final_grad[offset : offset + numel].view_as(p).clone()
            offset += numel
