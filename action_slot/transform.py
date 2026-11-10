
import os
import sys

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from typing import Any, Callable, Dict, List, Optional, Union
from torch import Tensor
import numpy as np
import torch
import torchvision.transforms as transforms
from BaseModel.core.registrar import TRANSFORMS
# from core.registry import build_from_cfg


@TRANSFORMS.register_module("Action_Slot_Transform")
class Action_Slot_Transform:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        
        if data['inputs']:
            data['inputs'] = self.to_np(data['inputs'])
            data['inputs'] = self.resize_videos(data['inputs'])
            data['inputs'] = torch.stack(data['inputs'], dim=0)


        if "bg_seg" in data['targets']:
            data['targets']['bg_seg'] = self.to_np_no_norm(data['targets']['bg_seg'])
            data['targets']['bg_seg'] = self.resize_masks(data['targets']['bg_seg'])
            data['targets']['bg_seg'] = torch.stack(data['targets']['bg_seg'], dim= 0)
        
        return data

    def resize_videos(self, data):
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode

        for i, img in enumerate(data):
            if isinstance(img, torch.Tensor) or hasattr(img, "resize"):
                data[i] = TF.resize(img, [256,768], interpolation=InterpolationMode.BILINEAR)
        return data
    
    def resize_masks(self, data):
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode
        for i, mask in enumerate(data):
            if isinstance(mask, torch.Tensor) or hasattr(mask, "resize"):
                data[i] = TF.resize(mask, [32,96], interpolation=InterpolationMode.NEAREST)
        return data
    def to_np(self, v):
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])

        for i, _ in enumerate(v):
            v[i] = transform(v[i])
        return v


    def to_np_no_norm(self, v):
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])
        for i, _ in enumerate(v):
            v[i] = transform(v[i])
        return v


    