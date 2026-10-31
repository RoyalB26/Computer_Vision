
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
from BaseModel.core.registrar import TRANSFORMS
# from core.registry import build_from_cfg


@TRANSFORMS.register_module("NormalizeBoxes")
class NormalizeBoxes:
    def __call__(self, data: Dict[str, Any], image_size: int = 320) -> Dict[str, Any]:
        if "target" in data and "boxes" in data['target']:
            data["target"]['boxes']/= image_size
        return data