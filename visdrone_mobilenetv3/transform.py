from typing import Any, Dict
from BaseModel.core.registrar import TRANSFORMS



class NormalizeBoxes:
    def __call__(self, data: Dict[str, Any], image_size: int = 320) -> Dict[str, Any]:
        if "target" in data and "boxes" in data['target']:
            data["target"]['boxes'] /= image_size
        return data
