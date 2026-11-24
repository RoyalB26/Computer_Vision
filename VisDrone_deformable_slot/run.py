"""Entry point to train VisDroneYOLO11DeformableSlot (Approach 2: Deformable High-Res Micro-Patch)."""
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Register modules into MODELS, DATASETS, LOSSES, METRICS, HOOKS
import VisDrone_deformable_slot.dataset
import VisDrone_deformable_slot.model
import VisDrone_yolov11.loss
import VisDrone_yolov11.metrics

from BaseModel.scripts.train import main as gpu_main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        sys.argv.extend(["--config", config_path])
    gpu_main()
