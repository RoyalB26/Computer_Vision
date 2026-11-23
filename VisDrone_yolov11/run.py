"""Entry point to train VisDrone YOLOv11 model."""
import os
import sys

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import task modules to register into MODELS, DATASETS, LOSSES, METRICS
import VisDrone_yolov11.dataset
import VisDrone_yolov11.loss
import VisDrone_yolov11.model
import VisDrone_yolov11.metrics

from BaseModel.scripts.train import main as gpu_main

if __name__ == "__main__":
    # If no configuration file passed via CLI, default to local config.yaml
    if len(sys.argv) == 1:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        sys.argv.extend(["--config", config_path])

    if "--tpu" in sys.argv:
        from VisDrone_yolov11.tpu_train import main as tpu_main
        tpu_main()
    else:
        gpu_main()
