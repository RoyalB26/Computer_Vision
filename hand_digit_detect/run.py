"""Entry point for COCO Object Detection."""
import os
import sys

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import custom task modules
import hand_digit_detect.dataset
import hand_digit_detect.model

from BaseModel.scripts.train import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        sys.argv.extend(["--config", config_path])

    main()
