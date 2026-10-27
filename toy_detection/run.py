"""Entry point demonstrating how a task-specific project uses BaseModel."""
import os
import sys

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import custom task modules to register them into DATASETS & MODELS
import toy_detection.dataset
import toy_detection.model

from BaseModel.scripts.train import main

if __name__ == "__main__":
    # If no arguments provided in CLI, pass default config
    if len(sys.argv) == 1:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        sys.argv.extend(["--config", config_path])

    main()
