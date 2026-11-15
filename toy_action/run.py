"""Entry point for action slot."""
import os
import sys

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import custom task modules
import toy_action.dataset
import toy_action.model

from BaseModel.scripts.train import main

if __name__ == "__main__":
    if "--config" not in sys.argv and "-c" not in sys.argv:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        sys.argv.extend(["--config", config_path])

    main()
