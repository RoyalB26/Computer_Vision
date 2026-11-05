"""
Root shortcut to run inference for hand digit detection model.

Usage:
    python inference.py --image path/to/image.jpg
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hand_digit_mobilenetv3.inference import main

if __name__ == "__main__":
    main()
