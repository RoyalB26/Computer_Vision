import os
import sys

# Thêm đường dẫn project vào sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CV_ROOT = os.path.dirname(CURRENT_DIR)
if CV_ROOT not in sys.path:
    sys.path.insert(0, CV_ROOT)

from BaseModel.scripts.train import main

if __name__ == "__main__":
    main()
