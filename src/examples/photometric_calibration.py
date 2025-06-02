import gsoup
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    print("Photometric Calibration Example")
    # offline steps for photometric calibration
    # 1. create patterns
    # 2. project patterns and acquire images
    # 3. find camera response function per channel
    # 4. linearize camera response function
    # 5. white balance camera channels
    # 6. find projector response function & color mixing matrix per-pixel
    # online steps for photometric calibration
    # 1. load/create patterns to project
    # 2. compute compensation images & project them
