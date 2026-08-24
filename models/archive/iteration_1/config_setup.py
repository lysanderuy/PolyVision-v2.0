#======= NOT USED AS OF THE MOMENT. IT IS IN THE binary_config ========

import os
from datetime import datetime
from detectron2.solver import WarmupMultiStepLR
# import iou_visualization

# You need these variables from your dataset registration
# Make sure dataset.py has been run first, or define them here:
DATA_SET_NAME = "SEAMaP-Binary-Full"  # From your dataset registration

# HYPERPARAMETERS
ARCHITECTURE = "faster_rcnn_R_50_FPN_3x"  # or faster_rcnn_R_101_FPN_3x
CONFIG_FILE_PATH = f"COCO-Detection/{ARCHITECTURE}.yaml"
MAX_ITER = 10000  # You might want to reduce this for CPU training
EVAL_PERIOD = 200
NUM_CLASSES = 2
BASE_LR = 0.01  # Consider reducing for CPU training

# OUTPUT DIR
OUTPUT_DIR_PATH = os.path.join(
    DATA_SET_NAME,
    ARCHITECTURE,
    datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
)

os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

print("Configuration Setup:")
print(f"Architecture: {ARCHITECTURE}")
print(f"Config file: {CONFIG_FILE_PATH}")
print(f"Max iterations: {MAX_ITER}")
print(f"Evaluation period: {EVAL_PERIOD}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Base learning rate: {BASE_LR}")
print(f"Output directory: {OUTPUT_DIR_PATH}")
print(f"Output directory created: {os.path.exists(OUTPUT_DIR_PATH)}")
