import os
from datetime import datetime
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.solver import WarmupMultiStepLR

# You need these variables - make sure previous scripts have been run
DATA_SET_NAME = "SEAMaP-Multi-class-100"
TRAIN_DATA_SET_NAME = f"{DATA_SET_NAME}-train"
TEST_DATA_SET_NAME = f"{DATA_SET_NAME}-test"

# HYPERPARAMETERS (from previous script)                          ASK KUYA JHURY ABOUT MULTICLASS CONFIGURATION
ARCHITECTURE = "faster_rcnn_R_50_FPN_3x"
CONFIG_FILE_PATH = f"COCO-Detection/{ARCHITECTURE}.yaml"
MAX_ITER = 10000  # Reduced for CPU training
EVAL_PERIOD = 200
NUM_CLASSES = 4
BASE_LR = 0.01  # Reduced for CPU training

# OUTPUT DIR
OUTPUT_DIR_PATH = os.path.join(
    DATA_SET_NAME,
    ARCHITECTURE,
    datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
)
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

# MODEL CONFIGURATION - CPU OPTIMIZED
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE_PATH))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE_PATH)

# Dataset configuration
cfg.DATASETS.TRAIN = (TRAIN_DATA_SET_NAME,)
cfg.DATASETS.TEST = (TEST_DATA_SET_NAME,)

# CPU-optimized settings
cfg.MODEL.DEVICE = "cpu"  # Force CPU usage
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Reduced for CPU
cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
cfg.DATALOADER.NUM_WORKERS = 0  # Set to 0 for CPU (avoid multiprocessing issues)
cfg.SOLVER.IMS_PER_BATCH = 2  # Reduced for CPU memory

# Input configuration
cfg.INPUT.MASK_FORMAT = 'bitmask'

# Solver configuration - CPU optimized
cfg.SOLVER.BASE_LR = BASE_LR
cfg.SOLVER.MAX_ITER = MAX_ITER
cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
cfg.SOLVER.WARMUP_ITERS = 1000  # Reduced for CPU
cfg.SOLVER.WARMUP_METHOD = "linear"
cfg.SOLVER.WARMUP_FACTOR = 0.001
cfg.SOLVER.STEPS = (2000, 5000, 8000)  # Adjusted for lower MAX_ITER
cfg.SOLVER.GAMMA = 0.1

# Model configuration
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
cfg.OUTPUT_DIR = OUTPUT_DIR_PATH

print("Model Configuration (CPU-Optimized):")
print(f"Device: {cfg.MODEL.DEVICE}")
print(f"Batch size per image: {cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}")
print(f"Images per batch: {cfg.SOLVER.IMS_PER_BATCH}")
print(f"Num workers: {cfg.DATALOADER.NUM_WORKERS}")
print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
print(f"Learning rate: {cfg.SOLVER.BASE_LR}")
print(f"Output directory: {cfg.OUTPUT_DIR}")

# Save config for inspection
with open(os.path.join(OUTPUT_DIR_PATH, "config.yaml"), "w") as f:
    f.write(cfg.dump())
print(f"Configuration saved to: {os.path.join(OUTPUT_DIR_PATH, 'config.yaml')}")