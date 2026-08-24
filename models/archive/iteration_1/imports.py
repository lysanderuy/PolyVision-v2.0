# COMMON LIBRARIES
import os
import cv2
from datetime import datetime

# Replace Google Colab's cv2_imshow with regular cv2 display
def cv2_imshow(image):
    """Display image using OpenCV (replacement for Colab's cv2_imshow)"""
    cv2.imshow('Image', image)
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows()  # Close window

# DATA SET PREPARATION AND LOADING
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# VISUALIZATION
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

# CONFIGURATION
from detectron2 import model_zoo
from detectron2.config import get_cfg

# EVALUATION
from detectron2.engine import DefaultPredictor

# TRAINING
from detectron2.engine import DefaultTrainer

print("All imports successful!")
print("Ready for CPU-based detectron2 work")