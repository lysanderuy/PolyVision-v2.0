import os
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

# Replace Google Colab's cv2_imshow with regular cv2 display
def cv2_imshow(image):
    """Display image using OpenCV (replacement for Colab's cv2_imshow)"""
    cv2.imshow('Dataset Visualization', image)
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows()  # Close window

# You'll need these variables from your dataset registration
# Make sure you've run dataset.py first, or define them here:
DATA_SET_NAME = "SEAMaP-Binary-Full"  # From your previous output
TRAIN_DATA_SET_NAME = f"{DATA_SET_NAME}-train"

# Get metadata and dataset
metadata = MetadataCatalog.get(TRAIN_DATA_SET_NAME)
dataset_train = DatasetCatalog.get(TRAIN_DATA_SET_NAME)

# Get first dataset entry
dataset_entry = dataset_train[0]
image = cv2.imread(dataset_entry["file_name"])

# Create visualizer
visualizer = Visualizer(
    image[:, :, ::-1],
    metadata=metadata,
    scale=0.8,
    instance_mode=ColorMode.IMAGE_BW
)

# Draw annotations
out = visualizer.draw_dataset_dict(dataset_entry)

# Display the image
cv2_imshow(out.get_image()[:, :, ::-1])

print(f"Displayed first image from: {TRAIN_DATA_SET_NAME}")
print(f"Image path: {dataset_entry['file_name']}")