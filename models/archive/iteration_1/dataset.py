import os
import cv2
from dotenv import load_dotenv
from roboflow import Roboflow
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

load_dotenv()  # reads ROBOFLOW_API_KEY from the project-root .env

# ADD cv2_imshow function for local display
def cv2_imshow(image, filename="visualization.jpg"):
    """Save image to file (replacement for Colab's cv2_imshow)"""
    cv2.imwrite(filename, image)
    print(f"Image saved as: {filename}")
    print("Open the file to view the visualization")


# rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
# project = rf.workspace("seamap").project("seamap-binary-full")
# dataset = project.version(2).download("coco")

#-------------- FINAL BINARY DATASET ----------------------
rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
project = rf.workspace("seamap").project("seamap-binary-full")
dataset = project.version(6).download("coco")

#-------------- FINAL MULTICLASS-100 BALANCE DATASET ----------------------
# rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
# project = rf.workspace("seamap").project("seamap-multi-class-100")  #seamap-multi-class-100
# dataset = project.version(1).download("coco")

# Dataset configuration
DATA_SET_NAME = dataset.name.replace(" ", "-")
ANNOTATIONS_FILE_NAME = "_annotations.coco.json"

print(f"Dataset downloaded to: {dataset.location}")
print(f"Dataset name: {dataset.name}")
print(f"Processed dataset name: {DATA_SET_NAME}")
print(f"Annotations file: {ANNOTATIONS_FILE_NAME}")

# rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
# project = rf.workspace("seamap").project("seamap-binary-balanced")
# dataset = project.version(1).download("coco")


##================= M U L T I - C L A S S =======================

##25%
# rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
# project = rf.workspace("seamap").project("seamap-multi-class-25")
# dataset = project.version(1).download("coco")

##75%

# rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
# project = rf.workspace("seamap").project("seamap-multi-class-75")
# dataset = project.version(1).download("coco")

## 100% Balance



## 100% ALL - IMBALANCE

# rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
# project = rf.workspace("seamap").project("seamap")
# dataset = project.version(10).download("coco")



##================= B I N A R Y =======================

# ORIGINAL -------------------
# rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
# project = rf.workspace("seamap").project("seamap-binary")
# dataset = project.version(9).download("coco")


# 25 % ------------------
# rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
# project = rf.workspace("seamap").project("seamap-binary-25")
# dataset = project.version(2).download("coco")

# 50 % ------------------
# rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
# project = rf.workspace("seamap").project("seamap-binary-50")
# dataset = project.version(2).download("coco")


# 75% ------------------------
# rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
# project = rf.workspace("seamap").project("seamap-binary-75")
# dataset = project.version(1).download("coco")


# TRAIN SET
TRAIN_DATA_SET_NAME = f"{DATA_SET_NAME}-train"
TRAIN_DATA_SET_IMAGES_DIR_PATH = os.path.join(dataset.location, "train")
TRAIN_DATA_SET_ANN_FILE_PATH = os.path.join(dataset.location, "train", ANNOTATIONS_FILE_NAME)

register_coco_instances(
    name=TRAIN_DATA_SET_NAME,
    metadata={},
    json_file=TRAIN_DATA_SET_ANN_FILE_PATH,
    image_root=TRAIN_DATA_SET_IMAGES_DIR_PATH
)

# TEST SET
TEST_DATA_SET_NAME = f"{DATA_SET_NAME}-test"
TEST_DATA_SET_IMAGES_DIR_PATH = os.path.join(dataset.location, "test")
TEST_DATA_SET_ANN_FILE_PATH = os.path.join(dataset.location, "test", ANNOTATIONS_FILE_NAME)

register_coco_instances(
    name=TEST_DATA_SET_NAME,
    metadata={},
    json_file=TEST_DATA_SET_ANN_FILE_PATH,
    image_root=TEST_DATA_SET_IMAGES_DIR_PATH
)

# VALID SET
VALID_DATA_SET_NAME = f"{DATA_SET_NAME}-valid"
VALID_DATA_SET_IMAGES_DIR_PATH = os.path.join(dataset.location, "valid")
VALID_DATA_SET_ANN_FILE_PATH = os.path.join(dataset.location, "valid", ANNOTATIONS_FILE_NAME)

register_coco_instances(
    name=VALID_DATA_SET_NAME,
    metadata={},
    json_file=VALID_DATA_SET_ANN_FILE_PATH,
    image_root=VALID_DATA_SET_IMAGES_DIR_PATH
)

print("All datasets registered successfully!")
print(f"Train dataset: {TRAIN_DATA_SET_NAME}")
print(f"Test dataset: {TEST_DATA_SET_NAME}")
print(f"Valid dataset: {VALID_DATA_SET_NAME}")

# List all registered datasets for this project
registered_datasets = [
    data_set
    for data_set
    in MetadataCatalog.list()
    if data_set.startswith(DATA_SET_NAME)
]

print("Registered datasets:")
for dataset_name in registered_datasets:
    print(f"  - {dataset_name}")
    

#-------------- VISUALIZE DATASET ----------------------
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