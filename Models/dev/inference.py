import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Import your dataset to get metadata
import dataset

def cv2_imshow(image, filename="prediction.jpg", output_dir="output"):
    """Save image to file"""
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    cv2.imwrite(output_path, image)
    print(f"Image saved as: {output_path}")

# Configuration - must match your training config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

# Set to your trained model path
MODEL_PATH = "../SEAMaP-Binary-Full/faster_rcnn_R_50_FPN_3x/2025-05-30-13-16-04/model_final.pth"
cfg.MODEL.WEIGHTS = MODEL_PATH

# Model configuration - must match training
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Your number of classes
cfg.MODEL.DEVICE = "cpu"  # Use CPU for inference
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # Confidence threshold

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Make sure training is complete and the path is correct")
    exit()

# Create predictor
print("Loading trained model...")
predictor = DefaultPredictor(cfg)
print("Model loaded successfully!")

# Get metadata for visualization
metadata = MetadataCatalog.get(dataset.TRAIN_DATA_SET_NAME)

# Example: Make prediction on a test image
def predict_on_image(image_path, output_dir="output"):
    """Make prediction on a single image"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Load image
    image = cv2.imread(image_path)
    
    # Make prediction
    print(f"Making prediction on: {image_path}")
    outputs = predictor(image)
    
    # Visualize results
    v = Visualizer(
        image[:, :, ::-1],
        metadata=metadata,
        scale=0.8
    )
    
    # Draw predictions
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Save result
    result_filename = f"prediction_{os.path.basename(image_path)}"
    cv2_imshow(out.get_image()[:, :, ::-1], result_filename, output_dir)
    
    # Print detection info
    instances = outputs["instances"]
    print(f"Found {len(instances)} detections")
    print(f"Confidence scores: {instances.scores.tolist()}")
    print(f"Classes: {instances.pred_classes.tolist()}")

# Example usage
if __name__ == "__main__":
    # Replace with path to your test image
    #test_image_path = "../images/img433_film_jpg.rf.87071a3a1820c99cff7e2f063d77cd96.jpg"
    #test_image_path = "images/img027_film_jpg.rf.7870684faa95d719fd1024e869c3dea0.jpg"
    test_image_path = "../images/filaments/img187_filament_jpg.rf.57edd73c4b5e65d23303e7b852361d18.jpg"
    
    # Or use an image from your dataset
    #dataset_train = dataset.DatasetCatalog.get(dataset.TRAIN_DATA_SET_NAME)
    #sample_image = dataset_train[0]["file_name"]
    
    print(f"Testing on sample image: {test_image_path}")
    predict_on_image(test_image_path)