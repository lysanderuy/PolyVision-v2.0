import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

# Import your dataset
import dataset

def cv2_imshow(image, filename="prediction.jpg"):
    """Save image to file (replacement for Colab's cv2_imshow)"""
    cv2.imwrite(filename, image)
    print(f"Image saved as: {filename}")

def setup_predictor(model_path):
    """Setup predictor with trained model"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Model configuration
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Your binary classification
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
    
    return DefaultPredictor(cfg)

def visualize_test_predictions(model_path, max_images=10, save_individual=True):
    """Visualize predictions on test dataset"""
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Setup predictor
    print("Loading trained model...")
    predictor = setup_predictor(model_path)
    print("Model loaded successfully!")
    
    # Get dataset and metadata
    TEST_DATA_SET_NAME = dataset.TEST_DATA_SET_NAME
    metadata = MetadataCatalog.get(dataset.TRAIN_DATA_SET_NAME)  # Use train metadata for classes
    dataset_test = DatasetCatalog.get(TEST_DATA_SET_NAME)
    
    print(f"Processing test dataset: {TEST_DATA_SET_NAME}")
    print(f"Total test images: {len(dataset_test)}")
    print(f"Will process first {min(max_images, len(dataset_test))} images")
    print(f"Classes: {metadata.thing_classes}")
    
    # Create output directory
    output_dir = "prediction_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images
    processed_count = 0
    
    for i, d in enumerate(dataset_test):
        if processed_count >= max_images:
            break
            
        print(f"\nProcessing image {i+1}/{len(dataset_test)}: {os.path.basename(d['file_name'])}")
        
        # Load image
        img = cv2.imread(d["file_name"])
        if img is None:
            print(f"  Warning: Could not load image {d['file_name']}")
            continue
        
        # Make prediction
        try:
            outputs = predictor(img)
            instances = outputs["instances"]
            
            # Print prediction info
            num_detections = len(instances)
            print(f"  Found {num_detections} detections")
            
            if num_detections > 0:
                scores = instances.scores.cpu().numpy()
                classes = instances.pred_classes.cpu().numpy()
                print(f"  Confidence scores: {[f'{s:.3f}' for s in scores]}")
                print(f"  Predicted classes: {classes.tolist()}")
            
            # Visualize
            visualizer = Visualizer(
                img[:, :, ::-1],
                metadata=metadata,
                scale=0.8
            )
            
            # Draw predictions
            out = visualizer.draw_instance_predictions(instances.to("cpu"))
            
            # Save individual image
            if save_individual:
                filename = f"prediction_{i+1:03d}_{os.path.basename(d['file_name'])}"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, out.get_image()[:, :, ::-1])
                print(f"  Saved: {filepath}")
            else:
                # Show image (you might want to comment this out for batch processing)
                cv2_imshow(out.get_image()[:, :, ::-1], f"prediction_{i+1:03d}.jpg")
            
            processed_count += 1
            
        except Exception as e:
            print(f"  Error processing image: {e}")
            continue
    
    print(f"\n=== Visualization Complete ===")
    print(f"Processed {processed_count} images")
    if save_individual:
        print(f"Images saved to: {output_dir}/")
    print("Check the output for model performance on test data!")

def visualize_with_ground_truth(model_path, max_images=5):
    """Visualize predictions alongside ground truth"""
    
    # Setup predictor
    predictor = setup_predictor(model_path)
    metadata = MetadataCatalog.get(dataset.TRAIN_DATA_SET_NAME)
    dataset_test = DatasetCatalog.get(dataset.TEST_DATA_SET_NAME)
    
    output_dir = "ground_truth_vs_predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, d in enumerate(dataset_test[:max_images]):
        print(f"Processing image {i+1}: {os.path.basename(d['file_name'])}")
        
        img = cv2.imread(d["file_name"])
        if img is None:
            continue
        
        # Ground truth visualization
        viz_gt = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
        out_gt = viz_gt.draw_dataset_dict(d)
        
        # Prediction visualization
        outputs = predictor(img)
        viz_pred = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
        out_pred = viz_pred.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Save both
        gt_path = os.path.join(output_dir, f"gt_{i+1:03d}_{os.path.basename(d['file_name'])}")
        pred_path = os.path.join(output_dir, f"pred_{i+1:03d}_{os.path.basename(d['file_name'])}")
        
        cv2.imwrite(gt_path, out_gt.get_image()[:, :, ::-1])
        cv2.imwrite(pred_path, out_pred.get_image()[:, :, ::-1])
        
        print(f"  Saved GT: {gt_path}")
        print(f"  Saved Pred: {pred_path}")

# Main execution
if __name__ == "__main__":
    # Update this path to your actual trained model
    MODEL_PATH = "SEAMaP-Binary-Full/faster_rcnn_R_50_FPN_3x/2025-05-30-13-14-08/model_final.pth"
    
    print("=== Prediction Visualization ===")
    print("Choose an option:")
    print("1. Visualize predictions only (saves individual images)")
    print("2. Visualize predictions vs ground truth")
    print("3. Quick preview (first 3 images)")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        max_imgs = int(input("How many images to process? (default 10): ") or "10")
        visualize_test_predictions(MODEL_PATH, max_images=max_imgs, save_individual=True)
    elif choice == "2":
        max_imgs = int(input("How many comparison images? (default 5): ") or "5")
        visualize_with_ground_truth(MODEL_PATH, max_images=max_imgs)
    elif choice == "3":
        visualize_test_predictions(MODEL_PATH, max_images=3, save_individual=True)
    else:
        print("Invalid choice, running quick preview...")
        visualize_test_predictions(MODEL_PATH, max_images=3, save_individual=True)