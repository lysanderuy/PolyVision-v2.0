import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import your dataset to ensure it's registered
import dataset

def setup_config_for_evaluation(model_path):
    """Setup configuration for evaluation"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Model configuration - must match training
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Your binary classification
    cfg.MODEL.DEVICE = "cpu"  # Use CPU
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Lower threshold for evaluation
    
    # Dataset configuration
    cfg.DATASETS.TEST = (dataset.TEST_DATA_SET_NAME,)
    cfg.DATALOADER.NUM_WORKERS = 0  # CPU optimization
    
    return cfg

def evaluate_model(model_path, output_dir="./evaluation_output"):
    """Evaluate trained model and generate confusion matrix"""
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Make sure training is complete and the path is correct")
        return
    
    # Setup configuration
    cfg = setup_config_for_evaluation(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create predictor
    print("Loading trained model for evaluation...")
    predictor = DefaultPredictor(cfg)
    print("Model loaded successfully!")
    
    # Get test dataset info
    test_dataset_name = dataset.TEST_DATA_SET_NAME
    test_metadata = MetadataCatalog.get(test_dataset_name)
    test_dataset = DatasetCatalog.get(test_dataset_name)
    
    print(f"Evaluating on dataset: {test_dataset_name}")
    print(f"Number of test images: {len(test_dataset)}")
    print(f"Classes: {test_metadata.thing_classes}")
    
    # Build test data loader
    val_loader = build_detection_test_loader(cfg, test_dataset_name)
    
    # Create evaluator
    evaluator = COCOEvaluator(test_dataset_name, ("bbox",), False, output_dir=output_dir)
    
    # Run evaluation
    print("Running evaluation...")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    # Print evaluation results
    print("\n=== Evaluation Results ===")
    if results and "bbox" in results:
        for key, value in results["bbox"].items():
            print(f"{key}: {value:.4f}")
    
    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    generate_confusion_matrix(predictor, test_dataset, test_metadata, output_dir)
    
    return results

def generate_confusion_matrix(predictor, test_dataset, metadata, output_dir):
    """Generate confusion matrix for binary classification"""
    
    gt_classes = []
    pred_classes = []
    
    print(f"Processing {len(test_dataset)} test images...")
    
    for i, dataset_dict in enumerate(test_dataset):
        if i % 50 == 0:  # Progress indicator
            print(f"Processed {i}/{len(test_dataset)} images")
        
        # Load image
        image_path = dataset_dict["file_name"]
        import cv2
        image = cv2.imread(image_path)
        
        if image is None:
            continue
            
        # Get ground truth
        annotations = dataset_dict.get("annotations", [])
        for ann in annotations:
            gt_classes.append(ann["category_id"])
        
        # Get predictions
        outputs = predictor(image)
        instances = outputs["instances"]
        
        # Filter predictions by confidence
        if len(instances) > 0:
            pred_classes.extend(instances.pred_classes.cpu().numpy().tolist())
        else:
            # If no predictions, assign background class
            pred_classes.extend([0] * len(annotations))  # Assuming 0 is background
    
    # Convert to binary classification (assuming class 0 and 1)
    # Adjust these mappings based on your actual class IDs
    gt_binary = [1 if cls > 0 else 0 for cls in gt_classes]
    pred_binary = [1 if cls > 0 else 0 for cls in pred_classes[:len(gt_binary)]]
    
    if len(gt_binary) == 0 or len(pred_binary) == 0:
        print("No valid predictions or ground truth found for confusion matrix")
        return
    
    # Ensure same length
    min_len = min(len(gt_binary), len(pred_binary))
    gt_binary = gt_binary[:min_len]
    pred_binary = pred_binary[:min_len]
    
    print(f"Computing confusion matrix with {len(gt_binary)} samples...")
    
    # Compute confusion matrix
    cm = confusion_matrix(gt_binary, pred_binary, labels=[0, 1])
    
    # Create display
    class_names = ['Background', 'Microplastic']  # Adjust based on your classes
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # Plot
    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Binary Classification')
    plt.tight_layout()
    
    # Save plot
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {cm_path}")
    plt.show()
    
    # Print metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n=== Classification Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")

# Main execution
if __name__ == "__main__":
    # Update this path to your actual trained model
    MODEL_PATH = "SEAMaP-Binary-Full/faster_rcnn_R_50_FPN_3x/2025-05-30-13-14-08/model_final.pth"
    
    # Check if model exists
    if os.path.exists(MODEL_PATH):
        print(f"Found model at: {MODEL_PATH}")
        evaluate_model(MODEL_PATH)
    else:
        print(f"Model not found at: {MODEL_PATH}")
        print("Please update MODEL_PATH to your actual trained model location")
        
        # Try to find model automatically
        base_dir = "SEAMaP-Binary-Full/faster_rcnn_R_50_FPN_3x"
        if os.path.exists(base_dir):
            subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if subdirs:
                latest_dir = max(subdirs)  # Get most recent training run
                auto_path = os.path.join(base_dir, latest_dir, "model_final.pth")
                if os.path.exists(auto_path):
                    print(f"Found model automatically: {auto_path}")
                    evaluate_model(auto_path)
                else:
                    print(f"No model_final.pth found in {os.path.join(base_dir, latest_dir)}")