"""
Simple Post-Training Model Evaluation Script
Uses existing registered datasets - no duplication
"""

import os
import torch
import glob
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Import to register datasets
import dataset

def evaluate_model():
    """Evaluate trained model using existing registered datasets"""
    
    print("=== Model Evaluation ===")
    
    # Get available datasets from what's currently registered
    all_datasets = MetadataCatalog.list()
    test_datasets = [d for d in all_datasets if 'test' in d and 'SEAMaP' in d]
    
    print("Available test datasets:")
    for i, dataset_name in enumerate(test_datasets):
        print(f"  {i+1}. {dataset_name}")
    
    if not test_datasets:
        print("❌ No SEAMaP test datasets found!")
        return
    
    # Use the first available test dataset
    test_dataset = test_datasets[0]
    print(f"\n✅ Using dataset: {test_dataset}")
    
    # Determine if binary or multiclass based on dataset name
    if 'Binary' in test_dataset:
        num_classes = 2
        model_type = "binary"
        architecture = "faster_rcnn_R_50_FPN_3x"
        base_path = "SEAMaP-Binary-Full"
    else:
        num_classes = 4  # multiclass
        model_type = "multiclass" 
        architecture = "faster_rcnn_R_50_FPN_3x"
        base_path = "SEAMaP-Multi-class-100"
    
    print(f"Model type: {model_type}")
    print(f"Number of classes: {num_classes}")
    
    # Find trained model
    model_pattern = os.path.join(base_path, architecture, "*", "model_final.pth")
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        print(f"❌ No trained model found in: {model_pattern}")
        return
    
    # OLD METHOD: Using file modification time (commented out for testing)
    # model_path = max(model_files, key=os.path.getmtime)
    
    # NEW METHOD: Use timestamp directory name to find latest model (same as InferenceBinary)
    def get_timestamp_from_path(file_path):
        # Extract the timestamp directory name (e.g., "2025-10-01-03-07-35")
        timestamp_dir = os.path.basename(os.path.dirname(file_path))
        return timestamp_dir
    
    # Sort by timestamp directory name (newest first)
    model_path = max(model_files, key=get_timestamp_from_path)
    print(f"✅ Found model: {model_path}")
    
    # Setup configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{architecture}.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = (test_dataset,)
    
    try:
        print("\n=== Running Evaluation ===")
        
        # Create predictor
        predictor = DefaultPredictor(cfg)
        print("✅ Model loaded successfully")
        
        # Create evaluator (simplified parameters to avoid deprecation warning)
        evaluator = COCOEvaluator(
            dataset_name=test_dataset,
            tasks=("bbox",),
            distributed=False,
            output_dir="./evaluation_output"
        )
        
        # Build test loader
        test_loader = build_detection_test_loader(cfg, test_dataset)
        print(f"✅ Test loader created")
        
        # Run evaluation
        print("Evaluating... (this may take a while)")
        with torch.no_grad():
            results = inference_on_dataset(predictor.model, test_loader, evaluator)
        
        # Display results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        if 'bbox' in results:
            bbox_results = results['bbox']
            
            # Main metrics
            overall_map = bbox_results.get('AP', 0)
            map50 = bbox_results.get('AP50', 0)
            map75 = bbox_results.get('AP75', 0)
            
            print(f"📊 DETECTION METRICS:")
            print(f"   • Overall mAP (IoU=0.50:0.95): {overall_map:.4f} ({overall_map:.1%})")
            print(f"   • mAP@0.50: {map50:.4f} ({map50:.1%})")
            print(f"   • mAP@0.75: {map75:.4f} ({map75:.1%})")
            
            # Performance rating for display
            if overall_map >= 0.9:
                rating_display = "🌟 Excellent"
                rating_text = "Excellent"
            elif overall_map >= 0.8:
                rating_display = "✅ Very Good"
                rating_text = "Very Good"
            elif overall_map >= 0.7:
                rating_display = "👍 Good"
                rating_text = "Good"
            elif overall_map >= 0.6:
                rating_display = "⚠️ Fair"
                rating_text = "Fair"
            else:
                rating_display = "❌ Needs Improvement"
                rating_text = "Needs Improvement"
            
            print(f"   • Performance Rating: {rating_display}")
            
            # Save results (without emojis for Windows compatibility)
            results_file = f"{model_type}_evaluation_results.txt"
            with open(results_file, "w", encoding="utf-8") as f:
                f.write("POST-TRAINING EVALUATION RESULTS\n")
                f.write("="*40 + "\n")
                f.write(f"Model Type: {model_type}\n")
                f.write(f"Dataset: {test_dataset}\n")
                f.write(f"Model Path: {model_path}\n")
                f.write(f"Overall mAP: {overall_map:.4f}\n")
                f.write(f"mAP@0.5: {map50:.4f}\n")
                f.write(f"mAP@0.75: {map75:.4f}\n")
                f.write(f"Performance: {rating_text}\n")
            
            print(f"📄 Results saved to: {results_file}")
            print("="*50)
            
        else:
            print("❌ No bbox results found")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

if __name__ == "__main__":
    evaluate_model()