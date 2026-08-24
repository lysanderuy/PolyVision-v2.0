#======= NOT USED AS OF THE MOMENT. IT IS IN THE custom_classes put into one file ========
import torch
from detectron2.evaluation import inference_context, COCOEvaluator
from detectron2.data import build_detection_test_loader

# Create your custom evaluator class - CPU optimized
class CustomEvaluator:
    def __init__(self, iou_thresholds=(0.3, 0.5, 0.75, 0.95)):
        self._iou_thresholds = iou_thresholds

    def __call__(self, trainer):
        # CPU-optimized: torch.no_grad() works on both GPU and CPU
        with inference_context(trainer.model), torch.no_grad():
            evaluator = COCOEvaluator(
                trainer.cfg.DATASETS.TEST[0], 
                trainer.cfg, 
                True,
                output_dir=trainer.cfg.OUTPUT_DIR  # CPU-friendly: specify output dir
            )
            val_loader = build_detection_test_loader(
                trainer.cfg, 
                trainer.cfg.DATASETS.TEST[0]
            )
            
            # Run evaluation
            print("Running custom evaluation...")
            results = evaluator.evaluate(val_loader)

        # Extract custom metrics for different IoU thresholds
        custom_metrics = {}
        
        # Handle different result formats that COCO evaluator might return
        if results is not None and "bbox" in results:
            for iou in self._iou_thresholds:
                # Try different possible metric names
                metric_key = f"bbox/AP{int(iou * 100)}"
                alt_metric_key = f"AP{int(iou * 100)}"
                
                if metric_key in results:
                    custom_metrics[f"AP{int(iou * 100)}"] = results[metric_key]
                elif alt_metric_key in results["bbox"]:
                    custom_metrics[f"AP{int(iou * 100)}"] = results["bbox"][alt_metric_key]
                else:
                    # Fallback: use overall AP if specific IoU not available
                    custom_metrics[f"AP{int(iou * 100)}"] = results.get("bbox/AP", 0.0)
                    
        else:
            # Fallback if evaluation fails
            print("Warning: Evaluation results not available, setting metrics to 0")
            for iou in self._iou_thresholds:
                custom_metrics[f"AP{int(iou * 100)}"] = 0.0

        print("Custom evaluation metrics:")
        for key, value in custom_metrics.items():
            print(f"  {key}: {value:.4f}")
            
        return custom_metrics

# Function to create and test the evaluator
def create_custom_evaluator(iou_thresholds=(0.3, 0.5, 0.75, 0.95)):
    """Create a CustomEvaluator with specified IoU thresholds"""
    evaluator = CustomEvaluator(iou_thresholds)
    print(f"CustomEvaluator created with IoU thresholds: {iou_thresholds}")
    return evaluator

print("CustomEvaluator class defined successfully!")
print("Use create_custom_evaluator() to create an instance")