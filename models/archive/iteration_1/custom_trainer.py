#======= NOT USED AS OF THE MOMENT. IT IS IN THE custom_classes put into one file ========
import os
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, build_detection_test_loader
from loss_eval_hook import LossEvalHook
# Import the LossEvalHook (assuming it's in the same directory)
# If you created loss_eval_hook.py, uncomment the next line:
# from loss_eval_hook import LossEvalHook

# If LossEvalHook is in the same file, you'll need to define it here or import it

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Build COCO evaluator for the given dataset"""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        
        # CPU-optimized: ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        return COCOEvaluator(
            dataset_name, 
            cfg, 
            True, 
            output_folder
        )

    def build_hooks(self):
        """Build training hooks including custom loss evaluation hook"""
        hooks = super().build_hooks()
        
        # CPU-optimized: Add LossEvalHook with proper configuration
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,  # Use self.cfg instead of cfg
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        
        print(f"CustomTrainer hooks built successfully!")
        print(f"Evaluation period: {self.cfg.TEST.EVAL_PERIOD}")
        print(f"Test dataset: {self.cfg.DATASETS.TEST[0]}")
        
        return hooks
    
    def build_optimizer(self, cfg, model):
        """CPU-optimized optimizer settings"""
        optimizer = super().build_optimizer(cfg, model)
        print(f"Optimizer created: {type(optimizer).__name__}")
        return optimizer

# Function to create and configure the trainer
def create_custom_trainer(cfg):
    """Create a CustomTrainer with the given configuration"""
    # Ensure we're using CPU
    if cfg.MODEL.DEVICE != "cpu":
        print(f"Warning: Model device was {cfg.MODEL.DEVICE}, forcing to CPU")
        cfg.MODEL.DEVICE = "cpu"
    
    trainer = CustomTrainer(cfg)
    print("CustomTrainer created successfully!")
    print(f"Output directory: {cfg.OUTPUT_DIR}")
    print(f"Training device: {cfg.MODEL.DEVICE}")
    
    return trainer

print("CustomTrainer class defined successfully!")
print("Use create_custom_trainer(cfg) to create a trainer instance")