import os
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg

# Import all configuration from binary_config.py
from binary_config import *

# Import all configuration from multiclass_config.py
# from multiclass_config import *

# Import your custom classes from custom_classes.py
from custom_classes import LossEvalHook, CustomEvaluator, CustomTrainer

# Import dataset registration (this will run dataset.py and register datasets)
import dataset

def main():
    print("=== Training Setup ===")
    print(f"Dataset: {DATA_SET_NAME}")
    print(f"Train dataset: {TRAIN_DATA_SET_NAME}")
    print(f"Test dataset: {TEST_DATA_SET_NAME}")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Max iterations: {MAX_ITER}")
    print(f"Output directory: {OUTPUT_DIR_PATH}")

    print("\n=== Starting Training ===")
    print("WARNING: Training on CPU will be VERY slow!")

    # Create trainer and start training
    try:
        print("Creating CustomTrainer...")
        trainer = CustomTrainer(cfg)  # cfg is already configured in model_config.py
        
        print("Loading pretrained weights...")
        trainer.resume_or_load(resume=False)
        
        print(f"Starting training for {MAX_ITER} iterations...")
        trainer.train()
        
        print("\n=== Training Completed Successfully! ===")
        print(f"Model saved in: {cfg.OUTPUT_DIR}")
        
    except KeyboardInterrupt:
        print("\n=== Training Stopped by User ===")
        
    except Exception as e:
        print(f"\n=== Training Failed ===")
        print(f"Error: {e}")
        
        # Save error info
        error_path = os.path.join(OUTPUT_DIR_PATH, "error_log.txt")
        with open(error_path, "w") as f:
            f.write(f"Training error: {str(e)}")
        print(f"Error details saved to: {error_path}")

    print(f"\nTraining session completed. Check {OUTPUT_DIR_PATH} for results.")

if __name__ == '__main__':
    main()