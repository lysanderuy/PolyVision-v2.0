# train.py (Enhanced Version)

# ==============================================================================
# 1. IMPORTS AND GLOBAL DEFINITIONS
# ==============================================================================

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger() # Set up detectron2's default logger

import argparse
import os
import glob
import cv2
from datetime import datetime
import torch
import json
import random

# Detectron2 specific imports
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.solver import build_optimizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper
from detectron2.data import MetadataCatalog
import sys
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from gpu_diagnostics import diagnose_gpu_support, format_diagnostic_lines, select_training_device

# Global seed configuration (edit this constant to reproduce a specific run)
DEFAULT_TRAINING_SEED = 1337

# ==============================================================================
# 2. CUSTOM TRAINER DEFINITION (For Similarity and Future Expansion)
#    We will use a CustomTrainer class similar to your example. In this setup,
#    it will behave identically to DefaultTrainer, but allows for easy
#    addition of custom hooks in the future if needed.
# ==============================================================================

from detectron2.engine import HookBase

class CancellationHook(HookBase):
    """Checks a threading.Event after every iteration and raises InterruptedError
    to break the training loop cleanly when the user clicks Cancel."""
    def __init__(self, cancel_event):
        self._cancel_event = cancel_event

    def after_step(self):
        if self._cancel_event.is_set():
            raise InterruptedError("Training cancelled by user.")


class CustomTrainer(DefaultTrainer):
    """
    Custom trainer that implements a sophisticated fine-tuning strategy:
    1.  PROGRESSIVE UNFREEZING: The earliest, most generic layers of the
        backbone ('stem' and 'res2') are completely frozen and do not train.

    2.  DIFFERENTIAL LEARNING RATES: The later, more specialized layers of
        the backbone ('res3', 'res4', 'res5') are trainable but at a
        learning rate 10x smaller than the rest of the model.

    3.  FULL LEARNING RATE: The model's "head" (FPN, RPN, ROI Heads), which
        is most specific to the detection task, trains at the full base LR.
    """

    def __init__(self, cfg, cancel_event=None):
        self._cancel_event = cancel_event
        super().__init__(cfg)

    def build_hooks(self):
        hooks = super().build_hooks()
        if self._cancel_event is not None:
            # Insert before the last hook (PeriodicWriter) so cancellation fires first
            hooks.insert(-1, CancellationHook(self._cancel_event))
        return hooks

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Overrides the default optimizer to set up freezing and differential LRs.
        """
        print("\n" + "="*50)
        print("--- Using CustomTrainer with Progressive Unfreezing ---")
        print("="*50)

        # --- Step 1: Freeze the early layers ---
        frozen_prefixes = ["backbone.bottom_up.stem", "backbone.bottom_up.res2"]
        for name, param in model.named_parameters():
            for prefix in frozen_prefixes:
                if name.startswith(prefix):
                    param.requires_grad = False
                    break # Move to the next parameter once a match is found
        
        print(f"Froze all layers with prefixes: {frozen_prefixes}")
        
        # --- Step 2: Set up differential learning rates for the remaining layers ---
        params = []
        print("\n--- Applying Differential Learning Rates for trainable layers ---")
        for key, value in model.named_parameters():
            if not value.requires_grad:
                # Skip parameters that were frozen in the step above
                continue
            
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY

            # Apply a smaller LR to the unfrozen, later backbone layers
            if "backbone" in key:
                lr = lr * 0.1
                print(f" -> [CHILLED LR] for layer: {key}")
            else:
                print(f" -> [FULL LR] for layer: {key}")
                
            params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})

        # AdamW is often a better choice for fine-tuning
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR)
        print("="*50 + "\n")
        return optimizer

    @classmethod
    def build_train_loader(cls, cfg):
        """
        This is the data loader logic. We will use a simple version with no
        aggressive augmentations, as they have proven to be harmful.
        """
        augs = [
            T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING),
            T.RandomFlip(prob=0.5, horizontal=True)
        ]
        print("\n--- Using simple data loader (Resize & Flip only) ---")
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
        return build_detection_train_loader(cfg, mapper=mapper)

# ==============================================================================
# 3. MAIN EXECUTION BLOCK
# ==============================================================================

def apply_global_seed(seed):
    """
    Apply deterministic seeds across Python, NumPy (if available), and Torch.
    """
    if seed is None:
        print("No seed override specified; using default randomness.")
        return

    print(f"Applying global seed: {seed}")
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # Determine seed priority: CLI overrides constant.
    seed = args.seed if args.seed is not None else DEFAULT_TRAINING_SEED
    apply_global_seed(seed)

    # --- 1. Register the Dataset ---
    dataset_name = "microplastic_retrain_run"
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
    register_coco_instances(dataset_name, {}, args.annotations_path, args.image_root)

    with open(args.annotations_path, "r") as f:
        cats = sorted(json.load(f)["categories"], key=lambda c: c["id"])
    id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    classes   = [c["name"] for c in cats]

    meta = MetadataCatalog.get(dataset_name)
    meta.thing_dataset_id_to_contiguous_id = id_to_idx
    meta.thing_classes = classes
    
    # --- 2. Build the Configuration ---
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(args.config_file)
    
    # CRITICAL: Apply the per-run overrides passed from the orchestrator.
    # This sets cfg.OUTPUT_DIR, cfg.MODEL.WEIGHTS, etc.
    cfg.merge_from_list(args.opts)
    print("USING MODEL.WEIGHTS =", cfg.MODEL.WEIGHTS)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.SEED = seed
    
    # --- 3. Set Final Config Values ---
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.AUGMENTATIONS_ENABLED = args.use_augmentation
    gpu_report = diagnose_gpu_support()
    for line in format_diagnostic_lines(gpu_report):
        print(line)
    cfg.MODEL.DEVICE = select_training_device(gpu_report)
    print(f"--- Training device selected: {cfg.MODEL.DEVICE} ---")
    
    # CRITICAL: We DO NOT modify cfg.OUTPUT_DIR here. It is now fully controlled
    # by the orchestrator script (Retrain.py). We just ensure it exists.
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(f"--- Training output will be saved to: {cfg.OUTPUT_DIR} ---")

    # --- 4. Train the Model ---
    cancel_event = getattr(args, 'cancel_event', None)
    trainer = CustomTrainer(cfg, cancel_event=cancel_event)
    trainer.resume_or_load(resume=False)
    print("--- Starting Training ---")
    trainer.train()
    print("--- Training Complete ---")

    # --- 5. Visualize Predictions ---
    # print("\n--- Visualizing predictions with the new model... ---")
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    # predictor = DefaultPredictor(cfg)

    # vis_output_dir = os.path.join(cfg.OUTPUT_DIR, "visualizations")
    # os.makedirs(vis_output_dir, exist_ok=True)
    
    # dataset_dicts = DatasetCatalog.get(dataset_name)
    # retrain_metadata = MetadataCatalog.get(dataset_name) # Get metadata for visualizer
    # for i, d in enumerate(dataset_dicts[:5]): # Visualize first 5 images
    #     img = cv2.imread(d["file_name"])
    #     outputs = predictor(img)

    #     visualizer = Visualizer(
    #         img[:, :, ::-1],
    #         metadata=retrain_metadata,
    #         scale=0.8
    #     )
    #     out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        
    #     file_basename = os.path.basename(d["file_name"])
    #     output_path = os.path.join(vis_output_dir, f"prediction_{file_basename}")
    #     cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
    #     print(f"Saved visualization to {output_path}")

    print("Script finished successfully.")


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    parser = argparse.ArgumentParser(description="Enhanced Retraining Script for PolyVision")
    parser.add_argument("--config-file", required=True, help="Path to the model's .yaml config file.")
    parser.add_argument("--image-root", required=True, help="Path to the directory containing retraining images.")
    parser.add_argument("--annotations-path", required=True, help="Path to the COCO-format .json annotation file.")
    
    parser.add_argument(
        "--freeze-backbone",
        action="store_true", # Makes it a flag, like --freeze-backbone
        help="If set, freezes the backbone layers of the model and only trains the head."
    )
    parser.add_argument(
        "--use-augmentation",
        action="store_true",
        help="If set, applies data augmentation during training."
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override training seed. If omitted, DEFAULT_TRAINING_SEED is used.",
    )
    args = parser.parse_args()
    main(args)
