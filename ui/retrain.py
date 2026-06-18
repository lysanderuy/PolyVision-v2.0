# Retrain.py

import os
import sys
import json
import sqlite3
import argparse
import logging
import threading
import shutil
import random
import yaml
import glob
import subprocess
from benchmark import main as benchmark_main
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from import_dialog import ImportDialog
from database import create_retraining_database
from datetime import datetime
from comparison_dialog import ComparisonDialog
from pathlib import Path

from retraining_runtime import train as retraining_train
from retraining_runtime.gpu_diagnostics import (
    CUDA_BROKEN,
    GPU_READY,
    HARDWARE_MISSING,
    HARDWARE_PRESENT_SOFTWARE_MISSING,
    RETRAINING_RUNTIME_BROKEN,
    diagnose_gpu_support,
    format_diagnostic_lines,
)
from retraining_runtime.ui_policy import BLOCKED, CPU_FALLBACK, READY, retraining_start_policy

import cv2
import time
from app_paths import models_path, storage_path, user_settings_path, resource_path, app_storage_dir
from base_marker import is_base_model_dir

BASE_OUTPUT_DIRECTORY = models_path()

RETRAIN_UI_STYLESHEET = """
QDialog {
    background: #eef1f4;
    font-family: Segoe UI, Arial, sans-serif;
}
QGroupBox {
    background: transparent;
    border: 1px solid #c0c0c0;
    border-radius: 5px;
    margin-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
}
QPushButton {
    color: #1f2933;
    background: #ffffff;
    border: 1px solid #b8c2cc;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background: #f1f7fb;
    border-color: #0f5f9e;
}
QPushButton:pressed {
    background: #e3edf4;
}
QPushButton:disabled {
    color: #9aa5b1;
    background: #eef1f4;
    border-color: #d5dbe1;
}
QPushButton#footerButton {
    padding: 6px 14px;
}
QPushButton#footerButton[buttonRole="danger"] {
    color: #b42318;
    background: #fff8f7;
    border-color: #efb0aa;
}
QPushButton#footerButton[buttonRole="danger"]:hover {
    background: #fff0ee;
    border-color: #d92d20;
}
QPushButton#footerButton[buttonRole="warning"] {
    color: #8a4b00;
    background: #fff8e8;
    border-color: #dfb76b;
}
QPushButton#footerButton[buttonRole="warning"]:hover {
    background: #fff0c2;
    border-color: #b7791f;
}
QPushButton#footerButton:disabled {
    color: #9aa5b1;
    background: #eef1f4;
    border-color: #d5dbe1;
}
QProgressBar {
    border: 1px solid #bfd0dc;
    border-radius: 9px;
    height: 18px;
    background: #edf3f7;
}
QProgressBar::chunk {
    border-radius: 8px;
    background: #0f5f9e;
}
QTextEdit {
    border: 1px solid #c0c0c0;
    border-radius: 0px;
    background: #ffffff;
}
#retrainSurface {
    background: #f0f0f0;
    border: 1px solid #c0c0c0;
}
#modelChoiceButton {
    min-width: 120px;
}
#dashboardCard {
    background: #ffffff;
    border: 1px solid #d2d8de;
    border-radius: 4px;
}
#dashboardTitle {
    color: #2b3640;
    font-weight: 700;
}
#dashboardLabel {
    color: #52616f;
}
#dashboardValue {
    color: #1f2933;
    font-weight: 700;
}
#mainStatusPanel {
    background: #f8fbfd;
    border: 1px solid #bfd0dc;
    border-left: 5px solid #0f5f9e;
    border-radius: 7px;
}
#mainStatusPercent {
    color: #0f5f9e;
    font-size: 17pt;
    font-weight: 800;
}
#mainStatusEta {
    color: #405160;
    font-weight: 700;
}
#mainStatusHealth {
    color: #107c41;
    background: #e1f4e9;
    border: 1px solid #a8d8b8;
    border-radius: 10px;
    padding: 2px 9px;
    font-weight: 700;
}
#mainStatusBottom {
    color: #52616f;
    font-weight: 600;
}
#highlightsPanel {
    background: #ffffff;
    border: 1px solid #c0c0c0;
    border-radius: 4px;
}
#currentStep {
    color: #17212b;
    font-size: 12pt;
    font-weight: 700;
}
#stepSummary {
    color: #52616f;
}
#highlightCard {
    background: #f8fafb;
    border: 1px solid #d2d8de;
    border-radius: 4px;
}
#highlightTitle {
    color: #2b3640;
    font-weight: 700;
}
#highlightItem_done {
    color: #107c41;
}
#highlightItem_active {
    color: #0f5f9e;
    font-weight: 600;
}
#highlightItem_next {
    color: #52616f;
}
#technicalLogToggle {
    color: #405160;
}
"""

def get_retraining_data(model_type_filter=None):
    db_path = storage_path('retrain_images.db')
    if not os.path.exists(db_path): return []
    try:
        connection = sqlite3.connect(db_path)
        c = connection.cursor()
        if model_type_filter == 'Multiclass':
            # Multiclass retraining only uses Multiclass-tagged images
            c.execute("SELECT image_name, is_microplastic, bounding_box, model_type FROM database_list WHERE model_type = 'Multiclass' ORDER BY rowid DESC")
        else:
            # Binary retraining uses ALL images (Binary + Multiclass)
            c.execute("SELECT image_name, is_microplastic, bounding_box, model_type FROM database_list ORDER BY rowid DESC")
        data = c.fetchall()
        connection.close()
        return data
    except Exception as e:
        print(f"Error reading retraining database: {e}")
        return []

def import_coco_data(image_dir, json_path, progress_callback):
    """
    Processes a COCO dataset and imports it into the project's retraining database.
    Returns the number of images successfully imported.
    """
    target_img_dir = models_path("retraining_images")
    db_path = storage_path("retrain_images.db")
    os.makedirs(target_img_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Determine model type from the categories array — more than 2 categories means Multiclass
    # This avoids the class_id=1 ambiguity (Filament in Multiclass vs Microplastic in Binary)
    file_model_type = 'Multiclass' if len(coco_data.get('categories', [])) > 2 else 'Binary'

    images_map = {img['id']: img for img in coco_data['images']}
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    
    images_to_process = list(images_map.values())
    total_images = len(images_to_process)
    imported_count = 0

    for i, image_info in enumerate(images_to_process):
        progress_callback.emit(f"Processing image {i+1}/{total_images}: {image_info['file_name']}")
        
        original_image_path = os.path.join(image_dir, image_info['file_name'])
        if not os.path.exists(original_image_path):
            progress_callback.emit(f"  -> Warning: Image not found. Skipping.")
            continue

        # Get current row count to create a unique name
        cursor.execute("SELECT COUNT(*) FROM database_list")
        next_idx = cursor.fetchone()[0]
        
        new_image_name = f"image_{next_idx}.png"
        target_image_path = os.path.join(target_img_dir, new_image_name)
        
        shutil.copy(original_image_path, target_image_path)
        
        is_mp = False
        bounding_boxes_for_db = []

        img_id = image_info['id']
        if img_id in annotations_by_image:
            is_mp = True # If there are annotations, it's a positive sample
            for ann in annotations_by_image[img_id]:
                # Convert COCO [x, y, w, h] to Detectron2 [xmin, ymin, xmax, ymax]
                x, y, w, h = ann['bbox']
                db_bbox = {
                    "bbox": [x, y, x + w, y + h],
                    "class_id": ann['category_id'],
                    "score": 1.0 # Ground truth
                }
                bounding_boxes_for_db.append(db_bbox)

        bounding_box_str = json.dumps(bounding_boxes_for_db)
        cursor.execute("INSERT INTO database_list (image_name, is_microplastic, bounding_box, model_type) VALUES (?, ?, ?, ?)",
                      (new_image_name, int(is_mp), bounding_box_str, file_model_type))
        
        imported_count += 1

    connection.commit()
    connection.close()
    return imported_count

class ImportThread(QThread):
    progress_update = pyqtSignal(str)
    import_finished = pyqtSignal(int) # Sends number of images imported

    def __init__(self, image_dir, json_path):
        super().__init__()
        self.image_dir = image_dir
        self.json_path = json_path

    def run(self):
        imported_count = import_coco_data(self.image_dir, self.json_path, self.progress_update)
        self.import_finished.emit(imported_count)

class _LineEmitter:
    """
    Captures train.py output and forwards each line as Qt log/progress signals
    so the UI stays updated without a subprocess.
    """
    def __init__(self, log_fn, progress_fn=None, total_iters=0):
        self._log = log_fn
        self._progress = progress_fn
        self._total = total_iters
        self._buf = ""

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.strip()
            if not stripped:
                continue
            self._log(stripped)
            if self._progress and "iter:" in stripped:
                try:
                    current = int(stripped.split("iter:")[1].strip().split()[0])
                    self._progress(current, self._total)
                except Exception:
                    pass

    def flush(self):
        if self._buf.strip():
            self._log(self._buf.strip())
            self._buf = ""

    def fileno(self):
        return sys.__stdout__.fileno()

    def isatty(self):
        return False


# NEW: The QThread class for handling the background process
class RetrainingThread(QThread):
    """
    Handles the full Train-Evaluate-Compare pipeline in the background.
    """
    # Signals for UI communication
    log_update = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)
    # NEW SIGNAL: Passes a dictionary with all results back to the UI.
    retraining_finished = pyqtSignal(dict)

    def __init__(self, model_type, config_overrides={}):
        super().__init__()
        self.model_type = model_type
        self.config_overrides = config_overrides # For HPO-optimized params
        self.process = None
        self._is_running = True
        self._cancel_event = threading.Event()
    
    #not used as of the moment. safely delete it to have clean code
    def _find_project_root_containing(self, marker_dir: str = "production_models") -> str:
        """Walk up from this file until we find a folder that contains `marker_dir`."""
        here = Path(__file__).resolve()
        for parent in [here.parent] + list(here.parents):
            if (parent / marker_dir).exists():
                return str(parent)
        # Fallback: current working directory
        return str(Path.cwd())

    def stop(self):
        """Signals the thread to stop. Sets the cancel event so the training
        hook fires at the end of the current iteration."""
        self.log_update.emit("Retraining cancellation requested...")
        self._is_running = False
        self._cancel_event.set()

    def cancelTraining(self, challenger_output_dir):
        """
        Cleans up any files and directories created during cancelled training.
        """
        try:
            if challenger_output_dir and os.path.exists(challenger_output_dir):
                self.log_update.emit(f"Cleaning up cancelled training files from: {challenger_output_dir}")
                shutil.rmtree(challenger_output_dir)
                self.log_update.emit("Cleanup completed successfully.")
            
            # Also clean up any temporary annotation files
            temp_annotation_file = models_path("retraining_data", "annotations_merged.json")
            if os.path.exists(temp_annotation_file):
                os.remove(temp_annotation_file)
                self.log_update.emit("Cleaned up temporary annotation file.")
            
            # Reset progress bar to 0
            self.progress_update.emit(0, 100)
                
        except Exception as e:
            self.log_update.emit(f"Warning: Could not complete cleanup. Error: {e}")

    # --- NEW HELPER METHODS ---

    def _redirect_logging_streams(self, stream):
        """
        Temporarily points existing Python logging StreamHandlers at the Qt
        emitter. Detectron2 can keep handlers from earlier imports, so stdout
        redirection alone is not enough for repeated retraining runs.
        """
        redirected = []
        loggers = [logging.getLogger()]
        loggers.extend(
            logger
            for logger in logging.Logger.manager.loggerDict.values()
            if isinstance(logger, logging.Logger)
        )
        for logger in loggers:
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is not stream:
                    redirected.append((handler, handler.stream))
                    handler.stream = stream
        return redirected

    def _restore_logging_streams(self, redirected_handlers):
        for handler, stream in reversed(redirected_handlers):
            handler.stream = stream

    def _find_latest_model_in_paths(self, path_list):
        if isinstance(path_list, str):
            path_list = [path_list]

        all_models = []
        for base_path in path_list:
            if not os.path.exists(base_path):
                continue
            # Accept timestamped subfolders...
            all_models.extend(glob.glob(os.path.join(base_path, "*", "model_final.pth")))
            # ...and a flat file directly inside the folder
            flat_path = os.path.join(base_path, "model_final.pth")
            if os.path.exists(flat_path):
                all_models.append(flat_path)

        if not all_models:
            return None
        return max(all_models, key=os.path.getmtime)

    def _find_champion_model(self):
        """
        Finds the current champion model in the base model directories.
        Returns (model_path, is_base_model) tuple.
        """
        # Define base model directory. Base identity comes from a marker file
        # (see base_marker.is_base_model_dir), NOT from a hardcoded timestamp.
        if self.model_type == 'Binary':
            base_model_dir = models_path("SEAMaP-Binary-Full", "faster_rcnn_R_50_FPN_3x")
        else:
            base_model_dir = models_path("SEAMaP-Multi-class-100", "faster_rcnn_R_50_FPN_3x")

        # Look for any model_final.pth in the base directory (including timestamp subdirectories)
        champion_model = self._find_latest_model_in_paths([base_model_dir])

        if champion_model:
            model_dir = os.path.dirname(champion_model)
            is_base = is_base_model_dir(model_dir, self.model_type)
            return champion_model, is_base

        return None, False

    def _run_benchmark(self, model_path, run_name):
        
        self.log_update.emit(f"\n--- Benchmarking '{run_name}' Model ---")
        
        if not model_path or not os.path.exists(model_path):
            self.log_update.emit(f"Warning: Model file not found. Cannot benchmark. Path provided: {model_path}")
            return None

        num_classes = 4 if self.model_type == "Multiclass" else 2
        
        # Convert to absolute paths to avoid working directory issues
        abs_model_path = os.path.abspath(model_path)
        
        #Final test set path
        if self.model_type == "Binary":
            test_set_json = models_path("SEAMaP-Binary-Full-6", "test", "_annotations.coco.json")
        else: #Multiclass
            test_set_json = models_path("SEAMaP-Multi-class-100-1", "test", "_annotations.coco.json")
        
        if not os.path.exists(test_set_json):
            self.log_update.emit(f"FATAL: Test set not found at '{test_set_json}'. Cannot benchmark.")
            return None

        try:
            # Create a mock args object
            class MockArgs:
                def __init__(self, model_path, annotations_path, num_classes, run_name):
                    self.model_path = model_path
                    self.annotations_path = annotations_path
                    self.num_classes = num_classes
                    self.run_name = run_name
            
            args = MockArgs(abs_model_path, test_set_json, num_classes, run_name)
            
            # Call benchmark function directly and get results in memory
            self.log_update.emit("Running benchmark evaluation...")
            results = benchmark_main(args)
            
            if results:
                ap_score = results.get('bbox', {}).get('AP', 'N/A')
                if isinstance(ap_score, (int, float)):
                    self.log_update.emit(f"--- Benchmark PASSED for {run_name}. AP: {ap_score:.2f} ---")
                else:
                    self.log_update.emit(f"--- Benchmark PASSED for {run_name}. AP: {ap_score} ---")
                return results
            else:
                self.log_update.emit(f"--- Benchmark returned no results for {run_name} ---")
                return None
                
        except Exception as e:
            self.log_update.emit(f"--- FAILED to run benchmark for {run_name}. Error: {e} ---")
            return None

    def prepare_coco_annotations(self):
        """
        Merges new data with a "bridge" dataset composed of both RANDOM and HARD
        examples from the original training set to improve fine-tuning robustness.
        """
        self.log_update.emit("--- Starting Data Preparation (Bridge Strategy) ---")
        
        # --- 1. Load New Data from DB ---
        self.log_update.emit("Loading new data from retrain_images.db...")
        new_data_from_db = get_retraining_data(self.model_type)
        num_new_samples = len(new_data_from_db)
        self.log_update.emit(f"Found {num_new_samples} new data points.")

        # --- 2. Load the Original Base Training Dataset ---
        if self.model_type == 'Binary':
            # Final Binary Model Paths
            base_annotation_path = models_path("SEAMaP-Binary-Full-6", "train", "_annotations.coco.json")
            base_image_root = models_path("SEAMaP-Binary-Full-6", "train") + "/"
            ranked_list_path = models_path("SEAMaP-Binary-Full-6", "hard_examples_ranked.json")
            
            # For testing onle
            # base_annotation_path = "../Models/retraining/original_datasets/binary_90_percent/train/_annotations.coco.json"
            # base_image_root = "../Models/retraining/original_datasets/binary_90_percent/train/"
            # ranked_list_path = "../Models/retraining/original_datasets/binary_90_percent/hard_examples_ranked.json"
        else: # Multiclass
            # Final Multiclass Model Paths
            base_annotation_path = models_path("SEAMaP-Multi-class-100-1", "train", "_annotations.coco.json")
            base_image_root = models_path("SEAMaP-Multi-class-100-1", "train") + "/"
            ranked_list_path = models_path("SEAMaP-Multi-class-100-1", "hard_examples_ranked.json")
            
            # For testing onle
            # base_annotation_path = "../Models/retraining/original_datasets/multiclass_90_percent/train/_annotations.coco.json"
            # base_image_root = "../Models/retraining/original_datasets/multiclass_90_percent/train/"
            # ranked_list_path = "../Models/retraining/original_datasets/multiclass_90_percent/hard_examples_ranked.json"

        if not os.path.exists(base_annotation_path):
            self.log_update.emit(f"FATAL ERROR: Base annotation file not found at {base_annotation_path}")
            raise FileNotFoundError(f"Base annotation file not found: {base_annotation_path}")

        self.log_update.emit(f"Loading base dataset from: {base_annotation_path}")
        with open(base_annotation_path, 'r') as f:
            base_coco = json.load(f)
        
        # --- 3. Build the "Bridge" Anchor Dataset ---
        # Use fixed 100% ratio for anchor data
        
        # --- 3a. Calculate Quotas for Random and Hard Anchors ---
        # Use all new samples as basis for anchor calculation
        num_random_to_keep = num_new_samples
        num_random_to_keep = min(num_random_to_keep, len(base_coco['images']))
        
        # We add a fixed ratio of HARD anchors (e.g., 50% of the random anchor count)
        HARD_ANCHOR_RATIO = 0.5 
        num_hard_to_keep = int(num_random_to_keep * HARD_ANCHOR_RATIO)
        
        # --- 3b. Select Random Anchor Images ---
        self.log_update.emit(f"Selecting {num_random_to_keep} RANDOM old images...")
        all_base_image_ids = [img['id'] for img in base_coco['images']]
        if num_random_to_keep > 0:
            random_ids = set(random.sample(all_base_image_ids, num_random_to_keep))
        else:
            random_ids = set()

        # --- 3c. Select Hard Anchor Images ---
        ranked_list_path = os.path.join(os.path.dirname(base_image_root), "../hard_examples_ranked.json")
        hard_ids = set()
        if num_hard_to_keep > 0 and os.path.exists(ranked_list_path):
            self.log_update.emit(f"Selecting top {num_hard_to_keep} HARDEST old images...")
            with open(ranked_list_path, 'r') as f:
                ranked_data = json.load(f)
            # Take the top N IDs from the ranked list
            hard_ids = set(ranked_data['ranked_image_ids'][:num_hard_to_keep])
        elif num_hard_to_keep > 0:
            self.log_update.emit(f"Warning: hard_examples_ranked.json not found. Skipping hard anchor selection.")

        # --- 3d. Combine and De-duplicate ---
        sampled_image_ids = random_ids | hard_ids # Set union automatically handles duplicates
        self.log_update.emit(f"Total unique anchor images selected: {len(sampled_image_ids)}")

        # Create the starting point for our final dataset
        final_coco = {
            "images": [img for img in base_coco['images'] if img['id'] in sampled_image_ids],
            "annotations": [ann for ann in base_coco['annotations'] if ann['image_id'] in sampled_image_ids],
            "categories": base_coco['categories']
        }
        
        # Correct image paths for the anchor images
        for img in final_coco['images']:
            img['file_name'] = os.path.abspath(os.path.join(base_image_root, img['file_name']))
        
        # --- 4. Merge the New Data (This part is unchanged) ---
        if not new_data_from_db:
            self.log_update.emit("No new data to merge.")
        
        max_image_id = max([img['id'] for img in final_coco['images']]) if final_coco['images'] else -1
        max_ann_id = max([ann['id'] for ann in final_coco['annotations']]) if final_coco['annotations'] else -1
        image_id_offset = max_image_id + 1
        annotation_id_offset = max_ann_id + 1
        new_annotations_count = 0
        for i, (image_name, is_mp, bbox_str, _) in enumerate(new_data_from_db):
            image_path = models_path("retraining_images", image_name)
            if not os.path.exists(image_path):
                self.log_update.emit(f"Warning: Image file not found, skipping: {image_path}")
                continue
            try:
                img_cv = cv2.imread(image_path)
                height, width, _ = img_cv.shape
            except Exception:
                self.log_update.emit(f"Warning: Could not read image {image_path}. Skipping.")
                continue
            new_image_id = i + image_id_offset
            final_coco["images"].append({"id": new_image_id, "file_name": image_path, "height": height, "width": width})
            if is_mp == 1 and bbox_str:
                try:
                    bboxes_list = json.loads(bbox_str)
                    for ann_idx, bbox_info in enumerate(bboxes_list):
                        x1, y1, x2, y2 = bbox_info['bbox']
                        w, h = x2 - x1, y2 - y1
                        category_id = bbox_info['class_id']
                        final_coco["annotations"].append({
                            "id": new_annotations_count + annotation_id_offset, "image_id": new_image_id,
                            "category_id": category_id, "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0
                        })
                        new_annotations_count += 1
                except (json.JSONDecodeError, TypeError):
                    self.log_update.emit(f"Warning: Could not decode bbox for {image_name}.")
        if self.model_type == 'Binary':
            self.log_update.emit("Binary model selected. Downcasting all categories to a single 'microplastic' class.")
            
            # Remap all annotation category_ids to 1
            final_coco['categories'] = [
                {"id": 0, "name": "other", "supercategory": "root"},
                {"id": 1, "name": "microplastic", "supercategory": "root"},
            ]

            # (B) annotations: force ALL binary GT boxes to id = 1 (the foreground class your champion uses)
            for ann in final_coco["annotations"]:
                ann["category_id"] = 1

        # --- 5. Save the Final Merged Annotation File ---
        self.log_update.emit("Merge complete. Saving final annotation file.")
        output_dir = models_path("retraining_data")
        os.makedirs(output_dir, exist_ok=True)
        final_annotations_path = models_path("retraining_data", "annotations_merged.json")
        with open(final_annotations_path, 'w') as f:
            json.dump(final_coco, f, indent=4)
            
        return final_annotations_path

    def prepare_training_command(self, annotations_path, output_dir):
        """
        Prepares arguments for the bundled retraining runtime, including HPO overrides.
        """
        config_file = models_path("Retraining", f"{self.model_type.lower()}_config.yaml")

        if not os.path.exists(config_file):
            return None, None

        # --- Build opts list ---
        opts = []
        for key, value in self.config_overrides.items():
            opts.extend([key, str(value)])
        opts.extend(["OUTPUT_DIR", output_dir])

        search_paths = []
        if self.model_type == 'Binary':
            search_paths.append(models_path("SEAMaP-Binary-Full", "faster_rcnn_R_50_FPN_3x"))
        else:
            search_paths.append(models_path("SEAMaP-Multi-class-100", "faster_rcnn_R_50_FPN_3x"))

        base_model_path = self._find_latest_model_in_paths(search_paths)
        if base_model_path:
            opts.extend(["MODEL.WEIGHTS", base_model_path])

        train_args = argparse.Namespace(
            config_file=config_file,
            annotations_path=annotations_path,
            image_root=".",
            use_augmentation=False,
            freeze_backbone=False,
            seed=None,
            opts=opts,
            cancel_event=self._cancel_event,
        )

        return retraining_train, train_args

    def run(self):
        """
        The main logic for the full Train, Evaluate, and Compare pipeline.
        """
        # This dictionary will hold all our results to pass back to the UI.
        final_result = {'success': False, 'message': 'Pipeline started...'}
        challenger_output_dir = None

        try:
            gpu_report = diagnose_gpu_support()
            for line in format_diagnostic_lines(gpu_report):
                self.log_update.emit(line)
            self.log_update.emit("")
            if not gpu_report.retraining_available:
                raise RuntimeError(f"Retraining runtime is unavailable: {gpu_report.reason}")

            # --- STAGE 1: DATA PREPARATION ---
            self.log_update.emit("Stage 1/4: Preparing training data...")
            annotations_path = self.prepare_coco_annotations()
            if not self._is_running:
                raise InterruptedError("Process cancelled during data preparation.")

            # --- STAGE 2: IDENTIFY CHAMPION MODEL ---
            self.log_update.emit("\nStage 2/4: Identifying current Champion model...")

            # Use the existing _find_champion_model method which searches in the correct base model directories
            champion_model_path, is_base_model = self._find_champion_model()
            if champion_model_path:
                champion_type = "Base Model" if is_base_model else "Retrained Model"
                self.log_update.emit(f"Champion model found: {champion_type} at {champion_model_path}")
            else:
                self.log_update.emit("No champion model found in base model directories.")


            # --- STAGE 3: TRAIN CHALLENGER MODEL ---
            self.log_update.emit("\nStage 3/4: Training new Challenger model...")
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            if self.model_type == 'Binary':
                # Save retrained models to base model directory with timestamp
                challenger_base_path = models_path("SEAMaP-Binary-Full", "faster_rcnn_R_50_FPN_3x")
            else: # Multiclass
                challenger_base_path = models_path("SEAMaP-Multi-class-100", "faster_rcnn_R_50_FPN_3x")
            
            # Create the final, unique directory for this specific run.
            challenger_output_dir = os.path.join(challenger_base_path, timestamp)
            
            # Prepare training args
            train_module, train_args = self.prepare_training_command(annotations_path, challenger_output_dir)

            if train_args is None:
                raise RuntimeError("Could not prepare the training arguments. Check that train.py and config files exist.")
            if not self._is_running:
                self.cancelTraining(challenger_output_dir)
                raise InterruptedError("Process cancelled before training started.")

            # Read MAX_ITER from config for accurate progress bar
            config_file_path = models_path("Retraining", f"{self.model_type.lower()}_config.yaml")
            total_iterations = 2000  # Default fallback
            try:
                with open(config_file_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    total_iterations = config_data.get('SOLVER', {}).get('MAX_ITER', 2000)
            except Exception as e:
                self.log_update.emit(f"Warning: Could not read MAX_ITER from config. Using default {total_iterations}. Error: {e}")
            total_iterations = int(self.config_overrides.get("SOLVER.MAX_ITER", total_iterations))

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            line_emitter = _LineEmitter(self.log_update.emit, self.progress_update.emit, total_iterations)
            redirected_handlers = []
            sys.stdout = line_emitter
            sys.stderr = line_emitter
            try:
                redirected_handlers = self._redirect_logging_streams(line_emitter)

                # Call the bundled runtime directly; no external Python code is loaded.
                train_module.main(train_args)
            finally:
                self._restore_logging_streams(redirected_handlers)
                line_emitter.flush()
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            if not self._is_running:
                self.cancelTraining(challenger_output_dir)
                raise InterruptedError("Training was cancelled.")
            
            challenger_model_path = os.path.join(challenger_output_dir, "model_final.pth")
            self.log_update.emit(f"Training complete. Verifying model file exists at: {challenger_model_path}")
            
            max_retries = 5
            retry_delay_seconds = 1
            
            for i in range(max_retries):
                if os.path.exists(challenger_model_path):
                    self.log_update.emit("Model file found. Proceeding to evaluation.")
                    break # Exit the loop, file is found
                else:
                    self.log_update.emit(f"Model file not yet visible. Retrying in {retry_delay_seconds} second(s)... (Attempt {i+1}/{max_retries})")
                    time.sleep(retry_delay_seconds)
            else: # This 'else' belongs to the 'for' loop. It runs if the loop finishes without a 'break'.
                raise FileNotFoundError(f"Challenger model file was not found after {max_retries} retries.")
            final_result['challenger_path'] = challenger_model_path
            self.log_update.emit(f"Challenger model trained successfully at: {challenger_model_path}")
            
            # --- STAGE 4: EVALUATION ---
            self.log_update.emit("\nStage 4/4: Evaluating Champion vs. Challenger...")
            
            # Benchmark the champion (if one exists)
            final_result['champion_scores'] = self._run_benchmark(champion_model_path, f"{self.model_type}_Champion")
            if not self._is_running:
                self.cancelTraining(challenger_output_dir)
                raise InterruptedError("Process cancelled during evaluation.")

            # Benchmark the new challenger
            final_result['challenger_scores'] = self._run_benchmark(challenger_model_path, f"{self.model_type}_Challenger_{timestamp}")
            
            final_result['success'] = True
            final_result['message'] = "Evaluation complete. Please review the results."

        except InterruptedError as e:
            self.log_update.emit("Training was interrupted by user.")
            if challenger_output_dir and os.path.exists(challenger_output_dir):
                self.cancelTraining(challenger_output_dir)
            final_result['success'] = False
            final_result['message'] = str(e)
        except Exception as e:
            self.log_update.emit(f"\n--- PIPELINE FAILED: An unexpected error occurred ---")
            self.log_update.emit(str(e))
            if challenger_output_dir and os.path.exists(challenger_output_dir):
                self.cancelTraining(challenger_output_dir)
            final_result['success'] = False
            final_result['message'] = f"An error occurred in the pipeline: {e}"
        finally:
            # Always emit the final result, whether success or failure.
            self.retraining_finished.emit(final_result)


class RetrainUI(QDialog):
    close_signal = pyqtSignal()
    settings_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_type = self._read_current_model_type()
        if self.model_type not in ("Binary", "Multiclass"):
            self.model_type = "Binary"
        self.retraining_thread = None
        self.ui_state = "idle"
        self.training_started_at = None
        self.current_stage = "Ready"
        self.current_progress = 0
        self.current_progress_total = 100
        self.latest_loss = None
        self.best_loss = None
        self.summary_counts = {
            "positive": 0,
            "negative": 0,
            "total": 0,
            "binary_usable": 0,
            "multiclass_ready": 0,
        }
        self.dashboard_values = {}
        self.highlight_labels = {}
        # The source repair script lives at project root; model paths use models_path().
        self.project_root = str(Path(__file__).resolve().parents[1])
        self.init_ui()
        self.load_data_summary()

    def _read_current_model_type(self):
        try:
            with open(user_settings_path(), "r") as f:
                settings_data = json.load(f)
                return settings_data.get("general_features", {}).get("model", "Binary")
        except (FileNotFoundError, json.JSONDecodeError): return "Binary"

    def init_ui(self):
        self.setWindowTitle(f"Retrain Model ({self.model_type})")
        self.setWindowIcon(QIcon(resource_path("res", "PolyVisionLogo.png")))
        self.setMinimumSize(980, 700)
        self.resize(1080, 760)
        self.setStyleSheet(RETRAIN_UI_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        surface = QFrame()
        surface.setObjectName("retrainSurface")
        surface_layout = QVBoxLayout(surface)
        surface_layout.setContentsMargins(14, 14, 14, 14)
        surface_layout.setSpacing(10)

        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        model_row = QHBoxLayout()
        model_label = QLabel("Select model to retrain:")
        self.binary_button = QPushButton("Binary")
        self.multiclass_button = QPushButton("Multiclass")
        for button in (self.binary_button, self.multiclass_button):
            button.setObjectName("modelChoiceButton")
            button.setCheckable(True)
            button.setMinimumWidth(120)
        model_row.addWidget(model_label)
        model_row.addStretch()
        model_row.addWidget(self.binary_button)
        model_row.addWidget(self.multiclass_button)
        model_layout.addLayout(model_row)
        self.model_accent = QFrame()
        self.model_accent.setFixedHeight(3)
        model_layout.addWidget(self.model_accent)

        summary_group = QGroupBox("Retraining Data Summary")
        summary_group.setObjectName("dashboardGroup")
        summary_layout = QHBoxLayout(summary_group)
        summary_layout.setSpacing(10)
        summary_layout.addWidget(
            self.dashboard_card(
                "Dataset",
                [
                    ("Positive", "dataset_positive", "0"),
                    ("Negative", "dataset_negative", "0"),
                    ("Total", "dataset_total", "0"),
                    ("Usable", "dataset_usable", "0"),
                ],
            )
        )
        summary_layout.addWidget(
            self.dashboard_card(
                "Run Plan",
                [
                    ("Model", "run_model", self.model_type),
                    ("Device", "run_device", "Check on start"),
                    ("Iterations", "run_iterations", "-"),
                    ("ETA", "run_eta", "-"),
                ],
                title_key="runtime",
            )
        )
        summary_layout.addWidget(
            self.dashboard_card(
                "Status",
                [
                    ("Phase", "status_phase", "Ready"),
                    ("Progress", "status_progress", "0%"),
                    ("Elapsed", "status_elapsed", "-"),
                    ("Readiness", "status_readiness", "Ready"),
                ],
                title_key="status",
            )
        )

        progress_group = QGroupBox("Retraining Progress")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(10)
        self.status_panel = self.main_status_panel()
        self.highlights_panel = self.progress_highlights()
        progress_layout.addWidget(self.status_panel)
        progress_layout.addWidget(self.highlights_panel)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        self.import_button = QPushButton("Import Dataset")
        self.reset_button = QPushButton("Reset Data")
        self.start_button = QPushButton("Start Retraining")
        self.cancel_button = QPushButton("Cancel")
        self.close_button = QPushButton("Close")

        self.import_button.setProperty("buttonRole", "secondary")
        self.reset_button.setProperty("buttonRole", "danger")
        self.start_button.setProperty("buttonRole", "primary")
        self.cancel_button.setProperty("buttonRole", "warning")
        self.close_button.setProperty("buttonRole", "secondary")
        for button in (
            self.import_button,
            self.reset_button,
            self.start_button,
            self.cancel_button,
            self.close_button,
        ):
            button.setObjectName("footerButton")
            button.setMinimumHeight(32)
            button.setMinimumWidth(96)
        self.start_button.setMinimumWidth(132)

        button_layout.addWidget(self.import_button, alignment=Qt.AlignLeft)
        button_layout.addWidget(self.reset_button, alignment=Qt.AlignLeft)
        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.close_button)

        surface_layout.addWidget(model_group)
        surface_layout.addWidget(summary_group)
        surface_layout.addWidget(progress_group, stretch=2)
        surface_layout.addLayout(button_layout)
        layout.addWidget(surface)

        self.binary_button.clicked.connect(lambda: self.set_model_type("Binary"))
        self.multiclass_button.clicked.connect(lambda: self.set_model_type("Multiclass"))
        self.start_button.clicked.connect(self.start_retraining)
        self.cancel_button.clicked.connect(self.cancel_retraining)
        self.close_button.clicked.connect(self.close)
        self.reset_button.clicked.connect(self.reset_retraining_data)
        self.import_button.clicked.connect(self.run_import_dialog)
        self.set_model_type(self.model_type, refresh_summary=False)

    def dashboard_card(self, title, rows, title_key=None):
        card = QFrame()
        card.setObjectName("dashboardCard")
        card.setMinimumWidth(210)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 10, 12, 10)
        card_layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("dashboardTitle")
        if title_key:
            self.dashboard_values[f"{title_key}_title"] = title_label
        card_layout.addWidget(title_label)

        for label_text, key, value in rows:
            row = QHBoxLayout()
            row.setSpacing(8)
            label = QLabel(label_text)
            label.setObjectName("dashboardLabel")
            value_label = QLabel(value)
            value_label.setObjectName("dashboardValue")
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.dashboard_values[key] = value_label
            row.addWidget(label)
            row.addStretch()
            row.addWidget(value_label)
            card_layout.addLayout(row)

        card_layout.addStretch()
        return card

    def set_dashboard_value(self, key, value):
        label = self.dashboard_values.get(key)
        if label:
            label.setText(str(value))

    def main_status_panel(self):
        panel = QFrame()
        panel.setObjectName("mainStatusPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(14, 10, 14, 10)
        panel_layout.setSpacing(8)

        top = QHBoxLayout()
        self.status_percent_label = QLabel("Ready")
        self.status_percent_label.setObjectName("mainStatusPercent")
        self.status_eta_label = QLabel("ETA -")
        self.status_eta_label.setObjectName("mainStatusEta")
        self.status_health_label = QLabel("Ready")
        self.status_health_label.setObjectName("mainStatusHealth")
        top.addWidget(self.status_percent_label)
        top.addSpacing(12)
        top.addWidget(self.status_eta_label)
        top.addStretch()
        top.addWidget(self.status_health_label)
        panel_layout.addLayout(top)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        panel_layout.addWidget(self.progress_bar)

        self.status_bottom_label = QLabel("")
        self.status_bottom_label.setObjectName("mainStatusBottom")
        panel_layout.addWidget(self.status_bottom_label)
        return panel

    def progress_highlights(self):
        panel = QFrame()
        panel.setObjectName("highlightsPanel")
        panel.setMinimumHeight(330)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(9)

        self.current_step_label = QLabel("Ready to start retraining")
        self.current_step_label.setObjectName("currentStep")
        self.current_step_label.setWordWrap(True)
        layout.addWidget(self.current_step_label)

        self.step_summary_label = QLabel("Review the run plan, then start when ready.")
        self.step_summary_label.setObjectName("stepSummary")
        self.step_summary_label.setWordWrap(True)
        layout.addWidget(self.step_summary_label)

        columns = QHBoxLayout()
        columns.setSpacing(10)
        columns.addWidget(self.highlight_column("Completed", "done", 4))
        columns.addWidget(self.highlight_column("In Progress", "active", 5))
        columns.addWidget(self.highlight_column("Next", "next", 4))
        layout.addLayout(columns)

        self.technical_button = QPushButton("Show Technical Log")
        self.technical_button.setCheckable(True)
        self.technical_button.setObjectName("technicalLogToggle")

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Courier", 9))
        self.log_view.setPlaceholderText("Training logs will appear here...")
        self.log_view.setVisible(False)
        self.log_view.setFixedHeight(170)

        def toggle_log(checked):
            self.log_view.setVisible(checked)
            self.technical_button.setText("Hide Technical Log" if checked else "Show Technical Log")

        self.technical_button.toggled.connect(toggle_log)
        layout.addWidget(self.technical_button, alignment=Qt.AlignRight)
        layout.addWidget(self.log_view)
        return panel

    def highlight_column(self, title, status, capacity):
        card = QFrame()
        card.setObjectName("highlightCard")
        card.setMinimumWidth(240)
        card.setMinimumHeight(150)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(7)

        title_label = QLabel(title)
        title_label.setObjectName("highlightTitle")
        layout.addWidget(title_label)

        self.highlight_labels[status] = []
        for _ in range(capacity):
            label = QLabel("")
            label.setObjectName(f"highlightItem_{status}")
            label.setWordWrap(True)
            label.setVisible(False)
            self.highlight_labels[status].append(label)
            layout.addWidget(label)

        layout.addStretch()
        return card
        
    def run_import_dialog(self):
        dialog = ImportDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            image_dir, json_path = dialog.get_paths()
            
            # Disable buttons during import
            self.update_action_state("importing")
            self.log_view.clear()

            # Run the import in a background thread
            self.import_thread = ImportThread(image_dir, json_path)
            self.import_thread.progress_update.connect(self.update_log)
            self.import_thread.import_finished.connect(self.on_import_finished)
            self.import_thread.start()

    def load_data_summary(self):
        data = get_retraining_data()
        
        total_count = len(data)
        pos_count = 0
        neg_count = 0
        multiclass_ready_count = 0
        max_class_id = 0
        parsed_rows = []

        for row in data:
            bbox_str = row[2]
            bboxes = []
            try:
                bboxes = json.loads(bbox_str)
            except (json.JSONDecodeError, TypeError):
                bboxes = []
            if not isinstance(bboxes, list):
                bboxes = []
            parsed_rows.append((row, bboxes))
            for bbox_info in bboxes:
                if isinstance(bbox_info, dict):
                    try:
                        class_id = int(bbox_info.get("class_id", 0) or 0)
                    except (TypeError, ValueError):
                        class_id = 0
                    max_class_id = max(max_class_id, class_id)

        is_db_multiclass = max_class_id > 1
        for row, bboxes in parsed_rows:
            is_mp = row[1]
            if is_mp == 1:
                pos_count += 1
                row_model_type = row[3] if len(row) > 3 else None
                has_boxes = len(bboxes) > 0
                if has_boxes and (row_model_type == "Multiclass" or (row_model_type is None and is_db_multiclass)):
                    multiclass_ready_count += 1
            else:
                neg_count += 1

        self.summary_counts = {
            "positive": pos_count,
            "negative": neg_count,
            "total": total_count,
            "binary_usable": total_count,
            "multiclass_ready": multiclass_ready_count,
        }
        self.update_dashboard_for_model()
        if self.ui_state == "idle":
            self.update_idle_progress_copy()
            self.update_action_state("idle")

    def _on_model_type_changed(self, model_type):
        self.set_model_type(model_type)

    def set_model_type(self, model_type, refresh_summary=True):
        if model_type not in ("Binary", "Multiclass"):
            model_type = "Binary"
        self.model_type = model_type
        self.setWindowTitle(f"Retrain Model ({self.model_type})")
        is_binary = self.model_type == "Binary"
        meta = self.model_meta()
        self.binary_button.setChecked(is_binary)
        self.multiclass_button.setChecked(not is_binary)
        self.binary_button.setStyleSheet(self.model_button_style("#0f5f9e", is_binary))
        self.multiclass_button.setStyleSheet(self.model_button_style("#0f7f45", not is_binary))
        self.model_accent.setStyleSheet(f"background: {meta['color']};")
        self.start_button.setStyleSheet(self.primary_button_style())
        self.update_status_accent()
        if refresh_summary:
            self.load_data_summary()
        else:
            self.update_dashboard_for_model()
            self.update_idle_progress_copy()
            self.update_action_state("idle")

    def model_meta(self):
        if self.model_type == "Multiclass":
            return {
                "color": "#0f7f45",
                "hover": "#0b6f3b",
                "pressed": "#095f33",
                "usable_key": "multiclass_ready",
                "usable_label": "multiclass-ready samples",
                "summary": "Review the Multiclass run plan, then start when ready.",
                "upcoming_train": "Train updated Multiclass model",
                "no_data_text": "No Multiclass Data",
            }
        return {
            "color": "#0f5f9e",
            "hover": "#0c5288",
            "pressed": "#0a4775",
            "usable_key": "binary_usable",
            "usable_label": "binary-relevant samples",
            "summary": "Review the Binary run plan, then start when ready.",
            "upcoming_train": "Train updated Binary model",
            "no_data_text": "No Data to Retrain",
        }

    def usable_sample_count(self):
        return int(self.summary_counts.get(self.model_meta()["usable_key"], 0))

    def update_dashboard_for_model(self):
        usable = self.usable_sample_count()
        self.set_dashboard_value("dataset_positive", self.summary_counts["positive"])
        self.set_dashboard_value("dataset_negative", self.summary_counts["negative"])
        self.set_dashboard_value("dataset_total", self.summary_counts["total"])
        self.set_dashboard_value("dataset_usable", usable)
        self.set_dashboard_value("run_model", self.model_type)
        self.set_dashboard_value("run_iterations", self.read_max_iter())
        if self.ui_state == "idle":
            self.set_dashboard_value("run_eta", "-")
            self.set_dashboard_value("status_phase", "Ready")
            self.set_dashboard_value("status_progress", "0%")
            self.set_dashboard_value("status_elapsed", "-")
            self.set_dashboard_value("status_readiness", "Ready" if usable > 0 else "Needs Data")
            self.set_dashboard_value("run_device", "Check on start")

    def read_max_iter(self, default=300):
        config_file_path = models_path("Retraining", f"{self.model_type.lower()}_config.yaml")
        try:
            with open(config_file_path, 'r') as f:
                config_data = yaml.safe_load(f)
                return int(config_data.get('SOLVER', {}).get('MAX_ITER', default))
        except Exception:
            return default

    def update_idle_progress_copy(self):
        meta = self.model_meta()
        usable = self.usable_sample_count()
        self.current_stage = "Ready"
        self.status_percent_label.setText("Ready")
        self.status_percent_label.setStyleSheet(
            f"color: {meta['color']}; font-size: 17pt; font-weight: 800;"
        )
        self.status_eta_label.setText("ETA -")
        self.status_health_label.setText("Ready" if usable > 0 else "Needs Data")
        self.status_bottom_label.setText(
            f"Waiting for confirmation - {usable} {meta['usable_label']} ready"
        )
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.current_step_label.setText(f"Ready to start {self.model_type} retraining")
        self.step_summary_label.setText(meta["summary"])
        if usable > 0:
            completed = [
                f"Found {usable} {meta['usable_label']}",
                "Loaded retraining configuration",
                "GPU check will run before training",
            ]
            active = ["Waiting for user confirmation"]
        else:
            completed = ["Loaded retraining configuration"]
            active = [f"No usable {self.model_type} retraining data found"]
        self.set_highlight_items("done", completed)
        self.set_highlight_items("active", active)
        self.set_highlight_items(
            "next",
            [
                "Prepare retraining dataset",
                meta["upcoming_train"],
                "Benchmark and compare results",
            ],
        )
        self.set_idle_log_message()
        self.update_status_accent()

    def set_idle_log_message(self):
        current_text = self.log_view.toPlainText().strip()
        idle_prefixes = (
            "Training logs will appear here",
            "No retraining data found",
            "Binary model selected.",
            "Multiclass model selected.",
        )
        if current_text and not current_text.startswith(idle_prefixes):
            return
        usable = self.usable_sample_count()
        if usable <= 0:
            self.log_view.setPlainText("No retraining data found. Please collect data for retraining.")
            return
        self.log_view.setPlainText(
            f"{self.model_type} model selected.\n"
            f"{usable} {self.model_meta()['usable_label']} are ready.\n"
            "Training logs will appear here after retraining starts."
        )

    def set_highlight_items(self, status, items):
        marker = {"done": "-", "active": "*", "next": ">"}.get(status, "-")
        labels = self.highlight_labels.get(status, [])
        for index, label in enumerate(labels):
            if index < len(items):
                label.setText(f"{marker} {items[index]}")
                label.setVisible(True)
            else:
                label.setVisible(False)

    def update_status_accent(self):
        meta = self.model_meta()
        self.status_panel.setStyleSheet(
            "QFrame#mainStatusPanel { "
            "background: #f8fbfd; "
            "border: 1px solid #bfd0dc; "
            f"border-left: 5px solid {meta['color']}; "
            "border-radius: 7px; } "
            "QProgressBar::chunk { "
            f"background: {meta['color']}; border-radius: 8px; }}"
        )

    def model_button_style(self, color, selected):
        if selected:
            return (
                f"QPushButton#modelChoiceButton {{ color: white; background: {color}; "
                f"border: 1px solid {color}; border-radius: 3px; padding: 5px 14px; }}"
            )
        return (
            "QPushButton#modelChoiceButton { color: #1f2933; background: #f5f5f5; "
            f"border: 1px solid {color}; border-radius: 3px; padding: 5px 14px; }} "
            "QPushButton#modelChoiceButton:hover { background: #eef6f7; }"
        )

    def primary_button_style(self):
        meta = self.model_meta()
        return (
            "QPushButton#footerButton { "
            "color: #ffffff; "
            f"background: {meta['color']}; "
            f"border: 1px solid {meta['color']}; "
            "border-radius: 4px; "
            "padding: 6px 14px; "
            "font-weight: 700; } "
            "QPushButton#footerButton:hover { "
            f"background: {meta['hover']}; border-color: {meta['hover']}; }} "
            "QPushButton#footerButton:pressed { "
            f"background: {meta['pressed']}; border-color: {meta['pressed']}; }} "
            "QPushButton#footerButton:disabled { "
            "color: #9aa5b1; background: #eef1f4; border-color: #d5dbe1; }"
        )

    def update_action_state(self, state=None):
        if state:
            self.ui_state = state
        usable = self.usable_sample_count()
        idle = self.ui_state == "idle"
        importing = self.ui_state == "importing"
        training = self.ui_state == "training"
        cancelling = self.ui_state == "cancelling"

        model_enabled = idle
        self.binary_button.setEnabled(model_enabled)
        self.multiclass_button.setEnabled(model_enabled)
        self.import_button.setEnabled(idle)
        self.reset_button.setEnabled(idle)
        self.start_button.setEnabled(idle and usable > 0)
        self.cancel_button.setEnabled(training)
        self.close_button.setEnabled(True)
        self.cancel_button.setText("Cancelling..." if cancelling else "Cancel")

        if idle:
            self.start_button.setText("Start Retraining" if usable > 0 else self.model_meta()["no_data_text"])
        elif importing:
            self.start_button.setText("Importing...")
        elif training:
            self.start_button.setText("Training...")
        elif cancelling:
            self.start_button.setText("Training...")

    def format_duration(self, seconds):
        seconds = max(0, int(seconds))
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        return f"{minutes}m {sec:02d}s"

    def launch_gpu_repair(self):
        if getattr(sys, "frozen", False):
            QMessageBox.warning(
                self,
                "GPU Repair Unavailable",
                "Do not install CUDA Toolkit or run repair scripts for this packaged application.\n\n"
                "Please contact your PolyVision support person for a corrected application build."
            )
            return

        repair_script = Path(self.project_root) / "packaging" / "repair_gpu_env.bat"
        if not repair_script.exists():
            QMessageBox.critical(
                self,
                "GPU Repair Missing",
                f"Repair script was not found:\n{repair_script}"
            )
            return

        confirmation = QMessageBox.question(
            self,
            "Technical Repair Confirmation",
            "This repair is intended for a technical administrator working on a source installation.\n\n"
            "It requires Microsoft C++ build tools, CUDA Toolkit 11.8/NVCC, and internet access. "
            "It may update packages in the project virtual environment after validation succeeds.\n\n"
            "Continue with the technical repair?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if confirmation != QMessageBox.Yes:
            return

        try:
            subprocess.Popen(
                ["cmd.exe", "/k", str(repair_script)],
                cwd=self.project_root,
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
        except Exception as exc:
            QMessageBox.critical(self, "GPU Repair Failed", f"Could not launch GPU repair:\n{exc}")
            return

        QMessageBox.information(
            self,
            "GPU Repair Started",
            "A repair window has been opened.\n\n"
            "Close PolyVision before continuing in that window, then restart PolyVision after repair completes."
        )

    def maybe_offer_gpu_repair(self):
        report = diagnose_gpu_support()
        is_frozen = getattr(sys, "frozen", False)
        policy = retraining_start_policy(report, is_frozen)
        message = "\n".join(format_diagnostic_lines(report))

        if policy == BLOCKED:
            if is_frozen:
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Critical)
                box.setWindowTitle("Retraining Unavailable")
                box.setText("PolyVision cannot safely start retraining on this computer.")
                box.setInformativeText(
                    "Do not install CUDA Toolkit or run repair scripts. "
                    "Please take a screenshot of this message and contact your PolyVision "
                    "support person for a corrected application build."
                )
                box.setDetailedText(message)
                box.addButton(QMessageBox.Close)
                box.exec_()
                return False

            box = QMessageBox(self)
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle("Retraining Runtime Repair Required")
            box.setText("PolyVision cannot safely retrain on either CPU or GPU.")
            if report.repair_recommended:
                box.setInformativeText(
                    "A technical administrator must repair this source installation before retraining. "
                    "The repair requires Microsoft C++ build tools and CUDA Toolkit 11.8/NVCC."
                )
                repair_button = box.addButton("Open Technical Repair", QMessageBox.AcceptRole)
                cancel_button = box.addButton(QMessageBox.Cancel)
                box.setDefaultButton(cancel_button)
            else:
                box.setInformativeText(
                    "Automatic GPU repair does not apply to this machine. "
                    "Ask a technical administrator to repair the source retraining environment."
                )
                repair_button = None
                box.addButton(QMessageBox.Close)
            box.setDetailedText(message)
            box.exec_()
            if repair_button is not None and box.clickedButton() == repair_button:
                self.launch_gpu_repair()
            return False

        if policy == READY:
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("GPU Retraining Unavailable")
        box.setText(
            "PolyVision cannot use the GPU for retraining, but CPU retraining is available. "
            "CPU retraining may take longer."
        )
        if policy == CPU_FALLBACK:
            box.setInformativeText(
                "Select Continue on CPU to start retraining now. "
                "Do not install CUDA Toolkit or run repair scripts. "
                "For GPU retraining, contact your PolyVision support person to check the "
                "NVIDIA graphics driver and application build."
            )
            repair_button = None
        else:
            box.setInformativeText(
                "You can continue on CPU now. A technical administrator can repair GPU support "
                "for this source installation."
            )
            repair_button = box.addButton("Open Technical Repair", QMessageBox.AcceptRole)
        box.setDetailedText(message)
        continue_button = box.addButton("Continue on CPU", QMessageBox.DestructiveRole)
        cancel_button = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(continue_button)
        box.exec_()

        clicked = box.clickedButton()
        if repair_button is not None and clicked == repair_button:
            self.launch_gpu_repair()
            return False
        if clicked == continue_button:
            return True
        if clicked == cancel_button:
            return False
        return False

    # NEW: Slot to handle the start button click
    def _models_dir_is_writable(self):
        """
        Verify the external Models folder (next to PolyVision.exe) can be written
        to before starting retraining. A real write-probe is used because os.access
        with W_OK is unreliable on Windows protected directories.
        """
        models_dir = models_path()
        try:
            os.makedirs(models_dir, exist_ok=True)
            probe = os.path.join(models_dir, ".write_test")
            with open(probe, "w") as f:
                f.write("")
            os.remove(probe)
            return True
        except OSError:
            return False

    def start_retraining(self):

        if not self._models_dir_is_writable():
            QMessageBox.critical(
                self,
                'Models Folder Not Writable',
                "PolyVision cannot write to its Models folder:\n\n"
                f"{models_path()}\n\n"
                "Retraining needs to save new model files here, but this location is "
                "read-only. This usually happens when PolyVision is installed in a "
                "protected folder such as 'Program Files'.\n\n"
                "Move the entire PolyVision folder to a writable location (for example "
                "your Desktop or Documents) and try again.",
            )
            return

        RECOMMENDED_MINIMUM_SAMPLES = 50
        self.load_data_summary()
        num_new_samples = self.usable_sample_count()

        if num_new_samples < RECOMMENDED_MINIMUM_SAMPLES:
            # The number of samples is below our recommendation, so we warn the user.
            warning_reply = QMessageBox.warning(
                self,
                'Low Data Warning',
                f"You have only collected {num_new_samples} usable {self.model_type} data samples.\n\n"
                f"It is recommended to have at least {RECOMMENDED_MINIMUM_SAMPLES} new samples "
                "to see a significant improvement in model performance.\n\n"
                "Do you want to continue with the retraining anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No # Default button is "No"
            )
            
            # If the user clicks "No", we simply stop the process here.
            if warning_reply == QMessageBox.No:
                return # Exit the function

        if not self.maybe_offer_gpu_repair():
            return

        reply = QMessageBox.question(self, 'Confirm Retraining', 
                                     "This will start the model retraining process, which can take a long time and consume significant computer resources. We recommend charging your device before proceeding.\n\nAre you sure you want to continue?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                with open(user_settings_path(), "r+") as f:
                    settings_data = json.load(f)
                    settings_data["general_features"]["model"] = self.model_type
                    f.seek(0)
                    json.dump(settings_data, f, indent=4)
                    f.truncate()
                self.settings_updated.emit()
            except Exception as e:
                self.log_view.append(f"Warning: Could not save setting to user_settings.json. {e}")

            max_iter = self.read_max_iter()
            self.training_started_at = time.time()
            self.current_stage = "Starting retraining"
            self.current_progress = 0
            self.current_progress_total = max_iter
            self.latest_loss = None
            self.best_loss = None
            self.log_view.clear()
            self.log_view.append(f"Loaded MAX_ITER = {max_iter} from config.")
            self.progress_bar.setMaximum(max_iter)
            self.progress_bar.setValue(0)
            self.dashboard_values["runtime_title"].setText("Runtime")
            self.dashboard_values["status_title"].setText("Training")
            self.set_dashboard_value("run_model", self.model_type)
            self.set_dashboard_value("run_iterations", f"0 / {max_iter}")
            self.set_dashboard_value("run_eta", "estimating...")
            self.set_dashboard_value("status_phase", "Starting")
            self.set_dashboard_value("status_progress", "0%")
            self.set_dashboard_value("status_elapsed", "0m 00s")
            self.set_dashboard_value("status_readiness", "Running")
            self.status_percent_label.setText("0%")
            self.status_eta_label.setText("ETA estimating...")
            self.status_health_label.setText("Starting")
            self.status_bottom_label.setText(f"{self.model_type} retraining is starting")
            self.current_step_label.setText("Starting retraining")
            self.step_summary_label.setText("Preparing the retraining pipeline.")
            self.set_highlight_items("done", ["User confirmed retraining"])
            self.set_highlight_items("active", ["Starting retraining thread"])
            self.set_highlight_items(
                "next",
                [
                    "Prepare retraining dataset",
                    "Train challenger model",
                    "Evaluate champion vs challenger",
                ],
            )
            self.update_action_state("training")
            
            self.retraining_thread = RetrainingThread(self.model_type)
            self.retraining_thread.log_update.connect(self.update_log)
            self.retraining_thread.progress_update.connect(self.update_progress)
            self.retraining_thread.retraining_finished.connect(self.on_retraining_finished)
            self.retraining_thread.start()

    # NEW: Slot to handle the cancel button click
    def cancel_retraining(self):
        if self.retraining_thread and self.retraining_thread.isRunning():
            self.retraining_thread.stop()
            self.status_health_label.setText("Cancelling")
            self.current_step_label.setText("Cancelling retraining")
            self.step_summary_label.setText("Waiting for the current training step to stop safely.")
            self.set_highlight_items("active", ["Cancellation requested", "Cleaning up generated files"])
            self.update_action_state("cancelling")
            
    # NEW: Slots to receive signals from the background thread
    @pyqtSlot(int)
    def on_import_finished(self, imported_count):
        self.log_view.append(f"\n--- Import complete. Successfully added {imported_count} images. ---")
        QMessageBox.information(self, "Import Complete", f"Successfully imported {imported_count} images into the retraining dataset.")
        
        self.ui_state = "idle"
        self.load_data_summary() # This will re-enable the start button if data exists

    @pyqtSlot(str)
    def update_log(self, message):
        self.log_view.append(message)
        self.update_status_from_log(message)

    @pyqtSlot(int, int)
    def update_progress(self, current_step, total_steps):
        if self.progress_bar.maximum() != total_steps:
            self.progress_bar.setMaximum(total_steps)
        self.progress_bar.setValue(current_step)
        self.current_progress = max(0, int(current_step))
        self.current_progress_total = max(1, int(total_steps))
        percent = int((self.current_progress / self.current_progress_total) * 100)
        elapsed = 0 if self.training_started_at is None else time.time() - self.training_started_at
        elapsed_text = self.format_duration(elapsed)

        if self.training_started_at is None or self.current_progress <= 0:
            eta_text = "estimating..."
            status_eta = "ETA estimating..."
        else:
            rate = self.current_progress / max(elapsed, 1)
            remaining_seconds = (self.current_progress_total - self.current_progress) / max(rate, 0.0001)
            eta_text = self.format_duration(remaining_seconds)
            status_eta = f"ETA {eta_text}"

        self.status_percent_label.setText(f"{percent}%")
        self.status_eta_label.setText(status_eta)
        self.status_bottom_label.setText(
            f"{self.current_stage} - Iteration {self.current_progress} / {self.current_progress_total}"
        )
        self.set_dashboard_value("run_iterations", f"{self.current_progress} / {self.current_progress_total}")
        self.set_dashboard_value("run_eta", eta_text)
        self.set_dashboard_value("status_progress", f"{percent}%")
        self.set_dashboard_value("status_elapsed", elapsed_text)
        if "Stage 3/4" in self.current_stage:
            active = [
                f"Iteration {self.current_progress} / {self.current_progress_total}",
                "Training challenger model",
            ]
            if self.latest_loss is not None:
                active.append(f"Current loss: {self.latest_loss:.3f}")
            if self.best_loss is not None:
                active.append(f"Best loss so far: {self.best_loss:.3f}")
            self.set_highlight_items("active", active)

    def update_status_from_log(self, message):
        stripped = message.strip()
        if not stripped:
            return

        if stripped.startswith("Selected device:"):
            device = stripped.split(":", 1)[1].strip()
            self.set_dashboard_value("run_device", "CUDA GPU" if device == "cuda" else device.upper())
            return

        if stripped.startswith("Status:"):
            status = stripped.split(":", 1)[1].strip()
            if status == GPU_READY:
                self.status_health_label.setText("GPU Ready")
            elif status in {HARDWARE_PRESENT_SOFTWARE_MISSING, CUDA_BROKEN}:
                self.status_health_label.setText("CPU Mode")
            elif status == HARDWARE_MISSING:
                self.status_health_label.setText("CPU Mode")
            elif status == RETRAINING_RUNTIME_BROKEN:
                self.status_health_label.setText("Runtime Error")
            return

        if "Stage 1/4" in stripped:
            self.set_training_stage(
                "Stage 1/4: Preparing training data",
                "Preparing the merged retraining dataset.",
                ["Checked GPU and training environment"],
                ["Preparing training data"],
                [
                    "Identify champion model",
                    "Train challenger model",
                    "Evaluate champion vs challenger",
                ],
            )
            return

        if "Stage 2/4" in stripped:
            self.set_training_stage(
                "Stage 2/4: Identifying current Champion model",
                "Finding the active model to benchmark against.",
                [
                    "Checked GPU and training environment",
                    "Prepared retraining data",
                ],
                ["Identifying champion model"],
                [
                    "Train challenger model",
                    "Evaluate champion vs challenger",
                    "Ask whether to deploy the new model",
                ],
            )
            return

        if "Stage 3/4" in stripped:
            self.set_training_stage(
                "Stage 3/4: Training new Challenger model",
                f"{self.model_type} model training is running.",
                [
                    "Checked GPU and training environment",
                    "Prepared retraining data",
                    "Identified champion model",
                ],
                ["Training challenger model"],
                [
                    "Verify model_final.pth was saved",
                    "Evaluate champion vs challenger",
                    "Ask whether to deploy the new model",
                ],
            )
            return

        if "Stage 4/4" in stripped:
            self.set_training_stage(
                "Stage 4/4: Evaluating Champion vs. Challenger",
                "Benchmarking both models before deployment.",
                [
                    "Checked GPU and training environment",
                    "Prepared retraining data",
                    "Trained challenger model",
                ],
                ["Benchmarking champion and challenger"],
                ["Review comparison results", "Choose whether to deploy"],
            )
            if self.current_progress_total > 0:
                self.progress_bar.setValue(self.current_progress_total)
                self.status_percent_label.setText("100%")
                self.set_dashboard_value("status_progress", "100%")
            self.status_eta_label.setText("ETA -")
            self.set_dashboard_value("run_eta", "-")
            return

        if "total_loss:" in stripped:
            try:
                loss_text = stripped.split("total_loss:", 1)[1].strip().split()[0]
                self.latest_loss = float(loss_text)
                self.best_loss = self.latest_loss if self.best_loss is None else min(self.best_loss, self.latest_loss)
            except Exception:
                pass
            return

        if "Training complete. Verifying model file exists" in stripped:
            self.set_highlight_items(
                "active",
                ["Training complete", "Verifying model_final.pth was saved"],
            )
            return

        if "Model file found. Proceeding to evaluation." in stripped:
            self.set_highlight_items(
                "done",
                [
                    "Checked GPU and training environment",
                    "Prepared retraining data",
                    "Trained challenger model",
                    "Verified model file",
                ],
            )
            return

        if "PIPELINE FAILED" in stripped or stripped.startswith("FATAL"):
            self.status_health_label.setText("Failed")
            self.set_dashboard_value("status_readiness", "Failed")
            self.current_step_label.setText("Retraining failed")
            self.step_summary_label.setText("No model was deployed. Review the technical log for details.")
            return

        if "Training was interrupted by user." in stripped:
            self.status_health_label.setText("Cancelled")
            self.current_step_label.setText("Retraining cancelled")
            self.step_summary_label.setText("Generated challenger files are being cleaned up.")

    def set_training_stage(self, title, summary, completed, active, upcoming):
        self.current_stage = title
        self.current_step_label.setText(title)
        self.step_summary_label.setText(summary)
        self.status_health_label.setText("Running normally")
        self.status_bottom_label.setText(title)
        self.set_dashboard_value("status_phase", title.split(":", 1)[0])
        self.set_dashboard_value("status_readiness", "Running")
        self.set_highlight_items("done", completed)
        self.set_highlight_items("active", active)
        self.set_highlight_items("next", upcoming)

    @pyqtSlot(dict)
    def on_retraining_finished(self, result):
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancel")
        
        if not result.get('success'):
            QMessageBox.critical(self, "Retraining Failed", result.get('message', 'An unknown error occurred.'))
            self.finishedTraining()  # Reset UI to idle state
            return

        # Launch the comparison dialog
        dialog = ComparisonDialog(self, result.get('champion_scores'), result.get('challenger_scores'))
        
        if dialog.exec_() == QDialog.Accepted:
            # User clicked "Yes", so we deploy the new model
            self.log_view.append("\n--- User approved deployment. Deploying new model... ---")
            self.deploy_model(result.get('challenger_path'))
            self.promote_to_base_dataset()
            self.cleanup_retraining_data()
            self.finishedTraining()  # Reset UI to idle state after deployment
        else:
            # User clicked "No" - reject and clean up the challenger model
            self.log_view.append("\n--- User rejected deployment. Cleaning up new model... ---")
            self.reject_model(result.get('challenger_path'))
            QMessageBox.information(self, "Deployment Rejected", "The new model has been removed and the current model will be retained.")
            self.promote_to_base_dataset()
            self.cleanup_retraining_data()
            self.finishedTraining()  # Reset UI to idle state after rejection

    def finishedTraining(self):
        self.training_started_at = None
        self.current_progress = 0
        self.current_progress_total = 100
        self.current_stage = "Ready"
        
        # Add completion message to logs
        self.log_view.append("\n" + "="*50)
        self.log_view.append("Training session completed. Ready for next operation.")
        self.log_view.append("="*50)
        self.dashboard_values["runtime_title"].setText("Run Plan")
        self.dashboard_values["status_title"].setText("Status")
        self.update_action_state("idle")
        self.load_data_summary()

    def promote_to_base_dataset(self):
        """
        Copies used retraining images into the base dataset train folder
        and appends their annotations to _annotations.coco.json.
        Called after both accept and reject.
        Binary: promotes ALL images, downcasts all annotations to class_id = 1.
        Multiclass: promotes only Multiclass-tagged images, uses original class_id.
        """
        self.log_view.append("\n--- Promoting new images to base dataset... ---")

        if self.model_type == 'Binary':
            base_train_dir = models_path("SEAMaP-Binary-Full-6", "train")
            used_data = get_retraining_data()  # Binary uses ALL images
        else:
            base_train_dir = models_path("SEAMaP-Multi-class-100-1", "train")
            used_data = get_retraining_data('Multiclass')  # Multiclass uses only Multiclass-tagged

        if not used_data:
            self.log_view.append("No new images to promote.")
            return

        annotation_path = os.path.join(base_train_dir, "_annotations.coco.json")
        if not os.path.exists(annotation_path):
            self.log_view.append(f"ERROR: Base annotation file not found at {annotation_path}. Skipping promotion.")
            return

        try:
            with open(annotation_path, 'r') as f:
                base_coco = json.load(f)

            max_image_id = max([img['id'] for img in base_coco['images']], default=-1)
            max_ann_id = max([ann['id'] for ann in base_coco['annotations']], default=-1)
            image_id_offset = max_image_id + 1
            ann_id_counter = max_ann_id + 1
            promoted_count = 0

            for i, (image_name, is_mp, bbox_str, _) in enumerate(used_data):
                src_path = models_path("retraining_images", image_name)
                if not os.path.exists(src_path):
                    self.log_view.append(f"Warning: Image not found, skipping: {image_name}")
                    continue

                # Copy image into base dataset train folder
                dst_path = os.path.join(base_train_dir, image_name)
                shutil.copy(src_path, dst_path)

                # Read image dimensions
                try:
                    img_cv = cv2.imread(src_path)
                    height, width = img_cv.shape[:2]
                except Exception:
                    self.log_view.append(f"Warning: Could not read dimensions for {image_name}. Skipping.")
                    continue

                new_image_id = image_id_offset + i
                base_coco['images'].append({
                    "id": new_image_id,
                    "file_name": image_name,
                    "height": height,
                    "width": width
                })

                # Add annotations only if image has bounding boxes
                if is_mp == 1 and bbox_str:
                    try:
                        bboxes_list = json.loads(bbox_str)
                        for bbox_info in bboxes_list:
                            x1, y1, x2, y2 = bbox_info['bbox']
                            w, h = x2 - x1, y2 - y1
                            # Binary: downcast all to class_id=1, Multiclass: use original
                            category_id = 1 if self.model_type == 'Binary' else bbox_info['class_id']
                            base_coco['annotations'].append({
                                "id": ann_id_counter,
                                "image_id": new_image_id,
                                "category_id": category_id,
                                "bbox": [x1, y1, w, h],
                                "area": w * h,
                                "iscrowd": 0
                            })
                            ann_id_counter += 1
                    except (json.JSONDecodeError, TypeError):
                        self.log_view.append(f"Warning: Could not decode bbox for {image_name}.")

                promoted_count += 1

            with open(annotation_path, 'w') as f:
                json.dump(base_coco, f, indent=4)

            self.log_view.append(f"--- Promoted {promoted_count} images to base dataset successfully. ---")

        except Exception as e:
            self.log_view.append(f"--- ERROR during base dataset promotion: {e} ---")

    def cleanup_retraining_data(self):
        """
        Deletes used retraining images and their db records after promotion.
        Binary: deletes ALL records and image files (both Binary and Multiclass were used).
        Multiclass: deletes only Multiclass-tagged records and image files, Binary stays intact.
        """
        self.log_view.append("\n--- Cleaning up retraining data... ---")

        db_path = storage_path('retrain_images.db')
        img_dir = models_path("retraining_images")

        try:
            if self.model_type == 'Binary':
                # Delete ALL image files
                if os.path.exists(img_dir):
                    for f in os.listdir(img_dir):
                        file_path = os.path.join(img_dir, f)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                # Delete ALL records from db
                if os.path.exists(db_path):
                    connection = sqlite3.connect(db_path)
                    c = connection.cursor()
                    c.execute("DELETE FROM database_list")
                    connection.commit()
                    connection.close()
                self.log_view.append("--- All retraining data cleared. Ready for fresh batch. ---")

            else:  # Multiclass
                if os.path.exists(db_path):
                    connection = sqlite3.connect(db_path)
                    c = connection.cursor()
                    # Fetch Multiclass image names before deleting
                    c.execute("SELECT image_name FROM database_list WHERE model_type = 'Multiclass'")
                    multiclass_images = [row[0] for row in c.fetchall()]
                    # Delete only Multiclass records
                    c.execute("DELETE FROM database_list WHERE model_type = 'Multiclass'")
                    connection.commit()
                    connection.close()
                    # Delete only Multiclass image files
                    for image_name in multiclass_images:
                        file_path = os.path.join(img_dir, image_name)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                self.log_view.append("--- Multiclass retraining data cleared. Binary images preserved. ---")

        except Exception as e:
            self.log_view.append(f"--- ERROR during cleanup: {e} ---")

    def deploy_model(self, challenger_model_path):
        """
        Deploys the challenger model according to specific rules:
        - Always preserve base model (specific timestamp directories)
        - Remove any other retrained models (previous champions)
        - Keep challenger as new active champion
        """
        if not challenger_model_path or not os.path.exists(challenger_model_path):
            QMessageBox.critical(self, "Deployment Error", f"Challenger model not found at: {challenger_model_path}")
            return

        try:
            # Define base model directory. The base model is identified by a marker
            # file (see base_marker.is_base_model_dir), NOT by a hardcoded timestamp.
            if self.model_type == 'Binary':
                base_model_dir = models_path("SEAMaP-Binary-Full", "faster_rcnn_R_50_FPN_3x")
            else:
                base_model_dir = models_path("SEAMaP-Multi-class-100", "faster_rcnn_R_50_FPN_3x")

            # Get challenger directory info
            challenger_source_dir = os.path.dirname(challenger_model_path)
            challenger_dir_name = os.path.basename(challenger_source_dir)

            self.log_view.append(f"Deploying challenger: {challenger_dir_name}")

            # STEP 1: Classify model folders. Keep the challenger and any protected
            # base model; mark every other previous champion for removal.
            retrained_models_to_remove = []
            protected_bases = []
            if os.path.exists(base_model_dir):
                self.log_view.append(f"Scanning directory: {base_model_dir}")
                for item in os.listdir(base_model_dir):
                    item_path = os.path.join(base_model_dir, item)
                    
                    # Skip if not a directory
                    if not os.path.isdir(item_path):
                        continue
                    
                    # Skip challenger directory
                    if item == challenger_dir_name:
                        self.log_view.append(f"Skipping challenger: {item}")
                        continue
                    
                    # Skip (and keep) any protected base model
                    if is_base_model_dir(item_path, self.model_type):
                        protected_bases.append(item)
                        self.log_view.append(f"Skipping protected base model: {item}")
                        continue

                    # Check if it has model_final.pth (valid model directory)
                    model_path = os.path.join(item_path, "model_final.pth")
                    if os.path.exists(model_path):
                        # This is a retrained model that should be removed
                        retrained_models_to_remove.append((item_path, item))
                        self.log_view.append(f"Marked for removal: {item}")
                    else:
                        self.log_view.append(f"Skipping {item} - no model_final.pth")

            # FAIL-SAFE: never delete when no base model can be positively identified.
            # Without this guard an unrecognized base would fall through to deletion.
            if not protected_bases:
                self.log_view.append(
                    "--- DEPLOYMENT HALTED: No protected base model found. "
                    "Refusing to delete any models to protect the base. ---")
                QMessageBox.warning(
                    self, "Deployment Halted",
                    "No protected base model could be identified, so no models were "
                    "removed (this protects your base model from accidental deletion).\n\n"
                    "Stamp the base model with tools/mark_base.py, then deploy again.\n\n"
                    "Your newly trained model is saved and is already the active model.")
                return
            
            # STEP 2: Remove old retrained models
            self.log_view.append(f"Found {len(retrained_models_to_remove)} retrained models to remove")
            for model_path, model_name in retrained_models_to_remove:
                try:
                    self.log_view.append(f"Removing old retrained model: {model_name}")
                    self._force_remove_directory(model_path)
                    self.log_view.append(f"Successfully removed: {model_name}")
                except Exception as e:
                    self.log_view.append(f"WARNING: Could not remove {model_name}: {e}")
                    # Continue with deployment even if removal fails
            
            # STEP 3: Verify challenger deployment
            if os.path.exists(challenger_model_path):
                self.log_view.append(f"Challenger model is now the active champion: {challenger_source_dir}")
                self.log_view.append(f"New active model verified at: {challenger_model_path}")
                QMessageBox.information(self, "Deployment Successful", 
                    f"New model has been successfully deployed!")
            else:
                raise FileNotFoundError(f"Challenger model not found at: {challenger_model_path}")
                
        except Exception as e:
            self.log_view.append(f"--- DEPLOYMENT FAILED: {e} ---")
            QMessageBox.critical(self, "Deployment Failed", f"Failed to deploy model: {str(e)}")

    def _force_remove_directory(self, dir_path):
        """
        Forcefully removes a directory with better error handling.
        """
        if not os.path.exists(dir_path):
            return
        
        import stat
        import time
        
        def handle_remove_readonly(func, path, exc):
            """Error handler for removing readonly files."""
            if os.path.exists(path):
                os.chmod(path, stat.S_IWRITE)
                func(path)
        
        # Try normal removal first
        try:
            shutil.rmtree(dir_path)
            return
        except PermissionError:
            pass
        
        # Try with readonly handler
        try:
            shutil.rmtree(dir_path, onerror=handle_remove_readonly)
            return
        except Exception:
            pass
        
        # Last resort: try to remove files individually
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    os.chmod(file_path, stat.S_IWRITE)
                    os.remove(file_path)
                except Exception:
                    pass
            for dir in dirs:
                try:
                    os.rmdir(os.path.join(root, dir))
                except Exception:
                    pass
        
        # Remove the main directory
        try:
            os.rmdir(dir_path)
        except Exception as e:
            raise Exception(f"Could not completely remove directory: {e}")

    def reject_model(self, challenger_model_path):
        """
        Rejects the challenger model by deleting it and its timestamp directory.
        This preserves the current champion and base model.
        """
        if not challenger_model_path or not os.path.exists(challenger_model_path):
            self.log_view.append("Challenger model path not found - nothing to clean up.")
            return

        try:
            # Remove the entire timestamp directory containing the rejected model
            challenger_dir = os.path.dirname(challenger_model_path)
            challenger_dir_name = os.path.basename(challenger_dir)
            
            # Safety check - only remove if it's a timestamp directory
            if '-' in challenger_dir_name and len(challenger_dir_name) == 19:  # Format: YYYY-MM-DD-HH-MM-SS
                self.log_view.append(f"Rejecting challenger model: {challenger_dir_name}")
                self._force_remove_directory(challenger_dir)
                self.log_view.append(f"Rejected model and directory removed: {challenger_dir}")
            else:
                # Fallback - just remove the model file if directory format is unexpected
                os.remove(challenger_model_path)
                self.log_view.append(f"Rejected model file removed: {challenger_model_path}")
                
        except Exception as e:
            self.log_view.append(f"--- ERROR during model rejection cleanup: {e} ---")

    def closeEvent(self, event):
        if self.retraining_thread and self.retraining_thread.isRunning():
            reply = QMessageBox.question(self, 'Retraining in Progress',
                                        "Retraining is currently in progress. Are you sure you want to close and cancel it?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.retraining_thread.stop()
                self.retraining_thread.log_update.disconnect()
                self.retraining_thread.progress_update.disconnect()
                self.retraining_thread.retraining_finished.disconnect()
                event.accept()
            else:
                event.ignore()
        else:
            self.close_signal.emit()
            event.accept()
    def reset_retraining_data(self):
        """ Deletes the retraining database and associated images after confirmation. """
        
        reply = QMessageBox.question(self, 'Confirm Reset', 
                                     "This will permanently delete all collected retraining images and their labels. This action cannot be undone.\n\nAre you sure you want to continue?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            db_path = storage_path('retrain_images.db')
            img_dir = models_path("retraining_images")
            
            # Delete the database file
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                    self.log_view.append(f"Successfully deleted {db_path}.")
                except OSError as e:
                    self.log_view.append(f"Error deleting database: {e}")
                    QMessageBox.critical(self, "Error", f"Could not delete the database file.\nMake sure the application is not using it.\n\nError: {e}")
                    return


            if os.path.exists(img_dir):
                try:
                    for filename in os.listdir(img_dir):
                        file_path = os.path.join(img_dir, filename)
                        os.remove(file_path)
                    self.log_view.append(f"Successfully cleared all images in {img_dir}.")
                except OSError as e:
                    self.log_view.append(f"Error clearing images: {e}")
                    QMessageBox.critical(self, "Error", f"Could not delete images in the retraining folder.\n\nError: {e}")
                    return

            # Re-create the empty database
            create_retraining_database(str(app_storage_dir()))
            self.log_view.append("A new, empty retraining database has been created.")
            
            # Refresh the summary view
            self.load_data_summary()
            
            QMessageBox.information(self, "Success", "Retraining data has been reset.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    retrain_ui = RetrainUI() 
    retrain_ui.show()
    sys.exit(app.exec_())
