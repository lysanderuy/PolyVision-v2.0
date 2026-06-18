# models/archive/iteration_1 — Iteration-1 training scripts

These are the original, manually-run training, evaluation, and visualization
scripts from the first iteration of the model pipeline. They were derived from a
Google Colab notebook (note the `cv2_imshow` shims that replace Colab's built-in).

**They are not part of the PolyVision application or its build.**

- Nothing in `ui/` imports from this folder.
- `models/` is not bundled into the packaged `.exe`.
- The live retraining pipeline is `ui/retraining_runtime/` + `models/retraining/`,
  which reads local COCO datasets directly — it does not use Roboflow.

They are kept for **reproducibility and provenance**: they record which Roboflow
dataset versions and which hyperparameters produced the iteration-1 models.

> **Base-model lineage:** the protected base models the app fine-tunes from
> (`SEAMaP-*/faster_rcnn_R_50_FPN_3x/2025-10-01-*/model_final.pth`) were trained
> on 2025-10-01 in a Docker GPU container using this iteration-1 config, starting
> from Detectron2's COCO-pretrained `faster_rcnn_R_50_FPN_3x` weights (not from
> scratch), then copied out via `verify_model_step_by_step.py`.

## How to run

These scripts use flat relative imports (`import dataset`,
`from binary_config import *`, etc.), so they only work when run from **inside
this folder**. Keep the files together; do not split them across directories.

`dataset.py` downloads from Roboflow and needs the `roboflow` package, which is
**no longer in `requirements.txt`**. To re-run the downloader:

```powershell
pip install roboflow
```

The Roboflow API key is read from `ROBOFLOW_API_KEY` via `python-dotenv`
(`load_dotenv()` in `dataset.py`). Copy `.env.example` to `.env` in this folder
and fill in the key — `.env` is gitignored and never committed. The key is no
longer hardcoded in the source.

## Contents

| File | Purpose |
|------|---------|
| `dataset.py` | Download SEAMaP datasets from Roboflow + register as COCO |
| `imports.py` | Common-import convenience bundle |
| `binary_config.py` / `multiclass_config.py` | Training hyperparameters |
| `config_setup.py` | Older config (superseded by `binary_config.py`) |
| `custom_classes.py` | Consolidated `LossEvalHook` + `CustomEvaluator` + `CustomTrainer` |
| `custom_evaluator.py` / `custom_trainer.py` / `loss_eval_hook.py` | Earlier split versions, merged into `custom_classes.py` |
| `train_model.py` | Main training entry point |
| `evaluate_trained_model.py` | Post-training COCO evaluation |
| `evaluation_fixed.py` | Evaluation + confusion matrix |
| `verify_model_step_by_step.py` | Verify a `.pth` file loads correctly |
| `visualize_dataset.py` | Draw ground-truth annotations |
| `visualize_predictions.py` | Draw model predictions |
| `iou_visualization.py` | IoU visualization |
| `check_ver.py` / `cudatest.py` | Torch / Detectron2 / CUDA environment checks |
