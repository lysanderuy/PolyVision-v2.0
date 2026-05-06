import os
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.engine import DefaultTrainer  
import detectron2.utils.comm as comm
import numpy as np
import torch
import time
import datetime
import logging
from detectron2.evaluation import COCOEvaluator

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            
            # CPU-optimized: Remove CUDA synchronization since we're using CPU
            
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

# Create your custom evaluator class
class CustomEvaluator:
    def __init__(self, iou_thresholds=(0.3, 0.5, 0.75, 0.95)):
        self._iou_thresholds = iou_thresholds

    def __call__(self, trainer):
        with inference_context(trainer.model), torch.no_grad():
            evaluator = COCOEvaluator(trainer.cfg.DATASETS.TEST[0], trainer.cfg, True)
            val_loader = build_detection_test_loader(trainer.cfg, trainer.cfg.DATASETS.TEST[0])
            metrics = evaluator.evaluate(val_loader)

        custom_metrics = {}
        for iou in self._iou_thresholds:
            custom_metrics[f"AP{iou * 100}"] = metrics[f"bbox/AP{iou}"]

        return custom_metrics

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,  # FIXED: Use self.cfg instead of undefined cfg
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks

# Test the classes
if __name__ == "__main__":
    print("✅ Custom Detectron2 classes loaded successfully!")
    print("Available classes:")
    print("- LossEvalHook")
    print("- CustomEvaluator") 
    print("- CustomTrainer")
    print("\nYou can now import these classes in your training scripts:")
    print("from custom_detectron_classes import LossEvalHook, CustomEvaluator, CustomTrainer")