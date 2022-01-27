import os
import torch
import time

from detectron2.engine import AMPTrainer, create_ddp_model
from detectron2.engine.hooks import LRScheduler, IterationTimer, PeriodicWriter, PeriodicCheckpointer
from detectron2.solver import build_optimizer, build_lr_scheduler
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils import comm
from datetime import datetime
from detectron2.utils.logger import PathManager
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from typing import Optional
from src.models.architectures import Detector, FastRCNN
from src.models.architectures import RegionProposalNetwork
from copy import deepcopy
from src.models.components.rpn import MyRPN


def default_writers(output_dir: str, max_iter: Optional[int] = None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    PathManager.mkdirs(output_dir)
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]


class RPNTrainer(AMPTrainer):
    def __init__(self, rpn: RegionProposalNetwork, data_loader, cfg):
        self.rpn = rpn
        self.distributed_rpn = create_ddp_model(self.rpn, find_unused_parameters=True)
        self.distributed_rpn.to(cfg.MODEL.DEVICE)
        self.data_loader = data_loader
        self.cfg = cfg
        self.max_iter = cfg.SOLVER.MAX_ITER

        # Create SGD Optimizer and scheduler
        self.optimizer = build_optimizer(cfg, self.rpn)
        self.scheduler = build_lr_scheduler(cfg, self.optimizer)

        super().__init__(self.rpn, self.data_loader, self.optimizer)

        # Create checkpoint and resume/load pre-trained rpn
        self.checkpointer = DetectionCheckpointer(self.rpn, cfg.OURS.CHECKPOINT_DIR)
        self.resume_or_load(resume=False)

        # Load pre-trained fast RCNN if step is equal to 2
        if self.rpn.step == 3:
            rcnn = FastRCNN.init_from_config(cfg)
            checkpointer_rcnn = DetectionCheckpointer(rcnn, cfg.OURS.CHECKPOINT_DIR_RCNN)
            self.resume_or_load(resume=False)
            checkpointer_rcnn.resume_or_load(cfg.OURS.MODELS.RCNN.WEIGHTS, resume=False)
            self.rpn.backbone = deepcopy(rcnn.backbone).to(self.cfg.MODEL.DEVICE)
            self.rpn.backbone.freeze_later(cfg.OURS.FREEZE_AT)

        # Register all hooks
        hooks = [
            LRScheduler(),
            IterationTimer(),
        ]
        if comm.is_main_process():
            hooks.append(PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))
            hooks.append(
                PeriodicWriter(
                    default_writers(os.path.join(self.cfg.OUTPUT_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
                                    self.max_iter), period=cfg.OURS.WRITERS.PERIOD))
        super().register_hooks(hooks)

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def train(self):
        super().train(self.start_iter, self.cfg.SOLVER.MAX_ITER)

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.rpn.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast():
            loss_dict = self.distributed_rpn(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()


class DetectorTrainer(AMPTrainer):
    def __init__(self, detector: Detector, data_loader, cfg):
        self.detector = detector
        self.distributed_detector = create_ddp_model(detector, find_unused_parameters=True)
        self.detector.rpn.to(cfg.MODEL.DEVICE)
        self.detector.rcnn.to(cfg.MODEL.DEVICE)
        self.detector.to(cfg.MODEL.DEVICE)

        self.data_loader = data_loader
        self.cfg = cfg
        self.max_iter = cfg.SOLVER.MAX_ITER

        # Create SGD Optimizer and scheduler
        self.optimizer = build_optimizer(cfg, self.detector)
        self.scheduler = build_lr_scheduler(cfg, self.optimizer)

        super().__init__(self.detector, self.data_loader, self.optimizer)

        ## Resume or load RCNN
        self.checkpointer_rcnn = DetectionCheckpointer(self.detector.rcnn, cfg.OURS.CHECKPOINT_DIR)
        self.resume_or_load(resume=False)
        if self.detector.step == 4 and not self.cfg.OURS.IS_TEST:
            self.detector.rcnn.backbone = deepcopy(self.detector.rpn.backbone)
            self.detector.rcnn.backbone.freeze_later(cfg.OURS.FREEZE_AT)

        # Register all hooks
        hooks = [
            LRScheduler(),
            IterationTimer(),
        ]
        if comm.is_main_process():
            hooks.append(PeriodicCheckpointer(self.checkpointer_rcnn, cfg.SOLVER.CHECKPOINT_PERIOD))
            hooks.append(
                PeriodicWriter(
                    default_writers(os.path.join(self.cfg.OUTPUT_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
                                    self.max_iter), period=cfg.OURS.WRITERS.PERIOD))
        super().register_hooks(hooks)

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer_rcnn.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer_rcnn.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.detector.rcnn.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast():
            loss_dict = self.distributed_detector(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def train(self):
        super().train(self.start_iter, self.cfg.SOLVER.MAX_ITER)