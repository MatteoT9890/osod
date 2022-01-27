import logging
import os
import tempfile
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from typing import Dict

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import create_ddp_model
from detectron2.utils.file_io import PathManager

from detectron2.utils.events import JSONWriter

from src.models.architectures import RegionProposalNetwork, Detector
from src.post_train.loop import Loop
from detectron2.utils import comm
from src.structures.metrics import ObjectnessMetrics, DetectionMetrics, ActivationMetrics
from src.tester.evaluator import PascalLoader, ClassImagesGT
import numpy as np


def default_writers(output_dir: str):
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
        JSONWriter(output_dir),
    ]

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

class PostRPNTrainer(Loop):
    def __init__(self, rpn: RegionProposalNetwork, data_loader, cfg):
        super().__init__()

        self.cfg = cfg
        self.rpn = rpn
        self.distributed_rpn = create_ddp_model(self.rpn, find_unused_parameters=True)
        self.distributed_rpn.to(cfg.MODEL.DEVICE)
        """
        We set the model to eval mode in the tester.
        """
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)

        # Create checkpoint and resume/load pre-trained model
        self.checkpointer = DetectionCheckpointer(self.rpn, cfg.OURS.CHECKPOINT_DIR)
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=False)
        self.rpn.eval()

        # Create metrics
        self.fg_objectness_metrics = ObjectnessMetrics(cfg)
        self.bg_objectness_metrics = ObjectnessMetrics(cfg)
        self.log_period = 20
        self.json_writer = JSONWriter(os.path.join(cfg.OUTPUT_DIR, "objectness_{}.json".format(
            "_".join([self.cfg.OURS.DATASETS.INFO.NAME, self.cfg.OURS.DATASETS.INFO.SPLIT])
        )))

    def apply_stack_context(self, stack: ExitStack):
        if isinstance(self.rpn, torch.nn.Module):
            stack.enter_context(inference_context(self.rpn))
        stack.enter_context(torch.no_grad())

    def before(self):
        logger = logging.getLogger(__name__)
        logger.info("Start inference on {} batches".format(len(self.data_loader)))

    def run_step(self):
        inputs = next(self._data_loader_iter)
        fg_scores, bg_scores = self.distributed_rpn(inputs)
        for fg_scores_i, bg_scores_i in zip(fg_scores, bg_scores):
            self.fg_objectness_metrics.update(fg_scores_i)
            self.bg_objectness_metrics.update(bg_scores_i)

    def after(self):
        self.fg_objectness_metrics.calculate_metrics()
        self.bg_objectness_metrics.calculate_metrics()
        if not comm.is_main_process():
            return

        self.storage.put_scalar("mean_fg_output", self.fg_objectness_metrics.output_mean)
        self.storage.put_scalar("std_fg_output", self.fg_objectness_metrics.output_std)
        self.storage.put_scalar("mean_fg_score", self.fg_objectness_metrics.mean_score)
        self.storage.put_scalar("mean_bg_output", self.bg_objectness_metrics.output_mean)
        self.storage.put_scalar("std_bg_output", self.bg_objectness_metrics.output_std)
        self.storage.put_scalar("mean_bg_score", self.bg_objectness_metrics.mean_score)
        self.json_writer.write()

class PostDetectorTrainer(Loop):
    def __init__(self, detector: Detector, data_loader, cfg):
        super().__init__()

        self.cfg = cfg
        self.detector = detector
        self.distributed_detector = create_ddp_model(self.detector, find_unused_parameters=True)
        self.distributed_detector.to(cfg.MODEL.DEVICE)

        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)

        # Create checkpoint and resume/load pre-trained model
        self.checkpointer = DetectionCheckpointer(self.detector.rcnn, cfg.OURS.CHECKPOINT_DIR_RCNN)
        self.checkpointer.resume_or_load(self.cfg.OURS.MODELS.RCNN.WEIGHTS, resume=False)
        self.detector.rcnn.eval()

        # Create metrics
        self._cpu_device = torch.device("cpu")
        self._predictions = defaultdict(list)
        self.activation_metrics: Dict[str, ActivationMetrics] = {}
        self.out_dir = cfg.OUTPUT_DIR

    def apply_stack_context(self, stack: ExitStack):
        if isinstance(self.detector, torch.nn.Module):
            stack.enter_context(inference_context(self.detector))
        stack.enter_context(torch.no_grad())

    def run_step(self):
        inputs = next(self._data_loader_iter)
        outputs = self.distributed_detector(inputs)
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

    def after(self):
        self.all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return

        classes_images_gt, (class_names, class_ids) = PascalLoader.load(self.cfg.OURS.MODELS.RCNN.POST_TRAIN_DATASET_NAME)
        predictions = defaultdict(list)
        for predictions_per_rank in self.all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del self.all_predictions

        with tempfile.TemporaryDirectory(prefix="openset_eval") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            for cls_id, cls_name in zip(class_ids, class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                class_activation = eval(
                    detpath=res_file_template,
                    class_images_gt=classes_images_gt[cls_name],
                    class_name=cls_name
                )
                self.activation_metrics[cls_name] = class_activation

        for cls_name in class_names:
            torch.save(self.activation_metrics[cls_name].mean, os.path.join(
                self.out_dir,
                cls_name + "_mean.pt"
            ))
            torch.save(torch.sigmoid(self.activation_metrics[cls_name].mean), os.path.join(
                self.out_dir,
                cls_name + "_mean_score.pt"
            ))
            torch.save(self.activation_metrics[cls_name].std, os.path.join(
                self.out_dir,
                cls_name + "_std.pt"
            ))

def eval(detpath, class_name, class_images_gt: ClassImagesGT, ovthresh=0.5):

    class_activation: ActivationMetrics = ActivationMetrics()
    # assumes detections are in detpath.format(classname)
    # read dets
    detfile = detpath.format(class_name)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    for d in range(nd):
        image_id = image_ids[d]
        ovmax = -np.inf
        R = class_images_gt.recs.get(image_id, None)
        if R is not None:
            bb = BB[d, :].astype(float)
            BBGT = R.bbox.astype(float)
        else:
            ## The current prediction corresponds to an id of another dataset
            BBGT = np.array([])

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                    (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                    + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                    - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            class_activation.update(torch.Tensor([confidence[d]]))

    class_activation.calculate_metrics()
    return class_activation
