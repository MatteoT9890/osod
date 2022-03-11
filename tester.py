import os
from contextlib import ExitStack, contextmanager
import datetime
from copy import deepcopy
from typing import List, Dict
import json

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetMapper
from detectron2.utils.file_io import PathManager

from detectron2.utils.events import TensorboardXWriter, JSONWriter

from src.tester.evaluator import UnifiedDatasetEvaluator, RecPrecAp, UdrUdpResult
from src.tester.hooks import PeriodicWriter
from detectron2.evaluation import print_csv_format
from detectron2.utils import comm
from detectron2.engine import create_ddp_model

from src.tester.test_loop import TestBase
from src.models.architectures import Detector, FastRCNN
from detectron2.data.build import build_detection_test_loader, get_detection_dataset_dicts
from detectron2.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator
import numpy as np

from src.utils.utils import write_list_json_as_csv, write_dict_as_json, save_object, load_object


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
        TensorboardXWriter(output_dir),
    ]


class Tester:
    def __init__(self, cfg, detector: Detector):
        #distributed_detector = detector
        distributed_detector = create_ddp_model(detector)
        detector.rcnn.to(cfg.MODEL.DEVICE)
        detector.rpn.to(cfg.MODEL.DEVICE)

        # Create checkpoint and resume/load pre-trained model
        checkpointer_rcnn = DetectionCheckpointer(detector.rcnn, cfg.OURS.CHECKPOINT_DIR_RCNN)
        checkpointer_rcnn.resume_or_load(cfg.OURS.MODELS.RCNN.WEIGHTS, resume=False)
        detector.rcnn.eval()

        self.test_dataset_names = cfg.OURS.TEST_DATASETS_REGISTERED_NAMES
        self.coco_dataset_name = "COCO2017"
        self.voc_dataset_name = "VOC2007_all"
        self.only_wic = cfg.OURS.TEST_ONLY_WIC
        self.out_dir = cfg.OUTPUT_DIR
        self.load_predictions = cfg.OURS.TEST_LOAD_PREDICTIONS
        if len(self.test_dataset_names) == 2:
            self.mode = "open_cwwr"
            voc_dataset_name, coco_dataset_name = self.test_dataset_names
            assert "VOC" in voc_dataset_name
            assert "COCO" in coco_dataset_name
        elif len(self.test_dataset_names) == 1:
            dataset_name = self.test_dataset_names[0]
            if "VOC" in dataset_name:
                self.mode = "cwwr"
            elif "COCO" in dataset_name:
                self.mode = "open"
            else:
                raise Exception("dataset name not recognized: {}".format(dataset_name))
        else:
            raise Exception("The config field `OURS.TEST_DATASETS_REGISTERED_NAMES` "
                            "must contain from 1 to 2 elements. Got {}".format(len(self.test_dataset_names)))

        self.evaluator = UnifiedDatasetEvaluator(
            voc_dataset_name=self.voc_dataset_name,
            coco_dataset_name=self.coco_dataset_name,
            mode=self.mode,
            out_dir=self.out_dir,
            model_reject=not(cfg.OURS.MODE == "baseline" and not cfg.OURS.USE_MSP))
        if self.only_wic:
            self.evaluator._predictions = []
        self.dataset_testers: List[DatasetTester] = []
        for i, dataset_name in enumerate(self.test_dataset_names):
            cfg.defrost()
            cfg.DATASETS.TEST = (cfg.OURS.TEST_DATASETS_REGISTERED_NAMES[i])
            cfg.freeze()
            dataloader = build_detection_test_loader(cfg)
            self.dataset_testers.append(DatasetTester(
                cfg,
                detector=detector,
                distributed_detector=distributed_detector,
                data_loader=dataloader,
                evaluator=self.evaluator,
                dataset_name=dataset_name,
                only_wic=self.only_wic
            ))

        self.cfg = cfg

    def test(self):
        model_out_dir = os.path.join(self.out_dir, "model_out")
        if not self.load_predictions:
            for dataset_tester in self.dataset_testers:
                dataset_tester.test(start_iter=0, max_iter=len(dataset_tester.data_loader))

            self.evaluator.collect_predictions()
            if self.only_wic and comm.is_main_process():
                PathManager.mkdirs(model_out_dir)
                save_object(self.evaluator.all_predictions, filename=os.path.join(
                    model_out_dir,
                    "predictions.pkl"
                ))
        if comm.is_main_process():
            ##Now evaluate

            if not self.only_wic:
                self.evaluator.evaluate()
                ## Create output dir
                ## Write results for each class in a CSV
                rec_prec_ap_results: List[RecPrecAp] = self.evaluator.rec_prec_ap_results
                header_rec_prec_ap = rec_prec_ap_results[0].__dict__.keys()
                rec_prec_ap = [result.__dict__ for result in rec_prec_ap_results]
                write_list_json_as_csv(data=rec_prec_ap, header=header_rec_prec_ap,
                                       filepath=os.path.join(self.cfg.OUTPUT_DIR, "rec_prec_ap.csv"))

                ## Write results for each class in a CSV
                udr_udp_results: List[UdrUdpResult] = self.evaluator.udr_udp_results
                if len(udr_udp_results) > 0:
                    header_udr_udp = udr_udp_results[0].__dict__.keys()
                    udr_udp = [result.__dict__ for result in udr_udp_results]
                    write_list_json_as_csv(data=udr_udp, header=header_udr_udp,
                                           filepath=os.path.join(self.cfg.OUTPUT_DIR, "udr_udp.csv"))
            else:
                if self.load_predictions:
                    self.evaluator.all_predictions = load_object(os.path.join(model_out_dir, "predictions.pkl"))
                eval_info = self.evaluator.evaluate_wic()
                Recalls_to_process = (0.1, 0.3, 0.5)
                wilderness = torch.arange(0, 5, 0.1).tolist()
                WIC_precision_values, wilderness_processed = self.evaluator.wic_analysis(
                    eval_info,
                    Recalls_to_process,
                    wilderness
                )
                save_object(WIC_precision_values.to("cpu"), filename=os.path.join(self.cfg.OUTPUT_DIR, "wic_precision_values.pkl"))
                save_object(wilderness_processed.to("cpu"), filename=os.path.join(self.cfg.OUTPUT_DIR, "wilderness_processed.pkl"))


    def test_with_precomputed_results(self):
        if comm.is_main_process():
            ##Now evaluate
            self.evaluator.evaluate_with_precomputed_results()

            ## Create output dir
            PathManager.mkdirs(self.cfg.OUTPUT_DIR)
            ## Write results for each class in a CSV
            rec_prec_ap_results: List[RecPrecAp] = self.evaluator.rec_prec_ap_results
            header_rec_prec_ap = rec_prec_ap_results[0].__dict__.keys()
            rec_prec_ap = [result.__dict__ for result in rec_prec_ap_results]
            write_list_json_as_csv(data=rec_prec_ap, header=header_rec_prec_ap,
                                   filepath=os.path.join(self.cfg.OUTPUT_DIR, "rec_prec_ap.csv"))

            ## Write results for each class in a CSV
            udr_udp_results: List[UdrUdpResult] = self.evaluator.udr_udp_results
            if len(udr_udp_results) > 0:
                header_udr_udp = udr_udp_results[0].__dict__.keys()
                udr_udp = [result.__dict__ for result in udr_udp_results]
                write_list_json_as_csv(data=udr_udp, header=header_udr_udp,
                                       filepath=os.path.join(self.cfg.OUTPUT_DIR, "udr_udp.csv"))

class DatasetTester(TestBase):
    def __init__(self, cfg, detector: Detector, distributed_detector, data_loader, evaluator, dataset_name, only_wic: bool = False):
        super().__init__()

        self.cfg = cfg
        self.detector = detector
        self.distributed_detector = distributed_detector

        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.evaluator: UnifiedDatasetEvaluator = evaluator
        self.dataset_name = dataset_name
        self.only_wic = only_wic

    def apply_stack_context(self, stack: ExitStack):
        if isinstance(self.detector, torch.nn.Module):
            stack.enter_context(inference_context(self.detector))
        stack.enter_context(torch.no_grad())

    def run_step(self):
        inputs = next(self._data_loader_iter)
        outputs = self.distributed_detector(inputs)
        if self.only_wic:
            self.evaluator.process_wic(inputs=inputs, outputs=outputs, dataset_name=self.dataset_name)
        else:
            self.evaluator.process(inputs=inputs, outputs=outputs, dataset_name=self.dataset_name)


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
