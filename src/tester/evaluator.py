import contextlib
import io
import json
import logging
from typing import List, Tuple, Dict

import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch

from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, BoxMode, Boxes, pairwise_iou
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.evaluator import DatasetEvaluator
from src.structures.metrics import DetectionMetrics
from src.utils.utils import get_max_iou, save_object, load_object


class VOCMetricsEvaluator:
    @staticmethod
    @lru_cache(maxsize=None)
    def parse_rec(filename):
        """Parse a PASCAL VOC xml file."""
        with PathManager.open(filename) as f:
            tree = ET.parse(f)
        objects = []
        for obj in tree.findall("object"):
            obj_struct = {}
            obj_struct["name"] = obj.find("name").text
            obj_struct["pose"] = obj.find("pose").text
            obj_struct["truncated"] = int(obj.find("truncated").text)
            obj_struct["difficult"] = int(obj.find("difficult").text)
            bbox = obj.find("bndbox")
            obj_struct["bbox"] = [
                int(bbox.find("xmin").text),
                int(bbox.find("ymin").text),
                int(bbox.find("xmax").text),
                int(bbox.find("ymax").text),
            ]
            objects.append(obj_struct)

        return objects

    @staticmethod
    def ap(rec, prec, use_07_metric=False):
        """Compute VOC AP given precision and recall. If use_07_metric is true, uses
        the VOC 07 11-point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.0
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([0.0], prec, [0.0]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


class MSPAnomalyEvaluator:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    """
    Args:
        - detection_instances: Output of the detector
        - gt_instances: GT assosiated to each image

    Behaviour: 
        The goal is to compute, for each image, two equal-sized arrays: y_pred, y_true. In this way we can compute AUROC SCORE.
        It matches each detected box to the most overlapped ground truth box. Then, to assign a prediction to 
        the detected box we distinguish three cases:
        - If the maximum overlap is below iou_threshold, then the match does not exist, 
          therefore the predicted class will be none of the known classes.
        - If the maximum overlap is above iou_threshold, the predicted class will be the class predicted by 
          the detector for that box.



    If the number of gt_boxes and detected boxes are not equal, then y_pred and y_true are padded with a value corresponding to none of the known classes.


    Returns:
        - y_pred (N,): class predictions made by the model for each image
        - y_true (N,): ground truth classes
    """

    def align_gt_with_detection(self, detection_instances, gt_instances, iou_thresh: float = 0.5):
        dt_boxes = detection_instances.pred_boxes.tensor.numpy().astype(float)
        dt_scores = detection_instances.scores
        dt_classes = detection_instances.pred_classes

        gt_boxes = gt_instances.gt_boxes.tensor.numpy().astype(float)
        gt_classes = gt_instances.gt_classes

        # Detect the size of the arrays to return
        max_size = max(len(dt_classes), len(gt_classes))

        # Initialize the prediction tensor
        y_pred = torch.full(size=max_size, fill_value=len(self.class_names))

        # Optionally, pad the ground truth tensor
        if len(gt_classes) < max_size:
            y_true = torch.cat(
                (gt_classes, torch.full(size=max_size - len(gt_classes), fill_value=len(self.class_names))))

        for idx in range(len(dt_boxes)):
            dt_box = dt_boxes[idx, :]
            maximum_iou_index = get_max_iou(boxes=gt_boxes, box=dt_box, thresh=iou_thresh)
            if maximum_iou_index != -1:
                y_pred[idx] = dt_classes[idx]

        self.y_pred = torch.cat((self.y_pred, y_pred))
        self.y_true = torch.cat((self.y_true, y_true))


class BoxImageGT:
    def __init__(self,
                 class_name: str,
                 difficult: int,
                 bbox: List[float],
                 class_id: int,
                 ann_id
                 ):
        self.class_name = class_name
        self.difficult = difficult
        self.bbox = bbox
        self.class_id = class_id
        self.ann_id = ann_id


class ClassImageGT:
    def __init__(self, bbox: np.array, difficult: np.array, det: List[bool]):
        self.bbox = bbox
        self.difficult = difficult
        self.det = det
        self.fn_unk_det = [False] * len(
            det)  # In order to calculate fn_unk for gt, as described by `Rethinking OW` paper. Used only for unknown class.


class ClassImagesGT:
    def __init__(self, npos: int, recs: Dict[str, ClassImageGT]):
        self.npos = npos
        self.recs = recs


class ImageToAnnotations:
    def __init__(self, bbox: np.array, difficult: np.array, det: List[bool], class_name: str, class_id: int):
        self.bbox = bbox
        self.difficult = difficult
        self.det = det
        self.class_name = class_name
        self.class_id = class_id


class PascalLoader:
    @staticmethod
    def load(dataset_name) -> Tuple[Dict[str, ClassImagesGT], Tuple[List[str], List[int]]]:
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        meta = MetadataCatalog.get(dataset_name)

        # Too many tiny files, download all to local for speed.
        annotation_dir_local = PathManager.get_local_path(
            os.path.join(meta.dirname, "Annotations/")
        )
        anno_file_template = os.path.join(annotation_dir_local, "{}.xml")
        image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        class_names = meta.thing_classes
        class_ids = meta.class_ids
        ##Load GT
        # first load gt
        # read list of images
        with PathManager.open(image_set_path, "r") as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        # load annots
        recs: Dict[str, List[BoxImageGT]] = {}
        for imagename in imagenames:
            recs[imagename] = PascalLoader.parse_rec(anno_file_template.format(imagename), class_names)

        # extract gt objects for each class
        classes_recs: Dict[str, ClassImagesGT] = {}
        for class_name in class_names:
            npos = 0
            class_recs: Dict[str, ClassImageGT] = {}
            for imagename in imagenames:
                R = [obj for obj in recs[imagename] if obj.class_name == class_name]
                bbox = np.array([x.bbox for x in R])
                difficult = np.array([x.difficult for x in R]).astype(np.bool)
                # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
                det = [False] * len(R)
                class_image_gt = ClassImageGT(bbox=bbox, difficult=difficult, det=det)
                npos = npos + sum(~difficult)
                class_recs[imagename] = class_image_gt
            class_images_gt = ClassImagesGT(npos=npos, recs=class_recs)
            classes_recs[class_name] = class_images_gt

        return classes_recs, (class_names, class_ids), recs

    @staticmethod
    def parse_rec(filename, class_names: List[str]) -> List[BoxImageGT]:
        """Parse a PASCAL VOC xml file."""
        with PathManager.open(filename) as f:
            tree = ET.parse(f)
        objects: List[BoxImageGT] = []
        count = 0
        for obj in tree.findall("object"):
            class_name = obj.find("name").text
            difficult = int(obj.find("difficult").text)
            bbox_struct = obj.find("bndbox")
            bbox = [
                float(bbox_struct.find("xmin").text),
                float(bbox_struct.find("ymin").text),
                float(bbox_struct.find("xmax").text),
                float(bbox_struct.find("ymax").text),
            ]
            box_image_gt = BoxImageGT(class_name=class_name, difficult=difficult, bbox=bbox,
                                      class_id=class_names.index(class_name), ann_id=count)
            count += 1
            objects.append(box_image_gt)

        return objects


class CocoLoader:
    """
    Evaluate COCO 2017 as single unknown class
    """

    @staticmethod
    def load(dataset_name) -> Tuple[Dict[str, ClassImagesGT], Tuple[List[str], List[int]]]:
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "COCO_2017_TEST"
        """
        dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        class_names = meta.thing_classes
        assert class_names == ["unknown"]
        class_ids = meta.class_ids
        from pycocotools.coco import COCO
        json_file = PathManager.get_local_path(meta.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)

        # sort indices for reproducible results
        img_ids = sorted(coco_api.imgs.keys())
        # imgs is a list of dicts, each looks something like:
        # {'license': 4,
        #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
        #  'file_name': 'COCO_val2014_000000001268.jpg',
        #  'height': 427,
        #  'width': 640,
        #  'date_captured': '2013-11-17 05:57:24',
        #  'id': 1268}
        imgs = coco_api.loadImgs(img_ids)
        # anns is a list[list[dict]], where each dict is an annotation
        # record for an object. The inner list enumerates the objects in an image
        # and the outer list enumerates over images. Example of anns[0]:
        # [{'segmentation': [[192.81,
        #     247.09,
        #     ...
        #     219.03,
        #     249.06]],
        #   'area': 1035.749,
        #   'iscrowd': 0,
        #   'image_id': 1268,
        #   'bbox': [192.81, 224.8, 74.73, 33.43],
        #   'category_id': 16,
        #   'id': 42986},
        #  ...]
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        imgs_anns = list(zip(imgs, anns))

        # load annots
        recs: Dict[str, List[BoxImageGT]] = {}
        for (img_dict, anno_dict_list) in imgs_anns:
            box_images_gt: List[BoxImageGT] = []
            for anno in anno_dict_list:
                if anno.get("iscrowd", 0) == 1:
                    continue

                box_image_gt = BoxImageGT(class_name="unknown", difficult=0, bbox=anno["bbox"], class_id=class_ids[0],
                                          ann_id=anno.get("id"))
                box_images_gt.append(box_image_gt)
            if len(box_images_gt) > 0:
                recs[img_dict["id"]] = box_images_gt

        # extract gt objects for each class (in this case only unknown, loop can be avoided)
        classes_recs: Dict[str, ClassImagesGT] = {}
        for class_name in class_names:
            npos = 0
            class_recs: Dict[str, ClassImageGT] = {}
            for img in imgs:
                image_id = img["id"]
                if recs.get(image_id, None) is None:
                    continue
                R = [obj for obj in recs[image_id] if obj.class_name == class_name]
                bbox = np.array([x.bbox for x in R])
                difficult = np.array([x.difficult for x in R]).astype(np.bool)
                # difficult = np.array([False for x in R]).astype(np.bool)  # avoid to treat all "difficult" as GT
                det = [False] * len(R)
                class_image_gt = ClassImageGT(bbox=bbox, difficult=difficult, det=det)
                npos = npos + sum(~difficult)
                class_recs[image_id] = class_image_gt
            class_images_gt = ClassImagesGT(npos=npos, recs=class_recs)
            classes_recs[class_name] = class_images_gt

        return classes_recs, (class_names, class_ids), recs


def eval(detpath, class_name, class_images_gt: ClassImagesGT, class_images_gt_unk: ClassImagesGT, unk_cls_name, mode,
         ovthresh=0.5):
    # assumes detections are in detpath.format(classname)

    # read dets
    detfile = detpath.format(class_name)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    padded_image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort image ids by confidence. There can be multiple equal image ids since 2 detections can be in the same image
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    padded_image_ids = [padded_image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(padded_image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    is_unk = np.zeros(nd)
    for d in range(nd):
        splitted_image_id = padded_image_ids[d].split("_")
        padding = splitted_image_id[0]
        image_id = splitted_image_id[1]
        if padding == "COCO":
            image_id = int(image_id)

        ovmax = -np.inf
        if mode == "open" or (mode == "open_cwwr" and padding == "COCO") and class_name != unk_cls_name:
            R_unk: ClassImageGT = class_images_gt_unk.recs.get(image_id, None)
            ovmax_unk = -np.inf
            if R_unk is not None:
                bb = BB[d, :].astype(float)
                BBGT_unk = R_unk.bbox.astype(float)
                if (len(R_unk.fn_unk_det) == sum(
                        R_unk.fn_unk_det)):  # Avoid to calculate intersection when all detections are discovered
                    BBGT_unk = np.array([])
            else:
                BBGT_unk = np.array([])

            if BBGT_unk.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT_unk[:, 0], bb[0])
                iymin = np.maximum(BBGT_unk[:, 1], bb[1])
                ixmax = np.minimum(BBGT_unk[:, 2], bb[2])
                iymax = np.minimum(BBGT_unk[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                ih = np.maximum(iymax - iymin + 1.0, 0.0)
                inters = iw * ih

                # union
                uni = (
                        (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                        + (BBGT_unk[:, 2] - BBGT_unk[:, 0] + 1.0) * (BBGT_unk[:, 3] - BBGT_unk[:, 1] + 1.0)
                        - inters
                )

                overlaps = inters / uni
                ovmax_unk = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax_unk > ovthresh:
                is_unk[d] = 1.0
                if not R_unk.difficult[jmax]:
                    if not R_unk.fn_unk_det[jmax]:
                        R_unk.fn_unk_det[jmax] = True
        if not (mode == "open" and class_name != unk_cls_name):
            R = class_images_gt.recs.get(image_id, None)
            if R is not None:
                bb = BB[d, :].astype(float)
                BBGT = R.bbox.astype(float)
            else:
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
                if not R.difficult[jmax]:
                    if not R.det[jmax]:
                        tp[d] = 1.0
                        R.det[jmax] = 1
                    else:
                        fp[d] = 1.0
            else:
                fp[d] = 1.0

    if not (mode == "open" and class_name != unk_cls_name):
        # compute metrics
        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)
        rec = tp_cumsum / float(class_images_gt.npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
        ap = DetectionMetrics.ap(rec=rec, prec=prec)

        num_tp = np.sum(tp)
        num_fp = np.sum(fp)
        recall = num_tp / class_images_gt.npos
        precision = num_tp / max((num_tp + num_fp), np.finfo(np.float64).eps)
        is_unk_cum = np.cumsum(is_unk)
    if mode == "open" and class_name == unk_cls_name:
        ## Calculate fn_unk_gt
        fn_unk_gt: int = 0
        for _, class_image_gt in class_images_gt_unk.recs.items():
            fn_unk_gt += sum(class_image_gt.fn_unk_det)
        udr = (num_tp + fn_unk_gt) / class_images_gt.npos
        udp = num_tp / (num_tp + fn_unk_gt)
        return recall, precision, ap, udr, udp, None, None, None

    if mode == "open" and class_name != unk_cls_name:
        return None, None, None, None, None, None, None, None

    if mode == "open_cwwr" and class_name == unk_cls_name:
        ## Calculate fn_unk_gt
        fn_unk_gt: int = 0
        for _, class_image_gt in class_images_gt_unk.recs.items():
            fn_unk_gt += sum(class_image_gt.fn_unk_det)
        udr = (num_tp + fn_unk_gt) / class_images_gt.npos
        udp = num_tp / (num_tp + fn_unk_gt)
        return recall, precision, ap, udr, udp, tp_cumsum, fp_cumsum, is_unk_cum, rec, np.sum(is_unk)

    return recall, precision, ap, None, None, tp_cumsum, fp_cumsum, is_unk_cum, rec, np.sum(is_unk)


class RecPrecAp:
    def __init__(self, class_name: str, rec: float, prec: float, ap: float):
        self.class_name = class_name
        self.rec = rec.item()
        self.prec = prec.item()
        self.ap = ap.item()


class UdrUdpResult:
    def __init__(self, udr: float, udp: float):
        self.udr = udr.item()
        self.udp = udp.item()


class UnifiedDatasetEvaluator(DatasetEvaluator):

    def __init__(self, voc_dataset_name: str, coco_dataset_name, mode: str, out_dir=None, thresh=50):
        self.thresh = thresh
        self.filename_unk_gt = "unk_classes_images.pkl"
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._predictions = defaultdict(list)
        self.mode = mode
        self.out_dir = out_dir
        self.voc_images_to_ann: Dict[str, List[BoxImageGT]]
        self.coco_images_to_ann: Dict[str, List[BoxImageGT]]
        self.voc_classes_images_gt: Dict[str, ClassImagesGT]
        self.unk_classes_images_gt: Dict[str, ClassImagesGT]
        self.wi = None
        self.wi_adjusted = None
        assert mode == "cwwr" or mode == "open" or mode == "open_cwwr"
        assert (voc_dataset_name is not None and coco_dataset_name is not None)
        assert out_dir is not None
        self.folder_model_results = os.path.join(out_dir, "model_out")
        PathManager.mkdirs(self.folder_model_results)
        if mode == "cwwr":
            classes_images_gt, class_identifiers, self.voc_images_to_ann = PascalLoader.load(voc_dataset_name)
            self.voc_classes_images_gt: Dict[str, ClassImagesGT] = classes_images_gt
            self.voc_classes_identifiers: Tuple[List[str], List[int]] = class_identifiers

            self._classnames = self.voc_classes_identifiers[0]
            self._classids = self.voc_classes_identifiers[1]

            self.unk_classes_images_gt: Dict[str, ClassImagesGT] = None
            meta_coco = MetadataCatalog.get(coco_dataset_name)
            self.unk_classes_identifiers: Tuple[List[str], List[int]] = (meta_coco.thing_classes, meta_coco.class_ids)
        elif mode == "open":
            classes_images_gt, class_identifiers, self.coco_images_to_ann = CocoLoader.load(coco_dataset_name)
            self.unk_classes_images_gt: Dict[str, ClassImagesGT] = classes_images_gt
            self.unk_classes_identifiers: Tuple[List[str], List[int]] = class_identifiers
            meta_voc = MetadataCatalog.get(voc_dataset_name)
            self.voc_classes_identifiers: Tuple[List[str], List[int]] = (meta_voc.thing_classes, meta_voc.class_ids)
            self._classnames = self.voc_classes_identifiers[0] + self.unk_classes_identifiers[0]
            self._classids = self.voc_classes_identifiers[1] + self.unk_classes_identifiers[1]
        else:
            self.unk_classes_images_gt, self.unk_classes_identifiers, self.coco_images_to_ann = CocoLoader.load(
                coco_dataset_name)
            self.voc_classes_images_gt, self.voc_classes_identifiers, self.voc_images_to_ann = PascalLoader.load(
                voc_dataset_name)
            self._classnames = self.voc_classes_identifiers[0] + self.unk_classes_identifiers[0]
            self._classids = self.voc_classes_identifiers[1] + self.unk_classes_identifiers[1]

        self.rec_prec_ap_results: List[RecPrecAp] = []
        self.udr_udp_results: List[UdrUdpResult] = []

    def process(self, inputs, outputs, dataset_name: str):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                if "VOC" in dataset_name:
                    # The inverse of data loading logic in `datasets/pascal_voc.py`
                    xmin += 1
                    ymin += 1
                    padding = "VOC"
                else:
                    padding = "COCO"
                new_image_id = "{}_{}".format(padding, image_id)
                self._predictions[cls].append(
                    f"{new_image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

    def collect_predictions(self):
        self.all_predictions = comm.gather(self._predictions, dst=0)

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """

        predictions = defaultdict(list)
        for predictions_per_rank in self.all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del self.all_predictions

        unk_cls_name = self.unk_classes_identifiers[0][0]
        dirname = self.folder_model_results
        res_file_template = os.path.join(dirname, "{}.txt")
        tp_plus_fp_cs_cum = []
        all_recs = []
        a_ose = []
        tp_css_cum = []
        fp_css_cum = []
        tp_oss_cum = []
        fp_oss_cum = []
        for cls_id, cls_name in zip(self._classids, self._classnames):
            lines = predictions.get(cls_id, [""])

            with open(res_file_template.format(cls_name), "w") as f:
                f.write("\n".join(lines))

            if self.mode == "open":
                class_images_gt = self.unk_classes_images_gt[unk_cls_name]
            elif cls_name == unk_cls_name:
                class_images_gt = self.unk_classes_images_gt[unk_cls_name]
            else:
                class_images_gt = self.voc_classes_images_gt[cls_name]
            rec, prec, ap, udr, udp, tp_cum, fp_cum, fp_os_cum, rec_cum, a_ose_cls = eval(
                detpath=res_file_template.format(cls_name),
                class_images_gt=class_images_gt,
                class_name=cls_name,
                class_images_gt_unk=self.unk_classes_images_gt[unk_cls_name],
                unk_cls_name=unk_cls_name,
                mode=self.mode,
                ovthresh=self.thresh / 100.0
            )
            if rec is not None:
                self.rec_prec_ap_results.append(RecPrecAp(class_name=cls_name, rec=rec, prec=prec, ap=ap))
                all_recs.append(rec_cum)
                a_ose.append((a_ose_cls, len(rec_cum)))
            if udr is not None and udp is not None:
                self.udr_udp_results.append(UdrUdpResult(udr=udr, udp=udp))

            if cls_name != unk_cls_name:
                tp_plus_fp_cs_cum.append(tp_cum + fp_cum)
                fp_oss_cum.append(fp_os_cum)
                tp_css_cum.append(tp_cum)
                fp_css_cum.append(fp_cum)
            else:
                tp_oss_cum.append(tp_cum)
        if self.mode == "open_cwwr":
            self.wi = self.compute_WI_at_many_recall_level(all_recs, tp_plus_fp_cs_cum, fp_oss_cum)
            self.wi_adjusted = self.compute_WI_adjusted_at_many_recall_level(all_recs, tp_css_cum, fp_css_cum, tp_oss_cum, fp_oss_cum)
            save_object(self.wi, os.path.join(self.out_dir, "wi.pkl"))
            save_object(self.wi_adjusted, os.path.join(self.out_dir, "wi_adjusted.pkl"))
            save_object(a_ose, os.path.join(self.out_dir, "a_ose.pkl"))

    def evaluate_with_precomputed_results(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """

        res_file_template = os.path.join(self.folder_model_results, "{}.txt")
        unk_cls_name = self.unk_classes_identifiers[0][0]
        tp_plus_fp_cs_cum = []
        all_recs = []
        a_ose = []
        tp_css_cum = []
        fp_css_cum = []
        tp_oss_cum = []
        fp_oss_cum = []
        for cls_id, cls_name in zip(self._classids, self._classnames):
            if self.mode == "open":
                class_images_gt = self.unk_classes_images_gt[unk_cls_name]
            elif cls_name == unk_cls_name:
                class_images_gt = self.unk_classes_images_gt[unk_cls_name]
            else:
                class_images_gt = self.voc_classes_images_gt[cls_name]
            rec, prec, ap, udr, udp, tp_cum, fp_cum, fp_os_cum, rec_cum, a_ose_cls = eval(
                detpath=res_file_template.format(cls_name),
                class_images_gt=class_images_gt,
                class_name=cls_name,
                class_images_gt_unk=self.unk_classes_images_gt[unk_cls_name],
                unk_cls_name=unk_cls_name,
                mode=self.mode,
                ovthresh=self.thresh / 100.0
            )
            if rec is not None:
                self.rec_prec_ap_results.append(RecPrecAp(class_name=cls_name, rec=rec, prec=prec, ap=ap))
                all_recs.append(rec_cum)
                a_ose.append((a_ose_cls, len(rec_cum)))
            if udr is not None and udp is not None:
                self.udr_udp_results.append(UdrUdpResult(udr=udr, udp=udp))

            if cls_name != unk_cls_name:
                tp_plus_fp_cs_cum.append(tp_cum + fp_cum)
                fp_oss_cum.append(fp_os_cum)
                tp_css_cum.append(tp_cum)
                fp_css_cum.append(fp_cum)
            else:
                tp_oss_cum.append(tp_cum)
        if self.mode == "open_cwwr":
            self.wi = self.compute_WI_at_many_recall_level(all_recs, tp_plus_fp_cs_cum, fp_oss_cum)
            self.wi_adjusted = self.compute_WI_adjusted_at_many_recall_level(all_recs, tp_css_cum, fp_css_cum, tp_oss_cum, fp_oss_cum)
            save_object(self.wi, os.path.join(self.out_dir, "wi.pkl"))
            save_object(self.wi_adjusted, os.path.join(self.out_dir, "wi_adjusted.pkl"))
            save_object(a_ose, os.path.join(self.out_dir, "a_ose.pkl"))

    def compute_WI_at_many_recall_level(self, recalls, tp_plus_fp_cs, fp_os):
        wi_at_recall = {}
        for r in range(1, 10):
            r = r / 10
            wi = self.compute_WI_at_a_recall_level(recalls, tp_plus_fp_cs, fp_os, recall_level=r)
            wi_at_recall[r] = wi
        return wi_at_recall

    def compute_WI_at_a_recall_level(self, recalls, tp_plus_fp_cs, fp_os, recall_level=0.5):
        tp_plus_fps = []
        fps = []
        for cls_id, rec in enumerate(recalls[:-1]):
            index = min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))
            tp_plus_fp = tp_plus_fp_cs[cls_id][index]
            tp_plus_fps.append(tp_plus_fp)
            fp = fp_os[cls_id][index]
            fps.append(fp)
        if len(tp_plus_fps) > 0:
            wi = np.mean(fps) / np.mean(tp_plus_fps)
        else:
            wi = 0
        return wi

    def compute_WI_adjusted_at_many_recall_level(self, recalls, tp_cs, fp_cs, tp_os, fp_os):
        wi_at_recall = {}
        for r in range(1, 10):
            r = r / 10
            wi = self.compute_WI_adjusted_at_a_recall_level(recalls, tp_cs, fp_cs, tp_os, fp_os, recall_level=r)
            wi_at_recall[r] = wi
        return wi_at_recall

    def compute_WI_adjusted_at_a_recall_level(self, recalls, tp_cs, fp_cs, tp_os, fp_os, recall_level=0.5):
        wi_adj = []
        index_tp_os = min(range(len(recalls[-1])), key=lambda i: abs(recalls[-1][i] - recall_level))
        for cls_id, rec in enumerate(recalls[:-1]):
            index = min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))
            tp_cs_cls = tp_cs[cls_id][index]
            fp_cs_cls = fp_cs[cls_id][index]
            tp_os_cls = tp_os[0][index_tp_os]
            fp_os_cls = fp_os[cls_id][index]

            wi_adj_cls = tp_cs_cls / (tp_cs_cls + fp_cs_cls) * (fp_os_cls + fp_cs_cls) / (tp_cs_cls + tp_os_cls)
            wi_adj.append(wi_adj_cls)
        if len(wi_adj) > 0:
            wi = np.mean(wi_adj)
        else:
            wi = 0
        return wi