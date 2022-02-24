# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from typing import Dict, List, Tuple, Union
import torch
from detectron2.data import MetadataCatalog
from detectron2.modeling import FastRCNNOutputLayers
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
import numpy as np

__all__ = ["MyFastRCNNOutputLayers"]

from src.utils.utils import swap, one_hot_matrix

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def _log_classification_stats(pred_logits, gt_classes, prefix="fast_rcnn"):
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)


class ClsLoss:
    def __init__(self, loss: str, n_classes: int):
        if loss != "ce" and loss != "bce":
            raise Exception("Loss not recognized. It can be 'ce' or 'bce'. Passed: {}".format(loss))
        self.loss = loss
        if self.loss == "bce":
            self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.n_classes = n_classes

    def compute_loss(self, activation: torch.Tensor, labels: torch.Tensor):
        if self.loss == "ce":
            return cross_entropy(activation, labels, reduction="mean")
        else:
            onehot = one_hot_matrix(labels=labels, n_classes=self.n_classes)
            return self.bce(activation, onehot)


class PostTrainFastRCNNOutputLayers(nn.Module):
    """
       Two linear layers for predicting Fast R-CNN outputs:

       1. proposal-to-detection box regression deltas
       2. classification scores
       """

    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            box2box_transform,
            num_classes: int,
            test_score_thresh: float = 0.0,
            test_nms_thresh: float = 0.5,
            test_topk_per_image: int = 100,
            device: str,
            smooth_l1_beta: float = 0.0,
            box_reg_loss_type: str = "smooth_l1",
            loss_weight: Union[float, Dict[str, float]] = 1.0,
            mode: str,
            unknown_regression: bool,
            is_test: bool = False
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou",
                "diou", "ciou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        self.mode = mode
        self.is_test = is_test
        self.unknown_regression = unknown_regression
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        box_dim = len(box2box_transform.weights)
        self.num_bbox_reg_classes = num_classes
        self.num_ll_classes = num_classes + 1  # known classes + bg
        if self.mode == "ours":
            # In this case we model fg unknown class too (hence + 1)
            self.num_ll_classes += 1
            if self.unknown_regression:
                # In this case we calculate a box for fg unknown class too
                self.num_bbox_reg_classes = num_classes + 1

        self.cls_score = nn.Linear(input_size, self.num_ll_classes)
        self.bbox_pred = nn.Linear(input_size, self.num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight
        self.device = device
        # Indices to swap bkg column with fg unknown column
        self.indices_swap_bk_fgunk = torch.LongTensor(list(range(self.num_classes + 2))).to(self.device)
        self.indices_swap_bk_fgunk[self.num_classes] = self.num_classes + 1
        self.indices_swap_bk_fgunk[self.num_classes + 1] = self.num_classes

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "device": cfg.MODEL.DEVICE,
            "test_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight": {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            "mode": cfg.OURS.MODE,
            "unknown_regression": cfg.OURS.UNKNOWN_REGRESSION,
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return self.fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_boxes(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # N x B
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # N x (K x B)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        return scores.split(num_inst_per_image, dim=0)

    def fast_rcnn_inference(
            self,
            boxes: List[torch.Tensor],
            scores: List[torch.Tensor],
            image_shapes: List[Tuple[int, int]],
            score_thresh: float,
            nms_thresh: float,
            topk_per_image: int,
    ):
        """
        Call `fast_rcnn_inference_single_image` for all images.

        Args:
            boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
                boxes for each image. Element i has shape (Ri, K * 4) if doing
                class-specific regression, or (Ri, 4) if doing class-agnostic
                regression, where Ri is the number of predicted objects for image i.
                This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
            scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
            image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
            score_thresh (float): Only return detections with a confidence score exceeding this
                threshold.
            nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
            topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
                all detections.

        Returns:
            instances: (list[Instances]): A list of N instances, one for each image in the batch,
                that stores the topk most confidence detections.
            kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
                the corresponding boxes/scores index in [0, Ri) from the input, for image i.
        """
        result_per_image = [
            self.fast_rcnn_inference_single_image(
                boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
            )
            for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
            if len(scores_per_image) > 0
        ]
        return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

    def fast_rcnn_inference_single_image(
            self,
            boxes,
            scores,
            image_shape: Tuple[int, int],
            score_thresh: float,
            nms_thresh: float,
            topk_per_image: int,
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Args:
            Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
            per image.

        Returns:
            Same as `fast_rcnn_inference`, but for only one image.
        """
        valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]

        max_activation = torch.max(scores, dim=1).values
        is_bkg = scores[:, self.num_classes] == max_activation
        mask_bkg = torch.logical_not(is_bkg)
        scores = scores[mask_bkg]

        scores = scores[:, :-2]
        num_bbox_reg_classes = boxes.shape[1] // 4
        # Convert to Boxes to use the `clip` function ...
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip(image_shape)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
        boxes = boxes[mask_bkg]

        # 1. Filter results based on detection scores. It can make NMS more efficient
        #    by filtering out low-confidence detections.
        filter_mask = torch.sigmoid(scores) > score_thresh  # R x K
        # R' x 2. First column contains indices of the R predictions;
        # Second column contains indices of classes.
        filter_inds = filter_mask.nonzero()
        if num_bbox_reg_classes == 1:
            boxes = boxes[filter_inds[:, 0], 0]
        else:
            boxes = boxes[filter_mask]
        scores = scores[filter_mask]

        # 2. Apply NMS for each class independently.
        keep = batched_nms(boxes, torch.sigmoid(scores), filter_inds[:, 1], nms_thresh)
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = filter_inds[:, 1]
        return result, filter_inds[:, 0]


def load_cls_activation_metrics(path: str, class_names: List[str], class_ids: List[int], device="cuda"):
    means = torch.empty(len(class_names))
    stds = torch.empty(len(class_names))
    mean_scores = torch.empty(len(class_names))
    for cls_name, cls_id in zip(class_names, class_ids):
        means[cls_id] = torch.load(os.path.join(path, cls_name + "_mean.pt"))
        mean_scores[cls_id] = torch.load(os.path.join(path, cls_name + "_mean_score.pt"))
        stds[cls_id] = torch.load(os.path.join(path, cls_name + "_std.pt"))
    return means.to(device), stds.to(device), mean_scores.to(device)


class MyFastRCNNOutputLayers(nn.Module):
    """
       Two linear layers for predicting Fast R-CNN outputs:

       1. proposal-to-detection box regression deltas
       2. classification scores
       """

    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            box2box_transform,
            num_classes: int,
            test_score_thresh: float = 0.0,
            test_nms_thresh: float = 0.5,
            test_topk_per_image: int = 100,
            device: str,
            smooth_l1_beta: float = 0.0,
            box_reg_loss_type: str = "smooth_l1",
            loss_weight: Union[float, Dict[str, float]] = 1.0,
            mode: str,
            unknown_regression: bool,
            use_msp: bool,
            cls_loss: str,
            fg_percentile: float,
            std_percentile: int = -1,
            is_test: bool = False,
            fg_rejection: bool = False,
            unk_score_rejection: bool = False
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou",
                "diou", "ciou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        self.mode = mode
        self.is_test = is_test
        self.unknown_regression = unknown_regression
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        box_dim = len(box2box_transform.weights)
        self.num_bbox_reg_classes = num_classes
        self.num_ll_classes = num_classes + 1  # known classes + bg
        if self.mode == "ours":
            # In this case we model fg unknown class too (hence + 1)
            self.num_ll_classes += 1
            if self.unknown_regression:
                # In this case we calculate a box for fg unknown class too
                self.num_bbox_reg_classes = num_classes + 1

        self.cls_score = nn.Linear(input_size, self.num_ll_classes)
        self.bbox_pred = nn.Linear(input_size, self.num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight
        self.device = device
        # Indices to swap bkg column with fg unknown column
        self.indices_swap_bk_fgunk = torch.LongTensor(list(range(self.num_classes + 2))).to(self.device)
        self.indices_swap_bk_fgunk[self.num_classes] = self.num_classes + 1
        self.indices_swap_bk_fgunk[self.num_classes + 1] = self.num_classes
        self.use_msp = use_msp
        self.cls_loss = ClsLoss(cls_loss, self.num_ll_classes)
        self.fg_percentile = fg_percentile
        if self.mode == "ours" and not self.use_msp and self.is_test:
            if self.fg_percentile == -1:
                raise Exception("fg_percentile cannot be -1 in testing mode")
        self.std_percentile = std_percentile
        self.fg_rejection = fg_rejection
        self.unk_score_rejection = unk_score_rejection
        if self.mode == "ours" and not self.use_msp and self.is_test:
            if self.fg_rejection:
                assert self.fg_percentile != -1
            assert self.fg_rejection or self.unk_score_rejection
    @classmethod
    def from_config(cls, cfg, input_shape):
        is_test = cfg.OURS.IS_TEST
        mode = cfg.OURS.MODE
        use_msp = cfg.OURS.USE_MSP


        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "device": cfg.MODEL.DEVICE,
            "test_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight": {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            "mode": mode,
            "unknown_regression": cfg.OURS.UNKNOWN_REGRESSION,
            "use_msp": cfg.OURS.USE_MSP,
            "cls_loss": cfg.OURS.MODELS.RCNN.CLS_LOSS,
            "fg_percentile": cfg.OURS.MODELS.RPN.FG_PERCENTILE,
            "is_test": is_test,
            "fg_rejection": cfg.OURS.REJECTION.FG,
            "unk_score_rejection": cfg.OURS.REJECTION.UNK_SCORE
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            if self.unknown_regression:
                gt_boxes = cat(
                    [cat((p.gt_boxes.tensor, p.proposal_boxes[p.gt_classes == self.num_classes + 1].tensor),
                         dim=0) if p.has("gt_boxes") else p.proposal_boxes.tensor for p in proposals],
                    dim=0,
                )
            else:
                gt_boxes = cat(
                    [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                    dim=0,
                )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls": self.cls_loss.compute_loss(scores, gt_classes),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            proposal_boxes/gt_boxes are tensors with the same shape (R, 4 or 5).
            pred_deltas has shape (R, 4 or 5), or (R, num_classes * (4 or 5)).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Clone in order to not affect gradients
        gt_classes_cloned = torch.clone(gt_classes)

        # Regression loss is only computed for foreground proposals (those matched to a GT)
        if self.unknown_regression:
            gt_classes_cloned = self.swap_gt_labels_array(gt_classes_cloned)
            fg_inds = nonzero_tuple((gt_classes_cloned >= 0) & (gt_classes_cloned <= self.num_classes))[0]
        else:
            fg_inds = nonzero_tuple((gt_classes_cloned >= 0) & (gt_classes_cloned < self.num_classes))[0]

        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_bbox_reg_classes, box_dim)[
                fg_inds, gt_classes_cloned[fg_inds]
            ]

        loss_box_reg = _dense_box_regression_loss(
            [proposal_boxes[fg_inds]],
            self.box2box_transform,
            [fg_pred_deltas.unsqueeze(0)],
            [gt_boxes[fg_inds]],
            ...,
            self.box_reg_loss_type,
            self.smooth_l1_beta,
        )

        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        objectness_logits = cat([p.objectness_logits for p in proposals], dim=0)
        objectness_logits = objectness_logits.split([len(p) for p in proposals])
        image_shapes = [x.image_size for x in proposals]
        return self.fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            objectness_logits
        )

    def predict_boxes(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # N x B
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # N x (K x B)

        ## If no unknown regression is made, the potential predicted box for the unknown is considered to be the proposal itself
        if not self.unknown_regression or self.mode is not "ours":
            predict_boxes = cat((predict_boxes, proposal_boxes), dim=1)  # N x ( (K+1) x B )
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]

        if self.mode == "ours":
            if self.use_msp:
                probs = F.softmax(scores[:, :-1], dim=1)

                ## Calculate MSP
                msp = torch.max(probs, dim=1).values
                probs = torch.cat((probs, msp.unsqueeze(dim=1)), dim=1)  # R x ( C + 2 )
            else:
                probs = scores
        else:
            probs = F.softmax(scores, dim=-1)
            msp = torch.max(probs, dim=1).values
            probs = torch.cat((probs, msp.unsqueeze(dim=1)), dim=1)  # R x ( C + 2 )

        return probs.split(num_inst_per_image, dim=0)

    def fast_rcnn_inference_single_image(
            self,
            boxes,
            scores,
            image_shape: Tuple[int, int],
            score_thresh: float,
            nms_thresh: float,
            topk_per_image: int,
            objectness_logits: torch.Tensor
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Args:
            Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
            per image.

        Returns:
            Same as `fast_rcnn_inference`, but for only one image.
        """
        valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]

        ## Scores shape: N x (C + 2)

        if self.mode == "ours" and self.use_msp:
            ## In this case the anomaly score is the last column itself, we only compute the mask for removing bkg detections.
            mask_bkg = scores[:, self.num_classes] != scores[:, -1]
            scores[:, -1] = 1 - scores[:, -1]
            max_softmax = torch.max(scores, dim=1).values
            is_unknown = scores[:, -1] == max_softmax
            not_unknown = torch.logical_not(is_unknown)

        elif self.mode == "ours":
            max_activation = torch.max(scores, dim=1).values
            is_unknown = torch.zeros(len(scores)).bool().to(self.device)

            # Se lo score massimo è unknown, allora sarà unknown.
            if self.unk_score_rejection:
                is_unknown = scores[:, self.num_classes + 1] == max_activation

            # Se la prob massima è background, allora si valuta se la RPN ha un objectness score più alta della media, se si è unknown.
            is_bkg = torch.logical_and(scores[:, self.num_classes] == max_activation,
                                       torch.logical_not(is_unknown))
            if self.fg_rejection:
                is_fg = objectness_logits > self.fg_percentile
                from_bkg_is_unknown = torch.logical_and(is_bkg, is_fg)
                is_bkg = torch.logical_and(is_bkg, torch.logical_not(is_fg))
                is_unknown = torch.logical_or(is_unknown, from_bkg_is_unknown)
            mask_bkg = torch.logical_not(is_bkg)
            # Avoid unknown to be selected
            not_unknown = torch.logical_not(is_unknown)
            scores[:, -1][not_unknown] = float("-inf")*torch.ones(len(not_unknown.nonzero(as_tuple=True)[0])).to(self.device)

            # Azzera gli altri in maniera tale che solo l'unknown viene selezionato
            if self.cls_loss.loss == "ce":
                scores = F.softmax(scores, dim=1)
            else:
                scores = torch.sigmoid(scores)
        elif self.mode == "baseline" and not self.use_msp:
            ## The first 21 columns contains softmax probability of 20 classes + bkg.
            ## The last column contains msp
            mask_bkg = scores[:, self.num_classes] != scores[:, -1]
            scores = scores[:, :-2]
            ##Remove last 4 coordinates indicating proposal coordinates, since no unknown exists hence are unuseful.
            boxes = boxes[:, :-4]
        elif self.mode == "baseline":
            ## The first 21 columns contains softmax probability of 20 classes + bkg.
            ## The last column contains msp
            mask_bkg = scores[:, self.num_classes] != scores[:, -1]
            scores[:, -1] = 1 - scores[:, -1]
            max_softmax = torch.max(scores, dim=1).values
            is_unknown = scores[:, -1] == max_softmax
            not_unknown = torch.logical_not(is_unknown)
        else:
            raise Exception("Training mode not recognized: {}".format(self.mode))

        ## Keep all rows on which background is not the maximum score
        #scores = scores[mask_bkg]


        # Swap the bkg and fg unk column, then remove bkg column in order to work with boxes
        if not (self.mode == "baseline" and not self.use_msp):
            # Azzera gli altri in maniera tale che solo l'unknown viene selezionato
            unknown_indices = is_unknown.nonzero(as_tuple=True)[0]
            scores[unknown_indices, :-1] = torch.zeros((len(unknown_indices), self.num_classes + 1)).to(self.device)
            not_unknown_indices = not_unknown.nonzero(as_tuple=True)[0]
            scores[not_unknown_indices, -1] = torch.zeros((len(not_unknown_indices))).to(self.device)
            scores = swap(scores, dim=1, idx=self.indices_swap_bk_fgunk)
            scores = scores[:, :-1]

        # Convert to Boxes to use the `clip` function ...
        num_bbox_reg_classes = boxes.shape[1] // 4
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip(image_shape)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x ( C + 1 ) x 4
        ## Remove all the rows on which background is the prediction
        #boxes = boxes[mask_bkg]
        filter_mask = scores > score_thresh  # R x K

        # R' x 2. First column contains indices of the R predictions;
        # Second column contains indices of classes.
        filter_inds = filter_mask.nonzero()
        boxes = boxes[filter_mask]
        scores = scores[filter_mask]

        # 2. Apply NMS for each class independently.
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores

        # Convert fg unk label to original value
        pred_classes = filter_inds[:, 1]
        pred_classes[pred_classes == self.num_classes] = self.num_classes + 1
        result.pred_classes = pred_classes
        return result, filter_inds[:, 0]

    def fast_rcnn_inference(
            self,
            boxes: List[torch.Tensor],
            scores: List[torch.Tensor],
            image_shapes: List[Tuple[int, int]],
            score_thresh: float,
            nms_thresh: float,
            topk_per_image: int,
            objectness_logits: List[torch.Tensor]
    ):
        """
        Call `fast_rcnn_inference_single_image` for all images.

        Args:
            boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
                boxes for each image. Element i has shape (Ri, K * 4) if doing
                class-specific regression, or (Ri, 4) if doing class-agnostic
                regression, where Ri is the number of predicted objects for image i.
                This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
            scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
            image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
            score_thresh (float): Only return detections with a confidence score exceeding this
                threshold.
            nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
            topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
                all detections.

        Returns:
            instances: (list[Instances]): A list of N instances, one for each image in the batch,
                that stores the topk most confidence detections.
            kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
                the corresponding boxes/scores index in [0, Ri) from the input, for image i.
        """
        result_per_image = [
            self.fast_rcnn_inference_single_image(
                boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image,
                objectness_logits_per_image
            )
            for scores_per_image, boxes_per_image, image_shape, objectness_logits_per_image in
            zip(scores, boxes, image_shapes, objectness_logits)
        ]
        return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

    def swap_gt_labels_array(self, gt_labels):
        bkg_mask = gt_labels == self.num_classes
        fgunk_mask = gt_labels == self.num_classes + 1
        gt_labels[bkg_mask] = self.num_classes + 1
        gt_labels[fgunk_mask] = self.num_classes
        return gt_labels
        """
        :param gt_labels: Array of length N containing int value, representing the label for the class 
        :return: 
        """
