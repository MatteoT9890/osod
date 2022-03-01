import logging
import numpy as np
from typing import Optional, Tuple, List, Dict
import torch
from detectron2.checkpoint import DetectionCheckpointer
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling import Backbone, build_backbone
from src.models.components.roi_heads import build_roi_heads
from detectron2.modeling import detector_postprocess
from detectron2.modeling import build_proposal_generator
from detectron2.modeling import META_ARCH_REGISTRY
from copy import deepcopy
from src.models.components.rpn import MyRPN


@META_ARCH_REGISTRY.register()
class RegionProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            proposal_generator: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
            vis_period: int = 0,
            post_training: bool = False,
            step: int
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.input_format = input_format
        self.vis_period = vis_period
        self.post_training = post_training
        self.step = step
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "post_training": cfg.OURS.POST_TRAINING,
            "step": cfg.OURS.STEP
        }

    @property
    def device(self):
        return self.pixel_mean.device

    @torch.no_grad()
    def generate_proposals(self, images: ImageList, gt_instances=None, features=None):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        if features is None:
            features = self.backbone(images.tensor)

        proposals, _ = self.proposal_generator(images, features, gt_instances)
        return proposals

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """

        images: ImageList = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        if self.post_training:
            return self.proposal_generator(images, features, gt_instances)

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_proposals(batched_inputs, proposals)

        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            if self.is_discovery:
                processed_results.append({"instances": r})
            else:
                processed_results.append({"proposals": r})
        return processed_results

    def preprocess_image(self, batched_inputs) -> ImageList:
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def visualize_proposals(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch


class FastRCNN(nn.Module):
    def __init__(self, backbone, roi_heads):
        super().__init__()
        self.backbone = backbone
        self.roi_heads = roi_heads

    @classmethod
    def init_from_config(cls, cfg):
        backbone = build_backbone(cfg)
        roi_heads = build_roi_heads(cfg, backbone.output_shape())
        return cls(backbone=backbone, roi_heads=roi_heads)

@META_ARCH_REGISTRY.register()
class Detector(nn.Module):
    """
        Generalized R-CNN. Any models that contains the following three components:
        1. Per-image feature extraction (aka backbone)
        2. Region proposal generation
        3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
            self,
            *,
            rcnn: FastRCNN,
            rpn: RegionProposalNetwork,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
            vis_period: int = 0,
            step: int
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.step = step
        self.rpn = rpn
        self.rcnn = rcnn
        self.odin = rcnn.roi_heads.box_predictor.use_odin
        self.gradnorm = rcnn.roi_heads.box_predictor.use_gradnorm

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        # Load pre-trained RPN
        rpn = RegionProposalNetwork(cfg)
        checkpointer_rpn = DetectionCheckpointer(rpn, cfg.OURS.CHECKPOINT_DIR_RPN)
        checkpointer_rpn.resume_or_load(cfg.OURS.MODELS.RPN.WEIGHTS, resume=False)
        rpn.eval()

        backbone = build_backbone(cfg)
        roi_heads = build_roi_heads(cfg, backbone.output_shape())
        rcnn = FastRCNN(backbone=backbone, roi_heads=roi_heads)
        return {
            "rcnn": rcnn,
            "rpn": rpn,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "step": cfg.OURS.STEP
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoi0.0014nts"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.rcnn.backbone(images.tensor)

        if self.step == 4:
            proposals = self.rpn.generate_proposals(images=images, gt_instances=gt_instances, features=features)
        else:
            proposals = self.rpn.generate_proposals(images=images, gt_instances=gt_instances, features=None)

        _, detector_losses = self.rcnn.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        return losses

    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        if self.odin:
            images.tensor.requires_grad = True
            for param in self.parameters():
                param.requires_grad = False

        if self.gradnorm:
            images.tensor.requires_grad = True
            for param in self.parameters():
                param.requires_grad = True

        self.zero_grad()

        features = self.rcnn.backbone(images.tensor)
        proposals = self.rpn.generate_proposals(images=images, gt_instances=None, features=features)
        results, _ = self.rcnn.roi_heads(images, features, proposals, None)
        if self.odin:
            results = self.repeat_inference(images, results)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return Detector._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def repeat_inference(
            self,
            images: ImageList,
            scores: List[torch.Tensor],
    ):
        """

        Parameters
        ----------
        scores
        scores : tensor
        """
        assert not self.training
        criterion = torch.nn.CrossEntropyLoss().to(images.device)
        scores = scores[0]
        before_perturbation_scores = deepcopy(scores.data)

        probs = scores - torch.max(scores, dim=1, keepdim=True)[0]
        probs = F.softmax(probs, dim=-1)
        labels = torch.argmax(probs, axis=1)

        scores = scores / 2.
        loss = criterion(scores, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(images.tensor.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Normalizing the gradient to the same space of image
        # fixme
        # Adding small perturbations to images
        images.tensor = torch.add(images.tensor.data, -0.0014, gradient)

        features = self.rcnn.backbone(images.tensor)
        proposals = self.rpn.generate_proposals(images=images, gt_instances=None, features=features)
        results, _ = self.rcnn.roi_heads(images, features, proposals, None, None, [before_perturbation_scores])
        return results


    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.rcnn.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
