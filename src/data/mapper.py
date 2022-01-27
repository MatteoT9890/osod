import copy
import logging
import numpy as np
import torch

from detectron2.config import configurable
import detectron2.data.detection_utils as utils


class DiscoveryDatasetMapper:
    @configurable
    def __init__(
            self,
            *,
            image_format: str,
            instance_mask_format: str = "polygon",
    ):
        """
        NOTE: this interface is experimental.

        Args:
            image_format: an image format supported by :func:`detection_utils.read_image`.
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
        """

        # fmt: off
        self.image_format = image_format
        self.instance_mask_format = instance_mask_format
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = {
            "image_format": cfg.INPUT.FORMAT,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        instances = utils.annotations_to_instances(
            dataset_dict["annotations"], image_shape, mask_format=self.instance_mask_format
        )
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        dataset_dict.pop("annotations", None)
        return dataset_dict
