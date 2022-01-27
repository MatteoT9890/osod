from typing import List, Tuple

from detectron2.data import build_detection_train_loader, build_detection_test_loader
from .loader import voc_register, coco_register, DataSplits

"""
Continuing in background, pid 14353.
Continuing in background, pid 14359.
Continuing in background, pid 14362.


"""


def build_ow_data_splits(dataset_name):
    if "VOC" in dataset_name:
        return {
            DataSplits.ALL: OWDataHandler(
                (
                    "aeroplane", "bicycle", "bottle", "car", "cat",
                    "cow", "chair", "dog", "diningtable", "motorbike",
                    "person", "sofa", "pottedplant", "train", "tvmonitor",
                    "boat", "bird", "bus", "horse", "sheep"
                ), "_".join([dataset_name, "all"])),
            DataSplits.KNOWN: OWDataHandler(
                (
                    "aeroplane", "bicycle", "bottle", "car", "cat",
                    "cow", "chair", "dog", "diningtable", "motorbike",
                    "person", "sofa", "pottedplant", "train", "tvmonitor",
                ), "_".join([dataset_name, "known"])),
            DataSplits.UNKNOWN: OWDataHandler(
                (
                    "boat", "bird", "bus", "horse", "sheep"
                ), "_".join([dataset_name, "unknown"])),
            DataSplits.KNOWN_AND_UNKNOWN: OWDataHandler(
                (
                    # First 15 are known
                    "aeroplane", "bicycle", "bottle", "car", "cat",
                    "cow", "chair", "dog", "diningtable", "motorbike",
                    "person", "sofa", "pottedplant", "train", "tvmonitor",

                    # All the others are marked as a single unknown
                    "unknown"
                ), "_".join([dataset_name, "known-and-unknown"]))
        }
    else:
        raise Exception("Dataset name not recognized")


class OWDataHandler:
    def __init__(self, class_names, dataset_name):
        self.class_names = class_names
        self.dataset_name = dataset_name


class DatasetHandler:
    def __init__(self, cfg):
        self.dataset_root = cfg.OURS.DATASETS.ROOT_DIR
        self.dataset_info = cfg.OURS.DATASETS.INFO
        if "VOC" in self.dataset_info.NAME:
            self.data_splits = build_ow_data_splits(self.dataset_info.NAME)

            voc_register(
                dirname="/".join([self.dataset_root, self.dataset_info.SUBDIR]),
                split=self.dataset_info.SPLIT,
                year=self.dataset_info.YEAR,
                data_splits=self.data_splits
            )
        elif "COCO" in self.dataset_info.NAME:
            coco_register(
                dirname="/".join([self.dataset_root, self.dataset_info.SUBDIR]),
                dataset_name=self.dataset_info.NAME,
                unk_id=cfg.MODEL.ROI_HEADS.NUM_CLASSES+1,
                filename=self.dataset_info.FILENAME
            )

        self.cfg = cfg

    @classmethod
    def search_dataset_for_name(cls, datasets, name: str):
        for dataset in datasets:
            if dataset['INFO']['NAME'] == name:
                return dataset

    @classmethod
    def register_test_datasets(cls, cfg):
        ours_datasets: List = cfg.OURS.TEST_DATASETS
        for dataset_name in cfg.OURS.TEST_DATASETS_NAMES:
            test_dataset = cls.search_dataset_for_name(ours_datasets, dataset_name)
            if "VOC" in dataset_name:
                data_splits = build_ow_data_splits(dataset_name)

                voc_register(
                    dirname="/".join([test_dataset['ROOT_DIR'], test_dataset['INFO']['SUBDIR']]),
                    split=test_dataset['INFO']['SPLIT'],
                    year=test_dataset['INFO']['YEAR'],
                    data_splits=data_splits
                )

            elif "COCO" in dataset_name:
                filename = test_dataset['INFO'].get("FILENAME", None)
                coco_register(
                    dirname="/".join([test_dataset['ROOT_DIR'], test_dataset['INFO']['SUBDIR']]),
                    dataset_name=dataset_name,
                    unk_id=cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1,
                    filename= filename if filename is not None else "WR1_Mixed_Unknowns.json"
                )

    def get_known_split(self) -> OWDataHandler:
        return self.data_splits[DataSplits.KNOWN]

    def get_all_split(self) -> OWDataHandler:
        return self.data_splits[DataSplits.ALL]

    def get_unknown_split(self) -> OWDataHandler:
        return self.data_splits[DataSplits.UNKNOWN]

    def get_known_and_unknown_split(self) -> OWDataHandler:
        return self.data_splits[DataSplits.KNOWN_AND_UNKNOWN]

    def get_train_loader(self):
        return build_detection_train_loader(self.cfg)

    def get_test_loader(self):
        return build_detection_test_loader(self.cfg)
