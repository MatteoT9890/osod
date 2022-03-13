import torch
from detectron2.config import \
    get_cfg  # import default model configuration: config/defaults.py, config/paths_catalog.py, yaml file
from detectron2.modeling.meta_arch import build_model
from src.trainer.trainer import DetectorTrainer
from src.data.handler import DatasetHandler
from src.tester.tester import Tester
from detectron2.engine import create_ddp_model
from detectron2.engine import default_argument_parser, default_setup, launch
from src.models.architectures import Detector
from src.models.components.roi_heads import MyRes5ROIHeads
from utils import *

import math


import torch


def setup(arguments):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(arguments.config_file)
    cfg.merge_from_list(arguments.opts)
    if cfg.OURS.DEBUG:
        cfg.DATALOADER.NUM_WORKERS = 0
    update_relative_path(cfg)
    cfg.freeze()
    default_setup(cfg, arguments)
    return cfg


def main(arguments):
    # Create config
    cfg = setup(arguments)
    # Register test datasets
    DatasetHandler.register_test_datasets(cfg)

    # Create the distributed data parallel model on CUDA
    detector = build_model(cfg)

    #Test
    tester = Tester(cfg=cfg, detector=detector)
    tester.test()


if __name__ == "__main__":
    #### coco
    # step1_faster_fg = torch.tensor([math.log(0.64 / (1 - 0.64))])
    # step1_faster_bg = torch.tensor([math.log(0.03 / (1 - 0.03))])
    # step3_faster_fg = torch.tensor([math.log(0.66 / (1 - 0.66))])
    # step3_faster_bg = torch.tensor([math.log(0.06 / (1 - 0.06))])
    # step3_our_fg = torch.tensor([math.log(0.74 / (1 - 0.74))])
    # step3_our_bg = torch.tensor([math.log(0.06 / (1 - 0.06))])
    #
    # step1_faster = torch.cat((step1_faster_fg, step1_faster_bg))
    # step3_faster = torch.cat((step3_faster_fg, step3_faster_bg))
    # step3_our = torch.cat((step3_our_fg, step3_our_bg))
    #
    # step1_faster_softmax = torch.nn.Softmax()(step1_faster)
    # step3_faster_softmax = torch.nn.Softmax()(step3_faster)
    # step3_our_softmax = torch.nn.Softmax()(step3_our)
    #
    #
    # ##### voc
    # step3_faster_fg = torch.tensor([math.log(0.8 / (1 - 0.8))])
    # step3_faster_bg = torch.tensor([math.log(0.06 / (1 - 0.06))])
    # step3_our_fg = torch.tensor([math.log(0.85 / (1 - 0.85))])
    # step3_our_bg = torch.tensor([math.log(0.06 / (1 - 0.06))])
    #
    # step3_faster = torch.cat((step3_faster_fg, step3_faster_bg))
    # step3_our = torch.cat((step3_our_fg, step3_our_bg))
    #
    # step3_faster_softmax = torch.nn.Softmax()(step3_faster)
    # step3_our_softmax = torch.nn.Softmax()(step3_our)


    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
