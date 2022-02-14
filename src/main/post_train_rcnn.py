import torch
from detectron2.config import \
    get_cfg  # import default model configuration: config/defaults.py, config/paths_catalog.py, yaml file
from detectron2.modeling.meta_arch import build_model
from src.post_train.post_trainer import PostRPNTrainer, PostDetectorTrainer
from src.data.handler import DatasetHandler
from detectron2.engine import default_argument_parser, default_setup, launch
from src.models.architectures import RegionProposalNetwork
from utils import *


def setup(arguments):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(arguments.config_file)
    cfg.merge_from_list(arguments.opts)
    update_relative_path(cfg)
    cfg.freeze()
    default_setup(cfg, arguments)
    return cfg


def main(arguments):
    # Create config
    cfg = setup(arguments)

    # Load only known data
    dataset_handler = DatasetHandler(cfg)
    post_train_dataloader = dataset_handler.get_test_loader()

    # Create the distributed data parallel model on CUDA
    model = build_model(cfg)

    # Train
    post_rpn_trainer = PostDetectorTrainer(detector=model, data_loader=post_train_dataloader, cfg=cfg)
    post_rpn_trainer.loop(0, len(post_train_dataloader))


if __name__ == "__main__":
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
