import torch
from detectron2.config import get_cfg  # import default model configuration: config/defaults.py, config/paths_catalog.py, yaml file
from detectron2.modeling.meta_arch import build_model
from src.trainer.trainer import RPNTrainer, DetectorTrainer
from src.data.handler import DatasetHandler
from detectron2.engine import default_argument_parser, default_setup, launch
from src.models.architectures import RegionProposalNetwork, FastRCNN, Detector
from src.models.components.roi_heads import MyRes5ROIHeads

def setup(arguments):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(arguments.config_file)
    cfg.merge_from_list(arguments.opts)
    cfg.freeze()
    default_setup(cfg, arguments)
    return cfg


def main(arguments):
    # Create config
    cfg = setup(arguments)

    # Load only known data
    dataset_handler = DatasetHandler(cfg)
    train_dataloader = dataset_handler.get_train_loader()

    # Create the distributed data parallel model on CUDA
    model = build_model(cfg)

    # Train
    if cfg.OURS.STEP == 1 or cfg.OURS.STEP == 3:
        trainer = RPNTrainer(rpn=model, data_loader=train_dataloader, cfg=cfg)
    elif cfg.OURS.STEP == 2 or cfg.OURS.STEP == 4:
        trainer = DetectorTrainer(detector=model, data_loader=train_dataloader, cfg=cfg)
    else:
        raise Exception("Training step not recognized. It must be one of 1,2,3,4. Actual step: {}".format(cfg.OURS.STEP))
    trainer.train()


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
