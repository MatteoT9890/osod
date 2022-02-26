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
from utils import update_relative_path
from src.utils import Logger

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
    logger = Logger(cfg.OUTPUT_DIR, 0)
    logger.save_cumulative_results()
    logger.close()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        1,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

