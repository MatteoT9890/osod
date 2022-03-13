# %%

import torchvision
from utils import *
from PIL import Image
import torch.nn.functional as functional


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
from detectron2.utils.visualizer import Visualizer

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
    #tester.test()
    dataset = tester.dataset_testers[1].data_loader.dataset
    for i in range(0, len(dataset)):
        data = dataset[i]
        img = data['image']
        bbox = data['instances']._fields['gt_boxes']
        visualizer = Visualizer(img_rgb=torch.permute(img, (1,2,0)).numpy())
        # output = visualizer.overlay_instances(boxes=bbox, alpha=0.5)
        output = visualizer.draw_box(box_coord=tuple(bbox.tensor[0].numpy()))
        output = Image.fromarray(output.img, 'RGB')

        output.save(f'/data/dariof/osod/coco_visual/{i}.png')
        print(i)
        # output.show()
# %%
if __name__ == '__main__':
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
