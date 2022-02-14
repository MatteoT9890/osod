import os
import detectron2

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = CUR_DIR.replace("src/main", "")

def update_relative_path(cfg):
    for entry in cfg:
        if isinstance(cfg[entry], detectron2.config.CfgNode):
            update_relative_path(cfg[entry])
        elif isinstance(cfg[entry], list):
            for idx, _ in enumerate(cfg[entry]):
                if isinstance(cfg[entry][idx], dict):
                    for key, val in cfg[entry][idx].items():
                        if isinstance(val, str) and '/home/mtarantino/thesis/oursowod/' in val:
                            cfg[entry][idx][key] = cfg[entry][idx][key].replace('/home/mtarantino/thesis/oursowod/', f"{ROOT_DIR}")
        elif isinstance(cfg[entry], str) and '/home/mtarantino/thesis/oursowod/' in cfg[entry]:
            cfg[entry] = cfg[entry].replace('/home/mtarantino/thesis/oursowod/', f"{ROOT_DIR}")