#!/bin/bash
PYTHONPATH='/home/dariof/osod'

bash $PYTHONPATH/start/setup.sh
  ## TODO: Automate the reading of RPN post-train metrics for training steps 2,3,4 with our strategy. Up to now it
  ##  is necessary to set them manually in the respective configuration files. ##
{
#  ## COMMON ##
#    # First step
#    CUDA_VISIBLE_DEVICES=0,1 python $PYTHONPATH/src/main/train.py --config=$PYTHONPATH/conf/common/train_first_step.yaml --num-gpus=2
#
#  ## BASELINE ##
#    # Second step
#    CUDA_VISIBLE_DEVICES=0,1 python $PYTHONPATH/src/main/train.py --config=$PYTHONPATH/conf/baseline/train_second_step_rcnn.yaml --num-gpus=2
#
#    # Third step
#    CUDA_VISIBLE_DEVICES=0,1 python $PYTHONPATH/src/main/train.py --config=$PYTHONPATH/conf/baseline/train_third_step_rpn.yaml --num-gpus=2
#
#    # Fourth step
#    CUDA_VISIBLE_DEVICES=0,1 python $PYTHONPATH/src/main/train.py --config=$PYTHONPATH/conf/baseline/train_fourth_step_rcnn.yaml --num-gpus=2

    # Test
#    CUDA_VISIBLE_DEVICES=0,1 python $PYTHONPATH/src/main/test.py --config=$PYTHONPATH/conf/baseline/test_open_cwwr_nomsp.yaml --num-gpus=2 #
#    CUDA_VISIBLE_DEVICES=0,1 python $PYTHONPATH/src/main/collect_results.py --config=$PYTHONPATH/conf/baseline/test_open_cwwr_nomsp.yaml --num-gpus=2 #
    CUDA_VISIBLE_DEVICES=0,1 python $PYTHONPATH/src/main/test.py --config=$PYTHONPATH/conf/baseline/test_open_cwwr_odin.yaml --num-gpus=2 #
    CUDA_VISIBLE_DEVICES=0,1 python $PYTHONPATH/src/main/collect_results.py --config=$PYTHONPATH/conf/baseline/test_open_cwwr_odin.yaml --num-gpus=2 #
}
{
  exit -1
}

