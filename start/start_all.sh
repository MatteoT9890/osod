#!/bin/bash
bash $PYTHONPATH/start/setup.sh
  ## TODO: Automate the reading of RPN post-train metrics for training steps 2,3,4 with our strategy. Up to now it
  ##  is necessary to set them manually in the respective configuration files. ##
{
  ## COMMON ##
    # First step
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/train.py --config=$PYTHONPATH/conf/common/train_first_step.yaml --num-gpus=4

  ## BASELINE ##
    # Second step
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/train.py --config=$PYTHONPATH/conf/baseline/train_second_step_rcnn.yaml --num-gpus=4

    # Third step
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/train.py --config=$PYTHONPATH/conf/baseline/train_third_step_rpn.yaml --num-gpus=4

    # Fourth step
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/train.py --config=$PYTHONPATH/conf/baseline/train_fourth_step_rcnn.yaml --num-gpus=4

    ## Test
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/test.py --config=$PYTHONPATH/conf/baseline/test_open_cwwr_nomsp.yaml --num-gpus=4 #
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/test.py --config=$PYTHONPATH/conf/baseline/test_open_cwwr.yaml --num-gpus=4 #

  ## OURS ##

  ## Common: Calculate the foreground/background activation metrics after first step, then set them into the specific configuration files ##
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/post_train_rpn.py --config=$PYTHONPATH/conf/baseline/post_train_first_step_rpn_voc.yaml --num-gpus=4 #
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/post_train_rpn.py --config=$PYTHONPATH/conf/baseline/post_train_first_step_rpn_coco.yaml --num-gpus=4 #

  ## 1std
    # Second step
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/train.py --config=$PYTHONPATH/conf/ours/one_std/noreg/train_second_step_rcnn.yaml --num-gpus=4 #

    # Third step
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/train.py --config=$PYTHONPATH/conf/ours/one_std/noreg/train_third_step_rpn.yaml --num-gpus=4 #

    # Post train third step
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/post_train_rpn.py --config=$PYTHONPATH/conf/ours/one_std/noreg/post_train_third_step_rpn_coco.yaml --num-gpus=4 #
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/post_train_rpn.py --config=$PYTHONPATH/conf/ours/one_std/noreg/post_train_third_step_rpn_voc.yaml --num-gpus=4 #

    # Fourth step
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/train.py --config=$PYTHONPATH/conf/ours/one_std/noreg/train_fourth_step_rcnn.yaml --num-gpus=4 #

    ## Test
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/test.py --config=$PYTHONPATH/conf/ours/one_std/noreg/test_open_cwwr_msp.yaml --num-gpus=4 #
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/test.py --config=$PYTHONPATH/conf/ours/one_std/noreg/test_open_cwwr_fg.yaml --num-gpus=4
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/test.py --config=$PYTHONPATH/conf/ours/one_std/noreg/test_open_cwwr_all.yaml --num-gpus=4
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/test.py --config=$PYTHONPATH/conf/ours/one_std/noreg/test_open_cwwr_unk_score.yaml --num-gpus=4

  ## 0std
    # Second step
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/train.py --config=$PYTHONPATH/conf/ours/zero_std/noreg/train_second_step_rcnn.yaml --num-gpus=4 #

    # Third step
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/train.py --config=$PYTHONPATH/conf/ours/zero_std/noreg/train_third_step_rpn.yaml --num-gpus=4 #

    # Post train third step
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/post_train_rpn.py --config=$PYTHONPATH/conf/ours/zero_std/noreg/post_train_third_step_rpn_coco.yaml --num-gpus=4 #
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/post_train_rpn.py --config=$PYTHONPATH/conf/ours/zero_std/noreg/post_train_third_step_rpn_voc.yaml --num-gpus=4 #

    # Fourth step
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/train.py --config=$PYTHONPATH/conf/ours/zero_std/noreg/train_fourth_step_rcnn.yaml --num-gpus=4 #

    ## Test
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/test.py --config=$PYTHONPATH/conf/ours/zero_std/noreg/test_open_cwwr_msp.yaml --num-gpus=4 #
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/test.py --config=$PYTHONPATH/conf/ours/zero_std/noreg/test_open_cwwr_fg.yaml --num-gpus=4
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/test.py --config=$PYTHONPATH/conf/ours/zero_std/noreg/test_open_cwwr_all.yaml --num-gpus=4
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHONPATH/src/main/test.py --config=$PYTHONPATH/conf/ours/zero_std/noreg/test_open_cwwr_unk_score.yaml --num-gpus=4

}
{
  exit -1
}

