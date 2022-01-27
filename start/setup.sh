## BASELINE ##
mkdir $PYTHONPATH/checkpoint/baseline
mkdir $PYTHONPATH/checkpoint/baseline/step1
mkdir $PYTHONPATH/checkpoint/baseline/step1/train
mkdir $PYTHONPATH/checkpoint/baseline/step2
mkdir $PYTHONPATH/checkpoint/baseline/step2/train
mkdir $PYTHONPATH/checkpoint/baseline/step3
mkdir $PYTHONPATH/checkpoint/baseline/step3/train
mkdir $PYTHONPATH/checkpoint/baseline/step4
mkdir $PYTHONPATH/checkpoint/baseline/step4/train
chmod 777 -R $PYTHONPATH/checkpoint/baseline

## OURS ##
mkdir $PYTHONPATH/checkpoint/ours
# One std no reg
mkdir $PYTHONPATH/checkpoint/ours/one_std
mkdir $PYTHONPATH/checkpoint/ours/one_std/noreg
mkdir $PYTHONPATH/checkpoint/ours/one_std/noreg/step2/
mkdir $PYTHONPATH/checkpoint/ours/one_std/noreg/step2/train
mkdir $PYTHONPATH/checkpoint/ours/one_std/noreg/step3/
mkdir $PYTHONPATH/checkpoint/ours/one_std/noreg/step3/train
mkdir $PYTHONPATH/checkpoint/ours/one_std/noreg/step3/post_train
mkdir $PYTHONPATH/checkpoint/ours/one_std/noreg/step4/
mkdir $PYTHONPATH/checkpoint/ours/one_std/noreg/step4/train

# Zero std no reg
mkdir $PYTHONPATH/checkpoint/ours/zero_std
mkdir $PYTHONPATH/checkpoint/ours/zero_std/noreg
mkdir $PYTHONPATH/checkpoint/ours/zero_std/noreg/step2/
mkdir $PYTHONPATH/checkpoint/ours/zero_std/noreg/step2/train
mkdir $PYTHONPATH/checkpoint/ours/zero_std/noreg/step3/
mkdir $PYTHONPATH/checkpoint/ours/zero_std/noreg/step3/train
mkdir $PYTHONPATH/checkpoint/ours/zero_std/noreg/step3/post_train
mkdir $PYTHONPATH/checkpoint/ours/zero_std/noreg/step4/
mkdir $PYTHONPATH/checkpoint/ours/zero_std/noreg/step4/train

#Change mod 777 ours dir
chmod 777 -R $PYTHONPATH/out
chmod 777 -R $PYTHONPATH/checkpoint