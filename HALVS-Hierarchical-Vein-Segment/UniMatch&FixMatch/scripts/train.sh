#!/bin/bash
# export NCCL_P2P_DISABLE= 1
# export NCCL_IB_DISABLE= 1
# export NCCL_P2P_LEVEL= NVL
# export NCCL_DEBUG= INFO

now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['leaf']
# method: ['unimatch_partial','unimatch', 'fixmatch','fixmatch_partial', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['6_6_60','18_18_180' ...]
dataset='leaf'
method='unimatch_partial'
exp='r101'
split='6_6_60'

# config=configs/${dataset}.yaml
# labeled_id_path=splits/$dataset/$split/labeled.txt
# unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
# partial_labeled_id_path=splits/$dataset/$split/partial_labeled.txt
# save_path=log/$dataset/$method/$exp/$split


mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --partial_labeled_id_path $partial_labeled_id_path --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log