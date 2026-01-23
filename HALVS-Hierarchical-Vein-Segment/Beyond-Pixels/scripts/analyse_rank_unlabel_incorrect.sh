#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
#####################################################method to analyse the wrong pseudo_labels
# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'coco']
# method: ['unimatch', 'fixmatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', 'u2pl_1_16', ...]. Please check directory './splits/$dataset' for concrete splits
dataset='pascal'
method='unimatch'
exp='r101'
split='92'
export CUDA_VISIBLE_DEVICES=$3
config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split
start_epoch=$4
end_epoch=$5

mkdir -p $save_path
echo "GPU used"
echo $3

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    analyse_unimatch_rank_unlabel_incorrect.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 --startepoch $start_epoch --endepoch $end_epoch  2>&1 