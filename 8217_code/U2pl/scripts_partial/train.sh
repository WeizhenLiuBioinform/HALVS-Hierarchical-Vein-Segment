#!/bin/bash
#export NCCL_P2P_DISABLE="1"
#export NCCL_IB_DISABLE="1"
export NCCL_P2P_LEVEL=NVL

now=$(date +"%Y%m%d_%H%M%S")
method='train_partial'
exp_path=experiments/18_72_360

mkdir -p $exp_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py --config=scripts/config.yaml --exp_path=$exp_path --seed 2 --port $2 2>&1 | tee $exp_path/seg_$now.log
