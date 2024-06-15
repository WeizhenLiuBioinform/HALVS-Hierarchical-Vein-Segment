import os
import subprocess
import time
import datetime
import torch
import torch.distributed as dist


def setup_distributed(backend="nccl", port=None):     # backend="gloo"
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        # init_method='env://', 
        # timeout=datetime.timedelta(seconds=5400)
    )
    return rank, world_size

def exclusive_loss(pred, exclude_label=None, mode='entropy'):

    if exclude_label is None:
        exclude_label = [0,1, 2]
    if mode == 'entropy':
        label = torch.zeros_like(pred)
        for i in exclude_label:
            label[:, i, :, :] = 1      # [0,1,1,0]
        pred_softmax = pred.softmax(dim=1)
        pred_exclusive_entropy = torch.log(1 + pred_softmax)
        loss = label * pred_exclusive_entropy

        loss = loss.sum(dim=1)
        return loss


