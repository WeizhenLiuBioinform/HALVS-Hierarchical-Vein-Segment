<!-- Our code is based on Unimatch, we added the PSSS module to it.-->

## Installation

```
conda create -n unimatch_Halvs python=3.10.4
conda activate unimatch_Halvs
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

## Pretrained Backbone

[ResNet-50](https://drive.google.com/file/d/1mqUrqFvTQ0k5QEotk4oiOFyP6B9dVZXS/view?usp=sharing) | [ResNet-101](https://drive.google.com/file/d/1Rx0legsMolCWENpfvE2jUScT3ogalMO8/view?usp=sharing) | [Xception-65](https://drive.google.com/open?id=1_j_mE07tiV24xXOJw4XDze0-a0NAhNVi)

```
├── ./pretrained

  ├── resnet50.pth

  ├── resnet101.pth

  └── xception.pth
```

## Dataset

```
├── [Halvs]

  ├── JPEGImages

  └── SegmentationClass
```

Update the data_root entry in the configs/leaf.yaml file to reflect the absolute path of the "Halvs" directory on your local machine.

## Train

```
sh scripts/train.sh <gpu_num> <port>
```

#modify these augments in `scripts/train.sh` if you want to try other methods.

#method: ['==unimatch_partial==',=='fixmatch_partial==','unimatch', 'fixmatch', 'supervised'].（Highlight are our method,the others are baselines.）

