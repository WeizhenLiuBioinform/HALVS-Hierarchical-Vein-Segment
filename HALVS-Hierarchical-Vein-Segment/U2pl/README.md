<!-- Our code is based on U2pl, we added the PSSS module to it.-->

## Installation
conda create -n u2pl python=3.10.4
conda activate u2pl
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```bash
pip install -r requirements.txt
```

## Usage
### Prepare Data

<details>
  <summary>For dataset HALVS</summary>

The dictionary structure of the dataset is as follows:

```angular2html
HALVS
├── JPEGImages
└── SegmentationClass
```

</details>

### Prepare Pretrained Backbone

Before training, modify ```model_urls``` in ```u2pl/models/resnet.py``` to ```</path/to/resnet101.pth>```

### Train a Fully-Supervised Model

We can train a model on HALVS with only ```6``` labeled images with full blades for supervision by:
```bash
cd experiments_sup/6
sh train.sh <num_gpu> <port>
```

To train on other data splits, please modify ``data_list`` in config.yaml

### Train a Semi-Supervised Model

We can train a model on HALVS with ```6``` labeled images and ```60``` unlabeled images with full blades using semi-supervised mehod U2PL by:
```bash
cd experiments_semi/6_6_60
sh train.sh <num_gpu> <port>
```

To train on other data splits, please modify ``data_list`` in config.yaml

### Train a Partially-Supervised Model

For instance, we can train a model on HALVS with ```18``` labeled images, ```72``` partially labeled images and ```360``` unlabeled images with full blades using U2PL with our PSSS module by:
```bash
sh scripts_partial/train.sh <num_gpu> <port>
```
To train on other data splits, please modify ``data_list`` in scripts_partial/config.yaml and ``exp_path`` in scripts_partial/train.sh

