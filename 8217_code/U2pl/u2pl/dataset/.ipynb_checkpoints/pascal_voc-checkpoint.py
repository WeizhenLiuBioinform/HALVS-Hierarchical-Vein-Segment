import copy
import math
import os
import os.path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from . import augmentation as psp_trsform
from .base import BaseDataset


class voc_dset(BaseDataset):
    def __init__(
        self, data_root, data_list, trs_form, seed=0, n_sup=10582, split="val"
    ):
        super(voc_dset, self).__init__(data_list)
        self.data_root = data_root
        self.transform = trs_form
        random.seed(seed)
        if len(self.list_sample) >= n_sup and split == "train":
            self.list_sample_new = random.sample(self.list_sample, n_sup)
        elif len(self.list_sample) < n_sup and split == "train":
            num_repeat = math.ceil(n_sup / len(self.list_sample))
            self.list_sample = self.list_sample * num_repeat

            self.list_sample_new = random.sample(self.list_sample, n_sup)
        else:
            self.list_sample_new = self.list_sample

    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.data_root, self.list_sample_new[index][0])
        label_path = os.path.join(self.data_root, self.list_sample_new[index][1])
        image = self.img_loader(image_path, "RGB")
        label = self.img_loader(label_path, "L")
        image, label = self.transform(image, label)
        return image[0], label[0, 0].long()

    def __len__(self):
        return len(self.list_sample_new)


def build_transfrom(cfg):
    trs_form = []
    mean, std, ignore_label = cfg["mean"], cfg["std"], cfg["ignore_label"]
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
    if cfg.get("resize", False):
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
    if cfg.get("rand_resize", False):
        trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        trs_form.append(
            psp_trsform.RandRotate(rand_rotation, ignore_label=ignore_label)
        )
    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        trs_form.append(psp_trsform.RandomGaussianBlur())
    if cfg.get("flip", False) and cfg.get("flip"):
        trs_form.append(psp_trsform.RandomHorizontalFlip())
    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=ignore_label)
        )
    return psp_trsform.Compose(trs_form)


def build_vocloader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 500)
    # build transform
    trs_form = build_transfrom(cfg)
    dset = voc_dset(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup)

    # build sampler
    sample = DistributedSampler(dset)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample,
        shuffle=False,
        pin_memory=False,
    )
    return loader


def build_voc_semi_loader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    using_partial = cfg.get("using_partial", False)
    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)

    n_sup = cfg.get("n_sup", 500)
    n_partial = cfg.get("n_partial", 0)
    n_unsup = 11800 - n_sup - n_partial
    n_max = max(n_sup, n_partial, n_unsup)

    # build transform
    trs_form = build_transfrom(cfg)
    trs_partial_form = build_transfrom(cfg)
    trs_form_unsup = build_transfrom(cfg)
    dset = voc_dset(cfg["data_root"], cfg["data_list"], trs_form, seed, n_max, split)

    if split == "val":
        # build sampler
        sample = DistributedSampler(dset)
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample,
            shuffle=False,
            pin_memory=True,
        )
        return loader

    else:
        # build sampler for unlabeled set
        data_list_unsup = cfg["data_list"].replace("labeled.txt", "unlabeled.txt")
        dset_unsup = voc_dset(
            cfg["data_root"], data_list_unsup, trs_form_unsup, seed, n_max, split
        )

        sample_sup = DistributedSampler(dset)
        loader_sup = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample_sup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        sample_unsup = DistributedSampler(dset_unsup)
        loader_unsup = DataLoader(
            dset_unsup,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample_unsup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        if using_partial:
            data_list_partial = cfg["data_list"].replace("labeled.txt", "partial.txt")
            dset_partial = voc_dset(
                cfg["data_root"], data_list_partial, trs_partial_form, seed, n_max, split
            )

            sample_partial = DistributedSampler(dset_partial)
            loader_partial = DataLoader(
                dset_partial,
                batch_size=batch_size,
                num_workers=workers,
                sampler=sample_partial,
                shuffle=False,
                pin_memory=True,
                drop_last=True,
            )

            return [loader_sup, loader_unsup, loader_partial]
        else:
            return [loader_sup, loader_unsup]
