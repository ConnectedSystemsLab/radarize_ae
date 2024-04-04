#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataloader import default_collate

sys.path.append("..")


class FlipRange(object):

    topics = [
        "radar_r_1",
        "radar_r_3",
        "radar_r_5",
        "radar_re_1",
        "radar_re_3",
        "radar_re_5",
        "depth_map",
    ]

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):

        if torch.rand(1).item() < self.prob:
            for topic in self.topics:
                if topic in sample:
                    sample[topic] = transforms.functional.hflip(sample[topic])

        return sample


class UNetDataset(Dataset):
    """UNet dataset."""

    topics = [
        "time",
        "pose_gt",
        "radar_r_1",
        "radar_r_3",
        "radar_r_5",
        "radar_re_1",
        "radar_re_3",
        "radar_re_5",
        "depth_map",
    ]

    def __init__(self, path, seq_len=1, transform=None):
        # Load files from .npz.
        self.path = path

        print(path)
        with np.load(path) as data:
            self.files = [k for k in data.files if k in self.topics]
            self.dataset = {k: data[k] for k in self.files}

        # Check if lengths are the same.
        for k in self.files:
            print(k, self.dataset[k].shape, self.dataset[k].dtype)
        lengths = [self.dataset[k].shape[0] for k in self.files]
        assert len(set(lengths)) == 1
        self.num_samples = lengths[0]

        # Set sequence length for stacking frames across time.
        self.seq_len = seq_len

        # Save transforms.
        self.transform = transform

    def __len__(self):
        return self.num_samples - self.seq_len

    def __getitem__(self, idx):
        sample = {}
        sample["time"] = torch.tensor(self.dataset["time"][idx : idx + self.seq_len])
        sample["radar_r_1"] = torch.from_numpy(
            np.concatenate(self.dataset["radar_r_1"][idx : idx + self.seq_len], axis=0)
        )
        sample["radar_r_3"] = torch.from_numpy(
            np.concatenate(self.dataset["radar_r_3"][idx : idx + self.seq_len], axis=0)
        )
        sample["radar_r_5"] = torch.from_numpy(
            np.concatenate(self.dataset["radar_r_5"][idx : idx + self.seq_len], axis=0)
        )
        sample["radar_re_1"] = torch.from_numpy(
            np.concatenate(self.dataset["radar_re_1"][idx : idx + self.seq_len], axis=0)
        )
        sample["radar_re_3"] = torch.from_numpy(
            np.concatenate(self.dataset["radar_re_3"][idx : idx + self.seq_len], axis=0)
        )
        sample["radar_re_5"] = torch.from_numpy(
            np.concatenate(self.dataset["radar_re_5"][idx : idx + self.seq_len], axis=0)
        )
        sample["depth_map"] = torch.from_numpy(
            self.dataset["depth_map"][idx + self.seq_len]
        )
        sample["pose_gt"] = torch.from_numpy(
            self.dataset["pose_gt"][idx : idx + self.seq_len]
        )

        if self.transform:
            sample = self.transform(sample)

        return sample


