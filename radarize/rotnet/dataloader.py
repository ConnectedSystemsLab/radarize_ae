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


class ReverseTime(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):

        if torch.rand(1).item() < self.prob:
            for k, v in sample.items():
                sample[k] = torch.flip(v, [0])

        return sample


class RotationDataset(Dataset):
    """Rotation dataset."""

    topics = ["time", "pose_gt", "radar_r_1", "radar_r_3", "radar_r_5", "radar_re"]

    def __init__(
        self, path, subsample_factor=1, seq_len=1, random_seq_len=False, transform=None
    ):

        # Load files from .npz.
        self.path = path

        print(path)
        with np.load(path) as data:
            self.files = [k for k in data.files if k in self.topics]
            self.dataset = {k: data[k][::subsample_factor] for k in self.files}

        # Check if lengths are the same.
        for k in self.files:
            print(k, self.dataset[k].shape, self.dataset[k].dtype)
        lengths = [self.dataset[k].shape[0] for k in self.files]
        assert len(set(lengths)) == 1
        self.num_samples = lengths[0]

        # Set sequence length for stacking frames across time.
        self.random_seq_len = random_seq_len
        self.seq_len = seq_len

        # Save transforms.
        self.transform = transform

    def __len__(self):
        return self.num_samples - self.seq_len

    def __getitem__(self, idx):
        sample = {}

        if self.random_seq_len:
            seq_len = np.random.randint(1, self.seq_len + 1)
        else:
            seq_len = self.seq_len

        sample["time"] = torch.tensor(self.dataset["time"][[idx, idx + seq_len]])
        sample["pose_gt"] = torch.tensor(self.dataset["pose_gt"][[idx, idx + seq_len]])
        sample["radar_r"] = torch.tensor(
            np.concatenate(
                [
                    self.dataset["radar_r_1"][idx],
                    self.dataset["radar_r_3"][idx],
                    self.dataset["radar_r_5"][idx],
                    self.dataset["radar_r_5"][idx + seq_len],
                    self.dataset["radar_r_3"][idx + seq_len],
                    self.dataset["radar_r_1"][idx + seq_len],
                ],
                axis=0,
            )
        )

        # sample['time'] = torch.tensor(self.dataset['time'][idx:idx+seq_len])
        # sample['pose_gt'] = torch.tensor(self.dataset['pose_gt'][idx:idx+seq_len])
        # sample['radar_r'] = torch.from_numpy(np.concatenate(self.dataset['radar_r'][idx:idx+seq_len], axis=0))

        if self.transform:
            sample = self.transform(sample)

        return sample


