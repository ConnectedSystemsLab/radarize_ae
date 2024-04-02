#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm
torch.multiprocessing.set_sharing_strategy('file_system')
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataloader import default_collate
sys.path.append("..")

class FlipFlow(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if torch.rand(1).item() < self.prob:
            sample['velo_gt'] = sample['velo_gt']*-1
            sample['radar_d'] = transforms.functional.vflip(sample['radar_d']) 
            sample['radar_de'] = transforms.functional.vflip(sample['radar_de'])

        return sample

class FlowDataset(Dataset):
    """Flow dataset."""

    topics = ['time', 
              'radar_d', 
              'radar_de', 
              'velo_gt']

    def __init__(self, 
                 path, 
                 subsample_factor=1,
                 transform=None):
        # Load files from .npz.
        self.path = path
        print(path)
        with np.load(path) as data:
            self.files = [k for k in data.files if k in self.topics]
            self.dataset = {k : data[k][::subsample_factor] for k in self.files} 

        # Check if lengths are the same.
        for k in self.files:
            print(k, self.dataset[k].shape, self.dataset[k].dtype)
        lengths = [self.dataset[k].shape[0] for k in self.files]
        assert len(set(lengths)) == 1
        self.num_samples = lengths[0] 

        # Save transforms.
        self.transform = transform
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = {k: torch.from_numpy(self.dataset[k][idx]) if type(self.dataset[k][idx]) is np.ndarray else self.dataset[k][idx] \
                for k in self.files}
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', help="Path to data directory")
    args = parser.parse_args()

    npz_paths = sorted(glob.glob(os.path.join(args.data_dir, '*.npz')))
    datasets = [FlowDataset(path) for path in npz_paths]
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    print(len(combined_dataset))

    # Test dataloader
    # dataloader = DataLoader(combined_dataset, batch_size=64,
    #                         shuffle=True, num_workers=4)

    # for i, batch in enumerate(tqdm(dataloader)):
    #     for k, v in batch.items():
    #         print(k, v.size(), v.dtype)

    # Visualize heatmaps.
    for path in npz_paths:
        print(f"Processing {path}...")

        dataset = FlowDataset(path, transform=FlipFlow())
        dataloader = DataLoader(dataset, 
                                batch_size=1, shuffle=False, num_workers=0)

        for i, batch in enumerate(dataloader):

            cv2.namedWindow('x_heatmap', cv2.WINDOW_KEEPRATIO) 
            cv2.namedWindow('y_heatmap', cv2.WINDOW_KEEPRATIO)

            cv2.imshow('x_heatmap', cv2.applyColorMap((batch['radar_d'][0][0].numpy()*255).astype(np.uint8), cv2.COLORMAP_TURBO))
            cv2.imshow('y_heatmap', cv2.applyColorMap((batch['radar_de'][0][0].numpy()*255).astype(np.uint8), cv2.COLORMAP_TURBO))

            cv2.waitKey(10)
