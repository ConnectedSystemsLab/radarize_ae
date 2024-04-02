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

class ReverseTime(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):

        if torch.rand(1).item() < self.prob:
            for k,v in sample.items():
                sample[k] = torch.flip(v,[0])

        return sample

class RotationDataset(Dataset):
    """Rotation dataset."""

    topics = ['time', 
              'pose_gt',
              'radar_r_1',
              'radar_r_3',
              'radar_r_5',
              'radar_re']

    def __init__(self, 
                 path, 
                 subsample_factor=1,
                 seq_len=1, 
                 random_seq_len=False, 
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
            seq_len = np.random.randint(1, self.seq_len+1)
        else:
            seq_len = self.seq_len

        sample['time']    = torch.tensor(self.dataset['time'][[idx,idx+seq_len]])
        sample['pose_gt'] = torch.tensor(self.dataset['pose_gt'][[idx,idx+seq_len]])
        sample['radar_r'] = torch.tensor(np.concatenate([
            self.dataset['radar_r_1'][idx],
            self.dataset['radar_r_3'][idx],
            self.dataset['radar_r_5'][idx],
            self.dataset['radar_r_5'][idx+seq_len],
            self.dataset['radar_r_3'][idx+seq_len],
            self.dataset['radar_r_1'][idx+seq_len],
        ], axis=0))

        # sample['time'] = torch.tensor(self.dataset['time'][idx:idx+seq_len])
        # sample['pose_gt'] = torch.tensor(self.dataset['pose_gt'][idx:idx+seq_len])
        # sample['radar_r'] = torch.from_numpy(np.concatenate(self.dataset['radar_r'][idx:idx+seq_len], axis=0))


        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    from scipy.spatial.transform import Rotation as R
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help="Path to data directory")
    args = parser.parse_args()

    npz_paths = sorted(glob.glob(os.path.join(args.data_dir, '*.npz')))

    # Visualize heatmaps.
    for path in npz_paths:
        print(f"Processing {path}...")

        dataset = RotationDataset(path, 
                                  seq_len=2, 
                                  transform=ReverseTime(0.5))
        dataloader = DataLoader(dataset, 
                                batch_size=2, 
                                shuffle=False, 
                                num_workers=0)

        for i, batch in enumerate(dataloader):

            for k, v in batch.items():
                print(k, v.size(), v.dtype)

            theta_1 = R.from_quat(batch['pose_gt'][:,0,3:].numpy()).as_euler('ZYX', degrees=False)[:,0:1]
            theta_2 = R.from_quat(batch['pose_gt'][:,-1,3:].numpy()).as_euler('ZYX', degrees=False)[:,0:1]
            print(theta_1.shape, theta_2.shape)

            # cv2.namedWindow('radar_r', cv2.WINDOW_KEEPRATIO) 
            # cv2.namedWindow('depth_map', cv2.WINDOW_KEEPRATIO)

            # cv2.imshow('radar_r', cv2.applyColorMap((batch['radar_r'][0][-1].numpy()*255).astype(np.uint8), cv2.COLORMAP_TURBO))
            # cv2.imshow('depth_map', cv2.applyColorMap((batch['depth_map'][0][0].numpy()*255).astype(np.uint8), cv2.COLORMAP_TURBO))

            cv2.waitKey(5000)
