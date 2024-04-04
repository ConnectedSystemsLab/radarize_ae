#!/usr/bin/env python3

import os
import sys

import argparse
import random

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.optim import lr_scheduler

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

from scipy.spatial.transform import Rotation as R

from radarize.config import cfg, update_config
from radarize.rotnet import dataloader, model
from radarize.rotnet.dataloader import RotationDataset


def normalize_angle(x):
    """Normalize angle to [-pi, pi]."""
    return np.arctan2(np.sin(x), np.cos(x))


def quat2yaw(q):
    """Convert quaternion to yaw angle.
    Args:
        q: (N, 4) array of quaternions
    Returns:
        yaw: (N, 1) array of yaw angles
    """
    if q.ndim == 1:
        return R.from_quat(q).as_euler("ZYX", degrees=False)[0]
    else:
        return R.from_quat(q).as_euler("ZYX", degrees=False)[..., 0:1]


def train(net, device, train_loader, optimizer, epoch):
    net.train()
    loss_plot = 0
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        for k, v in batch.items():
            batch[k] = v.to(device)

        x_1 = batch["radar_r"].to(torch.float32)
        x_2 = torch.flip(x_1, [1])

        theta_1 = quat2yaw(batch["pose_gt"][:, 0, 3:].cpu().numpy())
        theta_2 = quat2yaw(batch["pose_gt"][:, -1, 3:].cpu().numpy())
        delta_theta = normalize_angle(theta_2 - theta_1)
        y = torch.from_numpy(delta_theta).to(device).to(torch.float32)

        optimizer.zero_grad()
        y_pred_1 = net(x_1)
        y_pred_2 = net(x_2)
        loss = torch.sqrt((F.mse_loss(y_pred_1, y) + F.mse_loss(y_pred_2, -y)) / 2)
        loss_plot = loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % cfg["ROTNET"]["TRAIN"]["LOG_STEP"] == 0:
            print(
                f"Train Epoch: {epoch} [({100. * batch_idx / len(train_loader):.0f}%)]\t Loss: {loss.item():.6f}"
            )

    loss_plot /= len(train_loader)
    return loss_plot


def test(net, device, test_loader, scheduler):
    net.eval()

    test_l1_loss = 0
    test_mse_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)

            x = batch["radar_r"].to(torch.float32)
            x_ = torch.flip(x, [1])

            theta_1 = quat2yaw(batch["pose_gt"][:, 0, 3:].cpu().numpy())
            theta_2 = quat2yaw(batch["pose_gt"][:, -1, 3:].cpu().numpy())
            delta_theta = normalize_angle(theta_2 - theta_1)
            y = torch.from_numpy(delta_theta).to(device).to(torch.float32)

            y_pred = (net(x) - net(x_)) / 2

            test_l1_loss += F.l1_loss(y_pred, y).item()
            test_mse_loss += F.mse_loss(y_pred, y).item()

    test_l1_loss = test_l1_loss / len(test_loader)
    test_rmse_loss = np.sqrt(test_mse_loss / len(test_loader))
    print(f"\n[Test] MAE loss: {test_l1_loss:.4f} RMSE loss: {test_rmse_loss:.4f}")

    scheduler.step(test_rmse_loss)

    return test_rmse_loss


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", help="experiment configure file name", default=None, type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = args()
    update_config(cfg, args)

    # Prepare output directory.
    train_dir = os.path.join(cfg["OUTPUT_DIR"], cfg["ROTNET"]["MODEL"]["NAME"])
    os.makedirs(train_dir, exist_ok=True)

    # Set random seeds.
    random.seed(cfg["ROTNET"]["TRAIN"]["SEED"])
    np.random.seed(cfg["ROTNET"]["TRAIN"]["SEED"])
    torch.manual_seed(cfg["ROTNET"]["TRAIN"]["SEED"])

    # Set training params.
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {
        "batch_size": cfg["ROTNET"]["TRAIN"]["BATCH_SIZE"],
        "drop_last": True,
    }
    test_kwargs = {"batch_size": cfg["ROTNET"]["TEST"]["BATCH_SIZE"]}

    if use_cuda:
        cuda_kwargs = {
            "num_workers": 0,
            "shuffle": True,
            "worker_init_fn": lambda id: np.random.seed(
                id * cfg["ROTNET"]["TRAIN"]["SEED"]
            ),
        }
        train_kwargs.update(cuda_kwargs)
        cuda_kwargs = {
            "num_workers": 0,
            "shuffle": False,
            "worker_init_fn": lambda id: np.random.seed(
                id * cfg["ROTNET"]["TRAIN"]["SEED"]
            ),
        }
        test_kwargs.update(cuda_kwargs)

    # Prepare for the dataset
    print("Loading dataset...")
    train_paths = [
        os.path.join(cfg["DATASET"]["PATH"], x + ".npz")
        for x in cfg["DATASET"]["TRAIN_SPLIT"]
    ]
    train_datasets = [
        RotationDataset(
            path,
            subsample_factor=cfg["ROTNET"]["DATA"]["SUBSAMPLE_FACTOR"],
            seq_len=cfg["ROTNET"]["TRAIN"]["TRAIN_SEQ_LEN"],
            random_seq_len=cfg["ROTNET"]["TRAIN"]["TRAIN_RANDOM_SEQ_LEN"],
            transform=dataloader.ReverseTime(0.5),
        )
        for path in train_paths
    ]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    test_paths = [
        os.path.join(cfg["DATASET"]["PATH"], x + ".npz")
        for x in cfg["DATASET"]["VAL_SPLIT"]
    ]
    test_datasets = [
        RotationDataset(
            path,
            subsample_factor=cfg["ROTNET"]["DATA"]["SUBSAMPLE_FACTOR"],
            seq_len=cfg["ROTNET"]["TRAIN"]["VAL_SEQ_LEN"],
            random_seq_len=cfg["ROTNET"]["TRAIN"]["VAL_RANDOM_SEQ_LEN"],
        )
        for path in sorted(test_paths)
    ]
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model_kwargs = {
        "n_channels": cfg["ROTNET"]["MODEL"]["N_CHANNELS"],
        "n_outputs": cfg["ROTNET"]["MODEL"]["N_OUTPUTS"],
    }

    # Load network.
    net = getattr(model, cfg["ROTNET"]["MODEL"]["TYPE"])(**model_kwargs).to(device)

    optimizer = optim.AdamW(
        net.parameters(), lr=cfg["ROTNET"]["TRAIN"]["LR"], betas=(0.9, 0.999)
    )

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=10
    )

    train_loss_array = []
    test_loss_array = []

    least_test_loss = np.inf
    for epoch in range(1, cfg["ROTNET"]["TRAIN"]["EPOCHS"] + 1):
        train_loss = train(net, device, train_loader, optimizer, epoch)
        test_loss = test(net, device, test_loader, scheduler)

        train_loss_array.append(train_loss)
        test_loss_array.append(test_loss)

        plt.plot(np.array(train_loss_array), "b", label="Train Loss")
        plt.plot(np.array(test_loss_array), "r", label="Test Loss")
        plt.scatter(
            np.argmin(np.array(test_loss_array)),
            np.min(test_loss_array),
            s=30,
            color="green",
        )
        plt.title("Loss Plot, min:{:.3f}".format(np.min(test_loss_array)))
        plt.legend()
        plt.grid()
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.savefig(os.path.join(train_dir, "loss.jpg"))
        plt.close()
        # scheduler.step()
        if test_loss < least_test_loss:
            least_test_loss = test_loss
            torch.save(
                {
                    "model_name": cfg["ROTNET"]["MODEL"]["NAME"],
                    "model_type": type(net).__name__,
                    "model_state_dict": net.state_dict(),
                    "model_kwargs": model_kwargs,
                    "epoch": epoch,
                    "test_loss": test_loss,
                },
                os.path.join(train_dir, f"{cfg['ROTNET']['MODEL']['NAME']}.pth"),
            )
