#!/usr/bin/env python3

import os
import sys

import argparse
import random

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

from radarize.config import cfg, update_config
from radarize.unet import dataloader, model
from radarize.unet.dice_score import dice_loss


def train(net, device, train_loader, optimizer, epoch):
    net.train()
    loss_plot = 0
    for batch_idx, batch in enumerate(train_loader):
        for k, v in batch.items():
            batch[k] = v.to(device)
        # x = torch.cat([batch['radar_r'].to(torch.float32),
        #                batch['radar_re'].to(torch.float32)], dim=1)
        x = torch.cat(
            [
                batch["radar_r_1"],
                batch["radar_r_3"],
                batch["radar_r_5"],
                batch["radar_re_1"],
                batch["radar_re_3"],
                batch["radar_re_5"],
            ],
            dim=1,
        ).to(torch.float32)
        y = batch["depth_map"].to(torch.float32)
        y = torch.cat([y[:, 0:1, ...] == 0, y[:, 0:1, ...] == 1], dim=1).to(
            torch.float32
        )

        optimizer.zero_grad()
        y_pred = net(x)
        loss = cfg["UNET"]["TRAIN"]["BCE_WEIGHT"] * F.binary_cross_entropy(
            y_pred, y
        ) + cfg["UNET"]["TRAIN"]["DICE_WEIGHT"] * dice_loss(y_pred, y, multiclass=True)
        loss_plot += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % cfg["UNET"]["TRAIN"]["LOG_STEP"] == 0:
            print(
                f"Train Epoch: {epoch} [({100. * batch_idx / len(train_loader):.0f}%)]\t Loss: {loss.item():.6f}"
            )

    loss_plot /= len(train_loader)
    return loss_plot


def test(net, device, test_loader):
    net.eval()

    test_dice_loss = 0
    test_mse_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            # x = torch.cat([batch['radar_r'].to(torch.float32),
            #                batch['radar_re'].to(torch.float32)], dim=1)
            x = torch.cat(
                [
                    batch["radar_r_1"],
                    batch["radar_r_3"],
                    batch["radar_r_5"],
                    batch["radar_re_1"],
                    batch["radar_re_3"],
                    batch["radar_re_5"],
                ],
                dim=1,
            ).to(torch.float32)
            y = batch["depth_map"].to(torch.float32)

            y_pred = net(x)
            y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
            # y_pred = (y_pred == 1)

            test_dice_loss += dice_loss(y_pred, y, multiclass=True).item()
            test_mse_loss += F.mse_loss(y_pred, y).item()

    test_dice_loss = test_dice_loss / len(test_loader)
    test_mse_loss = test_mse_loss / len(test_loader)
    print(f"\n[Test] Dice loss: {test_dice_loss:.6f} MSE loss: {test_mse_loss:.6f}")
    return test_dice_loss


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
    train_dir = os.path.join(cfg["OUTPUT_DIR"], cfg["UNET"]["MODEL"]["NAME"])
    os.makedirs(train_dir, exist_ok=True)

    # Set random seeds.
    random.seed(cfg["UNET"]["TRAIN"]["SEED"])
    np.random.seed(cfg["UNET"]["TRAIN"]["SEED"])
    torch.manual_seed(cfg["UNET"]["TRAIN"]["SEED"])

    # Set training params.
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": cfg["UNET"]["TRAIN"]["BATCH_SIZE"], "drop_last": True}
    test_kwargs = {"batch_size": cfg["UNET"]["TEST"]["BATCH_SIZE"]}

    if use_cuda:
        cuda_kwargs = {
            "num_workers": 0,
            "shuffle": True,
            "worker_init_fn": lambda id: np.random.seed(
                id * cfg["UNET"]["TRAIN"]["SEED"]
            ),
        }
        train_kwargs.update(cuda_kwargs)
        cuda_kwargs = {
            "num_workers": 0,
            "shuffle": False,
            "worker_init_fn": lambda id: np.random.seed(
                id * cfg["UNET"]["TRAIN"]["SEED"]
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
        dataloader.UNetDataset(path, seq_len=1, transform=dataloader.FlipRange(0.5))
        for path in sorted(train_paths)
    ]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    test_paths = [
        os.path.join(cfg["DATASET"]["PATH"], x + ".npz")
        for x in cfg["DATASET"]["VAL_SPLIT"]
    ]
    test_datasets = [
        dataloader.UNetDataset(path, seq_len=1) for path in sorted(test_paths)
    ]
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model_kwargs = {
        "n_channels": cfg["UNET"]["MODEL"]["N_CHANNELS"],
        "n_classes": cfg["UNET"]["MODEL"]["N_CLASSES"],
    }

    # Load network.
    net = getattr(model, cfg["UNET"]["MODEL"]["TYPE"])(**model_kwargs).to(device)

    optimizer = optim.Adam(
        net.parameters(), lr=cfg["UNET"]["TRAIN"]["LR"], betas=(0.8, 0.9)
    )

    train_loss_array = []
    test_loss_array = []

    least_test_loss = np.inf
    for epoch in range(1, cfg["UNET"]["TRAIN"]["EPOCHS"] + 1):
        train_loss = train(net, device, train_loader, optimizer, epoch)
        test_loss = test(net, device, test_loader)
        print(train_loss, test_loss)

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
                    "model_name": cfg["UNET"]["MODEL"]["NAME"],
                    "model_type": type(net).__name__,
                    "model_state_dict": net.state_dict(),
                    "model_kwargs": model_kwargs,
                    "epoch": epoch,
                    "test_loss": test_loss,
                },
                os.path.join(train_dir, f"{cfg['UNET']['MODEL']['NAME']}.pth"),
            )
