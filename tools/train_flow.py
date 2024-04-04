#!/usr/bin/env python3

import os
import sys

from tqdm import tqdm
import argparse
import numpy as np
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

from radarize.flow import dataloader
from radarize.flow import model
from radarize.config import cfg, update_config


def train(net, device, train_loader, optimizer, epoch):
    net.train()
    loss_plot = 0
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        for k, v in batch.items():
            batch[k] = v.to(device)
        x = torch.cat([batch["radar_d"], batch["radar_de"]], axis=1).to(torch.float32)

        flow_gt = batch["velo_gt"].to(torch.float32)

        optimizer.zero_grad()
        flow_pred = net(x)

        flow_loss_x = F.mse_loss(flow_pred[:, 0], flow_gt[:, 0], reduction="mean")
        flow_loss_y = F.mse_loss(flow_pred[:, 1], flow_gt[:, 1], reduction="mean")
        loss = torch.sqrt((flow_loss_x + flow_loss_y) / 2.0)
        loss_plot += loss.item()

        loss.backward()
        optimizer.step()

        if batch_idx % cfg["FLOW"]["TRAIN"]["LOG_STEP"] == 0:
            print(
                f"Train Epoch: {epoch} [({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {np.sqrt(loss.item()):.6f} flow_loss_x: {np.sqrt(flow_loss_x.item()):.6f}, flow_loss_y: {np.sqrt(flow_loss_y.item()):.6f}"
            )
    loss_plot /= len(train_loader)
    loss_plot = np.sqrt(loss_plot)
    return loss_plot


def test(net, device, test_loader, scheduler):
    net.eval()

    test_loss_sum_mae = 0
    test_loss_sum_mse = 0

    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            x = torch.cat([batch["radar_d"], batch["radar_de"]], axis=1).to(
                torch.float32
            )
            flow_gt = batch["velo_gt"].to(torch.float32)
            flow_gt = flow_gt[:, :2]

            flow_pred = torch.squeeze(net(x))

            test_loss_sum_mae += F.l1_loss(flow_pred, flow_gt, reduction="mean").item()
            test_loss_sum_mse += F.mse_loss(flow_pred, flow_gt, reduction="mean").item()

    test_loss_mae = test_loss_sum_mae / len(test_loader)
    test_loss_rmse = np.sqrt(test_loss_sum_mse / len(test_loader))
    print(
        "\n[Test] L1 loss: {:.4f}, RMSE Loss: {:.4f}\n".format(
            test_loss_mae, test_loss_rmse
        )
    )

    scheduler.step(test_loss_rmse)

    return test_loss_rmse


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
    train_dir = os.path.join(cfg["OUTPUT_DIR"], cfg["FLOW"]["MODEL"]["NAME"])
    os.makedirs(train_dir, exist_ok=True)

    # Set random seeds.
    random.seed(cfg["FLOW"]["TRAIN"]["SEED"])
    np.random.seed(cfg["FLOW"]["TRAIN"]["SEED"])
    torch.manual_seed(cfg["FLOW"]["TRAIN"]["SEED"])

    # Set training params.
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": cfg["FLOW"]["TRAIN"]["BATCH_SIZE"], "drop_last": True}
    test_kwargs = {"batch_size": cfg["FLOW"]["TEST"]["BATCH_SIZE"]}

    if use_cuda:
        cuda_kwargs = {
            "num_workers": 0,
            "shuffle": True,
            "worker_init_fn": lambda id: np.random.seed(
                id * cfg["FLOW"]["TRAIN"]["SEED"]
            ),
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Prepare for the dataset
    print("Loading dataset...")
    train_paths = [
        os.path.join(cfg["DATASET"]["PATH"], x + ".npz")
        for x in cfg["DATASET"]["TRAIN_SPLIT"]
    ]
    train_datasets = [
        dataloader.FlowDataset(
            path,
            subsample_factor=cfg["FLOW"]["DATA"]["SUBSAMPLE_FACTOR"],
            transform=dataloader.FlipFlow(),
        )
        for path in sorted(train_paths)
    ]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    test_paths = [
        os.path.join(cfg["DATASET"]["PATH"], x + ".npz")
        for x in cfg["DATASET"]["VAL_SPLIT"]
    ]
    test_datasets = [
        dataloader.FlowDataset(
            path,
            subsample_factor=cfg["FLOW"]["DATA"]["SUBSAMPLE_FACTOR"],
            transform=None,
        )
        for path in sorted(test_paths)
    ]
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # Load network.
    model_kwargs = {
        "n_channels": cfg["FLOW"]["MODEL"]["N_CHANNELS"],
        "n_outputs": cfg["FLOW"]["MODEL"]["N_OUTPUTS"],
    }
    net = getattr(model, cfg["FLOW"]["MODEL"]["TYPE"])(**model_kwargs).to(device)

    optimizer = optim.Adam(net.parameters(), lr=cfg["FLOW"]["TRAIN"]["LR"])

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=10
    )

    train_loss_array = []
    test_loss_array = []

    least_test_loss = 1000
    for epoch in range(1, cfg["FLOW"]["TRAIN"]["EPOCHS"] + 1):
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
                    "model_name": cfg["FLOW"]["MODEL"]["NAME"],
                    "model_type": type(net).__name__,
                    "model_kwargs": model_kwargs,
                    "model_state_dict": net.state_dict(),
                    "epoch": epoch,
                    "test_loss": test_loss,
                },
                os.path.join(train_dir, f"{cfg['FLOW']['MODEL']['NAME']}.pth"),
            )
