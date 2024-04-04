#!/usr/bin/env python3

import os
import sys

import argparse

import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm

np.set_printoptions(precision=3, floatmode="fixed", sign=" ")

import matplotlib
matplotlib.use("Agg")
from collections import defaultdict

import imageio
import imageio.v2 as iio
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from radarize.config import cfg, update_config
from radarize.rotnet import dataloader, model
from radarize.rotnet.dataloader import RotationDataset
from radarize.utils import image_tools


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


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", help="experiment configure file name", default=None, type=str
    )
    parser.add_argument("--npz_path", help="Path to npz.", default=None, required=False)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    return args


def visualize_rotation(pred, gt):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16, 8))

    scale_factor = cfg["ROTNET"]["TEST"]["SEQ_LEN"] / 30

    ax[0].set_title(
        f"MAE x: {np.mean(np.abs(pred - gt)):.3f} RMSE x: {np.sqrt(np.mean((pred - gt)**2)):.3f}"
    )
    ax[0].plot(pred, label="pred", color="r")
    ax[0].plot(gt, label="gt", color="b")
    ax[0].set_ylim(-2.0 * scale_factor, 2.0 * scale_factor)
    ax[0].grid()

    ax[1].set_title(
        f"err mean x: {np.mean((pred - gt)):.3f} stdev: {np.std((pred - gt)):.3f}"
    )
    ax[1].plot(np.abs(pred - gt), label="err", color="g")
    ax[1].set_ylim(-0.0, 2.0 * scale_factor)

    fig.tight_layout()
    fig.legend()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image, np.uint8)
    rgb_image = image[:, :, :3]

    plt.close(fig)

    return rgb_image


def test(net, device, test_loader):
    net.eval()

    test_l1_loss = 0
    test_mse_loss = 0

    # runtimes = []

    d = defaultdict(list)
    m = {}

    with torch.no_grad():
        for batch in tqdm(test_loader):
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

            d["time"].append(batch["time"].cpu().numpy())
            d["rot_pred"].append(y_pred.cpu().numpy().astype(np.float64))
            d["rot_gt"].append(delta_theta)

    test_l1_loss = test_l1_loss / len(test_loader)
    test_rmse_loss = np.sqrt(test_mse_loss / len(test_loader))

    print(f"\n[Test] MAE loss: {test_l1_loss:.6f} RMSE loss: {test_rmse_loss:.6f}")
    # average_runtime = np.mean(runtimes)
    # print(f'Average runtime: {average_runtime:.3f} seconds')

    # For visualization
    for k, v in d.items():
        d[k] = np.concatenate(v, axis=0)
        print(k, d[k].shape, d[k].dtype)

    # Metrics
    # m['runtime'] = average_runtime
    m["l1_loss"] = test_l1_loss
    m["rmse_loss"] = test_rmse_loss

    return d, m


if __name__ == "__main__":
    args = args()
    update_config(cfg, args)

    device = torch.device("cpu" if args.no_cuda else "cuda")

    # Load Trained NN
    saved_model = torch.load(
        os.path.join(
            cfg["OUTPUT_DIR"],
            cfg["ROTNET"]["MODEL"]["NAME"],
            f"{cfg['ROTNET']['MODEL']['NAME']}.pth",
        )
    )
    model_type = saved_model["model_type"]
    model_name = saved_model["model_name"]
    print(model_name, device)
    state_dict = saved_model["model_state_dict"]
    model_kwargs = saved_model["model_kwargs"]
    net = getattr(model, model_type)(**model_kwargs).to(device)
    net.load_state_dict(state_dict)

    # Create output dir.
    test_res_dir = os.path.join(os.path.join(cfg["OUTPUT_DIR"], model_name))
    if not os.path.exists(test_res_dir):
        os.makedirs(test_res_dir)

    # Get list of bag files in root directory.
    if args.npz_path:
        npz_paths = [args.npz_path]
    else:
        npz_paths = sorted(
            [
                os.path.join(cfg["DATASET"]["PATH"], x + ".npz")
                for x in cfg["DATASET"]["TEST_SPLIT"]
            ]
        )

    all_metrics = defaultdict(list)

    for path in npz_paths:
        print(f"Processing {path}...")

        dataset = RotationDataset(
            path,
            subsample_factor=cfg["ROTNET"]["DATA"]["SUBSAMPLE_FACTOR"],
            seq_len=cfg["ROTNET"]["TEST"]["SEQ_LEN"],
            random_seq_len=False,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )
        d, m = test(net, device, test_loader)

        # Save output
        np.savez(os.path.join(test_res_dir, os.path.basename(path)), **d)

        # Save metrics
        fname = os.path.join(
            test_res_dir, os.path.basename(path).replace(".npz", ".txt")
        )
        with open(fname, "w") as f:
            for k, v in m.items():
                f.write(f"{k}: {v:.6f}\n")

        all_metrics["mae"].append(m["l1_loss"])
        all_metrics["rmse"].append(m["rmse_loss"])

        # Save plots
        fname = fname.replace(".txt", ".jpg")
        im = visualize_rotation(d["rot_pred"], d["rot_gt"])
        imageio.imwrite(fname, im)

    # Save average metrics
    fname = os.path.join(test_res_dir, "metrics.txt")
    with open(fname, "w") as f:
        for k, v in all_metrics.items():
            f.write(f"{k}: {np.mean(v):.6f}\n")
