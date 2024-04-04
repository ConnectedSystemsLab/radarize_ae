#!/usr/bin/env python3

import os
import sys

import argparse
import time

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

np.set_printoptions(precision=3, floatmode="fixed", sign=" ")

import matplotlib

matplotlib.use("Agg")
from collections import defaultdict

import imageio.v2 as iio
import matplotlib.pyplot as plt

from radarize.config import cfg, update_config
from radarize.unet import model
from radarize.unet.dataloader import UNetDataset
from radarize.unet.dice_score import dice_loss
from radarize.utils import image_tools


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


def confmap2range(confmap):
    device = confmap.device
    confmap = confmap.cpu().numpy()
    bin2range = np.linspace(0, cfg["DATASET"]["RR_MAX"], cfg["DATASET"]["RAMAP_RSIZE"])
    # confmap = np.squeeze(confmap)
    range = bin2range[np.argmin(confmap, axis=1)]
    # remove areas without wall
    range[np.min(confmap, axis=1) > 0.85] = cfg["DATASET"]["RR_MAX"]
    return torch.from_numpy(range).to(device)


def range2confmap(range):
    device = range.device
    range = range.cpu().numpy()
    bin_size = cfg["DATASET"]["RR_MAX"] / cfg["DATASET"]["RAMAP_RSIZE"]
    confmap = np.zeros((cfg["DATASET"]["RAMAP_RSIZE"], range.shape[0]))
    for i, r in enumerate(range):
        confmap[int(r // bin_size), i] = 1
    return torch.from_numpy(confmap).to(device)


def visualize_range(input, output, gt):
    fig = plt.figure(figsize=(9, 3))

    fig.add_subplot(1, 3, 1)
    plt.imshow(input, origin="lower", aspect="equal")
    plt.axis("off")
    plt.title("Radar Heatmap")

    fig.add_subplot(1, 3, 2)
    plt.imshow(output, origin="lower", aspect="equal")
    plt.axis("off")
    plt.title("Output")

    fig.add_subplot(1, 3, 3)
    plt.imshow(gt, origin="lower", aspect="equal")
    plt.axis("off")
    plt.title("Ground Truth")

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

    test_dice_loss = 0
    test_mse_loss = 0

    runtimes = []

    d = defaultdict(list)
    m = {}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
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

            tic = time.time()
            y_pred = net(x)
            y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
            # y_pred = (y_pred == 1)
            toc = time.time()
            # print(f'runtime: {toc - tic:.3f} s')
            runtimes.append(toc - tic)

            d["time"].append(batch["time"][:, -1].cpu().numpy())
            d["radar_r"].append(batch["radar_r_3"].cpu().numpy())
            d["depth_map_pred"].append(y_pred.cpu().numpy().astype(np.float64))
            d["depth_map"].append(y.cpu().numpy().astype(np.float64))

            test_dice_loss += dice_loss(y_pred, y, multiclass=True).item()
            test_mse_loss += F.mse_loss(y_pred, y).item()

    test_dice_loss = test_dice_loss / len(test_loader)
    test_mse_loss = test_mse_loss / len(test_loader)
    print(f"\n[Test] Dice loss: {test_dice_loss:.6f} MSE loss: {test_mse_loss:.6f}")
    average_runtime = np.mean(runtimes)
    print(f"Average runtime: {average_runtime:.3f} seconds")

    # For visualization
    for k, v in d.items():
        d[k] = np.concatenate(v, axis=0)
        print(k, d[k].shape, d[k].dtype)

    # Metrics
    m["runtime"] = average_runtime
    m["dice_loss"] = test_dice_loss
    m["mse_loss"] = test_mse_loss

    return d, m


if __name__ == "__main__":
    args = args()
    update_config(cfg, args)

    device = torch.device("cpu" if args.no_cuda else "cuda")

    # Load Trained NN
    saved_model = torch.load(
        os.path.join(
            cfg["OUTPUT_DIR"],
            cfg["UNET"]["MODEL"]["NAME"],
            f"{cfg['UNET']['MODEL']['NAME']}.pth",
        )
    )
    model_name = saved_model["model_name"]
    model_type = saved_model["model_type"]
    state_dict = saved_model["model_state_dict"]
    model_kwargs = saved_model["model_kwargs"]
    net = getattr(model, model_type)(**model_kwargs).to(device)
    net.load_state_dict(state_dict)
    print(model_name, device)

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

    for path in npz_paths:
        print(f"Processing {path}...")

        dataset = UNetDataset(path, seq_len=1)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )
        d, m = test(net, device, test_loader)

        # Visualize output
        # with iio.get_writer(os.path.join(test_res_dir, os.path.basename(path).replace('.npz', '.mp4')),
        #                     format='FFMPEG',
        #                     mode='I',
        #                     fps=30) as writer:
        #     for i in tqdm(range(len(d['time']))):
        #         radar_r        = d['radar_r'][i][-1,...]
        #         depth_map_pred = d['depth_map_pred'][i][0,...]
        #         depth_map      = d['depth_map'][i][0,...]

        #         writer.append_data(visualize_range(radar_r,
        #                                            depth_map_pred,
        #                                            depth_map))

        # Save output
        np.savez(
            os.path.join(test_res_dir, os.path.basename(path)),
            time=d["time"],
            depth_map=d["depth_map_pred"],
        )

        # Save metrics
        fname = os.path.join(
            test_res_dir, os.path.basename(path).replace(".npz", ".txt")
        )
        with open(fname, "w") as f:
            for k, v in m.items():
                f.write(f"{k}: {v:.6f}\n")
