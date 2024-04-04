#!/usr/bin/env python3

import os
import sys

import argparse
import torch
from tqdm import tqdm
import numpy as np

np.set_printoptions(precision=3, floatmode="fixed", sign=" ")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from radarize.flow import model
from radarize.flow.dataloader import FlowDataset
from radarize.utils import image_tools
from radarize.config import cfg, update_config


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


if __name__ == "__main__":
    args = args()
    update_config(cfg, args)

    device = torch.device("cpu" if args.no_cuda else "cuda")

    # Load Trained NN
    saved_model = torch.load(
        os.path.join(
            cfg["OUTPUT_DIR"],
            cfg["FLOW"]["MODEL"]["NAME"],
            f"{cfg['FLOW']['MODEL']['NAME']}.pth",
        )
    )
    model_name = saved_model["model_name"]
    model_type = saved_model["model_type"]
    model_kwargs = saved_model["model_kwargs"]
    state_dict = saved_model["model_state_dict"]
    net = getattr(model, model_type)(**model_kwargs).to(device)
    net.load_state_dict(state_dict)
    net.eval()

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

    mean_pred_mae = []
    mean_pred_rmse = []

    for path in npz_paths:
        print(f"Processing {path}...")

        dataset = FlowDataset(
            path,
            subsample_factor=cfg["FLOW"]["DATA"]["SUBSAMPLE_FACTOR"],
            transform=None,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )

        times = []
        flow_pred_xs, flow_pred_ys = [], []
        flow_gt_xs, flow_gt_ys = [], []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                curr_time = batch["time"].cpu().numpy()
                x = torch.cat([batch["radar_d"], batch["radar_de"]], axis=1).to(
                    torch.float32
                )
                flow_gt = batch["velo_gt"].cpu()

                flow_pred = net(x)
                flow_pred = flow_pred.cpu()
                flow_pred = torch.squeeze(flow_pred, dim=1)

                flow_x, flow_y = flow_pred[:, 0].numpy(), flow_pred[:, 1].numpy()
                flow_pred_xs.append(flow_x)
                flow_pred_ys.append(flow_y)

                flow_gt_x, flow_gt_y = flow_gt[:, 0].numpy(), flow_gt[:, 1].numpy()
                flow_gt_xs.append(flow_gt_x)
                flow_gt_ys.append(flow_gt_y)

                times.append(curr_time)

        flow_pred_xs, flow_pred_ys = np.squeeze(np.array(flow_pred_xs)), np.squeeze(
            np.array(flow_pred_ys)
        )
        flow_gt_xs, flow_gt_ys = np.squeeze(np.array(flow_gt_xs)), np.squeeze(
            np.array(flow_gt_ys)
        )

        # altitudes, altitudes_gt = np.array(altitudes), np.array(altitudes_gt)

        print(f"MAE x: {np.mean(np.abs(flow_pred_xs - flow_gt_xs)):.3f}")
        print(f"MAE y: {np.mean(np.abs(flow_pred_ys - flow_gt_ys)):.3f}")

        print(f"RMSE x: {np.sqrt(np.mean((flow_pred_xs - flow_gt_xs)**2)):.3f}")
        print(f"RMSE y: {np.sqrt(np.mean((flow_pred_ys - flow_gt_ys)**2)):.3f}")

        print(f"err_mean x: {np.mean((flow_pred_xs - flow_gt_xs)):.3f}")
        print(f"err_std x:  {np.std((flow_pred_xs - flow_gt_xs)):.3f}")

        print(f"err_mean y: {np.mean((flow_pred_ys - flow_gt_ys)):.3f}")
        print(f"err_std y:  {np.std((flow_pred_ys - flow_gt_ys)):.3f}")

        pred_mae = (
            np.mean(np.abs(flow_pred_xs - flow_gt_xs))
            + np.mean(np.abs(flow_pred_ys - flow_gt_ys))
        ) / 2
        pred_rmse = (
            np.sqrt(np.mean((flow_pred_xs - flow_gt_xs) ** 2))
            + np.sqrt(np.mean((flow_pred_ys - flow_gt_ys) ** 2))
        ) / 2

        mean_pred_mae.append(pred_mae)
        mean_pred_rmse.append(pred_rmse)

        fig, ax = plt.subplots(4, 1, sharex=True, figsize=(5, 8))

        ax[0].set_title(
            f"MAE x: {np.mean(np.abs(flow_pred_xs - flow_gt_xs)):.3f} RMSE x: {np.sqrt(np.mean((flow_pred_xs - flow_gt_xs)**2)):.3f}"
        )
        ax[0].plot(flow_gt_xs, label="velo_gt_x", color="b")
        ax[0].plot(flow_pred_xs, label="velo_x", color="r")
        ax[0].set_ylim(-1, 1)

        ax[1].set_title(
            f"err mean x: {np.mean((flow_pred_xs - flow_gt_xs)):.3f} stdev: {np.std((flow_pred_xs - flow_gt_xs)):.3f}"
        )
        ax[1].plot(flow_pred_xs - flow_gt_xs, label="err_x", color="g")
        ax[1].set_ylim(-1, 1)

        ax[2].set_title(
            f"MAE y: {np.mean(np.abs(flow_pred_ys - flow_gt_ys)):.3f} RMSE y: {np.sqrt(np.mean((flow_pred_ys - flow_gt_ys)**2)):.3f}"
        )
        ax[2].plot(flow_gt_ys, label="velo_gt_y", color="b")
        ax[2].plot(flow_pred_ys, label="velo_y", color="r")
        ax[2].set_ylim(-1, 1)

        ax[3].set_title(
            f"err mean y: {np.mean((flow_pred_ys - flow_gt_ys)):.3f} stdev: {np.std((flow_pred_ys - flow_gt_ys)):.3f}"
        )
        ax[3].plot(flow_pred_ys - flow_gt_ys, label="err_y", color="g")
        ax[3].set_ylim(-1, 1)

        fig.tight_layout()
        fig.legend()
        print(
            os.path.join(
                test_res_dir, os.path.basename(os.path.splitext(path)[0] + ".jpg")
            )
        )
        fig.savefig(
            os.path.join(
                test_res_dir, os.path.basename(os.path.splitext(path)[0] + ".jpg")
            )
        )
        plt.close(fig)

        d = {
            "time": times,
            "flow_pred_xs": flow_pred_xs,
            "flow_pred_ys": flow_pred_ys,
            "flow_gt_xs": flow_gt_xs,
            "flow_gt_ys": flow_gt_ys,
        }
        np.savez(
            os.path.join(
                test_res_dir, os.path.basename(os.path.splitext(path)[0] + ".npz")
            ),
            **d,
        )

    with open(os.path.join(test_res_dir, "metrics.txt"), "w") as f:
        f.writelines(s + "\n" for s in npz_paths)
        f.write(
            f"pred mae total {np.mean(mean_pred_mae):.3f} pred rmse total {np.mean(mean_pred_rmse):.3f}"
        )
        # print(f"pred mae total {np.mean(mean_pred_mae):.3f} pred rmse total {np.mean(mean_pred_rmse):.3f}")
