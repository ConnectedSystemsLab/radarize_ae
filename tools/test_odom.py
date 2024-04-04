#!/usr/bin/env python3

import os
import sys

import argparse

import numpy as np
import PIL.Image as Image
import torch
from tqdm import tqdm

np.set_printoptions(precision=3, floatmode="fixed", sign=" ")

import matplotlib as mpl
mpl.use("Agg")

from collections import defaultdict

import imageio
import imageio.v2 as iio
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset

import radarize.flow.model as translation_model
import radarize.rotnet.model as rotation_model
from radarize.config import cfg, update_config
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


def yaw2quat(yaw):
    """Convert yaw angle to quaternion.
    Args:
        yaw: (N, 1) array of yaw angles
    Returns:
        q: (N, 4) array of quaternions
    """
    padded = np.concatenate([yaw, np.zeros_like(yaw), np.zeros_like(yaw)], axis=1)
    return R.from_euler("ZYX", padded, degrees=False).as_quat()


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


def gen_arrow_head_marker(rot):
    """generate a marker to plot with matplotlib scatter, plot, ...
    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    rot=0: positive x direction
    Parameters
    ----------
    rot : float
        rotation in degree
        0 is positive x direction
    Returns
    -------
    arrow_head_marker : Path
        use this path for marker argument of plt.scatter
    scale : float
        multiply a argument of plt.scatter with this factor got get markers
        with the same size independent of their rotation.
        Paths are autoscaled to a box of size -1 <= x, y <= 1 by plt.scatter
    """
    arr = np.array([[0.1, 0.3], [0.1, -0.3], [1, 0], [0.1, 0.3]])  # arrow shape
    angle = rot / 180 * np.pi
    rot_mat = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    # scale
    x0 = np.amin(arr[:, 0])
    x1 = np.amax(arr[:, 0])
    y0 = np.amin(arr[:, 1])
    y1 = np.amax(arr[:, 1])
    scale = np.amax(np.abs([x0, x1, y0, y1]))
    codes = [
        mpl.path.Path.MOVETO,
        mpl.path.Path.LINETO,
        mpl.path.Path.LINETO,
        mpl.path.Path.CLOSEPOLY,
    ]
    arrow_head_marker = mpl.path.Path(arr, codes)
    return arrow_head_marker, scale


def compare_trajectory(pred, gt):
    """Plot trajectory with matplotlib
    Parameters
    ----------
    pred : np.ndarray
        shape (N, 3)
    gt : np.ndarray
        shape (N, 3)
    Returns
    -------
    rgb_image : np.ndarray
        shape (H, W, 3)
    """

    fig, ax = plt.subplots(1, 1, dpi=90, figsize=(9, 9))
    ax.plot(pred[:, 0], pred[:, 1], label="pred", color="r")
    ax.plot(gt[:, 0], gt[:, 1], label="gt", color="b")
    for i in range(0, pred.shape[0], 100):
        yaw = np.rad2deg(pred[i, 2])
        marker, scale = gen_arrow_head_marker(yaw)
        ax.scatter(pred[i, 0], pred[i, 1], marker=marker, s=(35 * scale) ** 2, c="g")
    for i in range(0, gt.shape[0], 100):
        yaw = np.rad2deg(gt[i, 2])
        marker, scale = gen_arrow_head_marker(yaw)
        ax.scatter(gt[i, 0], gt[i, 1], marker=marker, s=(35 * scale) ** 2, c="y")
    ax.set_aspect("equal")

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


class OdomDataset(Dataset):
    """Interface for .npz files."""

    topics = [
        "time",
        "radar_d",
        "radar_de",
        "radar_r_1",
        "radar_r_3",
        "radar_r_5",
        "radar_re_1",
        "radar_re_3",
        "radar_re_5",
        "pose_gt",
    ]

    def __init__(self, path, subsample_factor=1):
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

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = {
            k: (
                torch.from_numpy(self.dataset[k][idx])
                if type(self.dataset[k][idx]) is np.ndarray
                else self.dataset[k][idx]
            )
            for k in self.files
        }
        return sample


class OdomEstimator:
    def __init__(self, trans_model, rot_model, device="cuda:0"):
        self.device = device
        self.trans_model = trans_model.to(device)
        self.rot_model = rot_model.to(device)

    def __call__(self, dataset):
        """Estimate odometry from dataset.
        Parameters
        ----------
        dataset : OdomDataset
        Returns
        -------
        timestamps : np.ndarray
            shape (N,)
        states: np.ndarray
            shape (N, 3)
        """

        # States
        timestamps = []
        xs, ys, thetas = [], [], []
        last_kf = 0
        keyframes = []
        keyframe_poses = []

        # Create dataloader.
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # Parameters for keyframe creation.
        delay = cfg["ODOM"]["PARAMS"]["DELAY"]
        kf_delay = cfg["ODOM"]["PARAMS"]["KF_DELAY"]
        pos_thresh = cfg["ODOM"]["PARAMS"]["POS_THRESH"]
        yaw_thresh = cfg["ODOM"]["PARAMS"]["YAW_THRESH"]
        print(
            "delay",
            delay,
            "kf_delay",
            kf_delay,
            "pos_thresh",
            pos_thresh,
            "yaw_thresh",
            yaw_thresh,
        )

        # Loop over input trajectory.
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                # Read data.
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                curr_time = batch["time"].item()
                da_heatmap = torch.cat(
                    [batch["radar_d"][:, 0:1], batch["radar_de"][:, 0:1]], axis=1
                ).to(torch.float32)
                ra_heatmap = torch.cat(
                    [
                        batch["radar_r_1"][:, 0:1],
                        batch["radar_r_3"][:, 0:1],
                        batch["radar_r_5"][:, 0:1],
                    ],
                    axis=1,
                ).to(torch.float32)
                pose_gt = batch["pose_gt"].squeeze().cpu().numpy()

                # Initialize state for first few frames.
                if i < kf_delay * delay:
                    timestamps.append(curr_time)
                    xs.append(pose_gt[0])
                    ys.append(pose_gt[1])
                    thetas.append(quat2yaw(pose_gt[3:]))
                    if i % delay == 0:
                        last_kf = i
                        keyframes.append(ra_heatmap)
                        keyframe_poses.append(
                            [pose_gt[0], pose_gt[1], quat2yaw(pose_gt[3:])]
                        )
                    continue

                # Predict translation.
                vel_pred = self.trans_model(da_heatmap).squeeze().cpu().numpy()
                dt = curr_time - timestamps[-1]
                x_pred = (
                    xs[-1]
                    + dt * vel_pred[0] * np.cos(thetas[-1])
                    - dt * vel_pred[1] * np.sin(thetas[-1])
                )
                y_pred = (
                    ys[-1]
                    + dt * vel_pred[0] * np.sin(thetas[-1])
                    + dt * vel_pred[1] * np.cos(thetas[-1])
                )

                # Predict rotation.
                stacked_ra = torch.cat(
                    [keyframes[-kf_delay], torch.flip(ra_heatmap, [1])], axis=1
                ).to(torch.float32)
                stacked_ra_rev = torch.flip(stacked_ra, [1])
                rot_pred_f = self.rot_model(stacked_ra).item()
                rot_pred_r = self.rot_model(stacked_ra_rev).item()
                rot_pred = (rot_pred_f - rot_pred_r) / 2
                yaw_pred = keyframe_poses[-kf_delay][2] + rot_pred
                yaw_pred = normalize_angle(yaw_pred)

                # Create keyframe if necessary.
                time_constraint = (i - last_kf) % delay == 0
                pos_constraint = (
                    np.linalg.norm(
                        np.array(
                            [
                                x_pred - keyframe_poses[-kf_delay][0],
                                y_pred - keyframe_poses[-kf_delay][1],
                            ]
                        )
                    )
                    > pos_thresh
                )
                yaw_constraint = np.abs(yaw_pred - keyframe_poses[-1][2]) > yaw_thresh
                if time_constraint or yaw_constraint or pos_constraint:
                    last_kf = i
                    keyframes.append(ra_heatmap)
                    keyframe_poses.append([x_pred, y_pred, yaw_pred])

                # Update state.
                timestamps.append(curr_time)
                xs.append(x_pred)
                ys.append(y_pred)
                thetas.append(yaw_pred)

        timestamps = np.array(timestamps)
        states = np.stack([xs, ys, thetas], axis=1)

        return timestamps, states


if __name__ == "__main__":
    args = args()
    update_config(cfg, args)

    device = torch.device("cpu" if args.no_cuda else "cuda")

    # Load translation net
    saved_model = torch.load(
        os.path.join(
            cfg["OUTPUT_DIR"],
            cfg["ODOM"]["MODELS"]["TRANS"],
            f"{cfg['ODOM']['MODELS']['TRANS']}.pth",
        )
    )
    model_name = saved_model["model_name"]
    model_type = saved_model["model_type"]
    model_kwargs = saved_model["model_kwargs"]
    state_dict = saved_model["model_state_dict"]
    trans_net = getattr(translation_model, model_type)(**model_kwargs).to(device)
    trans_net.load_state_dict(state_dict)
    trans_net.eval()

    # Load rotation net
    saved_model = torch.load(
        os.path.join(
            cfg["OUTPUT_DIR"],
            cfg["ODOM"]["MODELS"]["ROT"],
            f"{cfg['ODOM']['MODELS']['ROT']}.pth",
        )
    )
    model_type = saved_model["model_type"]
    model_name = saved_model["model_name"]
    state_dict = saved_model["model_state_dict"]
    model_kwargs = saved_model["model_kwargs"]
    rot_net = getattr(rotation_model, model_type)(**model_kwargs).to(device)
    rot_net.load_state_dict(state_dict)
    rot_net.eval()

    estimator = OdomEstimator(trans_net, rot_net, device)

    # Create output dir.
    test_res_dir = os.path.join(
        os.path.join(cfg["OUTPUT_DIR"], cfg["ODOM"]["OUTPUT_DIR"])
    )
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

        dataset = OdomDataset(
            path, subsample_factor=cfg["ODOM"]["PARAMS"]["SUBSAMPLE_FACTOR"]
        )
        timestamps, pred = estimator(dataset)
        gt = dataset.dataset["pose_gt"]
        gt = np.concatenate([gt[:, 0:1], gt[:, 1:2], quat2yaw(gt[:, 3:])], axis=1)

        # Save trajectory as image.
        im = compare_trajectory(pred, gt)
        fname = os.path.join(
            test_res_dir, os.path.basename(path).replace(".npz", ".png")
        )
        imageio.imwrite(fname, im)

        # Save trajectory as TUM.
        timestamps = timestamps[:, None]
        pos = np.concatenate([pred[:, :2], np.zeros((pred.shape[0], 1))], axis=1)
        quats = yaw2quat(pred[:, 2:])
        trajectory = np.concatenate([timestamps, pos, quats], axis=1)
        fname = os.path.join(
            test_res_dir, os.path.basename(path).replace(".npz", ".txt")
        )
        np.savetxt(fname, trajectory, delimiter=" ")
