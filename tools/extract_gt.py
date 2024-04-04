#!/usr/bin/env python3

"""
Extract ground-truth from .bag file.
"""

import os
import sys

import argparse

import numpy as np
np.set_printoptions(precision=3, floatmode="fixed", sign=" ")

from radarize.config import cfg, update_config

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", help="experiment configure file name", default=None, type=str
    )
    parser.add_argument("--npz_path", help="Path to npz.", default=None, required=False)
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

    # Create output dir.
    out_dir = os.path.join(os.path.join(cfg["OUTPUT_DIR"], "gt"))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Get list of npz files.
    if args.npz_path:
        npz_paths = [args.npz_path]
    else:
        npz_paths = sorted(
            glob.glob(os.path.join(cfg["DATASET"]["TEST_PATH"], "*.npz"))
        )

    for npz_path in npz_paths:
        print(f"Processing {npz_path}...")
        # Load npz.
        with np.load(npz_path) as npz:
            data = {
                k: npz[k] for k in npz.files if k in ["time", "pose_gt", "depth_map"]
            }

        for k, v in data.items():
            print(f"{k}: {v.shape}")

        basename = os.path.basename(npz_path)

        # Save trajectory.
        trajectory = np.concatenate(
            [
                data["time"].reshape(-1, 1),
                data["pose_gt"],
            ],
            axis=1,
        )
        np.savetxt(
            os.path.join(out_dir, basename.replace(".npz", ".txt")),
            trajectory,
            delimiter=" ",
        )

        # Save depth map.
        np.savez(
            os.path.join(out_dir, basename),
            time=data["time"],
            depth_map=data["depth_map"],
        )
