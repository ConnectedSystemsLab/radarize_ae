#!/usr/bin/env python3

import os
import sys

import argparse
import glob
import multiprocessing
import subprocess

from radarize.config import cfg, update_config

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        help="experiment configure file name",
        default="configs/default.yaml",
        type=str,
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        default=1,
        help="Number of processes to use for parallel processing.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    return args


def run_commands(cmds, n_proc):
    with multiprocessing.Pool(n_proc) as pool:
        pool.map(subprocess.run, cmds)


if __name__ == "__main__":
    args = args()
    update_config(cfg, args)

    # Preprocess datasets. (bag -> npz)
    bag_paths = sorted(glob.glob(os.path.join(cfg["DATASET"]["PATH"], "*.bag")))
    bag_paths = [x for x in bag_paths if not os.path.exists(x.replace(".bag", ".npz"))]
    run_commands(
        [
            [f"tools/create_dataset.py", f"--cfg={args.cfg}", f"--bag_path={x}"]
            for x in bag_paths
        ],
        args.n_proc,
    )

    train_npz_paths = sorted(
        [
            os.path.join(cfg["DATASET"]["PATH"], os.path.basename(x) + ".npz")
            for x in cfg["DATASET"]["TRAIN_SPLIT"]
        ]
    )
    test_npz_paths = sorted(
        [
            os.path.join(cfg["DATASET"]["PATH"], os.path.basename(x) + ".npz")
            for x in cfg["DATASET"]["TEST_SPLIT"]
        ]
    )

    # Extract ground truth.
    run_commands(
        [
            ["tools/extract_gt.py", f"--cfg={args.cfg}", f"--npz_path={x}"]
            for x in test_npz_paths
        ],
        args.n_proc,
    )

    # Train flow models.
    subprocess.run(["tools/train_flow.py", f"--cfg={args.cfg}"], check=True)
    subprocess.run(["tools/test_flow.py", f"--cfg={args.cfg}"], check=True)

    # Train rotnet models.
    subprocess.run(["tools/train_rot.py", f"--cfg={args.cfg}"], check=True)
    subprocess.run(["tools/test_rot.py", f"--cfg={args.cfg}"], check=True)

    # Extract odometry.
    run_commands(
        [
            ["tools/test_odom.py", f"--cfg={args.cfg}", f"--npz_path={x}"]
            for x in test_npz_paths
        ],
        args.n_proc,
    )

    # Train UNet
    subprocess.run(["tools/train_unet.py", f"--cfg={args.cfg}"], check=True)
    run_commands(
        [
            ["tools/test_unet.py", f"--cfg={args.cfg}", f"--npz_path={x}"]
            for x in test_npz_paths
        ],
        args.n_proc,
    )

    ### Run Cartographer.
    # Get ground truth.
    subprocess.run(
        [
            "tools/run_carto.py",
            f"--cfg={args.cfg}",
            f"--n_proc=1",
            f"--odom=gt",
            f"--scan=gt",
            f"--params=default",
        ],
        check=True,
    )

    # RadarHD baseline.
    subprocess.run(
        [
            "tools/run_carto.py",
            f"--cfg={args.cfg}",
            f"--n_proc=1",
            f"--odom=gt",
            f"--scan=radarhd",
            f"--params=scan_only",
        ],
        check=True,
    )

    # RNIN + RadarHD baseline.
    subprocess.run(
        [
            "tools/run_carto.py",
            f"--cfg={args.cfg}",
            f"--n_proc=1",
            f"--odom=rnin",
            f"--scan=radarhd",
            f"--params=default",
        ],
        check=True,
    )

    # milliEgo + RadarHD baseline.
    subprocess.run(
        [
            "tools/run_carto.py",
            f"--cfg={args.cfg}",
            f"--n_proc=1",
            f"--odom=milliego",
            f"--scan=radarhd",
            f"--params=default",
        ],
        check=True,
    )

    # Our odometry + RadarHD baseline.
    subprocess.run(
        [
            "tools/run_carto.py",
            f"--cfg={args.cfg}",
            f"--n_proc=1",
            f"--odom=odometry",
            f"--scan=radarhd",
            f"--params=radar",
        ],
        check=True,
    )

    # Run radarize.
    subprocess.run(
        [
            "tools/run_carto.py",
            f"--cfg={args.cfg}",
            f"--n_proc=1",
            f"--odom=odometry",
            f"--scan=unet",
            f"--params=radar",
        ],
        check=True,
    )

