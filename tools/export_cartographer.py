#!/usr/bin/env python3

"""
Creates input .bags to feed into Cartographer.
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("agg")
import argparse
from collections import defaultdict

import rosbag
from rospy import Time
from nav_msgs.msg import Odometry
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from sensor_msgs.msg import LaserScan
from tqdm import tqdm

from radarize.config import cfg, update_config

def heatmap2range(heatmap, range_bins, range_min, range_max):
    """Convert range-azimuth heatmap to range scan."""

    # Assign range value to each angle bin
    range_value = np.linspace(range_min, range_max, range_bins)
    range_scan = range_value[np.argmin(heatmap, axis=0)]

    # Filter angle bins without collision
    range_scan[np.min(heatmap, axis=0) > 0.85] = range_max + 1

    return range_scan


def sensorDataPreprocessing(odom_path, scan_path, output_path):
    # Load odom from TUM format and scans
    odom_data = np.loadtxt(odom_path)
    scan_data = np.load(scan_path, allow_pickle=True)

    odom_times = odom_data[:, 0]
    odom_poses = odom_data[:, 1:]
    scan_times = scan_data["time"]
    scan_heatmaps = scan_data["depth_map"]
    # Trim scan to match odom
    valid_idx = np.logical_and(scan_times > odom_times[0], scan_times < odom_times[-1])
    scan_times = scan_times[valid_idx]
    scan_heatmaps = scan_heatmaps[valid_idx]

    print(odom_times[0], odom_times[-1])
    print(scan_times[0], scan_times[-1])
    print("odom_times", odom_times.shape)
    print("odom_poses", odom_poses.shape)
    print("scan_times", scan_times.shape)
    print("scan_heatmaps", scan_heatmaps.shape)

    # Interpolate odom to match scan times
    odom_p_interp = interp1d(
        odom_times, odom_poses[:, :3], kind="linear", axis=0, fill_value="extrapolate"
    )(scan_times)
    odom_q_interp = Slerp(odom_times, R.from_quat(odom_poses[:, 3:]))(scan_times)

    # Write to bag
    with rosbag.Bag(output_path, "w") as bag:
        last_ts = Time.from_sec(0.0)
        for i in tqdm(range(scan_heatmaps.shape[0])):

            ts = Time.from_sec(scan_times[i])
            if ts < last_ts:
                print("Warning: timestamp is not increasing")

            # Create laser scan message
            range_bins, angle_bins = scan_heatmaps[i].shape[1:]
            range_msg = LaserScan()
            range_msg.header.stamp = ts
            range_msg.header.frame_id = "horizontal_laser_link"
            range_msg.angle_min = np.deg2rad(cfg["DATASET"]["RA"]["RA_MIN"])
            range_msg.angle_max = np.deg2rad(cfg["DATASET"]["RA"]["RA_MAX"])
            range_msg.angle_increment = np.deg2rad(
                (cfg["DATASET"]["RA"]["RA_MAX"] - cfg["DATASET"]["RA"]["RA_MIN"])
                / angle_bins
            )
            range_msg.time_increment = 0.0
            range_msg.scan_time = 33e-3
            range_msg.range_min = cfg["DATASET"]["RA"]["RR_MIN"]
            range_msg.range_max = cfg["DATASET"]["RA"]["RR_MAX"]
            range_msg.ranges = heatmap2range(
                np.squeeze(scan_heatmaps[i]),
                range_bins,
                cfg["DATASET"]["RA"]["RR_MIN"],
                cfg["DATASET"]["RA"]["RR_MAX"],
            )

            bag.write("scan", range_msg, ts)

            # Create odometry message
            odom_msg = Odometry()
            odom_msg.header.stamp = ts
            odom_msg.header.frame_id = "map"
            odom_msg.child_frame_id = "base_link"
            odom_msg.pose.pose.position.x = odom_p_interp[i, 0]
            odom_msg.pose.pose.position.y = odom_p_interp[i, 1]
            odom_msg.pose.pose.position.z = odom_p_interp[i, 2]
            odom_msg.pose.pose.orientation.x = odom_q_interp[i].as_quat()[0]
            odom_msg.pose.pose.orientation.y = odom_q_interp[i].as_quat()[1]
            odom_msg.pose.pose.orientation.z = odom_q_interp[i].as_quat()[2]
            odom_msg.pose.pose.orientation.w = odom_q_interp[i].as_quat()[3]

            bag.write("odom", odom_msg, ts)


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
    parser.add_argument("--odom_path", help="Path to TUM odometry file.", required=True)
    parser.add_argument("--scan_path", help="Path to range scans.", required=True)
    parser.add_argument(
        "--output_path", help="Path to output directory.", required=True
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = args()
    update_config(cfg, args)

    print(f"Processing odom path {args.odom_path} + scan path {args.scan_path}...")

    sensorDataPreprocessing(args.odom_path, args.scan_path, args.output_path)
