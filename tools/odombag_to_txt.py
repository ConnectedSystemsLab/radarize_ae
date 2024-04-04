#!/usr/bin/env python3

"""
Extract trajectory from .bag file.
"""

import os
import sys
import rosbag

import argparse

import numpy as np
from tqdm import tqdm

np.set_printoptions(precision=3, floatmode="fixed", sign=" ")


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag_path", help="Path to .bag odometry file.", default=None, required=False
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    return args


def extract_msg(bag):

    pose_topic = "trajectory_0"

    pose_ts, pose_msgs = [], []
    last_ts = None

    for topic, msg, ts in tqdm(
        bag.read_messages([pose_topic]), total=bag.get_message_count([pose_topic])
    ):
        curr_ts = ts.secs + 1e-9 * ts.nsecs
        if last_ts is None:
            last_ts = curr_ts
            continue
        elif curr_ts - last_ts < 33e-3:
            continue
        else:
            pose_msgs.append(
                np.array(
                    [
                        msg.transform.translation.x,
                        msg.transform.translation.y,
                        msg.transform.translation.z,
                        msg.transform.rotation.x,
                        msg.transform.rotation.y,
                        msg.transform.rotation.z,
                        msg.transform.rotation.w,
                    ]
                )
            )
            pose_ts.append(curr_ts)
            last_ts = curr_ts

    pose_ts = np.array(pose_ts)
    pose_msgs = np.array(pose_msgs)
    return pose_ts, pose_msgs


if __name__ == "__main__":
    args = args()

    print(f"Processing {args.bag_path}...")

    # Open bag file.
    bag = rosbag.Bag(args.bag_path)

    pose_ts, pose_msgs = extract_msg(bag)

    # Save trajectory.
    trajectory = np.concatenate(
        [
            pose_ts.reshape(-1, 1),
            pose_msgs,
        ],
        axis=1,
    )
    np.savetxt(
        os.path.join(args.bag_path.replace(".bag", ".txt")), trajectory, delimiter=" "
    )
