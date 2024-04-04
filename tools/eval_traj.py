#!/usr/bin/env python3

"""
Compare trajectories
https://github.com/MichaelGrupp/evo/blob/master/notebooks/metrics.py_API_Documentation.ipynb
"""

import os
import sys

import pprint
from evo.core import metrics
from evo.tools import file_interface
from evo.core import sync
import copy
import numpy as np
import argparse

from evo.tools import plot
import matplotlib.pyplot as plt

from radarize.config import cfg, update_config

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
    parser.add_argument(
        "--input", help="Directory name <odom>_<scan>_<params>.", required=True
    )
    args = parser.parse_args()

    return args


def load_trajs(ref_file, est_file):
    traj_ref = file_interface.read_tum_trajectory_file(ref_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)

    max_diff = 0.01

    # Sync trajectories.
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)
    # Align trajectories w/o scaling.
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=False, correct_only_scale=False)
    # Align origin.
    traj_est_aligned_origin = copy.deepcopy(traj_est)
    traj_est_aligned_origin.align_origin(traj_ref)

    # fig = plt.figure()
    # traj_by_label = {
    #     "estimate (align origin)": traj_est_aligned_origin,
    #     "estimate (aligned)": traj_est_aligned,
    #     "reference": traj_ref
    # }
    # plot.trajectories(fig, traj_by_label, plot.PlotMode.xy)
    # plt.savefig(est_file.replace('/output/', '/result/').replace('.txt', '.png'))
    # plt.close(fig)

    data = (traj_ref, traj_est_aligned)

    return data


# return mean
def get_stat(metric_type, pose_relation_type, data):
    metric_types = ["ape", "rpe"]
    pose_relation_types = ["translation", "rotation_angle", "rotation", "full"]
    if metric_type not in metric_types:
        raise ValueError("Invalid metric type. Expected one of: %s" % metric_types)
    if pose_relation_type not in pose_relation_types:
        raise ValueError(
            "Invalid pose_relation type. Expected one of: %s" % pose_relation_types
        )

    if pose_relation_type == "translation":
        pose_relation = metrics.PoseRelation.translation_part
    if pose_relation_type == "rotation_angle":
        pose_relation = metrics.PoseRelation.rotation_angle_rad
    if pose_relation_type == "rotation":
        pose_relation = metrics.PoseRelation.rotation_part
    if pose_relation_type == "full":
        pose_relation = metrics.PoseRelation.full_transformation

    if metric_type == "ape":
        metric = metrics.APE(pose_relation)
    if metric_type == "rpe":
        delta = 1
        delta_unit = metrics.Unit.frames
        all_pairs = False
        metric = metrics.RPE(
            pose_relation=pose_relation,
            delta=delta,
            delta_unit=delta_unit,
            all_pairs=all_pairs,
        )

    metric.process_data(data)
    return metric.get_statistic(metrics.StatisticsType.mean)


if __name__ == "__main__":
    args = args()
    update_config(cfg, args)

    ape_translation = np.array([])
    ape_rotation_angle = np.array([])
    ape_rotation = np.array([])
    ape_full = np.array([])
    rpe_translation = np.array([])
    rpe_rotation_angle = np.array([])
    rpe_rotation = np.array([])
    rpe_full = np.array([])

    # Create dir.
    result_dir = os.path.join(cfg["OUTPUT_DIR"], args.input, "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for x in cfg["DATASET"]["TEST_SPLIT"]:
        ref_file = os.path.join(
            cfg["OUTPUT_DIR"], "gt_gt_default", "output", f"{x}.txt"
        )
        est_file = os.path.join(cfg["OUTPUT_DIR"], args.input, f"{x}.txt")
        data = load_trajs(ref_file, est_file)
        ape_translation = np.append(
            ape_translation, get_stat("ape", "translation", data)
        )
        rpe_translation = np.append(
            rpe_translation, get_stat("rpe", "translation", data)
        )
        ape_rotation_angle = np.append(
            ape_rotation_angle, get_stat("ape", "rotation_angle", data)
        )
        rpe_rotation_angle = np.append(
            ape_rotation_angle, get_stat("rpe", "rotation_angle", data)
        )

    np.savez(
        os.path.join(cfg["OUTPUT_DIR"], args.input, "result", "traj_eval.npz"),
        ape_trans=ape_translation,
        rpe_trans=rpe_translation,
        ape_rot=ape_rotation_angle,
        rpe_rot=rpe_rotation_angle,
    )

    print(f"{cfg['OUTPUT_DIR']}{args.input} ape_trans: ", np.mean(ape_translation))
    print(f"{cfg['OUTPUT_DIR']}{args.input} rpe_trans: ", np.mean(rpe_translation))
    print(f"{cfg['OUTPUT_DIR']}{args.input} ape_rot: ", np.mean(ape_rotation_angle))
    print(f"{cfg['OUTPUT_DIR']}{args.input} rpe_rot: ", np.mean(rpe_rotation_angle))

    if args.input != "gt_gt_default":
        gt_gt_default = np.load(
            os.path.join(
                cfg["OUTPUT_DIR"], "gt_gt_default/output", "result", "traj_eval.npz"
            )
        )

    # Plot APE and RPE tables.

    fig = plt.figure()
    # Make sure to sort the x values before plotting.
    x = np.array(cfg["DATASET"]["TEST_SPLIT"])
    idxs = np.argsort(x)
    x = x[idxs]
    plt.bar(x, ape_translation[idxs])
    if args.input != "gt_gt_default":
        plt.bar(x, gt_gt_default["ape_trans"][idxs])
    plt.xlabel("Dataset")
    plt.ylabel("APE translation")
    ax = plt.gca()
    plt.draw()
    ax.set_xticks(ax.get_xticks(), x, rotation=45, ha="right")
    spacing = 0.5
    fig.subplots_adjust(bottom=spacing)
    plt.title(f"{args.input} mAPE: {np.mean(ape_translation):.3f}")
    plt.savefig(os.path.join(cfg["OUTPUT_DIR"], args.input, "result", "ape_trans.png"))
    plt.close(fig)

    fig = plt.figure()
    # Make sure to sort the x values before plotting.
    x = np.array(cfg["DATASET"]["TEST_SPLIT"])
    idxs = np.argsort(x)
    x = x[idxs]
    plt.bar(x, rpe_translation[idxs])
    if args.input != "gt_gt_default":
        plt.bar(x, gt_gt_default["rpe_trans"][idxs])
    plt.xlabel("Dataset")
    plt.ylabel("RPE translation")
    ax = plt.gca()
    plt.draw()
    ax.set_xticks(ax.get_xticks(), x, rotation=45, ha="right")
    spacing = 0.5
    fig.subplots_adjust(bottom=spacing)
    plt.title(f"{args.input} mRPE: {np.mean(rpe_translation):.3f}")
    plt.savefig(os.path.join(cfg["OUTPUT_DIR"], args.input, "result", "rpe_trans.png"))
    plt.close(fig)

    fig = plt.figure()
    # Make sure to sort the x values before plotting.
    x = np.array(cfg["DATASET"]["TEST_SPLIT"])
    idxs = np.argsort(x)
    x = x[idxs]
    plt.bar(x, ape_rotation_angle[idxs])
    if args.input != "gt_gt_default":
        plt.bar(x, gt_gt_default["ape_rot"][idxs])
    plt.xlabel("Dataset")
    plt.ylabel("APE rotation angle")
    ax = plt.gca()
    plt.draw()
    ax.set_xticks(ax.get_xticks(), x, rotation=45, ha="right")
    spacing = 0.5
    fig.subplots_adjust(bottom=spacing)
    plt.title(f"{args.input} mAPE: {np.mean(ape_rotation_angle):.3f}")
    plt.savefig(os.path.join(cfg["OUTPUT_DIR"], args.input, "result", "ape_rot.png"))
    plt.close(fig)

    fig = plt.figure()
    # Make sure to sort the x values before plotting.
    x = np.array(cfg["DATASET"]["TEST_SPLIT"])
    idxs = np.argsort(x)
    x = x[idxs]
    plt.bar(x, rpe_rotation_angle[idxs])
    if args.input != "gt_gt_default":
        plt.bar(x, gt_gt_default["rpe_rot"][idxs])
    plt.xlabel("Dataset")
    plt.ylabel("RPE rotation angle")
    ax = plt.gca()
    plt.draw()
    ax.set_xticks(ax.get_xticks(), x, rotation=45, ha="right")
    spacing = 0.5
    fig.subplots_adjust(bottom=spacing)
    plt.title(f"{args.input} mRPE: {np.mean(rpe_rotation_angle):.3f}")
    plt.savefig(os.path.join(cfg["OUTPUT_DIR"], args.input, "result", "rpe_rot.png"))
    plt.close(fig)
