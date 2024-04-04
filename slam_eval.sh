#!/bin/bash

set -e

> log
for d in "gt_gt_default" "gt_radarhd_scan_only" "rnin_radarhd_default" "milliego_radarhd_default" "odometry_radarhd_radar" "odometry_unet_radar"
do
    for f in "main_0" "main_1" "main_2" "main_3" "main_4"
    do 
        ./tools/eval_traj.py --cfg=configs/$f.yaml --input=$d/output | tee -a log
    done
done

> slam_result.txt
for d in "gt_radarhd_scan_only" "rnin_radarhd_default" "milliego_radarhd_default" "odometry_radarhd_radar" "odometry_unet_radar"
do
    for f in "ape_trans" "ape_rot" "rpe_trans" "rpe_rot"
    do 
        average=$(echo "($(cat log | grep $d | grep $f | cut -d' ' -f4 | paste -s -d+))/5" | bc -l)
        echo "$d $f ${average}" | tee -a slam_result.txt
    done
done
