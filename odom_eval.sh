#!/bin/bash

set -e

> log
for d in "gt_gt_default/output" "milliego" "rnin" "odometry"
do
    for f in "main_0" "main_1" "main_2" "main_3" "main_4"
    do 
        ./tools/eval_traj.py --cfg=configs/$f.yaml --input=$d | tee -a log
    done
done

> odom_result.txt
for d in "milliego" "rnin" "odometry"
do
    for f in "ape_trans" "ape_rot" "rpe_trans" "rpe_rot"
    do 
        average=$(echo "($(cat log | grep $d | grep $f | cut -d' ' -f4 | paste -s -d+))/5" | bc -l)
        echo "$d $f ${average}" | tee -a odom_result.txt
    done
done
