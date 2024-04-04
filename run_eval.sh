#!/bin/bash

set -e

for file in configs/*.yaml; do
    if [ "$file" != "configs/default.yaml" ]; then
        echo "Running $file"
        python main_eval.py --cfg="$file"
    fi
done

# Get odometry results.
./odom_eval.sh

# Get SLAM results.
./slam_eval.sh
