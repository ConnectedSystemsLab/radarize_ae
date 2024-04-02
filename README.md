# Radarize

## Prerequisites

Note: Tested on Ubuntu 18.04 with CUDA >= 11.1 capable GPU (RTX 3090).

1. Install conda environment with 
```shell script
conda env create -f env.yaml
```

2. Install [cartographer_ros](https://google-cartographer-ros.readthedocs.io/en/latest/compilation.html).

3. Copy ```configuration_files/``` and ```launch/``` into cartographer workspace.

## Dataset Preparation

- Download the dataset from the [Box link]() and unzip into this directory.
- Download the saved models+outputs from [Box link]() and unzip into this directory.

### Sensor Arrangement
<img src="calib/coords_1843.png" width="300" />

### Rosbag Format

```shell script
topics:      /camera/depth/image_rect_raw/compressedDepth    4307 msgs    : sensor_msgs/CompressedImage
             /radar0/radar_data                              1800 msgs    : xwr_raw_ros/RadarFrameFull 
             /ti_mmwave/radar_scan_pcl_0                     1800 msgs    : sensor_msgs/PointCloud2    
             /tracking/fisheye1/image_raw/compressed         1801 msgs    : sensor_msgs/CompressedImage
             /tracking/fisheye2/image_raw/compressed         1800 msgs    : sensor_msgs/CompressedImage
             /tracking/imu                                  11972 msgs    : sensor_msgs/Imu            
             /orb_slam3                                      1787 msgs    : geometry_msgs/PoseStamped  
             /tracking/odom/sample                          11971 msgs    : nav_msgs/Odometry
```

- `/tracking/odom/sample`: T265 VIO baseline/pseudo-groundtruth.
- `/camera/depth/image_rect_raw/compressedDepth`: Depth camera pseudo-groundtruth.
- `/tracking/fisheye1/image_raw/compressed`:  Black-white fisheye image from left camera.
- `/tracking/fisheye2/image_raw/compressed`:  Black-white fisheye image from right camera.
- `/tracking/imu`: Linearly interpolated IMU samples.
- `/ti_mmwave/radar_scan_pcl_0`:  Radar point cloud.
- `/radar0/radar_data`: Raw DSP samples from radar.

## Training the Models 
1. Source conda environment.
```shell script
conda activate radarize_ae
```

2. Source cartographer_ros environment.
```shell script
source <cartographer_workspace>/install_isolated/setup.bash
```

2. Run the top-level script

```shell script
./main.py --cfg=<config_file>
```

Each model (and its associated evaluation plots) will be located in a separate folder.

## Getting Metrics

First, run 
```shell script
./slam_eval.sh
```
to get SLAM metrics.

Second, run
```shell script
./odom_eval.sh
```
to get odometry metrics.


