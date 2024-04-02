# Cartographer Setup

## Installation

1. Install [Cartographer ROS](https://google-cartographer-ros.readthedocs.io/en/latest/compilation.html).

2. Go to install directory (assuming it is `${HOME}/cartographer_ws/install_isolated/share/cartographer_ros`)

    a. Copy files under `configuration_files` and `launch` to their respective folders.

    b. Each launch file runs Cartographer with a different set of hyperparameters. 
    * `known_poses` means mapping with known trajectory. 
    * `scan_only` relies on pure scan matching. 
    * `balanced` combines both scan matching and odometry prior.

3. Source the environment by running 
```shell script
source cartographer_ws/install_isolated/setup.bash
```

## Running Cartographer
Try running on example bag (`csl_basement_0.bag`). This example bag file contains two topics `odom` and `scan` which are from GT pose and depth map. You can replace these topics from any other source (i.e. milliEgo for `odom`, milliMap for `scan`).

* To run with rviz visualization (for debugging):
```shell script 
roslaunch cartographer_ros demo_backpack_2d.launch configuration_basename:=balanced.lua bag_filename:=${HOME}/cartographer_ws/gt_gt_csl_basement_0.bag
```

* To run with no visualization (to get results):
```shell script 
roslaunch cartographer_ros offline_scan_only.launch configuration_basenames:=balanced.lua bag_filenames:=${HOME}/cartographer_ws/gt_gt_csl_basement_0.bag
```
This generates results as `gt_gt_csl_basement_0.bag.pbstream`.

* To extract resulting trajectory from `.pbstream`:
```shell script
cartographer_dev_pbstream_trajectories_to_rosbag -input gt_gt_csl_basement_0.bag.pbstream -output trajectory.bag
```
You can use `evo_traj` to visualize the `.bag` file. <img src="scan_only_trajectories.png" width="600" />

* To extract resulting map from `.pbstream`:
```shell script
cartographer_pbstream_to_ros_map -pbstream_filename gt_gt_csl_basement_0.bag.pbstream
```
This outputs map as a `.pgm` file. 

## Comparing Outputs

* Need to use `evo_ape` and `evo_rpe` to compute trajectory error. 
* Need to find a way to evaluate metrics between two maps (`.pgm` files).


### Disadvantage of Pure Scan Matching

<img src="gt_gt_scan_only_trajectories.png" width="300" />
<img src="gt_gt_balanced_trajectories.png" width="300" />

As you can see, `scan_only` fails when going down a hallway because scan matching becomes degenerate in that environment. So the trajectory is too short.

