## Radarize: Enhancing Radar SLAM with Generalizable Doppler-Based Odometry [MobiSys'24]


https://github.com/ConnectedSystemsLab/radarize_ae/assets/14133352/b713b0d6-a548-4776-8d54-673ec6a2543d


### Prerequisites

- Ubuntu 20.04
- ROS Noetic
- Conda with Python 3.8+
- CUDA >= 11.3 capable GPU.
- ImageMagick

### Setup

1. Install conda environment with  ```conda env create -f env.yaml```.
2. Source environment ```conda activate radarize_ae``` and then ```pip install -e .```.
3. Install [cartographer_ros](https://google-cartographer-ros.readthedocs.io/en/latest/compilation.html). Inside ```cartographer/```, ```configuration_files/``` and ```launch/``` into ```<catkin_ws>/install_isolated/share/cartographer_ros/```.
5. Source conda environment (if not already) and cartographer_ros environment.
```shell script
conda activate radarize_ae
source <catkin_ws>/install_isolated/setup.bash
```

### Dataset Preparation

1. Download the dataset ```dataset.zip``` from the [link](https://zenodo.org/records/11093859) and unzip into this directory.
2. Download the saved models+outputs ```eval.zip``` from [link](https://zenodo.org/records/11093859) and unzip into this directory.

### Evaluation

To generate results in the paper, run the top-level script
```shell script
./run_eval.sh 
```
Then,
1. Run ```./slam_eval.sh``` to get the SLAM metrics.
2. Run ```./odom_eval.sh``` to get the odometry metrics.

### Training from Scratch

```shell script
./run.sh 
```

