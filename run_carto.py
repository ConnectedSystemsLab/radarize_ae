#!/usr/bin/env python3

import argparse
import glob
import multiprocessing
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath('src'))

from subprocess import Popen
from src.config import cfg, update_config

os.path.dirname(os.path.realpath(__file__)) 


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='configs/default.yaml',
                        type=str)
    parser.add_argument('--n_proc',
                        type=int,
                        default=4,
                        help="Number of processes to use for parallel processing.")
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--odom', 
                        help="Odometry type.", 
                        required=True)
    parser.add_argument('--scan', 
                        help="Scan type.", 
                        required=True)
    parser.add_argument('--params', 
                        help="Cartographer parameters.", 
                        required=True)
    parser.add_argument('--demo',
                        action='store_true',
                        help="Turn Rviz on")
    args = parser.parse_args()

    return args

def run_commands(cmds, n_proc):
    print("Commands are: ")
    print(cmds)
    with multiprocessing.Pool(n_proc) as pool:
        pool.map(subprocess.run, cmds)

if __name__ == '__main__':
    args = args()
    update_config(cfg, args)

    print(f"Running cartographer on Odometry {args.odom} and Scan {args.scan} with Params {args.params}...")

    # Create dir.
    carto_in_dir = os.path.join(cfg['OUTPUT_DIR'],  
                                args.odom+"_"+args.scan+"_"+args.params, 
                                'input')
    carto_out_dir = os.path.join(cfg['OUTPUT_DIR'], 
                                 args.odom+"_"+args.scan+"_"+args.params, 
                                 'output')
    if not os.path.exists(carto_in_dir):
        os.makedirs(carto_in_dir)
    if not os.path.exists(carto_out_dir):
        os.makedirs(carto_out_dir)
 
    # Assume these dir exist, and .txt are all TUM files
    odom_dir = os.path.join(cfg['OUTPUT_DIR'], 
                            args.odom) 
    scan_dir = os.path.join(cfg['OUTPUT_DIR'], 
                            args.scan)

    # Convert odom + scans into bags.
    run_commands([['./export_cartographer.py',
                   f'--cfg={args.cfg}',
                   f'--odom_path={os.path.join(odom_dir, x+".txt")}',
                   f'--scan_path={os.path.join(scan_dir, x+".npz")}',
                   f'--output_path={os.path.join(carto_in_dir, x+".bag")}'] for x in cfg['DATASET']['TEST_SPLIT']], args.n_proc)

    # Run cartographer on all test bags.
    for x in cfg['DATASET']['TEST_SPLIT']:
        if args.demo:
            subprocess.run(['roslaunch', 
                          'cartographer_ros', 
                          'demo_backpack_2d.launch',
                          f'configuration_basename:={args.params}.lua',
                          f'bag_filename:={os.path.abspath(os.path.join(carto_in_dir, x+".bag"))}'])
        else:
            subprocess.run(['roslaunch', 
                          'cartographer_ros', 
                          'offline_backpack_2d.launch',
                          f'configuration_basenames:={args.params}.lua',
                          f'bag_filenames:={os.path.abspath(os.path.join(carto_in_dir, x+".bag"))}'])

    # Convert cartographer output to rosbag.
    run_commands([['cartographer_dev_pbstream_trajectories_to_rosbag',
                    f'-input={os.path.join(carto_in_dir, x+".bag.pbstream")}',
                    f'-output={os.path.join(carto_out_dir, x+".bag")}'] for x in cfg['DATASET']['TEST_SPLIT']], args.n_proc)
    # Convert rosbag to TUM format.
    run_commands([['./odombag_to_txt.py',
                    f'--bag_path={os.path.join(carto_out_dir, x+".bag")}'] for x in cfg['DATASET']['TEST_SPLIT']], args.n_proc)

    # Evaluate trajectory.
    # subprocess.run(['./eval_traj.py',
    #                 f'--cfg={args.cfg}',
    #                 f'--input={args.odom}_{args.scan}_{args.params}/output'], check=True)

    # Convert cartographer output to PNG.
    run_commands([['cartographer_pbstream_to_ros_map',
                    f'-pbstream_filename={os.path.join(carto_in_dir, x+".bag.pbstream")}',
                    f'-map_filestem={os.path.join(carto_out_dir, x)}']  for x in cfg['DATASET']['TEST_SPLIT']], args.n_proc)
    subprocess.run(['mogrify', '-format', 'png', 
                    os.path.join(carto_out_dir,"*.pgm")], check=True)
    

