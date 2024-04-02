#!/usr/bin/env python3

"""
Converts .bag files to dataset.
"""

import os
import sys

sys.path.insert(0, os.path.abspath('src'))

import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
import open3d as o3d
import rosbag
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

np.set_printoptions(precision=3, floatmode='fixed', sign=' ')

from sensor_msgs import point_cloud2

from config import cfg, update_config
from utils import dsp, grid_map, image_tools, radar_config


def create_radar_bev(bag,
                     radar_params,
                     radar_topic='/radar0/radar_data',
                     radar_buffer_len=1,
                     angle_res=1, angle_range=43,
                     normalization_range=[0.0, 18.0],
                     warp_cartesian=False):
    """Create radar range-azimuth grid maps."""

    heatmap_ts, heatmap_msgs = [], []

    # Get radar params
    range_bias = radar_params['range_bias']
    range_max  = radar_params['range_max']
    range_bins = radar_params['n_samples']

    radar_buffer = deque(maxlen=radar_buffer_len)

    for i, (topic, msg, ts) in tqdm(enumerate(bag.read_messages([radar_topic])),
                                    total=bag.get_message_count(radar_topic)):

        # 1843/1843AOP
        radar_cube = dsp.reshape_frame(msg)

        # Accumulate radar cubes in buffer.
        radar_buffer.append(radar_cube)
        if len(radar_buffer) < radar_buffer.maxlen:
            continue
        radar_cube = np.concatenate(radar_buffer, axis=0)

        # Choose antennas for range-azimuth heatmap.
        radar_cube_a = radar_cube[:, :8, :]
        radar_cube_a = dsp._tdm(radar_cube_a, 2, 4)

        # All images should be C x H x W
        heatmap = np.stack([
            dsp.compute_range_azimuth(radar_cube_a, 
                                      angle_res, 
                                      angle_range, 
                                      'capon'),
        ])

        if warp_cartesian:
            # Warp heatmap to cartesian coordinates.
            heatmap = np.stack([
                np.rot90(
                    image_tools.polar2cartesian(
                        x,
                        np.linspace(
                            range_bias,
                            range_max,
                            x.shape[0]
                        ),
                        np.linspace(
                            np.deg2rad(angle_range),
                            -np.deg2rad(angle_range),
                            x.shape[1]
                        ),
                        np.linspace(
                            0.,
                            range_max,
                            x.shape[0]
                        ),
                        np.arange(
                            -range_max*np.sin(np.deg2rad(angle_range)),
                            range_max*np.sin(np.deg2rad(angle_range)),
                            range_max/x.shape[0]
                        )
                    )
                )
            for x in heatmap])


        heatmap = dsp.normalize(heatmap,
                                min_val=normalization_range[0],
                                max_val=normalization_range[1])

        # cv2.namedWindow('range_azimuth', cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('range_azimuth', image_tools.normalize_and_color(heatmap[0]))
        # cv2.waitKey(1)

        heatmap_msgs.append(heatmap)
        heatmap_ts.append(ts.secs + 1e-9 * ts.nsecs)

    heatmap_ts = np.array(heatmap_ts)
    return heatmap_ts, heatmap_msgs

def create_radar_bev_elevation(bag,
                               radar_params,
                               radar_topic='/radar0/radar_data',
                               radar_buffer_len=1,
                               angle_res=1, angle_range=43,
                               normalization_range=[0.0, 18.0],
                               warp_cartesian=False):
    """Create radar range-azimuth grid maps."""

    heatmap_ts, heatmap_msgs = [], []

    # Get radar params
    range_bias = radar_params['range_bias']
    range_max  = radar_params['range_max']
    range_bins = radar_params['n_samples']

    radar_buffer = deque(maxlen=radar_buffer_len)

    for i, (topic, msg, ts) in tqdm(enumerate(bag.read_messages([radar_topic])),
                                    total=bag.get_message_count(radar_topic)):

        # 1843/1843AOP
        radar_cube = dsp.reshape_frame(msg)

        # Accumulate radar cubes in buffer.
        radar_buffer.append(radar_cube)
        if len(radar_buffer) < radar_buffer.maxlen:
            continue
        radar_cube = np.concatenate(radar_buffer, axis=0)

        # Choose antennas for range-azimuth heatmap.
        radar_cube_e = (radar_cube[:, 2:6, :] + radar_cube[:, 8:12, :]) / 2  # [2,3,4,5,8,9,10,11]
        radar_cube_e = dsp._tdm(radar_cube_e, 2, 4)


        # All images should be C x H x W
        heatmap = np.stack([
            dsp.compute_range_azimuth(radar_cube_e, 
                                      angle_res, 
                                      angle_range, 
                                      'capon')
        ])

        if warp_cartesian:
            # Warp heatmap to cartesian coordinates.
            heatmap = np.stack([
                np.rot90(
                    image_tools.polar2cartesian(
                        x,
                        np.linspace(
                            range_bias,
                            range_max,
                            x.shape[0]
                        ),
                        np.linspace(
                            np.deg2rad(angle_range),
                            -np.deg2rad(angle_range),
                            x.shape[1]
                        ),
                        np.linspace(
                            0.,
                            range_max,
                            x.shape[0]
                        ),
                        np.arange(
                            -range_max*np.sin(np.deg2rad(angle_range)),
                            range_max*np.sin(np.deg2rad(angle_range)),
                            range_max/x.shape[0]
                        )
                    )
                )
            for x in heatmap])


        heatmap = dsp.normalize(heatmap,
                                min_val=normalization_range[0],
                                max_val=normalization_range[1])

        # cv2.namedWindow('range_azimuth_elevation', cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('range_azimuth_elevation', image_tools.normalize_and_color(heatmap[0]))
        # cv2.waitKey(1)

        heatmap_msgs.append(heatmap)
        heatmap_ts.append(ts.secs + 1e-9 * ts.nsecs)

    heatmap_ts = np.array(heatmap_ts)
    return heatmap_ts, heatmap_msgs

def create_radar_doppler(bag,
                         radar_params,
                         radar_topic='/radar0/radar_data',
                         resize_shape=(181, 60),
                         radar_buffer_len=3,
                         range_subsampling_factor=1,
                         angle_res=1, angle_range=90,
                         normalization_range=[10.0, 25.0]):
    """Create doppler-azimuth heatmaps. """

    heatmap_ts, heatmap_msgs = [], []

    radar_buffer = deque(maxlen=radar_buffer_len)

    for i, (topic, msg, ts) in tqdm(enumerate(bag.read_messages([radar_topic])),
                                    total=bag.get_message_count(radar_topic)):

        # Convert radar msg to radar cube.
        radar_cube = dsp.reshape_frame(msg)

        # Accumulate radar cubes in buffer.
        radar_buffer.append(radar_cube)
        if len(radar_buffer) < radar_buffer.maxlen:
            continue
        radar_cube = np.concatenate(radar_buffer, axis=0)

        radar_cube_h = radar_cube[::1]
        heatmap_h = dsp.preprocess_1d_radar_1843(radar_cube_h,
                                                 angle_res, angle_range,
                                                 range_subsampling_factor,
                                                 normalization_range[0], 
                                                 normalization_range[1],
                                                 resize_shape)
        heatmap_h = np.fliplr(heatmap_h)

        # radar_cube_m = radar_cube[::2]
        # heatmap_m = dsp.preprocess_1d_radar_1843(radar_cube_m,
        #                                          angle_res, angle_range,
        #                                          range_subsampling_factor,
        #                                          normalization_range[0], 
        #                                          normalization_range[1],
        #                                          resize_shape)
        # heatmap_m = np.fliplr(heatmap_m)

        # radar_cube_l = radar_cube[::4]
        # heatmap_l = dsp.preprocess_1d_radar_1843(radar_cube_l,
        #                                          angle_res, angle_range,
        #                                          range_subsampling_factor,
        #                                          normalization_range[0], 
        #                                          normalization_range[1],
        #                                          resize_shape)

        # heatmap_l = np.fliplr(heatmap_l)

        # heatmap = np.zeros((3, resize_shape[1], resize_shape[0]))
        # heatmap[0], heatmap[1], heatmap[2] = heatmap_h, heatmap_m, heatmap_l

        # cv2.namedWindow('doppler_azimuth', cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('doppler_azimuth', image_tools.normalize_and_color(heatmap_l))
        # cv2.waitKey(1)

        heatmap = np.stack([
            heatmap_h,
            # heatmap_m,
            # heatmap_l,
        ])

        heatmap_msgs.append(heatmap)
        heatmap_ts.append(ts.secs + 1e-9 * ts.nsecs)

    heatmap_ts = np.array(heatmap_ts)
    return heatmap_ts, heatmap_msgs

def create_radar_doppler_elevation(bag,
                                   radar_params,
                                   radar_topic='/radar0/radar_data',
                                   resize_shape=(181, 60),
                                   radar_buffer_len=3,
                                   range_subsampling_factor=1,
                                   angle_res=1, angle_range=90,
                                   normalization_range=[10.0, 25.0]):
    """Create doppler-azimuth heatmaps with elevation beamforming. """

    heatmap_ts, heatmap_msgs = [], []

    radar_buffer = deque(maxlen=radar_buffer_len)

    for i, (topic, msg, ts) in tqdm(enumerate(bag.read_messages([radar_topic])),
                                    total=bag.get_message_count(radar_topic)):

        # Convert radar msg to radar cube.
        radar_cube = dsp.reshape_frame(msg)

        # Accumulate radar cubes in buffer.
        radar_buffer.append(radar_cube)
        if len(radar_buffer) < radar_buffer.maxlen:
            continue
        radar_cube = np.concatenate(radar_buffer, axis=0)

        # Do elevation beamforming.
        radar_cube_e = (radar_cube[:, 2:6, :] + radar_cube[:, 8:12, :]) / 2  # [2,3,4,5,8,9,10,11]

        radar_cube_h = radar_cube_e[::1]
        heatmap_h = dsp.preprocess_1d_radar_1843(radar_cube_h,
                                                 angle_res, angle_range,
                                                 range_subsampling_factor,
                                                 normalization_range[0], 
                                                 normalization_range[1],
                                                 resize_shape)
        heatmap_h = np.fliplr(heatmap_h)

        # radar_cube_m = radar_cube_e[::2]
        # heatmap_m = dsp.preprocess_1d_radar_1843(radar_cube_m,
        #                                          angle_res, angle_range,
        #                                          range_subsampling_factor,
        #                                          normalization_range[0], 
        #                                          normalization_range[1],
        #                                          resize_shape)
        # heatmap_m = np.fliplr(heatmap_m)

        # radar_cube_l = radar_cube_e[::4]
        # heatmap_l = dsp.preprocess_1d_radar_1843(radar_cube_l,
        #                                          angle_res, angle_range,
        #                                          range_subsampling_factor,
        #                                          normalization_range[0], 
        #                                          normalization_range[1],
        #                                          resize_shape)

        # heatmap_l = np.fliplr(heatmap_l)

        # heatmap = np.zeros((3, resize_shape[1], resize_shape[0]))
        # heatmap[0], heatmap[1], heatmap[2] = heatmap_h, heatmap_m, heatmap_l

        # cv2.namedWindow('doppler_azimuth_e', cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('doppler_azimuth_e', image_tools.normalize_and_color(heatmap_h))
        # cv2.waitKey(1)

        heatmap = np.stack([
            heatmap_h,
            # heatmap_m,
            # heatmap_l,
        ])

        heatmap_msgs.append(heatmap)
        heatmap_ts.append(ts.secs + 1e-9 * ts.nsecs)

    heatmap_ts = np.array(heatmap_ts)
    return heatmap_ts, heatmap_msgs



def create_camera_fpv(bag,
                      camera_topic='/tracking/fisheye1/image_raw/compressed'):
    """Extract camera images from bag file."""

    camera_ts, camera_msgs = [], []

    for i, (topic, msg, ts) in tqdm(enumerate(bag.read_messages([camera_topic])),
                                    total=bag.get_message_count(camera_topic)):

        # Read depth image.
        camera_img = image_tools.it.convert_to_cv2(msg)
        camera_img = image_tools.image_resize(camera_img, height=424)

        # cv2.namedWindow('camera', cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('camera', camera_img)
        # cv2.waitKey(1)

        camera_msgs.append(camera_img)
        camera_ts.append(ts.secs + 1e-9 * ts.nsecs)

    camera_ts = np.array(camera_ts)
    return camera_ts, camera_msgs

def create_depth_bev(bag,
                     depth_intrinsics,
                     radar_params,
                     depth_topic='/camera/depth/image_rect_raw/compressed',
                     pcd_subsampling_factor=1,
                     angle_range=43, angle_bins=88,
                     warp_cartesian=False):

    """Create top-down occupancy grid maps from depth sensors readings.

       255 - Free space.
       127 - Unobserved.
       0.0 - Obstacle.
    """

    depthmap_ts, depthmap_msgs = [], []

    # Get radar params
    range_max  = radar_params['range_max']
    range_bins = radar_params['n_samples']
    range_bias = radar_params['range_bias']

    def depth2pcd(depth, intrinsic, depth_trunc=3.0):
        depth_img = o3d.geometry.Image(depth)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_img, 
                                                              intrinsic, 
                                                              depth_trunc=depth_trunc)
        return np.asarray(pcd.points)


    for i, (topic, msg, ts) in tqdm(enumerate(bag.read_messages([depth_topic])),
                                    total=bag.get_message_count(depth_topic)):

        # Read depth image.
        depth_img = image_tools.it.convert_depth_to_cv2(msg)

        # Deproject depth images into pointcloud.
        pcd = depth2pcd(depth_img, depth_intrinsics, range_max)

        # Filter out points away from horizontal.
        idxs = np.logical_and(pcd[:,1] > -0.03, pcd[:,1] < 0.03)
        pcd = pcd[idxs]

        # Remap from X,Z to X,Y.
        pcd = np.delete(pcd, 1, 1)

        # Subsample pcds
        pcd = pcd[::pcd_subsampling_factor]

        # Generate occupancy grid from pcd.
        if warp_cartesian:
            depthmap = grid_map.generate_ray_casting_grid_map(pcd, 
                                                              range_max, 
                                                              range_bins)
        else:
            depthmap = grid_map.generate_ray_casting_polar_map(pcd,
                np.linspace(
                    0,
                    range_max,
                    range_bins,
                ),
                np.linspace(
                    np.deg2rad(-angle_range),
                    np.deg2rad(angle_range),
                    angle_bins, 
                ))

        # All images should be C x H x W
        depthmap = np.stack([
            depthmap,
        ])

        # cv2.namedWindow('depth', cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('depth', image_tools.normalize_and_color(depthmap[0]))
        # cv2.waitKey(1)

        depthmap_msgs.append(depthmap)
        depthmap_ts.append(ts.secs + 1e-9*ts.nsecs)

    depthmap_ts = np.array(depthmap_ts)
    return depthmap_ts, depthmap_msgs

def create_pcd_bev(bag,
                   radar_params,
                   pcd_topic='/ti_mmwave/radar_scan_pcl_0',
                   angle_range=43, angle_bins=88,
                   warp_cartesian=False):

    """Create top-down point cloud from radar.
    """

    pcd_ts, pcd_msgs = [], []

    # Get radar params
    range_max  = radar_params['range_max']
    range_bins = radar_params['n_samples']

    for i, (topic, msg, ts) in tqdm(enumerate(bag.read_messages([pcd_topic])),
                                    total=bag.get_message_count(pcd_topic)):

        # Read point cloud.
        pcd = [p[:3] for p in point_cloud2.read_points(msg, skip_nans=True)]
        pcd = np.array(pcd)
        if pcd.size == 0:
            continue

        # Filter out points away from horizontal.
        # idxs = np.logical_and(pcd[:,2] > -0.30, pcd[:,2] < 0.30)
        # pcd = pcd[idxs,:]

        # Filter out point near origin.
        idxs = np.linalg.norm(pcd, axis=1) > 0.2
        pcd = pcd[idxs,:]

        if pcd.size == 0:
            continue

        # Delete Z axis.
        pcd = pcd[:,:2]

        # Rotate X and Y.
        pcd = np.hstack((-pcd[:,1:2:], pcd[:,0:1]))

        # Generate occupancy grid from pcd.
        if warp_cartesian:
            pcdmap = grid_map.generate_ray_casting_grid_map(pcd, range_max, range_bins)
        else:
            pcdmap = grid_map.generate_ray_casting_polar_map(pcd,
                np.linspace(
                    0,
                    range_max,
                    range_bins
                ),
                np.linspace(
                    np.deg2rad(-angle_range),
                    np.deg2rad(angle_range),
                    angle_bins,
                ))

        # All images should be C x H x W
        pcdmap = np.stack([
            pcdmap,
        ])

        # cv2.namedWindow('pcd', cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('pcd', image_tools.normalize_and_color(pcdmap[0]))
        # cv2.waitKey(33)

        pcd_msgs.append(pcdmap)
        pcd_ts.append(ts.secs + 1e-9*ts.nsecs)

    pcd_ts = np.array(pcd_ts)
    return pcd_ts, pcd_msgs

def create_imu(bag,
               imu_topic='/tracking/imu'):
    imu_ts, imu_msgs = [], []

    for i, (topic, msg, ts) in tqdm(enumerate(bag.read_messages([imu_topic])),
                                    total=bag.get_message_count(imu_topic)):
        # Read imu angular velocity.
        imu_msg = np.array([msg.linear_acceleration.x,
                            msg.linear_acceleration.y,
                            msg.linear_acceleration.z,
                            msg.angular_velocity.x,
                            msg.angular_velocity.y,
                            msg.angular_velocity.z])

        imu_msgs.append(imu_msg)
        imu_ts.append(ts.secs + 1e-9 * ts.nsecs)

    imu_ts = np.array(imu_ts)
    return imu_ts, imu_msgs

def create_body_velo_gt(bag,
                        pose_topic='/tracking/odom/sample'):

    body_velo_gt_ts, body_velo_gt_msgs = [], []
    last_ts = None
    last_position = None
    last_rotation = None

    for (topic, msg, ts) in tqdm(bag.read_messages([pose_topic]),
                                 total=bag.get_message_count([pose_topic])):
        curr_ts = ts.secs + 1e-9*ts.nsecs
        pose = msg.pose.pose
        position = np.array([pose.position.x, 
                             pose.position.y, 
                             pose.position.z])
        rotation = R.from_quat([pose.orientation.x, 
                                pose.orientation.y, 
                                pose.orientation.z, 
                                pose.orientation.w]).as_matrix()
        # Downsample to 30fps.
        if last_ts is None:
            last_ts = curr_ts 
            last_position = position
            last_rotation = rotation
            continue
        elif curr_ts - last_ts < 33e-3:
            continue
        else:
            elapsed = curr_ts - last_ts

            velo = (position - last_position)/elapsed
            body_velo = last_rotation.T @ velo

            # Filter body velocity changes that are too big.
            if body_velo_gt_msgs and np.linalg.norm(body_velo-body_velo_gt_msgs[-1]) > 0.5:
                body_velo = body_velo_gt_msgs[-1]

            body_velo_gt_msgs.append(body_velo)
            body_velo_gt_ts.append(curr_ts)

            last_ts = curr_ts 
            last_position = position
            last_rotation = rotation

    body_velo_gt_ts   = np.array(body_velo_gt_ts)
    body_velo_gt_msgs = np.array(body_velo_gt_msgs)
    return body_velo_gt_ts, body_velo_gt_msgs

def create_pose_gt(bag,
                   pose_topic='/tracking/odom/sample'):

    pose_ts, pose_msgs = [], []
    last_ts = None

    for (topic, msg, ts) in tqdm(bag.read_messages([pose_topic]),
                                 total=bag.get_message_count([pose_topic])):
        curr_ts = ts.secs + 1e-9*ts.nsecs 
        if last_ts is None:
            last_ts = curr_ts 
            continue
        elif curr_ts - last_ts < 33e-3:
            continue
        else:
            pose_msgs.append(np.array([msg.pose.pose.position.x,
                                       msg.pose.pose.position.y,
                                       msg.pose.pose.position.z,
                                       msg.pose.pose.orientation.x,
                                       msg.pose.pose.orientation.y,
                                       msg.pose.pose.orientation.z,
                                       msg.pose.pose.orientation.w]))
            pose_ts.append(curr_ts)
            last_ts = curr_ts 

    pose_ts   = np.array(pose_ts)
    pose_msgs = np.array(pose_msgs)
    return pose_ts, pose_msgs

def sync2topic(unpacked_bag, sync_topic):
    """ Interpolate everything to sync_topic timestamps. """

    d = defaultdict(lambda: [])
    time_begin = unpacked_bag['pose_gt'][0][0]

    for sync_ts in unpacked_bag[sync_topic][0]:
        if sync_ts < time_begin:
            continue
        for topic, (topic_ts, topic_msgs) in unpacked_bag.items():
            i_topic = np.argmin(np.abs(sync_ts - topic_ts))
            d[topic].append(topic_msgs[i_topic])
        d['time'].append(sync_ts)

    for k, v in d.items():
        d[k] = np.stack(v)

    return d

def args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=None, 
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--bag_path', help="Path to bag directory.", required=True)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = args()
    update_config(cfg, args)
    cfg = cfg['DATASET']

    print(f"Processing {args.bag_path}...")

    # Get sensor parameters.
    with open(cfg['RADAR_CONFIG'], 'r') as f:
        radar_config = radar_config.RadarConfig(f.readlines())
    radar_params = radar_config.get_params()

    intrinsics = np.loadtxt(cfg['DEPTH_INTRINSICS'])
    depth_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=int(intrinsics[0]),
                                                         height=int(intrinsics[1]),
                                                         fx=intrinsics[2],
                                                         fy=intrinsics[3],
                                                         cx=intrinsics[4],
                                                         cy=intrinsics[5])

    # Open bag file.
    bag = rosbag.Bag(args.bag_path)

    # DA heatmaps 
    da_ts, da_heatmaps           = create_radar_doppler(bag, radar_params,
                                                        radar_topic=cfg['RADAR_TOPIC'],
                                                        resize_shape=cfg['DA']['RESIZE_SHAPE'],
                                                        radar_buffer_len=cfg['DA']['RADAR_BUFFER_LEN'],
                                                        range_subsampling_factor=cfg['DA']['RANGE_SUBSAMPLING_FACTOR'])

    da_e_ts, da_e_heatmaps        = create_radar_doppler_elevation(bag, radar_params,
                                                                   radar_topic=cfg['RADAR_TOPIC'],
                                                                   resize_shape=cfg['DA']['RESIZE_SHAPE'],
                                                                   radar_buffer_len=cfg['DA']['RADAR_BUFFER_LEN'],
                                                                   range_subsampling_factor=cfg['DA']['RANGE_SUBSAMPLING_FACTOR'])

    # RA heatmaps

    ra_1_ts, ra_1_heatmaps       = create_radar_bev(bag, radar_params,
                                                    radar_topic=cfg['RADAR_TOPIC'],
                                                    radar_buffer_len=1,
                                                    angle_range=cfg['RA']['RA_MAX'])
    ra_3_ts, ra_3_heatmaps       = create_radar_bev(bag, radar_params,
                                                    radar_topic=cfg['RADAR_TOPIC'],
                                                    radar_buffer_len=3,
                                                    angle_range=cfg['RA']['RA_MAX'])
    ra_5_ts, ra_5_heatmaps       = create_radar_bev(bag, radar_params,
                                                    radar_topic=cfg['RADAR_TOPIC'],
                                                    radar_buffer_len=5,
                                                    angle_range=cfg['RA']['RA_MAX'])

    ra_e_1_ts, ra_e_1_heatmaps   = create_radar_bev_elevation(bag, radar_params,
                                                              radar_topic=cfg['RADAR_TOPIC'],
                                                              radar_buffer_len=1,
                                                              angle_range=cfg['RA']['RA_MAX'])

    ra_e_3_ts, ra_e_3_heatmaps   = create_radar_bev_elevation(bag, radar_params,
                                                              radar_topic=cfg['RADAR_TOPIC'],
                                                              radar_buffer_len=3,
                                                              angle_range=cfg['RA']['RA_MAX'])

    ra_e_5_ts, ra_e_5_heatmaps   = create_radar_bev_elevation(bag, radar_params,
                                                              radar_topic=cfg['RADAR_TOPIC'],
                                                              radar_buffer_len=5,
                                                              angle_range=cfg['RA']['RA_MAX'])

    # Depth maps.
    depth_ts, depth_maps         = create_depth_bev(bag, depth_intrinsics, radar_params,
                                                    depth_topic=cfg['DEPTH_TOPIC'])

    depth_u_ts, depth_maps_u     = create_depth_bev(bag, depth_intrinsics, radar_params,
                                                    depth_topic=cfg['DEPTH_TOPIC'],
                                                    angle_bins=88*8)

    # Point clouds.
    pcd_ts, pcd_maps             = create_pcd_bev(bag, radar_params,
                                                  pcd_topic=cfg['PCD_TOPIC'])

    # Camera images.
    # cam_ts, cam_imgs           = create_camera_fpv(bag,
    #                                                camera_topic=cfg['CAMERA_TOPIC'])

    # IMU data.
    imu_ts, imu_msgs             = create_imu(bag,
                                              imu_topic=cfg['IMU_TOPIC'])

    # Calculate GT
    pose_ts, pose_msgs           = create_pose_gt(bag, 
                                                  pose_topic=cfg['POSE_TOPIC'])

    body_velo_ts, body_velo_msgs = create_body_velo_gt(bag,
                                                       pose_topic=cfg['POSE_TOPIC'])

    # Keep frames in dict.
    frames = defaultdict(lambda: [])

    frames['pose_gt']               = [pose_ts,      pose_msgs]
    frames['velo_gt']               = [body_velo_ts, body_velo_msgs]
    frames['depth_map']             = [depth_ts,     depth_maps]
    frames['depth_map_u']           = [depth_u_ts,   depth_maps_u]
    frames['pcd_map']               = [pcd_ts,       pcd_maps]
    # frames['camera']                = [cam_ts, cam_imgs]
    frames['radar_d']               = [da_ts,   da_heatmaps] # doppler heatmap
    frames['radar_de']              = [da_e_ts, da_e_heatmaps]
    frames['radar_r_1']             = [ra_1_ts, ra_1_heatmaps] 
    frames['radar_r_3']             = [ra_3_ts, ra_3_heatmaps] # range_heatmap
    frames['radar_r_5']             = [ra_5_ts, ra_5_heatmaps] 
    frames['radar_re_1']            = [ra_e_1_ts, ra_e_1_heatmaps]
    frames['radar_re_3']            = [ra_e_3_ts, ra_e_3_heatmaps]
    frames['radar_re_5']            = [ra_e_5_ts, ra_e_5_heatmaps]
    frames['imu']                   = [imu_ts,  imu_msgs]

    # Check if any keys are empty.
    for k, v in frames.items():
        print(k, v[0].shape)

    # Remove empty keys.
    frames = {k: v for k, v in frames.items() if len(v[0]) > 0}

    # Synchronize to radar timestamps.
    synced_bag = sync2topic(frames, cfg['SYNC_TOPIC'])

    # Check if any keys are empty.
    for k, v in synced_bag.items():
        print(k, v.shape, v.dtype)

    # Save to .npz
    np.savez(os.path.splitext(args.bag_path)[0] + '.npz', **synced_bag)
