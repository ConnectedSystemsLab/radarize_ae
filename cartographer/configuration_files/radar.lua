-- Copyright 2016 The Cartographer Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "base_link",
  published_frame = "base_link",
  odom_frame = "odom",
  provide_odom_frame = false,
  publish_frame_projected_to_2d = false,
  use_pose_extrapolator = true,
  use_odometry = true,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 33e-3,
  pose_publish_period_sec = 33e-3,
  trajectory_publish_period_sec = 33e-3,
  rangefinder_sampling_ratio = .5,
  odometry_sampling_ratio = .5,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}

MAP_BUILDER.use_trajectory_builder_2d = true
TRAJECTORY_BUILDER_2D.use_imu_data = false
TRAJECTORY_BUILDER_2D.min_range = 0.3
TRAJECTORY_BUILDER_2D.max_range = 4.284
-- TRAJECTORY_BUILDER_2D.missing_data_ray_length = 99999
TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 1

TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = false
-- TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.0
-- TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(35.)
-- TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 1.
-- TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 1.

-- TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.probability_grid_range_data_inserter.hit_probability = 0.8
-- TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.probability_grid_range_data_inserter.miss_probability = 0.4
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.08
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 30

TRAJECTORY_BUILDER_2D.ceres_scan_matcher.occupied_space_weight = 70.
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight    = 600.
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight       = 40.

TRAJECTORY_BUILDER_2D.ceres_scan_matcher.ceres_solver_options.use_nonmonotonic_steps = true 
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.ceres_solver_options.max_num_iterations = 100
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.ceres_solver_options.num_threads = 1

-- TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds = 1.0
-- TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters = 1.0
-- TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.3)

POSE_GRAPH.optimize_every_n_nodes = 90

-- POSE_GRAPH.constraint_builder.max_constraint_distance = 5.
-- POSE_GRAPH.constraint_builder.sampling_ratio = 1.
-- POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.linear_search_window = 5.
-- POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.angular_search_window = math.rad(45.)
-- POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.branch_and_bound_depth = 7
-- POSE_GRAPH.constraint_builder.min_score = 0.7
-- POSE_GRAPH.constraint_builder.ceres_scan_matcher.occupied_space_weight = 80.
-- POSE_GRAPH.constraint_builder.ceres_scan_matcher.translation_weight = 40.
-- POSE_GRAPH.constraint_builder.ceres_scan_matcher.rotation_weight = 1.
-- POSE_GRAPH.constraint_builder.loop_closure_translation_weight = 1e5
-- POSE_GRAPH.constraint_builder.loop_closure_rotation_weight = .1e5

-- POSE_GRAPH.matcher_translation_weight = 5.0e3
-- POSE_GRAPH.matcher_rotation_weight = 1.6e3

-- POSE_GRAPH.optimization_problem.local_slam_pose_translation_weight = 1.0e5
-- POSE_GRAPH.optimization_problem.local_slam_pose_rotation_weight = 2.0e5
-- POSE_GRAPH.optimization_problem.odometry_translation_weight = 1e6
-- POSE_GRAPH.optimization_problem.odometry_rotation_weight = 4e5
-- POSE_GRAPH.optimization_problem.huber_scale = 1.
-- POSE_GRAPH.optimization_problem.ceres_solver_options.max_num_iterations = 100
-- POSE_GRAPH.optimization_problem.ceres_solver_options.num_threads = 1

-- POSE_GRAPH.max_num_final_iterations = 100


return options

