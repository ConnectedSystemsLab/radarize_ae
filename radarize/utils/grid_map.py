#!/usr/bin/env python3

"""Useful functions for grid mapping.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from numba import njit


@njit(cache=True)
def bresenham(start, end):
    """
    Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Bresenham's Line Algorithm
    Produces a np.array from start and end (original from roguebasin.com)
    >>> points1 = bresenham((4, 4), (6, 10))
    >>> print(points1)
    np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points


@njit(cache=True)
def flood_fill(occupancy_map, center_point, value):
    """
    center_point: starting point (x,y) of fill
    occupancy_map: occupancy map generated from Bresenham ray-tracing
    """
    # Fill empty areas with queue method
    sx, sy = occupancy_map.shape
    fringe = []
    fringe.insert(0, center_point)
    while fringe:
        n = fringe.pop()
        nx, ny = n
        # West
        if nx > 0:
            if occupancy_map[nx - 1, ny] == 0.5:
                occupancy_map[nx - 1, ny] = value
                fringe.insert(0, (nx - 1, ny))
        # East
        if nx < sx - 1:
            if occupancy_map[nx + 1, ny] == 0.5:
                occupancy_map[nx + 1, ny] = value
                fringe.insert(0, (nx + 1, ny))
        # North
        if ny > 0:
            if occupancy_map[nx, ny - 1] == 0.5:
                occupancy_map[nx, ny - 1] = value
                fringe.insert(0, (nx, ny - 1))
        # South
        if ny < sy - 1:
            if occupancy_map[nx, ny + 1] == 0.5:
                occupancy_map[nx, ny + 1] = value
                fringe.insert(0, (nx, ny + 1))


@njit(cache=True)
def ray_cast(grid, start, end, value):

    beam = bresenham(start, end)  # line

    x_w, y_w = grid.shape
    valid_x = np.logical_and(beam[:, 0] >= 0, beam[:, 0] <= x_w - 1)
    valid_y = np.logical_and(beam[:, 1] >= 0, beam[:, 1] <= y_w - 1)
    valid_mask = np.logical_and(valid_x, valid_y)
    valid_beam = beam[valid_mask]

    # grid[valid_beam] = value
    for pt in valid_beam:
        grid[pt[0], pt[1]] = value


@njit(cache=True)
def generate_ray_casting_grid_map(points, range_max, range_bins, hfov=39):
    """
    The breshen boolean tells if it's computed with bresenham ray casting
    (True) or with flood fill (False)
    """

    x_w = 2 * range_bins
    y_w = range_bins
    xy_resolution = range_max / range_bins
    min_x = -range_max
    max_x = range_max
    min_y = 0
    max_y = range_max
    center_x = range_bins
    center_y = 0
    center = np.array([center_x, center_y])

    # Initialize occupancy map.
    occupancy_map = (np.ones((x_w, y_w)) * 255).astype(np.uint8)
    # print((int(np.sqrt(2)*range_bins*np.sin(np.deg2rad(hfov))),  \
    #        int(np.sqrt(2)*range_bins*np.cos(np.deg2rad(hfov)))))
    ray_cast(
        occupancy_map,
        (center_x, center_y),
        (
            center_x - int(np.sqrt(2) * range_bins * np.sin(np.deg2rad(hfov))),
            center_y + int(np.sqrt(2) * range_bins * np.cos(np.deg2rad(hfov))),
        ),
        255,
    )
    # cv2.namedWindow('depth', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('depth', occupancy_map)
    # cv2.waitKey()
    # print((int(np.sqrt(2)*range_bins*np.sin(np.deg2rad(hfov))),  \
    #        int(np.sqrt(2)*range_bins*np.cos(np.deg2rad(hfov)))))
    ray_cast(
        occupancy_map,
        (center_x, center_y),
        (
            center_x + int(np.sqrt(2) * range_bins * np.sin(np.deg2rad(hfov))),
            center_y + int(np.sqrt(2) * range_bins * np.cos(np.deg2rad(hfov))),
        ),
        255,
    )
    # cv2.namedWindow('depth', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('depth', occupancy_map)
    # cv2.waitKey()
    flood_fill(occupancy_map, (center_x, y_w - 1), 255)  # unoccupied 255
    # cv2.namedWindow('depth', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('depth', occupancy_map)
    # cv2.waitKey()

    # Occupancy grid computed with bresenham ray casting
    for p in points:
        # Cull points farther than max range.
        if np.linalg.norm(p) >= range_max:
            continue

        x, y = p
        x_, y_ = (p / np.linalg.norm(p)) * np.sqrt(2) * range_max

        # x, y coordinate of the the free area
        ix = int(round((x - min_x) / xy_resolution))
        iy = int(round((y - min_y) / xy_resolution))

        # x, y coordinate of the unobserved area
        ix_ = int(round((x_ - min_x) / xy_resolution))
        iy_ = int(round((y_ - min_y) / xy_resolution))

        ray_cast(occupancy_map, (ix, iy), (ix_, iy_), 127)  # unobserved 127

        # Obstacle
        if ix < 1 or ix >= x_w - 1 or iy < 1 or iy >= y_w - 1:
            continue
        occupancy_map[ix][iy] = 0  # occupied area 0
        occupancy_map[ix + 1][iy] = 0  # extend the occupied area
        occupancy_map[ix][iy + 1] = 0  # extend the occupied area
        occupancy_map[ix + 1][iy + 1] = 0  # extend the occupied area

    return np.rot90(occupancy_map)


@njit(cache=True)
def generate_ray_casting_polar_map(points, range_axis, angle_axis):
    """
    The breshen boolean tells if it's computed with bresenham ray casting
    (True) or with flood fill (False)
    """

    x_w = len(range_axis)
    y_w = len(angle_axis)

    # Initialize occupancy map.
    occupancy_map = (np.ones((x_w, y_w)) * 1).astype(np.uint8)

    # Occupancy grid computed with bresenham ray casting
    for p in points:
        # Cull points farther than max range.

        range_p = np.linalg.norm(p)
        angle_p = -1 * np.arctan2(p[0], p[1])

        if range_p >= range_axis[-1]:
            continue
        if angle_p <= angle_axis[0] or angle_p >= angle_axis[-1]:
            continue

        # x, y coordinate of the the free area
        range_temp = np.zeros(x_w)
        for idx in range(x_w):
            range_temp[idx] = abs(range_p - range_axis[idx])
        ix = np.argmin(range_temp)
        angle_temp = np.zeros(y_w)
        for idy in range(y_w):
            angle_temp[idy] = abs(angle_p - angle_axis[idy])
        iy = np.argmin(angle_temp)

        # Obstacle
        if ix < 1 or ix >= x_w - 1 or iy < 1 or iy >= y_w - 1:
            continue
        occupancy_map[ix][iy] = 0  # occupied area 0
        occupancy_map[ix + 1][iy] = 0  # extend the occupied area
        occupancy_map[ix][iy + 1] = 0  # extend the occupied area
        occupancy_map[ix + 1][iy + 1] = 0  # extend the occupied area

    return occupancy_map
