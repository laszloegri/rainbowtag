"""
This is a Python implementation of the RainbowTag fiducial marker published here:
https://ieeexplore.ieee.org/document/9743123

The project was funded by NSERC and Indus.ai (acquired by Procore https://www.procore.com/en-ca)
through Concordia University.

This code can be used for research purposes. For other purposes, 
please check with Concordia University, Indus.ai and NSERC.

Code author: Laszlo Egri
"""

__author__ = "Laszlo Egri"
__copyright__ = "Copyright 2019, Concordia University, Indus.ai, NSERC"


import numpy as np
import numba as nb
import cv2


def get_perspective_transform_100_square(pt0, pt1, pt2, pt3):
    pts0 = np.float32([[0, 0], \
                       [100, 0], \
                       [0, 100], \
                       [100, 100]])
    pts1 = np.float32([pt0, pt1, pt2, pt3])
    m = cv2.getPerspectiveTransform(pts1, pts0)
    return(m)


#@nb.jit(cache=True)
def compute_step_vector(p0, p1, factor):
    row_step = (p0[0] - p1[0]) * factor
    col_step = (p0[1] - p1[1]) * factor
    return(np.array([row_step, col_step]))


#@nb.jit(cache=True)
def line_intersection(p0, p1, q0, q1):
    x1 = p0[0]
    y1 = p0[1]
    x2 = p1[0]
    y2 = p1[1]
    x3 = q0[0]
    y3 = q0[1]
    x4 = q1[0]
    y4 = q1[1]

    d = float((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    if d == 0:
        return None
    t_num = float((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4))
    u_num = float(-((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)))

    t = t_num / d
    u = u_num / d

    # if not ((0 <= u) and (u <= 1) and (0 <= t) and (t <= 1)):
    #     return None

    return(np.array([x1 + t*(x2 - x1), y1 + t*(y2 - y1)]))


def scaled_rotated_grid(size, angle, scale, center):
    grid = np.empty((size * size, 2), np.float32)

    c = 0
    for i in range(size):
        for j in range(size):
            grid[c] = j, i
            c = c + 1

    grid = grid - ((size - 1) / 2, (size - 1) / 2)

    angle = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    c = 0
    for i in range(size):
        for j in range(size):
            grid[c] = np.matmul(rotation_matrix, grid[c])
            c = c + 1

    grid = grid * scale
    grid = grid + (center, center)
    return(grid)


def rotate(pts, ang):
    rad = np.deg2rad(ang)
    rot_mat = np.array([[np.cos(rad), np.sin(rad)], [- np.sin(rad), np.cos(rad)]])
    return(np.dot(pts, rot_mat))


# Assumes first input is a LMS_quantization_number x 2 numpy array
def rotate_points_around_point(pts, center, ang):
    translated_pts = pts - [center[0], center[1]]
    rotated_pts = rotate(translated_pts, ang)
    return(rotated_pts + [center[0], center[1]])


# Finds centroid of contour
def get_contour_centroid(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        if len(contour.shape) == 3:
            return(contour[0][0][0], contour[0][0][1])
        else:
            return (contour[0][0], contour[0][1])
    # centroid from img moments
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return(cx, cy)


def scale_points_around_point(pts, point, factor):
    translated_pts = pts - point
    scaled_translated_pts = factor * translated_pts
    result = scaled_translated_pts + point
    result = np.array(result, dtype=np.int)
    return(result)


# Assumes first input is a LMS_quantization_number x 2 numpy array
def rotate_points_around_point(pts, center, ang):
    translated_pts = pts - [center[0], center[1]]
    rotated_pts = rotate(translated_pts, ang)
    return(rotated_pts + [center[0], center[1]])


def scale_points_around_centroid(pts, factor):
    pts_convex_hull = cv2.convexHull(np.array(pts, dtype = int))
    pts_centroid = get_contour_centroid(pts_convex_hull)
    translated_pts = pts - pts_centroid
    scaled_translated_pts = factor * translated_pts
    result = scaled_translated_pts + pts_centroid
    result = np.array(result, dtype=np.int)
    return(result)


def translate_contour(contour, vector):
    translated_contour = contour + vector
    return(translated_contour)


def scale_contour(contour, factor):
    c = get_contour_centroid(contour)
    translated_contour = translate_contour(contour, (-c[0], -c[1]))
    scaled_contour = translated_contour * factor
    scaled_contour = translate_contour(scaled_contour, c)
    scaled_contour = np.array(scaled_contour, dtype=np.int)
    return(scaled_contour)