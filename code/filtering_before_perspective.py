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


import numba as nb
import cv2
import numpy as np
from numba.typed import List
from filtering_based_on_color import *

def reorder_blobs_ccw_and_do_basic_checks(five_clique_list):
    reordered_five_cliques = []
    for five_clique in five_clique_list:
        clique_convex_hull = cv2.convexHull(np.array([[five_clique[0][0], five_clique[0][1]],
                                                      [five_clique[1][0], five_clique[1][1]],
                                                      [five_clique[2][0], five_clique[2][1]],
                                                      [five_clique[3][0], five_clique[3][1]],
                                                      [five_clique[4][0], five_clique[4][1]]]),
                                            clockwise=False)

        # SWAPPED ORDER OF TESTS FOR LOGICAL FLOW
        # Fix and check colors (red vs magenta, yellow chroma)
        five_clique = correct_red_magenta_color_codes_of_clique(five_clique)
        if five_clique is None:
            continue

        # One patch should be surrounded by 4 others, otherwise skip
        if clique_convex_hull.shape[0] == 4:
            reordered_clique = reorder_surrounding_patches_ccw(five_clique,
                                                               clique_convex_hull)
        else:
            continue


        # Check angle and edge length bounds of quadrilateral determined
        # by the centroids of the four blobs.
        if apply_angle_and_edge_length_filters_to_five_clique(five_clique):
            reordered_five_cliques.append(reordered_clique)
        else:
            continue

    return(reordered_five_cliques)


# Note that we don't move the center blob!
# @nb.jit(cache=True)
def rotate_five_clique_list_so_top_is_first(five_clique_list):
    n = len(five_clique_list)
    rotated_five_clique_list = []
    for i in range(n):
        five_clique = five_clique_list[i]
        top_index = np.argmin(five_clique[:,0])
        rotated_five_clique = np.empty(five_clique.shape, dtype=np.int32)
        for j in range(4):
            rotated_five_clique[j] = five_clique[(top_index + j) % 4]
        rotated_five_clique[4] = five_clique[4]

        rotated_five_clique_list.append(rotated_five_clique)
    # print("ROTATED")
    # print(rotated_five_clique_list)
    # print("ORIGINAL")
    # print(five_clique_list)
    return(rotated_five_clique_list)


# We reorder the rows clockwise and put the middle last in the list.
@nb.jit(cache=True)
def reorder_surrounding_patches_ccw(five_clique, clique_convex_hull):
    # For the 4 surrounding blobs
    for i in range(4):
        # Find the corresponding row of clique array
        for j in range(i + 1, 5):
            if clique_convex_hull[i][0][0] == five_clique[j][0] and \
                    clique_convex_hull[i][0][1] == five_clique[j][1]:
                # No choice, but to write a workaround because
                # Numba does not fully support numpy.
                # We swap the rows.
                # Last row will be center patch.
                row_i = np.copy(five_clique[i])
                row_j = five_clique[j]
                five_clique[i] = row_j
                five_clique[j] = row_i
                continue
    return(five_clique)


@nb.jit
def compute_angle(p0, p1, p2):
    A0 = p1[0] - p0[0]
    A1 = p1[1] - p0[1]
    B0 = p2[0] - p0[0]
    B1 = p2[1] - p0[1]

    AB = (A0*B0) + (A1*B1)
    A_l = np.sqrt(A0**2 + A1**2)
    B_l = np.sqrt(B0**2 + B1**2)

    ang = np.rad2deg(np.arccos(AB / (A_l * B_l)))

    return(ang)


# Angle test for surrounding 4 blobs.
# Fifth blob not used.
@nb.jit(cache=True)
def do_clique_angle_test(five_clique, min_angle=45, max_angle=135):
    for i in range(4):
        angle = compute_angle(five_clique[(1 + i) % 4][0:2],
                              five_clique[(0 + i) % 4][0:2],
                              five_clique[(2 + i) % 4][0:2])
        if (angle < min_angle) or angle > (max_angle):
            return(False)
    return(True)


# Side length test for quadrilateral defined
# by centroid of 4 blobs.
@nb.jit(cache=True)
def do_side_length_uniformity_test(five_clique,
                                   max_opposite_edge_length_factor_squared = 3.0,
                                   max_neighboring_edge_length_factor_squared = 6.0):
    lengths = np.empty(4, nb.int32)
    for i in range(4):
        p1_x = five_clique[i][0]
        p1_y = five_clique[i][1]
        p2_x = five_clique[(i + 1) % 4][0]
        p2_y = five_clique[(i + 1) % 4][1]
        lengths[i] = np.sqrt(np.square(p1_x - p2_x) + np.square(p1_y - p2_y))

    # Opposite edge length test
    o_longer_1 = np.maximum(lengths[0], lengths[2])
    o_shorter_1 = np.minimum(lengths[0], lengths[2])
    o_longer_2 = np.maximum(lengths[1], lengths[3])
    o_shorter_2 = np.minimum(lengths[1], lengths[3])
    if (o_longer_1 > (o_shorter_1 * max_opposite_edge_length_factor_squared)) or \
            (o_longer_2 > (o_shorter_2 * max_opposite_edge_length_factor_squared)):
        return(False)

    # Neighboring edge length test
    for i in range(4):
        edge_length = lengths[i]
        next_edge_length = lengths[(i + 1) % 4]
        longer = np.maximum(edge_length, next_edge_length)
        shorter = np.minimum(edge_length, next_edge_length)
        if (longer > (shorter * max_neighboring_edge_length_factor_squared)):
            return (False)
    return(True)


@nb.jit(cache=True)
def apply_angle_and_edge_length_filters_to_five_clique(five_clique):
    clique_angle_test = do_clique_angle_test(five_clique)
    side_length_uniformity_test = do_side_length_uniformity_test(five_clique)
    good = clique_angle_test and side_length_uniformity_test
    return(good)