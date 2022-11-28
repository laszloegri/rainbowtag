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
import time
import cv2
import numpy as np
from contour_methods import compute_size_of_intersection_of_contours
from shapely.geometry import Polygon

x_translate = 100
y_translate = 100

translate = np.array([[[x_translate, y_translate]],
                      [[x_translate, y_translate]],
                      [[x_translate, y_translate]],
                      [[x_translate, y_translate]]])

square_1 = np.array([[[20, 0]], [[40, 0]], [[40, 20]], [[20, 20]]]) + translate
square_2 = np.array([[[0, 20]],[[20, 20]], [[20, 40]], [[0, 40]]]) + translate
square_3 = np.array([[[20, 40]], [[40, 40]], [[40, 60]], [[20, 60]]]) + translate
square_4 = np.array([[[40, 20]], [[60, 20]], [[60, 40]], [[40, 40]]]) + translate
square_5 = np.array([[[20, 20]], [[40, 20]], [[40, 40]], [[20, 40]]]) + translate

squares = [square_1,
          square_2,
          square_3,
          square_4,
          square_5]


# Computes perspective map to template, applies them to the five blobs of clique,
# and returns a list containing the five blobs (numpy array).
def apply_perspective_to_clique_blobs(clique,
                                      filtered_contours):

    # target_centroids = np.array([[31 + x_translate, 10 + y_translate],
    #                              [10 + x_translate, 31 + y_translate],
    #                              [31 + x_translate, 52 + y_translate],
    #                              [52 + x_translate, 31 + y_translate]],
    #                             dtype=np.float32)

    # EXACT TEMPLATE NOT GOOD
    target_centroids = np.array([[30 + x_translate, 10 + y_translate],
                                 [10 + x_translate, 30 + y_translate],
                                 [30 + x_translate, 50 + y_translate],
                                 [50 + x_translate, 30 + y_translate]],
                                dtype=np.float32)

    blob_centroids = np.array([[clique[0][1], clique[0][0]],
                        [clique[1][1], clique[1][0]],
                        [clique[2][1], clique[2][0]],
                        [clique[3][1], clique[3][0]]],
                       dtype=np.float32)

    perps_transform_from_blobs_to_std = cv2.getPerspectiveTransform(blob_centroids,
                                    target_centroids)

    blob_1 = np.array(filtered_contours[clique[0][6]], dtype=np.float32)
    blob_2 = np.array(filtered_contours[clique[1][6]], dtype=np.float32)
    blob_3 = np.array(filtered_contours[clique[2][6]], dtype=np.float32)
    blob_4 = np.array(filtered_contours[clique[3][6]], dtype=np.float32)
    blob_5 = np.array(filtered_contours[clique[4][6]], dtype=np.float32)

    standard_blob_1  = cv2.perspectiveTransform(blob_1, perps_transform_from_blobs_to_std)
    standard_blob_2  = cv2.perspectiveTransform(blob_2, perps_transform_from_blobs_to_std)
    standard_blob_3  = cv2.perspectiveTransform(blob_3, perps_transform_from_blobs_to_std)
    standard_blob_4  = cv2.perspectiveTransform(blob_4, perps_transform_from_blobs_to_std)
    standard_blob_5  = cv2.perspectiveTransform(blob_5, perps_transform_from_blobs_to_std)

    standard_contours = [np.array(standard_blob_1, dtype=np.int),
                    np.array(standard_blob_2, dtype=np.int),
                    np.array(standard_blob_3, dtype=np.int),
                    np.array(standard_blob_4, dtype=np.int),
                    np.array(standard_blob_5, dtype=np.int)]

    return(standard_contours)


# THIS IS FOR ONE CANDIDATE AFTER PERSPECTIVE TRANSFORM IS APPLIED.
# Returns -1 if candidate this not pass the test.
# Otherwise, a goodness score for the candidate is
# returned.
def score_and_threshold_candidate(standard_contours):

    # Go over 5 blobs
    r0 = 0
    r1 = 0
    r2 = 0
    for contour, square in zip(standard_contours, squares):
        r = compute_size_of_intersection_of_contours(square, contour, 260, 260)
        percentage_of_blob_in_box = r[2] / r[1]
        percentage_of_template_occupied = r[2] / r[0]
        percentage_of_template = r[1] / r[0]

        if (percentage_of_blob_in_box < 0.50) or\
                (percentage_of_template_occupied < 0.05) or\
                (percentage_of_template > 2.0):
            # print("PERCENTAGE OF BLOB IN TEMPLATE:")
            # print(percentage_of_blob_in_box)
            # print("PERCENTAGE OF TEMPLATE OCCUPIED:")
            # print(percentage_of_template_occupied)
            # print("PERCENTAGE OF TEMPLATE:")
            # print(percentage_of_template)
            return((-1, -1, -1))

        r0 = r0 + percentage_of_blob_in_box
        r1 = r1 + percentage_of_template_occupied
        r2 = r2 + percentage_of_template

    return((r0, r1, r2))


# Computes scores for each clique in perspective and
# prunes standard_clique_list and five_clique_list at the same
# time based on score thresholds.
def filter_candidate_lists_based_on_scores(five_clique_list,
                                           filtered_contours):
    if len(five_clique_list) == 0:
        return([], [], [])

    standard_clique_list = []
    pruned_five_clique_list = []
    score_list = []

    for five_clique in five_clique_list:
        standard_clique = apply_perspective_to_clique_blobs(five_clique,
                                                            filtered_contours)
        t = score_and_threshold_candidate(standard_clique)
        if t[0] != -1:
            standard_clique_list.append(standard_clique)
            pruned_five_clique_list.append(five_clique)
            score_list.append(t)
    return(standard_clique_list,
           pruned_five_clique_list,
           score_list)


# Returns True  if any of the lines connecting opposite blobs of std_clique_1
# intersects the same lines in std_clique_1.
def is_overlapping_old(std_clique_1, std_clique_2):
    p1 = std_clique_1[0][1::-1]
    p2 = std_clique_1[1][1::-1]
    p3 = std_clique_1[2][1::-1]
    p4 = std_clique_1[3][1::-1]
    q1 = std_clique_2[0][1::-1]
    q2 = std_clique_2[1][1::-1]
    q3 = std_clique_2[2][1::-1]
    q4 = std_clique_2[3][1::-1]

    if line_segment_intersection(p1, p3, q1, q3) or \
            line_segment_intersection(p1, p3, q2, q4) or \
            line_segment_intersection(p2, p4, q2, q4) or \
            line_segment_intersection(p2, p4, q2, q4):
        return(True)
    else:
        return(False)


def is_overlapping(std_clique_1, std_clique_2):
    p1 = std_clique_1[0][1::-1]
    p2 = std_clique_1[1][1::-1]
    p3 = std_clique_1[2][1::-1]
    p4 = std_clique_1[3][1::-1]
    q1 = std_clique_2[0][1::-1]
    q2 = std_clique_2[1][1::-1]
    q3 = std_clique_2[2][1::-1]
    q4 = std_clique_2[3][1::-1]

    c1 = Polygon([p1, p2, p3, p4])
    c2 = Polygon([q1, q2, q3, q4])

    return(c1.intersects(c2))

# Returns True if test_clique is dominated by another clique.
# If all scores are the same for two cliques, then the test clique
# is dominated if its id is larger than the other clique.
# That is, if we have several cliques with the exact same score,
# we keep the ony with the lowest index and throw away the rest.
def is_dominated_by_other_cliques_in_list(five_clique_list,
                                          five_clique_score_list,
                                          test_clique,
                                          test_clique_id,
                                          test_clique_score):

    for id, (list_five_clique, list_five_clique_score) in\
            enumerate(zip(five_clique_list, five_clique_score_list)):
        if (test_clique_id != id):
            if is_overlapping(test_clique, list_five_clique):

                # percentage_of_blob_in_box = r[2] / r[1]
                # percentage_of_template_occupied = r[2] / r[0]
                # percentage_of_template = r[1] / r[0]
                # r[0] square size = 441
                # r[1] total blob size
                # r[2] size of intersection og square and blob

                primary_list_score = list_five_clique_score[0] +\
                                     1.2 * list_five_clique_score[1]
                primary_test_score = test_clique_score[0] +\
                                     1.2 * test_clique_score[1]
                secondary_list_score = list_five_clique_score[2]
                secondary_test_score = test_clique_score[2]

                # True if test clique is dominated.
                if (primary_list_score > primary_test_score) or \
                        ((primary_list_score == primary_test_score) and \
                        (secondary_list_score > secondary_test_score)):
                    return(True)
                # If there is a tie between test clique,
                # and clique, then the test clique is dominated
                # clique is earlier in the list. (We want to keep only the
                # smallest index cliques if there are many with the same score.)
                elif (primary_list_score == primary_test_score) and \
                    (secondary_list_score == secondary_test_score) and \
                    test_clique_id > id:
                    return(True)
    return(False)


# For any clique
def remove_overlapping_dominated_cliques(standard_clique_list,
                                         five_clique_list,
                                         five_clique_score_list):
    pruned_standard_clique_list = []
    pruned_five_clique_list = []
    pruned_score_list = []

    for test_clique_id,\
        (standard_clique,
         test_clique,
         test_clique_score) in enumerate(zip(standard_clique_list,
                                             five_clique_list,
                                             five_clique_score_list)):

        if not is_dominated_by_other_cliques_in_list(five_clique_list,
                                                     five_clique_score_list,
                                                     test_clique,
                                                     test_clique_id,
                                                     test_clique_score):
            pruned_standard_clique_list.append(standard_clique)
            pruned_five_clique_list.append(test_clique)
            pruned_score_list.append(test_clique_score)

    return(pruned_standard_clique_list,
           pruned_five_clique_list,
           pruned_score_list)


###########################################
###### INTERSECTION OF LINE SEGMENTS ######
###########################################

# Returns true if q is on the line segment p1-p2
def is_point_on_line_segment(p1, p2, q):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    x3 = q[0]
    y3 = q[1]

    # If p1 = p2, then q must be p1
    if x1 == x2 and y1 == y2:
        if x3 == x1 and y3 == y1:
            return(True)
        else:
            return(False)

    # If x coordinates of p1 and p2 are the same, line segment
    # is vertical.
    if x1 == x2 and y1 != y2:
        if x3 == x1:
            u = (y3 - y1) / (y2 - y1)
            if u >= 0 and u <= 1:
                return(True)
        return(False)

    # If y coordinates of p1 and p2 are the same, line segment
    # is horizontal.
    if y2 == y1 and x1 != x2:
        if y3 == y1:
            u = (x3 - x1) / (x2 - x1)
            if u >= 0 and u <= 1:
                return(True)
        return(False)

    # Otherwise we are in the general case.
    u1 = (x3 - x1) / (x2 - x1)
    u2 = (y3 - y1) / (y2 - y1)
    if u1 != u2:
        return(False)
    else:
        if not (0 <= u1 and u1 <= 1):
            return(False)

    return(True)


# Methods returns true if and only if two lines intersect.
# We use standard xy-coordinate system but reflection and rotation
# does not change intersection status, so we are ok.
#@nb.jit(cache=True)
def line_segment_intersection(p1, p2, p3, p4):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    x3 = p3[0]
    y3 = p3[1]
    x4 = p4[0]
    y4 = p4[1]

    determinant = float((x2 - x1) * (y3 - y4) - (x3 - x4) * (y2 - y1))

    # If segments are parallel
    if determinant == 0:
        if is_point_on_line_segment(p1, p2, p3):
            return(True)
        if is_point_on_line_segment(p1, p2, p4):
            return (True)
        if is_point_on_line_segment(p3, p4, p1):
            return (True)
        if is_point_on_line_segment(p3, p4, p2):
            return (True)
        return(False)

    determinant_u = float((x3 - x1) * (y3 - y4) - (x3 - x4) * (y3 - y1))
    determinant_v = float((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    u = determinant_u / determinant
    v = determinant_v / determinant

    if not ((0 <= u) and (u <= 1) and (0 <= v) and (v <= 1)):
        return(False)

    # Interesection point, assuming xy-coordinate system
    # np.array([x1 + u * (x2 - x1), y1 + v * (y2 - y1)])

    return(True)