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


import cv2
import numpy as np

def draw_4_clique_boundaries(img, four_clique_list):
    for clique in four_clique_list:
        pt1_row = clique[0][0]
        pt1_col = clique[0][1]
        pt2_row = clique[1][0]
        pt2_col = clique[1][1]
        pt3_row = clique[2][0]
        pt3_col = clique[2][1]
        pt4_row = clique[3][0]
        pt4_col = clique[3][1]
        sequence_of_points = np.array([[[pt1_col, pt1_row],
                            [pt2_col, pt2_row],
                            [pt3_col, pt3_row],
                            [pt4_col, pt4_row]]], dtype=np.int32)
        #print(sequence_of_points)
        cv2.polylines(img, sequence_of_points, True, (0,255,255))
    return(img)


def draw_5_clique_with_index_clique_index(img, five_clique_list, contour_list, clique_index):
    if clique_index >= len(five_clique_list):
        clique_index = len(five_clique_list) - 1
    if clique_index > -1:
        clique = five_clique_list[clique_index]
        for i in range(5):
            if clique[i][3] == 1:
                color = (255, 0, 0)
            if clique[i][3] == 2:
                color = (255, 255, 0)
            if clique[i][3] == 3:
                color = (0, 255, 0)
            if clique[i][3] == 4:
                color = (0, 0, 255)
            if clique[i][3] == 5:
                color = (255, 0, 255)
            cv2.drawContours(img, [contour_list[clique[i][6]]], -1, color, -1)


def draw_5_clique_list(img,
                       five_clique_list,
                       contour_list):
    for clique in five_clique_list:
        for i in range(5):
            if clique[i][3] == 1:
                color = (255, 0, 0)
            if clique[i][3] == 2:
                color = (255, 255, 0)
            if clique[i][3] == 3:
                color = (0, 255, 0)
            if clique[i][3] == 4:
                color = (0, 0, 255)
            if clique[i][3] == 5:
                color = (255, 0, 255)
            cv2.drawContours(img, [contour_list[clique[i][6]]], -1, color, -1)


def nice_draw_5_clique_list(img,
                       five_clique_list,
                       contour_list):
    for clique in five_clique_list:
        for i in range(5):
            if clique[i][3] == 1:
                color = (255, 0, 0)
            if clique[i][3] == 2:
                color = (255, 255, 0)
            if clique[i][3] == 3:
                color = (0, 255, 0)
            if clique[i][3] == 4:
                color = (0, 0, 255)
            if clique[i][3] == 5:
                color = (255, 0, 255)
            cv2.drawContours(img, [contour_list[clique[i][6]]], 0, color, 0)

            cv2.line(img,
                     (clique[i][1]-3, clique[i][0]),
                     (clique[i][1]+3, clique[i][0]),
                     (0,255,255), 1)
            cv2.line(img,
                     (clique[i][1], clique[i][0]-3),
                     (clique[i][1], clique[i][0]+3),
                     (0,255,255), 1)



def draw_standard_5_clique(img,
                           five_clique_list,
                           standard_clique_contours_list,
                           clique_index):
    if clique_index >= len(five_clique_list):
        clique_index = len(five_clique_list) - 1
    if clique_index > -1:
        clique = five_clique_list[clique_index]
        std_contours = standard_clique_contours_list[clique_index]
        for i in range(5):
            if clique[i][3] == 1:
                color = (255, 0, 0)
            if clique[i][3] == 2:
                color = (255, 255, 0)
            if clique[i][3] == 3:
                color = (0, 255, 0)
            if clique[i][3] == 4:
                color = (0, 0, 255)
            if clique[i][3] == 5:
                color = (255, 0, 255)
            cv2.drawContours(img, [std_contours[i]], -1, color, -1)


# Extracts color codes from 5 clique and produces a string descriptor.
# Descriptor starts with red if there is red (exluding central blob), and
# otherwise starts with magenta.
def get_marker_code(five_clique):
    first_index = -1
    for i in range(4):
        if five_clique[i][3] == 1:
            first_index = i
            break
        if five_clique[i][3] == 5:
            first_index = i

    code = ''
    for i in range(4):
        current = five_clique[(i + first_index) % 4][3]
        if current == 1:
            code = code + 'R'
        if current == 2:
            code = code + 'Y'
        if current == 3:
            code = code + 'G'
        if current == 4:
            code = code + 'B'
        if current == 5:
            code = code + 'M'
    return(code)


def get_marker_codes(five_clique_list):
    marker_codes = []
    for five_clique in five_clique_list:
        code = get_marker_code(five_clique)
        marker_codes.append(code)
    return(marker_codes)


def draw_standard_5_clique_list(img,
                                five_clique_list,
                                standard_clique_contours_list):
    for id, (five_clique, std_contours) in enumerate(zip(five_clique_list,
                                                       standard_clique_contours_list)):
        ID = get_marker_code(five_clique)
        for i in range(5):
            if five_clique[i][3] == 1:
                color = (255, 0, 0)
            if five_clique[i][3] == 2:
                color = (255, 255, 0)
            if five_clique[i][3] == 3:
                color = (0, 255, 0)
            if five_clique[i][3] == 4:
                color = (0, 0, 255)
            if five_clique[i][3] == 5:
                color = (255, 0, 255)
            cv2.drawContours(img, [std_contours[i] + (id * 140, 0)], -1, color, -1)
            cv2.putText(img, ID, (100 + id * 140, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)

    for (index, clique) in enumerate(five_clique_list):
        cv2.line(img, (clique[0][1], clique[0][0]), (160 + index * 140, 210), (0, 255, 255), 1)


# Draw colored detected blobs
def draw_all_detected_blobs(img,
                            blobs_with_stats,
                            filtered_contours):
    for i in range(blobs_with_stats.shape[0]):
        if blobs_with_stats[i][3] == 1:
            color = (255,0,0)
        if blobs_with_stats[i][3] == 2:
            color = (255, 255, 0)
        if blobs_with_stats[i][3] == 3:
            color = (0, 255, 0)
        if blobs_with_stats[i][3] == 4:
            color = (0, 0, 255)
        if blobs_with_stats[i][3] == 5:
            color = (255, 0, 255)
        cv2.drawContours(img, [filtered_contours[blobs_with_stats[i][6]]], -1, color, -1)


def draw_squares(img, squares, num):
    numpy_squares = np.array(squares)
    for i in range(num):
        cv2.drawContours(img, numpy_squares + (i * 140, 0), -1, (0, 0, 0), 3)

