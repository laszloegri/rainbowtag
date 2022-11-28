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

center_point_x = 400
center_point_y = 400

# Takes the centroids of the 4 blobs, c1, c2, c3, c4. Finds the perspective transform
# matrix from c1, c2, c3, c4 to (0,0), (0,100), (100,100), (100,0).
# It also applies the transform to the points of the blob contours.
# Now all stickers are in the same reference plane with standardized size,
# so we increased "invariance", it will be easier to analyze shape (shape info unrelated to
# centroid location.)
# The transformed blobs are stored in a list.

def move_clique_blobs_to_standard_reference_points_using_perspective_transform_aux(clique,
                                                                               filtered_contours,
                                                                               center_point_x = 400,
                                                                               center_point_y = 400,
                                                                               standard_square_side_length = 100):
    # Let's compute perspective transform
    reference_pts = np.array([[center_point_x, center_point_y],
                             [center_point_x, center_point_y + standard_square_side_length],
                             [center_point_x + standard_square_side_length, center_point_y + standard_square_side_length],
                             [center_point_x + standard_square_side_length, center_point_y]], dtype=np.float32)

    blob_centroids = np.array([[clique[0][1], clique[0][0]],
                        [clique[1][1], clique[1][0]],
                        [clique[2][1], clique[2][0]],
                        [clique[3][1], clique[3][0]]],
                       dtype=np.float32)

    m = cv2.getPerspectiveTransform(blob_centroids,
                                    reference_pts)

    b1 = np.array(filtered_contours[clique[0][6]], dtype=np.float32)
    b2 = np.array(filtered_contours[clique[1][6]], dtype=np.float32)
    b3 = np.array(filtered_contours[clique[2][6]], dtype=np.float32)
    b4 = np.array(filtered_contours[clique[3][6]], dtype=np.float32)

    standard_b1  = cv2.perspectiveTransform(b1, m)
    standard_b2  = cv2.perspectiveTransform(b2, m)
    standard_b3  = cv2.perspectiveTransform(b3, m)
    standard_b4  = cv2.perspectiveTransform(b4, m)

    standard_cts = [np.array(standard_b1, dtype=np.int),
           np.array(standard_b2, dtype=np.int),
           np.array(standard_b3, dtype=np.int),
           np.array(standard_b4, dtype=np.int)]

    return(standard_cts)


def move_clique_blobs_to_standard_reference_points_using_perspective_transform(clique_list,
                                                                               filtered_contours,
                                                                               center_point_x = center_point_x,
                                                                               center_point_y = center_point_y,
                                                                               standard_square_side_length = 100):
    if len(clique_list) == 0:
        return([])
    standard_clique_list = []
    for clique in clique_list:
        standard_clique = move_clique_blobs_to_standard_reference_points_using_perspective_transform_aux(clique,
                                                                                                         filtered_contours,
                                                                                                         center_point_x,
                                                                                                         center_point_y,
                                                                                                         standard_square_side_length)
        standard_clique_list.append(standard_clique)
    return(standard_clique_list)


def filter_standard_blobs_by_shape_perspective(four_cliques_list,
                                               standard_clique_list,
                                               img,
                                               center_point_shift = 400):
    curvy_test = True
    do_blob_box_property_tests = True
    do_other_blob_tests = True

    blob_box_margin = 75
    max_blob_side_ratio_perspective = 4
    max_curviness_perspective = 5.5
    min_blob_area_in_standard_clique_thr = 430
    min_mean_blob_area_in_standard_clique_thr = 1100
    blob_area_difference_factor = 10
    max_rect_ang_diff = 130
    min_opposite_min_area_bounding_rectangle_distance = 40

    filtered_cliques = []
    filtered_standard_cliques = []
    clique_score_table = []

    for standard_clique, clique in zip(standard_clique_list, four_cliques_list):
        b1 = standard_clique[0]
        b2 = standard_clique[1]
        b3 = standard_clique[2]
        b4 = standard_clique[3]

        # AREA TESTS
        b1_area = cv2.contourArea(b1)
        b2_area = cv2.contourArea(b2)
        b3_area = cv2.contourArea(b3)
        b4_area = cv2.contourArea(b4)
        mean_blob_area_in_standard_clique = np.mean([b1_area, b2_area, b3_area, b4_area])
        if mean_blob_area_in_standard_clique < min_mean_blob_area_in_standard_clique_thr:
            continue

        # print("HERE")
        # print(b1_area)
        # print(b2_area)
        # print(b3_area)
        # print(b4_area)
        min_blob_area_in_standard_clique = np.min([b1_area, b2_area, b3_area, b4_area])

        # We need the following values
        rect_around_b1 = cv2.minAreaRect(b1)
        rect_around_b2 = cv2.minAreaRect(b2)
        rect_around_b3 = cv2.minAreaRect(b3)
        rect_around_b4 = cv2.minAreaRect(b4)

        if do_other_blob_tests:
            # MIN AREA TO CENTRAL QUAD RATIO TEST (central quad has area 10000, because it is standardized)
            if min_blob_area_in_standard_clique < min_blob_area_in_standard_clique_thr:
                continue

            max_blob_area = np.max([b1_area, b2_area, b3_area, b4_area])
            if min_blob_area_in_standard_clique * blob_area_difference_factor < max_blob_area:
                continue


            # Too long blob
            long_1 = np.maximum(rect_around_b1[1][0], rect_around_b1[1][1])
            short_1 = np.minimum(rect_around_b1[1][0], rect_around_b1[1][1])
            if short_1 == 0:
                continue
            if (float(long_1) / short_1) > max_blob_side_ratio_perspective:
                continue

            long_2 = np.maximum(rect_around_b2[1][0], rect_around_b2[1][1])
            short_2 = np.minimum(rect_around_b2[1][0], rect_around_b2[1][1])
            if short_2 == 0:
                continue
            if (float(long_2) / short_2) > max_blob_side_ratio_perspective:
                continue

            long_3 = np.maximum(rect_around_b3[1][0], rect_around_b3[1][1])
            short_3 = np.minimum(rect_around_b3[1][0], rect_around_b3[1][1])
            if short_3 == 0:
                continue
            if (float(long_3) / short_3) > max_blob_side_ratio_perspective:
                continue

            long_4 = np.maximum(rect_around_b4[1][0], rect_around_b4[1][1])
            short_4 = np.minimum(rect_around_b4[1][0], rect_around_b4[1][1])
            if short_4 == 0:
                continue
            if (float(long_4) / short_4) > max_blob_side_ratio_perspective:
                continue

        # Curvy blob
        if curvy_test:
            perimeter_1 = cv2.arcLength(b1, True)
            estimated_perimeter_1 = np.sqrt(b1_area)
            # print(perimeter_1)
            # print(estimated_perimeter_1)
            if (float(perimeter_1) / estimated_perimeter_1) > max_curviness_perspective:
                continue

            perimeter_2 = cv2.arcLength(b2, True)
            estimated_perimeter_2 = np.sqrt(b2_area)
            # print(perimeter_2)
            # print(estimated_perimeter_2)
            if (float(perimeter_2) / estimated_perimeter_2) > max_curviness_perspective:
                continue

            perimeter_3 = cv2.arcLength(b3, True)
            estimated_perimeter_3 = np.sqrt(b3_area)
            # print(perimeter_3)
            # print(estimated_perimeter_3)
            if (float(perimeter_3) / estimated_perimeter_3) > max_curviness_perspective:
                continue

            perimeter_4 = cv2.arcLength(b4, True)
            estimated_perimeter_4 = np.sqrt(b4_area)
            # print(perimeter_4)
            # print(estimated_perimeter_4)
            if (float(perimeter_4) / estimated_perimeter_4) > max_curviness_perspective:
                continue


        # Let's collect blob min and max coordinates. First we che
        # Blob 1
        if do_blob_box_property_tests:
            min_x_1 = np.min(b1[:, :, 0])
            max_x_1 = np.max(b1[:, :, 0])
            min_y_1 = np.min(b1[:, :, 1])
            max_y_1 = np.max(b1[:, :, 1])
            if min_x_1 < (center_point_shift - blob_box_margin) or\
                    max_x_1 > (center_point_shift + blob_box_margin) or\
                    min_y_1 < (center_point_shift - blob_box_margin) or\
                    max_y_1 > (center_point_shift + blob_box_margin):
                continue

            # Blob 2
            min_x_2 = np.min(b2[:, :, 0])
            max_x_2 = np.max(b2[:, :, 0])
            min_y_2 = np.min(b2[:, :, 1])
            max_y_2 = np.max(b2[:, :, 1])
            if min_x_2 < (center_point_shift - blob_box_margin) or\
                    max_x_2 > (center_point_shift + blob_box_margin) or\
                    min_y_2 < (center_point_shift + 100 - blob_box_margin) or\
                    max_y_2 > (center_point_shift + 100 + blob_box_margin):
                continue

            # Blob 3
            min_x_3 = np.min(b3[:, :, 0])
            max_x_3 = np.max(b3[:, :, 0])
            min_y_3 = np.min(b3[:, :, 1])
            max_y_3 = np.max(b3[:, :, 1])
            if min_x_3 < (center_point_shift + 100 - blob_box_margin) or\
                    max_x_3 > (center_point_shift + 100 + blob_box_margin) or\
                    min_y_3 < (center_point_shift + 100 - blob_box_margin) or\
                    max_y_3 > (center_point_shift + 100 + blob_box_margin):
                continue

            # Blob 4
            min_x_4 = np.min(b4[:, :, 0])
            max_x_4 = np.max(b4[:, :, 0])
            min_y_4 = np.min(b4[:, :, 1])
            max_y_4 = np.max(b4[:, :, 1])
            if min_x_4 < (center_point_shift + 100 - blob_box_margin) or\
                    max_x_4 > (center_point_shift + 100 + blob_box_margin) or\
                    min_y_4 < (center_point_shift - blob_box_margin) or\
                    max_y_4 > (center_point_shift + blob_box_margin):
                continue


        # Check min area rectangle angles
        rect_ang_diff_1 = np.abs(rect_around_b1[2] + 45)
        rect_ang_diff_2 = np.abs(rect_around_b2[2] + 45)
        rect_ang_diff_3 = np.abs(rect_around_b3[2] + 45)
        rect_ang_diff_4 = np.abs(rect_around_b4[2] + 45)
        rect_ang_diff_score = rect_ang_diff_1 + rect_ang_diff_2 + rect_ang_diff_3 + rect_ang_diff_4
        # print("ONE")
        # print(rect_ang_diff_1)
        # print(rect_around_b1[2])
        # print("TWO")
        # print(rect_ang_diff_2)
        # print(rect_around_b2[2])
        # print("THEE")
        # print(rect_ang_diff_3)
        # print(rect_around_b3[2])
        # print("FOUR")
        # print(rect_ang_diff_4)
        # print(rect_around_b4[2])
        # print(rect_ang_diff_score)

        box_1_corners = cv2.boxPoints(rect_around_b1)
        box_1_corners = np.int0(box_1_corners)
        box_2_corners = cv2.boxPoints(rect_around_b2)
        box_2_corners = np.int0(box_2_corners)
        box_3_corners = cv2.boxPoints(rect_around_b3)
        box_3_corners = np.int0(box_3_corners)
        box_4_corners = cv2.boxPoints(rect_around_b4)
        box_4_corners = np.int0(box_4_corners)

        # Are opposite min area rectangles far enough from each other?

        good = are_opposite_min_area_rectangles_far(box_1_corners,
                                                    box_2_corners,
                                                    box_3_corners,
                                                    box_4_corners,
                                                    min_opposite_min_area_bounding_rectangle_distance)
        if not good:
            continue

        if rect_ang_diff_score > max_rect_ang_diff:
            continue

        # print("HERE")
        # print(rect_around_b1)
        color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        cv2.drawContours(img, [box_1_corners], 0, color, 2)
        cv2.drawContours(img, [box_2_corners], 0, color, 2)
        cv2.drawContours(img, [box_3_corners], 0, color, 2)
        cv2.drawContours(img, [box_4_corners], 0, color, 2)

        filtered_cliques.append(clique)
        filtered_standard_cliques.append(standard_clique)


        # We compute a goodness core and pass it on to clique_score_table.
        # Blob area uniformity score:
        # area_std = np.std(np.array([b1_area,
        #                             b2_area,
        #                             b3_area,
        #                             b4_area]))
        # clique_scores = np.array([rect_ang_diff_score, area_std, ratio])

        clique_score_table.append(rect_ang_diff_score)

    return(filtered_cliques,
           filtered_standard_cliques,
           clique_score_table,
           img)


@nb.njit
def are_opposite_min_area_rectangles_far(box_1_corners,
                                         box_2_corners,
                                         box_3_corners,
                                         box_4_corners,
                                         min_opposite_min_area_bounding_rectangle_distance):
    #min_13 = np.inf
    for i in range(4):
        for j in range(4):
            x1 = box_1_corners[i][0]
            y1 = box_1_corners[i][1]
            x2 = box_3_corners[j][0]
            y2 = box_3_corners[j][1]
            d = np.sqrt(np.square((x1 - x2)) + np.square((y1 - y2)))
            if d < min_opposite_min_area_bounding_rectangle_distance:
                return(False)
                # return(False, 0)
            # if d < min_13:
            #     min_13 = d

    min_24 = np.inf
    for i in range(4):
        for j in range(4):
            x1 = box_2_corners[i][0]
            y1 = box_2_corners[i][1]
            x2 = box_4_corners[j][0]
            y2 = box_4_corners[j][1]
            d = np.sqrt(np.square((x1 - x2)) + np.square((y1 - y2)))
            if d < min_opposite_min_area_bounding_rectangle_distance:
                return(False)
            #     return(False, 0)
            # if d < min_24:
            #     min_24 = d

    # ratio = 0
    # if min_13 < min_24:
    #     ratio = min_13 / min_24
    # else:
    #     ratio = min_24 / min_13
    #return(True, ratio)
    return(True)