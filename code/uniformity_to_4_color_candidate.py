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


from patch_uniformity_checks import *
from color_processing_methods import convert_img_to_6_channels
from parameters import max_area_difference_factor, \
    max_relative_distance, \
    max_blob_side_ratio, \
    max_blob_area, \
    max_curviness, \
    sampling_box_size, \
    LMS_quantization_number


@nb.njit
def replace_marker_by_three_by_three_square_original(uniformity_map):
    h, w = uniformity_map.shape
    h_e = h + sampling_box_size - 1
    w_e = w + sampling_box_size - 1
    replaced = np.zeros((h_e, w_e), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            if uniformity_map[row, col] == True:
                for row_sw in range(sampling_box_size-1):
                    for col_sw in range(sampling_box_size-1):
                        replaced[row + row_sw, col + col_sw] = 1
    return(replaced)

# TEST
# THIS IS NOT DOING MUCH SINCE THE INNER FOR LOOPS WERE CHANGED TO 1
# SIMPLIFY
@nb.njit
def replace_marker_by_three_by_three_square(uniformity_map):
    h, w = uniformity_map.shape
    h_e = h + sampling_box_size - 1
    w_e = w + sampling_box_size - 1
    replaced = np.zeros((h_e, w_e), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            if uniformity_map[row, col] == True:
                for row_sw in range(1):
                    for col_sw in range(1):
                        replaced[row + row_sw, col + col_sw] = 1
    return(replaced)



@nb.njit
def add_right_bottom_pixels_to_connected_component_original(cc, i):
    h, w = cc.shape
    fat_cc = np.zeros((h+1, w+1), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            if cc[row, col] == i:
                fat_cc[row, col] = 1
                fat_cc[row+1, col] = 1
                fat_cc[row, col+1] = 1
                fat_cc[row+1, col+1] = 1
    return(fat_cc)


@nb.njit
def add_right_bottom_pixels_to_connected_component(cc, i):
    h, w = cc.shape
    fat_cc = np.zeros((h + sampling_box_size - 1,
                       w + sampling_box_size - 1),
                      dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            if cc[row, col] == i:
                for row_sw in range(sampling_box_size):
                    for col_sw in range(sampling_box_size):
                        fat_cc[row + row_sw, col + col_sw] = 1
    return(fat_cc)


"""
Makes a list of centroids and areas of blobs.
All contours are also returned for later visualization.
(No color info is used yet but it is recorded for later tests.)
"""
def find_blobs_with_stats(img_7_channels,
                          max_blob_side_ratio=max_blob_side_ratio,
                          max_blob_area=max_blob_area,
                          max_curviness=max_curviness):

    lab_L_img = img_7_channels[0]
    lab_chroma_img = img_7_channels[1]
    ipt_hue_angle_img = img_7_channels[2]
    ipt_hue_code_map = img_7_channels[3]

    adaptive_high_chroma_mask = compute_adaptive_high_chroma_mask(lab_L_img,
                                                                  lab_chroma_img,
                                                                  ipt_hue_code_map)
    processed, _, _ = compute_sliding_window_uniformity(
        ipt_hue_angle_img,
        lab_chroma_img,
        max_chroma_difference_red,
        max_chroma_difference_yellow,
        max_chroma_difference_green,
        max_chroma_difference_blue,
        max_chroma_difference_magenta,
        lab_L_img,
        max_intensity_difference,
        ipt_hue_code_map,
        adaptive_high_chroma_mask)

    three_by_three_separated_components_map = replace_marker_by_three_by_three_square(processed)
    label_map = cv2.connectedComponentsWithStats(three_by_three_separated_components_map)

    # We make an array to store the blobs. Note that
    # since we filter blobs, not the whole array will be filled.
    # Entries:in
    # 0: row
    # 1: col
    # 2: area
    # 3: ipt_hue_code
    # 4: ipt_hue_angle_img
    # 5: lab_chroma
    # 6: ID in all_contours list below, for later retrieval

    blobs_with_stats = np.empty((label_map[0] - 1, 7), dtype=np.int32)
    filtered_contours = []
    all_contours = []

    # For each "connected" component. (0 is background.)
    index_of_filtered_contour = 0
    for i in range(1, label_map[0]):
        stats = label_map[2]
        x = stats[i][0]
        y = stats[i][1]
        w = stats[i][2]
        h = stats[i][3]
        component = add_right_bottom_pixels_to_connected_component(label_map[1][y:y+h, x:x+w], i)

        # Let's start with determining if the component
        # is not proportional in shape
        contours = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x, y))

        # If the component has a hole, we have more than one contours.
        # We want to ignore these areas.
        for ctr in contours[0]:
            all_contours.append(ctr)

        if len(contours[0]) != 1:
            continue

        contour = contours[0][0]

        # Test
        rect_around_blob = cv2.minAreaRect(contour)
        long = np.maximum(rect_around_blob[1][0], rect_around_blob[1][1])
        short = np.minimum(rect_around_blob[1][0], rect_around_blob[1][1])
        if short == 0:
            continue
        if (float(long) / short) > max_blob_side_ratio:
            continue

        # 2 is stats matrix, 1 is first connected component (0 is background), and 4 is
        # index of area
        # Compute all statistics
        comp_ccws = cv2.connectedComponentsWithStats(component)
        area = comp_ccws[2][1][4]

        # Test
        if area > max_blob_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        estimated_perimeter = np.sqrt(area)
        if (float(perimeter) / estimated_perimeter) > max_curviness:
            continue

        # 3 is centroid matrix, 1 is first components (0 is background)
        centroid = comp_ccws[3][1]
        row = centroid[1].astype(np.int) + y
        col = centroid[0].astype(np.int) + x

        blobs_with_stats[index_of_filtered_contour] = np.array([row,
                                                                col,
                                                                area,
                                                                ipt_hue_code_map[row, col],
                                                                ipt_hue_angle_img[row, col],
                                                                lab_chroma_img[row, col],
                                                                index_of_filtered_contour])
        filtered_contours.append(contour)
        index_of_filtered_contour = index_of_filtered_contour + 1

    # Adjust size of blobs_with_stats to reduced size
    blobs_with_stats = blobs_with_stats[0:index_of_filtered_contour]
    return(blobs_with_stats,
           index_of_filtered_contour,
           filtered_contours,
           all_contours,
           processed,
           lab_chroma_img,
           lab_L_img)


@nb.njit
def pairwise_blob_compatibility_table(blobs_with_stats,
                                      number_of_filtered_contours,
                                      max_area_difference_factor = max_area_difference_factor,
                                      max_relative_distance = max_relative_distance):
    adjacency_matrix = np.zeros((number_of_filtered_contours, number_of_filtered_contours), dtype=nb.boolean)
    squared_max_relative_distance = np.square(max_relative_distance)
    num_pairs = 0
    for col in nb.prange(number_of_filtered_contours):
        for row in nb.prange(number_of_filtered_contours):

            # The matrix is symmetric, we just have to compute above diagonal
            # and copy symmetrically below.
            if row >= col:
                continue

            # Blobs cannot be too different in size
            area_1 = blobs_with_stats[col][2]
            area_2 = blobs_with_stats[row][2]
            if np.maximum(area_1, area_2) > (max_area_difference_factor * np.minimum(area_1, area_2)):
                continue

            # Blobs cannot be too far (relative to their sizes)
            #squared_distance_unit = np.square((sqrt_area_1 + sqrt_area_2) / 2)
            squared_distance_unit = np.minimum(area_1, area_2)
            x1 = blobs_with_stats[col][1]
            y1 = blobs_with_stats[col][0]
            x2 = blobs_with_stats[row][1]
            y2 = blobs_with_stats[row][0]
            squared_distance = np.square(x1 - x2) + np.square(y1 - y2)
            if squared_distance > squared_max_relative_distance * squared_distance_unit:
                continue

            # Blobs should have different colors
            color_code_1 = blobs_with_stats[col][3]
            color_code_2 = blobs_with_stats[row][3]

            # if color_code_1 == color_code_2:
            #     continue
            # IF WE ALLOW R-R AND M-M CONNECTIONS:
            if (color_code_1 == color_code_2):
                if (color_code_1 != 1) and (color_code_1 != 5):
                    continue

            adjacency_matrix[col, row] = True
            adjacency_matrix[row, col] = True

            num_pairs = num_pairs + 1

    return(adjacency_matrix)