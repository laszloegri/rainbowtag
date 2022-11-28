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


print("Initializing, please wait. This might take a few minutes.")

from adjacency_matrix_methods import *
from filtering_before_perspective import *
from drawing_methods import *
from uniformity_to_4_color_candidate import *
from template_matching import *
from parameters import images_folder,\
    file_extension,\
    rotate_image,\
    resize_img,\
    resize_height,\
    resize_width,\
    add_black_borders,\
    black_border_size, \
    sRGB_to_LMS_table, \
    LMS_to_master_table
from dynamic_color_master_table import dynamic_convert_img_to_7_channels,\
    compute_LMS_gain_coefficients_for_XYZ_white_point_to_D65
from disk_operations import *
from colormath.color_objects import XYZColor
from imageio import imread, imwrite
from time import time

verbose = False
run_aruco = False

if run_aruco:
    dictionary4 = cv2.aruco.Dictionary_create(30, 4, 0)
    dictionary6 = cv2.aruco.Dictionary_create(30, 6, 0)


def process_frame(img,
                  CIEXYZ_white_point,
                  frame_name):

    # if rotate_image:
    #     img = imutils.rotate_bound(img, 315)

    processed_img = np.copy(img)

    # Adapt colors
    gain_coeff_L, \
    gain_coeff_M, \
    gain_coeff_S = compute_LMS_gain_coefficients_for_XYZ_white_point_to_D65(CIEXYZ_white_point)

    # Note that the gain coefficients are applied
    img_7_channels = dynamic_convert_img_to_7_channels(img,
                                                       sRGB_to_LMS_table,
                                                       LMS_to_master_table,
                                                       gain_coeff_L,
                                                       gain_coeff_M,
                                                       gain_coeff_S,
                                                       LMS_quantization_number)

    # Detection begins
    blobs_with_stats,\
    number_of_filtered_contours,\
    filtered_contours,\
    all_contours,\
    processed,\
    lab_chroma_img,\
    lab_L_img = find_blobs_with_stats(img_7_channels)

    adjacency_matrix = pairwise_blob_compatibility_table(blobs_with_stats, number_of_filtered_contours)
    no_low_degree_adjacency_matrix = repeatedly_remove_low_degree_vertices_from_adjacency_table(adjacency_matrix)
    no_isolated_adjacency_matrix, no_isolated_blobs_with_stats = remove_isolated_vertices(no_low_degree_adjacency_matrix, blobs_with_stats)

    reordered_adjacency_matrix, \
    reordered_blobs_with_stats, \
    color_class_sizes, \
    color_class_starting_indices, \
    = reorder_rows_and_columns_to_RMYGB(no_isolated_adjacency_matrix, no_isolated_blobs_with_stats)


    # WE HAVE AN ADJACENCY MATRIX, WE ALLOWED R-R AND M-M EDGES.
    # WE ALSO HAVE THE STATS ARRAY, BLOBS_WITH_STATS.
    # WE NEED TO FIND ALL 4-CLIQUES THAT CONTAIN ALL DIFFERENT COLORS,
    # OR THOSE THAT CONTAIN R-R AND TWO OTHER COLORS, OR M-M AND 2 OTHER COLORS.
    # COLOR INFO IS 3RD ENTRY OF BLOBS_WITH_STATS ARRAY.

    five_clique_list = find_5_color_cliques_in_ordered_adjacency_graph(
        reordered_adjacency_matrix,
        reordered_blobs_with_stats,
        color_class_starting_indices)

    # if verbose:
    #     print("Number of 5 cliques before any filtering:")
    #     print(len(five_clique_list))

    five_clique_list = reorder_blobs_ccw_and_do_basic_checks(five_clique_list)

    # NEW
    # Rotate lists so that blob with smallest y coordinate (highest on screen)
    # is first. This way it is easier to read.
    five_clique_list = rotate_five_clique_list_so_top_is_first(five_clique_list)

    standard_clique_contours_list,\
    five_clique_list,\
    score_list\
        = filter_candidate_lists_based_on_scores(five_clique_list,
                                                 filtered_contours)

    # if verbose:
    #     print("Number of 5 cliques after removing low score cliques:")
    #     print(len(five_clique_list))

    standard_clique_contours_list, \
    five_clique_list, \
    score_list \
        = remove_overlapping_dominated_cliques(standard_clique_contours_list,
                                               five_clique_list,
                                               score_list)

    # if verbose:
    #     print("Number of 5 cliques after removing dominated cliques:")
    #     print(len(five_clique_list))

    # DRAWING
    # Draw cross made out of squares
    print()
    print("\033[31;1;4mIMAGE FILE PROCESSED:\033[0m " + frame_name)
    num = len(five_clique_list)
    print("There are (is) " + str(num) + " marker(s) detected in this image:")
    print()
    for marker in five_clique_list:
        marker_code = get_marker_code(marker)
        print("\033[36mMarker ID:\033[0m " + str(marker_code))
        print("Note that coordinates are for images that are resized to 1920 x 1080.")
        print("Detected patch center locations:")
        print("Row: " + str(marker[0][0]) + "  Column: " + str(marker[0][1]))
        print("Row: " + str(marker[1][0]) + "  Column: " + str(marker[1][1]))
        print("Row: " + str(marker[2][0]) + "  Column: " + str(marker[2][1]))
        print("Row: " + str(marker[3][0]) + "  Column: " + str(marker[3][1]))
        print("Row: " + str(marker[4][0]) + "  Column: " + str(marker[4][1]))
        print()

    if True:
        draw_squares(processed_img, squares, num)

    # Draw blobs in perspective
        draw_standard_5_clique_list(processed_img,
                                    five_clique_list,
                                    standard_clique_contours_list)

        nice_draw_5_clique_list(processed_img,
                                five_clique_list,
                                filtered_contours)

    marker_codes = get_marker_codes(five_clique_list)

    # Draw original detected blobs
    # draw_5_clique_list(processed_img,
    #                    five_clique_list,
    #                    filtered_contours)

    # Draw all detected blobs
    # draw_all_detected_blobs(processed_img,
    #                         blobs_with_stats,
    #                         filtered_contours)

    # Aruco
    if run_aruco:

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 6x6 Aruco
        res = cv2.aruco.detectMarkers(gray, dictionary6)
        if len(res[0]) > 0:
            cv2.aruco.drawDetectedMarkers(processed_img, res[0], res[1])

        A6_total = len(res[0])
        A6 = 0
        for k in range(len(res[0])):
            if (res[1][k][0] == 0) or (res[1][k][0] == 1):
                A6 = A6 + 1

        # 4x4 Aruco
        res = cv2.aruco.detectMarkers(gray, dictionary4)
        if len(res[0]) > 0:
            cv2.aruco.drawDetectedMarkers(processed_img, res[0], res[1])
        A4_total = len(res[0])
        A4 = 0
        for k in range(len(res[0])):
            if (res[1][k][0] == 0) or (res[1][k][0] == 1):
                A4 = A4 + 1

    RYMB_RGYM = 0
    RYGB_RYBG = 0
    for code in marker_codes:
        if code == "RYMB" or code == "RGYM":
            RYMB_RGYM = RYMB_RGYM + 1
        if code == "RYGB" or code == "RYBG":
            RYGB_RYBG = RYGB_RYBG + 1

    if not run_aruco:
        A6 = None
        A4 = None
        A6_total = None
        A4_total = None

    return (processed_img, (RYMB_RGYM, A6, A4, RYGB_RYBG, A6_total, A4_total, len(marker_codes)))


# We have a processing method for a frame.
# We have to load the relevant white point, run the method
# it on all frames, and save the images in the correct folders.

def process_and_save():

    # Get white point for this condition
    white_point = np.load(os.path.join(images_folder, 'white_point.npy'))
    white_point_xyz = XYZColor(white_point[0], white_point[1], white_point[2])

    print("Using white point:")
    print(white_point_xyz)


    #processed_frame_folder = os.path.join(images_folder, subdirectory, frame_folder + "_Processed")
    processed_frame_folder = os.path.join(images_folder, "processed_images")

    if not os.path.exists(processed_frame_folder):
        os.makedirs(processed_frame_folder)

    # read each frame, process it, then save it to
    frames = get_files_in_folder(images_folder)
    for img_file in frames:
        if os.path.splitext(img_file)[1] == file_extension:
            full_img_file_name = os.path.join(images_folder, img_file)
            img = imread(full_img_file_name)
            if resize_img:
                img = cv2.resize(img,
                                 (resize_width, resize_height),
                                 interpolation=cv2.INTER_LINEAR)

            # Processing
            processed_data = process_frame(img,
                                           white_point_xyz,
                                           full_img_file_name)
            processed_img = processed_data[0]
            detection_counts = processed_data[1]

            processed_file_name = os.path.splitext(img_file)[0] + file_extension
            imwrite(os.path.join(processed_frame_folder, processed_file_name), processed_img)

            # EXCEL RECORDING
            # row_label = str(img_file[0:4]).zfill(4)
            # detections_excel_row = [row_label,
            #                         str(detection_counts[0]),
            #                         str(detection_counts[1]),
            #                         str(detection_counts[2]),
            #                         str(detection_counts[3]),
            #                         str(detection_counts[4]),
            #                         str(detection_counts[5]),
            #                         str(detection_counts[6]),]
            # writer = csv.writer(csvfile)
            # writer.writerow(detections_excel_row)
            # EXCEL RECORDING OVER

process_and_save()