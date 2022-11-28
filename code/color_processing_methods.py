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
from colormath.color_conversions import *
from colormath.color_objects import *

bradford_matrix = np.array([[0.8951000, 0.2664000, -0.1614000],
                            [-0.7502000, 1.7135000, 0.0367000],
                            [0.0389000, -0.0685000, 1.0296000]], dtype=np.float)

inverse_bradford_matrix = np.array([[0.9869929, -0.1470543, 0.1599627],
                                    [0.4323053, 0.5183603, 0.0492912],
                                    [-0.0085287, 0.0400428, 0.9684867]], dtype=np.float)

# Helper methods
def convert_CIEXYZ_to_bradford_LMS(xyz_color):
    t_xyz = xyz_color.get_value_tuple()
    xyz_column_vector = np.array([[t_xyz[0]],
                                  [t_xyz[1]],
                                  [t_xyz[2]]])
    LMS = np.matmul(bradford_matrix, xyz_column_vector)
    return(np.array([LMS[0][0],
                     LMS[1][0],
                     LMS[2][0]]))


def convert_bradford_LMS_to_CIEXYZ(L, M, S):
    LMS_col_vec = np.array([[L],
                            [M],
                            [S]])
    xyz_column_vector = np.matmul(inverse_bradford_matrix, LMS_col_vec)

    # Illuminant needs to be specified because colormath uses d50 by default
    # when we instantiate an XYZColor object.
    xyz_color = XYZColor(xyz_column_vector[0][0],
                         xyz_column_vector[1][0],
                         xyz_column_vector[2][0], illuminant='d65')
    return(xyz_color)


def convert_sRGB_to_bradford_LMS(sRGB_color):
    # Colormath assumes D65 since source color space sRGB_sample_white has D65 white point.
    # The point is that the white points are not changed when going from sRGB_sample_white to XYZ.
    xyz_color = convert_color(sRGB_color, XYZColor)
    L, M, S = convert_CIEXYZ_to_bradford_LMS(xyz_color)
    return(np.array([L, M, S]))


def compute_LMS_gain_coefficients_CIEXYZ_whites(source_white_CIEXYZ, target_white_CIEXYZ):

    # Compute multiplicative gain in cone space
    source_L, source_M, source_S = convert_CIEXYZ_to_bradford_LMS(source_white_CIEXYZ)
    target_L, target_M, target_S = convert_CIEXYZ_to_bradford_LMS(target_white_CIEXYZ)

    # Let's compute change
    gain_coeff_l = target_L / source_L
    gain_coeff_m = target_M / source_M
    gain_coeff_s = target_S / source_S

    return(gain_coeff_l, gain_coeff_m, gain_coeff_s)


# This method takes as input a set of CIEXYZ COLORS. It converts
# each into LMS space. Takes the average of the LMS coordinates.
# It converts the resulting average LMS color to CIEXYZ and
# returns it.
def find_average_of_CIEXYZ_colors(CIEXYZ_colors):
    illuminant = CIEXYZ_colors[0].illuminant
    t = numpy.array([0,0,0], float)
    n = len(CIEXYZ_colors)
    for color in CIEXYZ_colors:
        t = t + color.get_value_tuple()
    average_CIEXYZ_color = t / n
    average_CIEXYZ_color_cm = XYZColor(average_CIEXYZ_color[0],
                                       average_CIEXYZ_color[1],
                                       average_CIEXYZ_color[2],
                                       illuminant=illuminant)
    return(average_CIEXYZ_color_cm)


# Magenta to red correction, just a linear separator
def linear_hue_correction(hue_code, cx, cy):
    corrected_hue_code = hue_code
    if hue_code == 5:
        if cy > 0.2143 * cx + 0.09265:
            corrected_hue_code = 1
    return (corrected_hue_code)


def compute_ipt_hue_code(ipt_hue_defs,
                         hue_angle,
                         ipt_p,
                         ipt_t,
                         do_hue_red_magenta_hue_correction=False):

    # Computes a hue code based on the inputs.
    # Hue codes:
    # code 1: red
    # code 2: yellow
    # code 3: green
    # code 4: blue
    # code 5: magenta

    ipt_hue_code = None
    # RED
    # Check if interval crosses 0
    if ipt_hue_defs['red'][0] <= ipt_hue_defs['red'][1]:
        if (ipt_hue_defs['red'][0] <= hue_angle) and (hue_angle < ipt_hue_defs['red'][1]):
            ipt_hue_code = 1
    else:
        # Otherwise we break down the interval to before and after 0.
        if ((ipt_hue_defs['red'][0] <= hue_angle) and ipt_hue_defs['red'][0] < 360) or \
                ((0 <= hue_angle) and (hue_angle < ipt_hue_defs['red'][1])):
            ipt_hue_code = 1

    # YELLOW
    if (ipt_hue_defs['yellow'][0] <= hue_angle) and (hue_angle < ipt_hue_defs['yellow'][1]):
        ipt_hue_code = 2

    # GREEN
    if (ipt_hue_defs['green'][0] <= hue_angle) and (hue_angle < ipt_hue_defs['green'][1]):
        ipt_hue_code = 3

    # BLUE
    if (ipt_hue_defs['blue'][0] <= hue_angle) and (hue_angle < ipt_hue_defs['blue'][1]):
        ipt_hue_code = 4

    # MAGENTA (same as red)
    if ipt_hue_defs['magenta'][0] <= ipt_hue_defs['magenta'][1]:
        if (ipt_hue_defs['magenta'][0] <= hue_angle) and (hue_angle < ipt_hue_defs['magenta'][1]):
            ipt_hue_code = 5
    else:
        if ((ipt_hue_defs['magenta'][0] <= hue_angle) and (hue_angle < 360)) or \
            ((0 <= hue_angle) and (hue_angle <= ipt_hue_defs['magenta'][1])):
            ipt_hue_code = 5

    # HUE CORRECTION
    if do_hue_red_magenta_hue_correction:
        ipt_hue_code = linear_hue_correction(ipt_hue_code, ipt_p, ipt_t)

    return(ipt_hue_code)



# This method produces and save the following numpy array.
# Input is 256 x 256 x 256, one input for each sRGB_sample_white color.
# The output has 4 channels:
# 0: CIELAB L component
# 1: CIELAB chroma
# 2: IPT hue angle
# 3: IPT hue code (0:red, 1:yellow, 2:green, 3:blue, 4:magenta) with correction for red and magenta.
# The table is computed such that the white point CIEXYZ_white_point given as input
# is mapped to the D65 white point, using Bradford color adaptation.


def make_color_master_table(source_white_XYZ,
                            ipt_hue_defs,
                            color_tables_folder,
                            file_name,
                            do_hue_correction=False):

    # Let's figure out Bradford gain coefficients to go to D65 white.
    gain_coeff_L,\
    gain_coeff_M,\
    gain_coeff_S = compute_LMS_gain_coefficients_CIEXYZ_whites(source_white_XYZ, D65_XYZ)

    color_master_table = np.empty((9, 256, 256, 256), dtype=np.int)
    for color_r in range(256):
        for color_g in range(256):
            for color_ in range(256):

                # Specify sRGB_sample_white color
                srgb_color = sRGBColor(color_r / 255.0, color_g / 255.0, color_ / 255.0)

                # Convert sRGB_sample_white color to Bradford LMS space so that we can do chromatic adaptation.
                color_L, color_M, color_S = convert_sRGB_to_bradford_LMS(srgb_color)

                # Apply gain
                adapted_color_L = gain_coeff_L * color_L
                adapted_color_M = gain_coeff_M * color_M
                adapted_color_S = gain_coeff_S * color_S

                # Go back to XYZ
                adapted_xyz_color = convert_bradford_LMS_to_CIEXYZ(adapted_color_L,
                                                                   adapted_color_M,
                                                                   adapted_color_S)

                # Now we convert the adapted color to:
                # CIELAB L
                # CIELAB chroma
                # IPT hue angle
                # IPT corrected hue code

                # CIELAB L
                lab_color = convert_color(adapted_xyz_color, LabColor)
                t_lab = lab_color.get_value_tuple()
                Lab_L = t_lab[0] / 100.0 * 255
                Lab_L = np.int(np.round(Lab_L, 0))

                # CIELAB chroma
                Lab_a = t_lab[1]
                Lab_b = t_lab[2]

                chroma = np.sqrt(np.square(Lab_a) + np.square(Lab_b))
                chroma = np.int(np.round(chroma, 0))

                # SCALE IT SINCE WE SCALED IT BEFORE. MAKE IT CONSISTENT???
                Lab_a = np.round(Lab_a / 100.0 * 127 + 128, 0)
                Lab_b = np.round(Lab_b / 100.0 * 127 + 128, 0)

                # IPT hue angle
                ipt_color = convert_color(adapted_xyz_color, IPTColor)
                t_ipt = ipt_color.get_value_tuple()
                ipt_p_value = t_ipt[1]
                ipt_t_value = t_ipt[2]
                ipt_angle_rad = np.arctan2(ipt_t_value, ipt_p_value)
                ipt_hue_angle = np.rad2deg(ipt_angle_rad)
                ipt_hue_angle = np.int(np.mod(np.round(ipt_hue_angle, 0), 360))

                # Find IPT hue code, apply linear correction if switch is on
                ipt_hue_code = compute_ipt_hue_code(ipt_hue_defs,
                                                    ipt_hue_angle,
                                                    ipt_p_value,
                                                    ipt_t_value,
                                                    do_hue_red_magenta_hue_correction=do_hue_correction)

                # Compute adapted sRGB_sample_white values
                # Since illuminant of adapted_xyz_color is
                # d65, we are good to convert to sRGB_sample_white
                adapted_srgb_color = convert_color(adapted_xyz_color, sRGBColor)
                t_adapted_srgb_color = adapted_srgb_color.get_value_tuple()
                adapted_r = np.int(np.round(np.clip(t_adapted_srgb_color[0] * 255, 0, 255), 0))
                adapted_g = np.int(np.round(np.clip(t_adapted_srgb_color[1] * 255, 0, 255), 0))
                adapted_b = np.int(np.round(np.clip(t_adapted_srgb_color[2] * 255, 0, 255), 0))

                # Set array values
                color_master_table[0][color_r, color_g, color_] = Lab_L
                color_master_table[1][color_r, color_g, color_] = Lab_a
                color_master_table[2][color_r, color_g, color_] = Lab_b
                color_master_table[3][color_r, color_g, color_] = chroma
                color_master_table[4][color_r, color_g, color_] = ipt_hue_angle
                color_master_table[5][color_r, color_g, color_] = ipt_hue_code
                color_master_table[6][color_r, color_g, color_] = adapted_r
                color_master_table[7][color_r, color_g, color_] = adapted_g
                color_master_table[8][color_r, color_g, color_] = adapted_b

        print("Producing master color table. Progress: " + str(color_r) + " / 255")
    # Save results
    np.save(color_tables_folder + file_name + ".npy", color_master_table)


@nb.njit(parallel=True)
def convert_img_to_6_channels(img, color_master_table):
    h, w = img.shape[:2]
    lab_img = np.empty((h, w, 3), dtype=np.uint8)
    lab_chroma_img = np.empty((h, w), dtype=np.uint8)
    ipt_hue_angle_img = np.empty((h, w), dtype=np.int16)
    ipt_hue_code = np.empty((h, w), dtype=np.uint8)
    #high_chroma_map = np.zeros((h, w), dtype=nb.boolean)
    for col in nb.prange(w):
        for row in nb.prange(h):
            x, y, z = img[row, col]
            t = color_master_table[:, x, y, z]
            lab_img[row, col, 0] = t[0]
            lab_img[row, col, 1] = t[1]
            lab_img[row, col, 2] = t[2]
            lab_chroma_img[row, col] = t[3]
            ipt_hue_angle_img[row, col] = t[4]
            ipt_hue_code[row, col] = t[5]

    return(lab_img, lab_chroma_img, ipt_hue_angle_img, ipt_hue_code)


@nb.njit(parallel=True)
def compute_adaptive_high_chroma_mask(lab_intensity_table,
                                      lab_chroma_table,
                                      ipt_hue_code_table):
    h, w = lab_intensity_table.shape
    mask = np.zeros((h, w), dtype=nb.boolean)
    for col in nb.prange(w):
        for row in nb.prange(h):
            L = lab_intensity_table[row, col]
            chroma = lab_chroma_table[row, col]
            hue_code = ipt_hue_code_table[row, col]

            # Everything under 10 is considered black.
            if L >= 10:

                # We set different threshold per color

                # FOR DARKER BLUE IN DARK, VERY SENSITIVE
                # if (hue_code == 4):
                #     if L <= 50:
                #         chroma_thr = 10
                #     else:
                #         chroma_thr = 0.05 * L + 7.5

                # For a bit brighter blue.
                if (hue_code == 4):
                    if L <= 50:
                        chroma_thr = 12
                    else:
                        chroma_thr = 0.07 * L + 7.5

                # Red
                if (hue_code == 1):
                    if L <= 50:
                        chroma_thr = 20
                    else:
                        chroma_thr = 0.35 * L + 2.5
                # if (hue_code == 1):
                #     if L <= 50:
                #         chroma_thr = 12
                #     else:
                #         chroma_thr = 0.2 * L + 2.0

                # Green
                if (hue_code == 3):
                    if L <= 50:
                        chroma_thr = 7
                    else:
                        chroma_thr = 0.04 * L + 5.0

                # Yellow
                if (hue_code == 2):
                    if L <= 50:
                        chroma_thr = 12
                    else:
                        chroma_thr = 0.2 * L + 2.0
                # if (hue_code == 2):
                #     if L <= 50:
                #         chroma_thr = 12
                #     else:
                #         chroma_thr = 0.1 * L + 7.0

                # Magenta
                if (hue_code == 5):
                    if L <= 50:
                        chroma_thr = 15
                    else:
                        chroma_thr = 0.05 * L + 12.5

                # Threshold it
                if chroma >= chroma_thr:
                    mask[row, col] = True
    return(mask)


@nb.njit(parallel=True)
def get_adapted_sRGB_img(img, color_master_table):
    h, w = img.shape[:2]
    adapted_sRGB_img = np.empty((h, w, 3), dtype=np.uint8)
    for col in nb.prange(w):
        for row in nb.prange(h):
            color_r, color_g, color_b = img[row, col]
            adapted_sRGB_img[row, col, 0] = color_master_table[6, color_r, color_g, color_b]
            adapted_sRGB_img[row, col, 1] = color_master_table[7, color_r, color_g, color_b]
            adapted_sRGB_img[row, col, 2] = color_master_table[8, color_r, color_g, color_b]
    return(adapted_sRGB_img)