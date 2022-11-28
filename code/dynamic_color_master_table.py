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
from color_processing_methods import *
from parameters import color_tables_folder,\
    create_color_tables,\
    D65_XYZ,\
    L_min,\
    L_max,\
    M_min,\
    M_max,\
    S_max,\
    S_min,\
    LMS_quantization_number,\
    ipt_hue_defs
import os


def make_sRGB_to_LMS_table(color_tables_folder):
    sRGB_to_LMS_table = np.empty((256, 256, 256, 3), np.float)
    for r in range(256):
        for g in range(256):
            for b in range(256):

                # scale sRGB values to [0,1] interval
                srgb_color = sRGBColor(r / 255.0, g / 255.0, b / 255.0)

                # Convert sRGB_sample_white color to Bradford LMS space so that we can do chromatic adaptation.
                color_L, color_M, color_S = convert_sRGB_to_bradford_LMS(srgb_color)
                sRGB_to_LMS_table[r, g, b] = (color_L, color_M, color_S)
        print("Producing sRGB_to_LMS_table. Progress: " + str(r) + " / 255")
    np.save(os.path.join(color_tables_folder, "sRGB_to_LMS_table.npy"), sRGB_to_LMS_table)


# This is used when making the table
# n = LMS_quantization_number
def change_0_n_interval_to_LMS(L_n, M_n, S_n):

    L_step = (L_max - L_min) / np.float(LMS_quantization_number)
    M_step = (M_max - M_min) / np.float(LMS_quantization_number)
    S_step = (S_max - S_min) / np.float(LMS_quantization_number)

    scaled_L = L_step * L_n
    scaled_M = M_step * M_n
    scaled_S = S_step * S_n

    return(scaled_L, scaled_M, scaled_S)



# Table entries are labelled with quantized LMS values
def make_LMS_to_master_table(color_tables_folder,
                             do_hue_correction=False):
    m = LMS_quantization_number + 1
    LMS_to_master_table = np.empty((7, m, m, m), dtype=np.int)
    for L_i in range(m):
        for M_i in range(m):
            for S_i in range(m):
                L, M, S = change_0_n_interval_to_LMS(L_i, M_i, S_i)

                # Go back to XYZ
                xyz_color = convert_bradford_LMS_to_CIEXYZ(L, M, S)

                # Now we convert the xyz_color to:
                # CIELAB L
                # CIELAB a
                # CIELAB b
                # CIELAB chroma
                # IPT hue angle
                # IPT corrected hue code

                # CIELAB L
                lab_color = convert_color(xyz_color, LabColor)
                t_lab = lab_color.get_value_tuple()

                # CIELAB chroma
                Lab_a = t_lab[1]
                Lab_b = t_lab[2]

                chroma = np.sqrt(np.square(Lab_a) + np.square(Lab_b))
                chroma = np.int(np.round(chroma, 0))

                # RGB colors map to the ranges (numbers are determined using
                # colormath package):
                # L: 0-100
                # a: -86.1827 - 98.2343
                # b: -107.8602 - 94.4880
                # We convert these ranges to 0-255.

                # L:
                Lab_L = t_lab[0] / 100.0 * 255
                Lab_L = np.int(np.round(Lab_L, 0))

                # a:
                # Lab_a = (Lab_a + 86.1827) / 184.417 * 255.0
                # Lab_a = np.int(np.round(Lab_a, 0))

                # b:
                # Lab_b = (Lab_b + 107.8602) / 202.3482 * 255.0
                # Lab_b = np.int(np.round(Lab_b, 0))

                # IPT hue angle
                ipt_color = convert_color(xyz_color, IPTColor)
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
                # Since illuminant of xyz_color is
                # d65, we are good to convert to sRGB_sample_white
                srgb_color = convert_color(xyz_color, sRGBColor)
                t_srgb_color = srgb_color.get_value_tuple()
                r = np.int(np.round(np.clip(t_srgb_color[0] * 255, 0, 255), 0))
                g = np.int(np.round(np.clip(t_srgb_color[1] * 255, 0, 255), 0))
                b = np.int(np.round(np.clip(t_srgb_color[2] * 255, 0, 255), 0))

                # Set array values
                LMS_to_master_table[0][L_i, M_i, S_i] = Lab_L
                # LMS_to_master_table[1][L_i, M_i, S_i] = Lab_a
                # LMS_to_master_table[2][L_i, M_i, S_i] = Lab_b
                LMS_to_master_table[1][L_i, M_i, S_i] = chroma
                LMS_to_master_table[2][L_i, M_i, S_i] = ipt_hue_angle
                LMS_to_master_table[3][L_i, M_i, S_i] = ipt_hue_code
                LMS_to_master_table[4][L_i, M_i, S_i] = r
                LMS_to_master_table[5][L_i, M_i, S_i] = g
                LMS_to_master_table[6][L_i, M_i, S_i] = b

        print("Producing LMS_to_master_table. Progress: " + str(L_i) + " / " + str(LMS_quantization_number))
    # Save results
    np.save(os.path.join(color_tables_folder, "LMS_to_master_table.npy"), LMS_to_master_table)


# Input L, M and S are "normal" LMS cone responses.
# They are scaled to the range [0, LMS_quantization_number]
@nb.njit
def change_LMS_interval_to_0_n(L, M, S, LMS_quantization_number):

    L_step = nb.float32(LMS_quantization_number) / (L_max - L_min)
    M_step = nb.float32(LMS_quantization_number) / (M_max - M_min)
    S_step = nb.float32(LMS_quantization_number) / (S_max - S_min)

    rounded_L = np.round(L_step * L, 0)
    rounded_M = np.round(M_step * M, 0)
    rounded_S = np.round(S_step * S, 0)

    int_L = np.int(rounded_L)
    int_M = np.int(rounded_M)
    int_S = np.int(rounded_S)

    L_n = np.maximum(np.minimum(int_L, LMS_quantization_number), 0)
    M_n = np.maximum(np.minimum(int_M, LMS_quantization_number), 0)
    S_n = np.maximum(np.minimum(int_S, LMS_quantization_number), 0)

    return(L_n, M_n, S_n)


# This is the main color conversion method.
# It takes the image, converts it to the Bradford LMS
# cone response space. It applies the color adaptation:
# multiplies the L,M,S values by the gain coefficients.
# Then it converts the adapted LMS image into a variety
# f outputs: lab, chroma of lab, ipt hue angle map, and
# ipt hue code.
@nb.njit(parallel=True)
def dynamic_convert_img_to_7_channels(img,
                                      sRGB_to_LMS_table,
                                      LMS_to_master_table,
                                      gain_coeff_L,
                                      gain_coeff_M,
                                      gain_coeff_S,
                                      LMS_quantization_number):
    h, w = img.shape[:2]
    lab_L_img = np.empty((h, w), dtype=np.uint8)
    lab_chroma_img = np.empty((h, w), dtype=np.uint8)
    ipt_hue_angle_img = np.empty((h, w), dtype=np.int16)
    ipt_hue_code = np.empty((h, w), dtype=np.uint8)
    sRGB_img = np.empty((h, w, 3), dtype=np.uint8)

    for col in nb.prange(w):
        for row in nb.prange(h):

            # Get sRGB_sample_white color from image
            color_r, color_g, color_b = img[row, col]

            # Look up the LMS cone responses
            color_L, color_M, color_S = sRGB_to_LMS_table[color_r, color_g, color_b]

            # Apply adaptation
            adapted_color_L = gain_coeff_L * color_L
            adapted_color_M = gain_coeff_M * color_M
            adapted_color_S = gain_coeff_S * color_S

            # We have to convert so that we can use out lookup table
            adapted_L_n,\
            adapted_M_n,\
            adapted_S_n = change_LMS_interval_to_0_n(adapted_color_L,
                                                     adapted_color_M,
                                                     adapted_color_S,
                                                     LMS_quantization_number)

            t = LMS_to_master_table[:, adapted_L_n, adapted_M_n, adapted_S_n]

            lab_L_img[row, col] = t[0]
            lab_chroma_img[row, col] = t[1]
            ipt_hue_angle_img[row, col] = t[2]
            ipt_hue_code[row, col] = t[3]
            sRGB_img[row, col] = t[4:7]

    return(lab_L_img,
           lab_chroma_img,
           ipt_hue_angle_img,
           ipt_hue_code,
           sRGB_img)


@nb.njit(parallel=True)
def dynamic_get_adapted_sRGB_img(img,
                                 sRGB_to_LMS_table,
                                 LMS_to_master_table,
                                 gain_coeff_L,
                                 gain_coeff_M,
                                 gain_coeff_S,
                                 LMS_quantization_number):
    h, w = img.shape[:2]
    adapted_sRGB_img = np.empty((h, w, 3), dtype=np.uint8)
    for col in nb.prange(w):
        for row in nb.prange(h):

            # Get sRGB_sample_white color from image
            color_r, color_g, color_b = img[row, col]

            # Look up the LMS cone responses
            color_L, color_M, color_S = sRGB_to_LMS_table[color_r, color_g, color_b]

            # Apply adaptation
            adapted_color_L = gain_coeff_L * color_L
            adapted_color_M = gain_coeff_M * color_M
            adapted_color_S = gain_coeff_S * color_S

            # We have to convert so that we can use out lookup table
            adapted_L_n, \
            adapted_M_n, \
            adapted_S_n = change_LMS_interval_to_0_n(adapted_color_L,
                                                     adapted_color_M,
                                                     adapted_color_S,
                                                     LMS_quantization_number)

            t = LMS_to_master_table[:, adapted_L_n, adapted_M_n, adapted_S_n]

            adapted_sRGB_img[row, col, 0] = t[6]
            adapted_sRGB_img[row, col, 1] = t[7]
            adapted_sRGB_img[row, col, 2] = t[8]

    return(adapted_sRGB_img)


# Divides all components of XYZ_color by Y component, so
# that Y component is 1.
def normalize_CIEXYZ_color(XYZ_color):
    XYZ_color_t = XYZ_color.get_value_tuple()
    Y_value = XYZ_color_t[1]
    normalized_XYZ_color = XYZColor(XYZ_color_t[0] / Y_value,
                                    XYZ_color_t[1] / Y_value,
                                    XYZ_color_t[2] / Y_value,
                                    illuminant=XYZ_color.illuminant)
    return(normalized_XYZ_color)


def convert_sRGB_to_normalized_CIEXYZ(sRGB_color):
    r_color = sRGB_color[0]
    g_color = sRGB_color[1]
    b_color = sRGB_color[2]
    sRGB_cm = sRGBColor(r_color / 255.0,
                        g_color / 255.0,
                        b_color / 255.0)
    # The native illuminant if sRGB is D65, so the illuminant for
    # the converted XYZ_color is also D65.
    XYZ_color = convert_color(sRGB_cm, XYZColor)
    normalized_XYZ_color = normalize_CIEXYZ_color(XYZ_color)

    return(normalized_XYZ_color)


def compute_LMS_gain_coefficients_for_sRGB_white_point_to_D65(sRGB_white_point):
    normalized_source_white_XYZ = convert_sRGB_to_normalized_CIEXYZ(sRGB_white_point)
    gain_coeff_L, \
    gain_coeff_M, \
    gain_coeff_S = compute_LMS_gain_coefficients_CIEXYZ_whites(normalized_source_white_XYZ,
                                                               D65_XYZ)
    return(gain_coeff_L, gain_coeff_M, gain_coeff_S)


def compute_LMS_gain_coefficients_for_XYZ_white_point_to_D65(CIEXYZ_white_point):
    normalized_source_white_XYZ = normalize_CIEXYZ_color(CIEXYZ_white_point)
    gain_coeff_L, \
    gain_coeff_M, \
    gain_coeff_S = compute_LMS_gain_coefficients_CIEXYZ_whites(normalized_source_white_XYZ,
                                                               D65_XYZ)
    return(gain_coeff_L, gain_coeff_M, gain_coeff_S)


#######################
### TABLE TEST CODE ###
#######################

# USED BY COLOR MARKER
if create_color_tables:
    print("Creating the color tables might take a few hours, please wait.")
    make_sRGB_to_LMS_table(color_tables_folder)
    make_LMS_to_master_table(color_tables_folder)

# sRGB_to_lms = np.load(color_tables_folder + "sRGB_to_LMS_table.npy")
# lms_to_master = np.load(color_tables_folder + "LMS_to_master_table.npy")
#
# L, M, S = sRGB_to_lms[10,100,5]
# print(L, M, S)
#
# L_n, M_n, S_n = change_LMS_interval_to_0_n(L, M, S)
# print(L_n, M_n, S_n)
#
# r,g,b = lms_to_master[6:9,L_n,M_n,S_n]
# print(r,g,b)