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


from color_processing_methods import compute_adaptive_high_chroma_mask
import numpy as np
import numba as nb
from video_loop import *
#from parameters import color_master_table
from parameters import ipt_hue_defs as ihd
from parameters import sampling_box_size


sampling_box_size = sampling_box_size
max_intensity_difference = 90

max_chroma_difference_red = 19
max_chroma_difference_yellow = 21
max_chroma_difference_green = 19
max_chroma_difference_blue = 19
max_chroma_difference_magenta = 19

hue_variance_constant = 1 / 9.0
blue_hue_variance_constant = 1 / 9.0
yellow_hue_variance_constant = 1 / 9.0
magenta_hue_variance_constant = 1 / 9.0
green_hue_variance_constant = 1 / 7.0
#hue_span_constant = 1/3.0
red_range = ihd['red'][1] - ihd['red'][0]
r_cw = ihd['red'][0]
r_acw = ihd['red'][1]
red_var_thr = (red_range * hue_variance_constant)**2
#red_max_span = red_range * hue_span_constant

yellow_range = ihd['yellow'][1] - ihd['yellow'][0]
y_cw = ihd['yellow'][0]
y_acw = ihd['yellow'][1]
yellow_var_thr = (yellow_range * yellow_hue_variance_constant)**2
#yellow_max_span = yellow_range * hue_span_constant

green_range = ihd['green'][1] - ihd['green'][0]
g_cw = ihd['green'][0]
g_acw = ihd['green'][1]
green_var_thr = (green_range * green_hue_variance_constant)**2
#green_max_span = green_range * hue_span_constant

blue_range = ihd['blue'][1] - ihd['blue'][0]
b_cw = ihd['blue'][0]
b_acw = ihd['blue'][1]
blue_var_thr = (blue_range * blue_hue_variance_constant)**2
#blue_max_span = blue_range * hue_span_constant

magenta_range = 360 - ihd['magenta'][0] + ihd['magenta'][1]
m_cw = ihd['magenta'][0]
m_acw = ihd['magenta'][1]
magenta_var_thr = (magenta_range * magenta_hue_variance_constant)**2
#magenta_max_span = magenta_range * hue_span_constant

@nb.njit(parallel=True)
def compute_sliding_window_uniformity(hue_angle_map,
                                      chroma_map,
                                      max_chroma_difference_red,
                                      max_chroma_difference_yellow,
                                      max_chroma_difference_green,
                                      max_chroma_difference_blue,
                                      max_chroma_difference_magenta,
                                      intensity_map,
                                      max_intensity_difference,
                                      hue_code_map,
                                      adaptive_high_chroma_mask,
                                      sampling_box_size=sampling_box_size):
    h, w = hue_angle_map.shape
    h_sw = h - (sampling_box_size - 1)
    w_sw = w - (sampling_box_size - 1)
    s2 = sampling_box_size**2

    uniformity_map = np.zeros((h_sw, w_sw), dtype=nb.boolean)
    hue_angle_sw = np.empty((h_sw, w_sw, s2), dtype=nb.float32)
    #active_hue_sw = np.empty((h_sw, w_sw, sampling_box_size**2), dtype=nb.float32)
    chroma_sw = np.empty((h_sw, w_sw, s2), dtype=nb.float32)
    high_chroma_sw = np.empty((h_sw, w_sw, s2), dtype=nb.boolean)
    intensity_sw = np.empty((h_sw, w_sw, s2), dtype=nb.float32)
    hue_code_sw = np.empty((h_sw, w_sw, s2), dtype=nb.uint8)

    sliding_window_colors = np.zeros((h_sw, w_sw, 3), dtype=nb.uint8)
    sliding_window_color_codes = np.zeros((h_sw, w_sw), dtype=nb.uint8)

    # For each sliding window location
    for y in nb.prange(h_sw):
        for x in nb.prange(w_sw):

            # LOCAL COPY OF SLIDING WINDOW
            # For each point in the sliding window
            for y_sw in range(sampling_box_size):
                for x_sw in range(sampling_box_size):
                    hue_angle_sw[y, x][y_sw * sampling_box_size + x_sw] = hue_angle_map[y_sw + y, x_sw + x]
                    chroma_sw[y, x][y_sw * sampling_box_size + x_sw] = chroma_map[y_sw + y, x_sw + x]
                    high_chroma_sw[y, x][y_sw * sampling_box_size + x_sw] = adaptive_high_chroma_mask[y_sw + y, x_sw + x]
                    intensity_sw[y, x][y_sw * sampling_box_size + x_sw] = intensity_map[y_sw + y, x_sw + x]
                    hue_code_sw[y, x][y_sw * sampling_box_size + x_sw] = hue_code_map[y_sw + y, x_sw + x]

            # HUE VARIANCE
            # Check if the arc that contains 0 is larger than 180 degrees
            sw_0_min = np.min(hue_angle_sw[y, x])
            sw_0_max = np.max(hue_angle_sw[y, x])
            # If condition below is false,
            # then arc containing 180 degree point
            # and all color points is 180 degrees or more.
            side_of_0_large = (sw_0_min + (360 - sw_0_max)) > 180
            if side_of_0_large:
                active_hue_sw = hue_angle_sw[y, x]
                #active_sw_min = sw_0_min
                #active_sw_max = sw_0_max
                mean_angle = np.mean(active_hue_sw)
                sw_variance = np.var(active_hue_sw)
            else:
                rotated_hue_sw = np.mod((hue_angle_sw[y, x] + 180), 360)
                sw_180_min = np.min(rotated_hue_sw)
                sw_180_max = np.max(rotated_hue_sw)
                side_of_180_large = (sw_180_min + (360 - sw_180_max)) > 180
                if side_of_180_large:
                    active_hue_sw = rotated_hue_sw
                    #active_sw_min = sw_180_min
                    #active_sw_max = sw_180_max
                    mean_angle = np.mod((np.mean(rotated_hue_sw) + 180), 360)
                    sw_variance = np.var(active_hue_sw)


            # VARIANCE threshold depends on average hue
            if r_cw <= mean_angle and mean_angle < r_acw:
                variance_thr = red_var_thr
                sliding_window_colors[y, x] = (255, 0, 0)
                sliding_window_color_codes[y, x] = 1
            elif y_cw <= mean_angle and mean_angle < y_acw:
                variance_thr = yellow_var_thr
                sliding_window_colors[y, x] = (255, 255, 0)
                sliding_window_color_codes[y, x] = 2
            elif g_cw <= mean_angle and mean_angle < g_acw:
                variance_thr = green_var_thr
                sliding_window_colors[y, x] = (0, 255, 0)
                sliding_window_color_codes[y, x] = 3
            elif b_cw <= mean_angle and mean_angle < b_acw:
                variance_thr = blue_var_thr
                sliding_window_colors[y, x] = (0, 0, 255)
                sliding_window_color_codes[y, x] = 4
            elif (m_cw <= mean_angle) or (mean_angle < m_acw):
                variance_thr = magenta_var_thr
                sliding_window_colors[y, x] = (255, 0, 255)
                sliding_window_color_codes[y, x] = 5

            # CHROMA
            # 1. LARGE CHROMA DIFFERENCE IS DEFINED AS NOT UNIFORM. THRESHOLD IS SEPARATE FOR EACH COLOR.
            # 2. IF MAX CHROMA IS SMALL, WE IGNORE THAT WINDOW BY SAYING IT IS NOT
            # UNIFORM
            chroma_uniform = False
            if sliding_window_color_codes[y, x] == 1:
                if (np.max(chroma_sw[y, x]) - np.min(chroma_sw[y,x])) <= max_chroma_difference_red:
                    chroma_uniform = True
            if sliding_window_color_codes[y, x] == 2:
                if (np.max(chroma_sw[y, x]) - np.min(chroma_sw[y,x])) <= max_chroma_difference_yellow:
                    chroma_uniform = True
            if sliding_window_color_codes[y, x] == 3:
                if (np.max(chroma_sw[y, x]) - np.min(chroma_sw[y,x])) <= max_chroma_difference_green:
                    chroma_uniform = True
            if sliding_window_color_codes[y, x] == 4:
                if (np.max(chroma_sw[y, x]) - np.min(chroma_sw[y,x])) <= max_chroma_difference_blue:
                    chroma_uniform = True
            if sliding_window_color_codes[y, x] == 5:
                if (np.max(chroma_sw[y, x]) - np.min(chroma_sw[y,x])) <= max_chroma_difference_magenta:
                    chroma_uniform = True

            # HIGH ENOUGH CHROMA IN SLIDING WINDOW
            high_chroma_present_sw = np.any(high_chroma_sw[y,x])

            # INTENSITY
            intensity_uniform = False
            if np.max(intensity_sw[y,x]) - np.min(intensity_sw[y,x]) <= max_intensity_difference:
                intensity_uniform = True

            # HUE VARIANCE
            hue_variance_low = False
            if (sw_variance <= variance_thr):
                hue_variance_low = True

            # UNIQUE HUES
            few_hues = False
            r_present = 0
            y_present = 0
            g_present = 0
            b_present = 0
            m_present = 0
            for sw_index in range(s2):
                pixel_hue_code = hue_code_sw[y,x][sw_index]
                if pixel_hue_code == 1:
                    r_present = 1
                if pixel_hue_code == 2:
                    y_present = 1
                if pixel_hue_code == 3:
                    g_present = 1
                if pixel_hue_code == 4:
                    b_present = 1
                if pixel_hue_code == 5:
                    m_present = 1
            hue_presence_sum = r_present + y_present + g_present + b_present + m_present
            if (hue_presence_sum == 1) or ((hue_presence_sum == 2) and (r_present == 1) and (m_present == 1)):
                few_hues = True

            uniformity_map[y, x] = chroma_uniform and \
                                   intensity_uniform and \
                                   hue_variance_low and \
                                   few_hues and \
                                   high_chroma_present_sw

            # uniformity_map[y, x] = chroma_uniform and \
            #                        intensity_uniform and \
            #                        hue_variance_low and \
            #                        high_chroma_present_sw

    return(uniformity_map, sliding_window_colors, sliding_window_color_codes)


@nb.njit
def expand_and_colorize_sliding_window_markers(sliding_window_markers, sliding_window_colors):
    h, w =sliding_window_markers.shape
    h_e = h + sampling_box_size - 1
    w_e = w + sampling_box_size - 1
    expanded_colorized_mask = np.zeros((h_e, w_e), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            if sliding_window_markers[row, col] == True:
                for row_sw in range(sampling_box_size-1):
                    for col_sw in range(sampling_box_size-1):
                        expanded_colorized_mask[row + row_sw, col + col_sw] = sliding_window_colors[row, col]
    return(expanded_colorized_mask)


@nb.njit
def compute_hue_blob_masks(processed, sliding_window_color_codes):
    expanded_colorized_mask = expand_and_colorize_sliding_window_markers(processed, sliding_window_color_codes)


    h, w = expanded_colorized_mask.shape
    R = np.zeros((h, w), dtype=nb.boolean)
    Y = np.zeros((h, w), dtype=nb.boolean)
    G = np.zeros((h, w), dtype=nb.boolean)
    B = np.zeros((h, w), dtype=nb.boolean)
    M = np.zeros((h, w), dtype=nb.boolean)

    for row in range(h):
        for col in range(w):
            if expanded_colorized_mask[row, col] == 1:
                R[row, col] = True
            if expanded_colorized_mask[row, col] == 2:
                Y[row, col] = True
            if expanded_colorized_mask[row, col] == 3:
                G[row, col] = True
            if expanded_colorized_mask[row, col] == 4:
                B[row, col] = True
            if expanded_colorized_mask[row, col] == 5:
                M[row, col] = True

    return(R, Y, G, B, M)