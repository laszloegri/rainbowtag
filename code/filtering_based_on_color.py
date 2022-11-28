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
from numba.typed import List
import numpy as np
from parameters import ipt_hue_defs
from geometry import *


red_cw = ipt_hue_defs['red'][0]
#red_ccw = ipt_hue_defs['red'][1]
#magenta_cw = ipt_hue_defs['magenta'][0]
magenta_ccw = ipt_hue_defs['magenta'][1]


# Chroma sampling grid
# abstract_chroma_sampling_points = scaled_rotated_grid(5, 45, 70 / (4 * np.sqrt(2)), 50)
# s_chroma = abstract_chroma_sampling_points.shape
# abstract_chroma_sampling_points = np.reshape(abstract_chroma_sampling_points,
#                                              (s_chroma[0],
#                                               1,
#                                               s_chroma[1]))
#
# # Brightness sampling
# abstract_brightness_sampling_points = scaled_rotated_grid(11, 45, 120 / (10 * np.sqrt(2)), 50)
# s_brightness = abstract_brightness_sampling_points.shape
# abstract_brightness_sampling_points = np.reshape(abstract_brightness_sampling_points,
#                                                  (s_brightness[0],
#                                                   1,
#                                                   s_brightness[1]))


@nb.jit(cache=True)
def correct_red_magenta_color_codes_of_clique(five_clique,
                                              red_cw = red_cw,
                                              magenta_ccw = magenta_ccw):

    # First we find the indices of the various colors blobs.
    first_rm_found = False
    for i in range(5):
        if (five_clique[i][3] == 1) or (five_clique[i][3] == 5):
            if not first_rm_found:
                first_rm_found = True
                rm_1_index = i
            else:
                rm_2_index = i
        if (five_clique[i][3] == 2):
            yellow_index = i
        # if (five_clique[i][3] == 3):
        #     green_index = i
        if (five_clique[i][3] == 4):
            blue_index = i

    # YELLOW CHECK - HARMFUL IN SOME CIRCUMSTANCES
    # if five_clique[blue_index][5] >= five_clique[yellow_index][5] + 5:
    #     # print("Yellow has low chroma compared to blue.")
    #     # print(five_clique)
    #     return(None)

    # BLOB 1 AND BLOB 2 HUE ANGLES ARE EXACTLY THE SAME, NO GOOD
    if ((five_clique[rm_1_index][4] - 180) % 360) == ((five_clique[rm_2_index][4] - 180) % 360):
        return(None)

    # Otherwise make blob_1 the counterclockwise direction (red direction)
    if ((five_clique[rm_1_index][4] - 180) % 360) < ((five_clique[rm_2_index][4] - 180) % 360):
        temp = rm_1_index
        rm_1_index = rm_2_index
        rm_2_index = temp

    # If colors are at least 5 degrees into the red and magenta range, we return it
    amount_of_red_in_safe_range = np.mod(five_clique[rm_1_index][4] - 180, 360) - \
                                  np.mod(red_cw - 180, 360)
    amount_of_magenta_in_safe_range = np.mod(magenta_ccw - 180, 360) -\
                                      np.mod(five_clique[rm_2_index][4] - 180, 360)

    # IF RED ANF MAGENTA ARE IN THEIR RANGES
    # If red and magenta are sufficiently into their ranges, it's good.
    red_and_magenta_well_in_their_intervals = \
        (amount_of_red_in_safe_range >= 5) and (amount_of_magenta_in_safe_range >= 5)
    if red_and_magenta_well_in_their_intervals:
        return(five_clique)

    # If red and magenta are in their ranges and they have at least 10 degrees difference, it's good.
    red_and_magenta_barely_in_their_intervals_but_far = \
        (amount_of_red_in_safe_range >= 1) and \
        (amount_of_magenta_in_safe_range >= 1) and \
        ((((five_clique[rm_1_index][4] - 180) % 360) - ((five_clique[rm_2_index][4] - 180) % 360)) >= 10)
    if red_and_magenta_barely_in_their_intervals_but_far:
        return(five_clique)


    # If colors are in the same interval, but at least 15 degrees
    # from each other, accept it, and adjust the color codes in the five_clique.
    same_interval_but_far = np.abs(((five_clique[rm_1_index][4] - 180) % 360) - \
                                   ((five_clique[rm_2_index][4] - 180) % 360)) >= 15
    if same_interval_but_far:
        five_clique[rm_1_index][3] = 1
        five_clique[rm_2_index][3] = 5
        #print("RED AND MAGENTA ARE AT LEAST 15 DEGREES AWAY BUT NOT IN THEIR RANGES.")
        #print(five_clique)
        return(five_clique)

    # Otherwise we require at least 10 degrees of difference and a larger chroma difference
    # of 10. We adjust rm_1_index to point at the blob with a more red hue (counter clockqise)
    same_interval_medium_distance_large_chroma_diff = \
        ((five_clique[rm_1_index][4] - 180) % 360) - ((five_clique[rm_2_index][4] - 180) % 360) >= 10 \
        and\
        (five_clique[rm_2_index][5] + 15 <= five_clique[rm_1_index][5])
    if same_interval_medium_distance_large_chroma_diff:
        five_clique[rm_1_index][3] = 1
        five_clique[rm_2_index][3] = 5
        #print("RED-MAGENTA AT LEAST 10 DEGREE DIFFERENCE AND AT LEAST 15 CHROMA DIFFERENCE.")
        #print(five_clique)
        return (five_clique)

    # If all conditions fail, we reject
    return(None)