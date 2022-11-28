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


from colormath.color_objects import XYZColor
import numpy as np
import os


###################################################################################
### Adjust parameters here as explained in CODE_SETUP file ########################
###################################################################################

images_folder = os.path.join("..", "images")

file_extension = ".jpg"
#file_extension = ".bmp"
create_color_tables = True

color_tables_folder = os.path.join("..", "color_tables")

#LMS_quantization_number = 255
LMS_quantization_number = 511

###################################################################################
###################################################################################
###################################################################################


if not os.path.exists(color_tables_folder):
    os.makedirs(color_tables_folder)

factor = 1.0
camera_w = 1920
camera_h = 1080

resize_img = True
resize_height = int(camera_h * factor)
resize_width = int(camera_w * factor)
sampling_box_size = 4

rotate_image = False
add_black_borders = False
black_border_size = 151

#fixed_XYZ_white_point = XYZColor(.9423, 1.0000, 1.0302)
given_fixed_white_point = False

max_area_difference_factor = 10
max_relative_distance = 6
max_blob_side_ratio = 6
max_blob_area = 5000
max_curviness = 7

# COLOR
ipt_hue_defs = {'red':(5, 70),
                'yellow':(70, 110),
                'green':(110, 186),
                'blue':(186, 265),
                'magenta':(265, 5)
                }

# Precisely one should be true
get_white_point_by_clicking_single = False
get_white_point_by_clicking_multiple = True

show_score_when_there_is_a_tie = False

if not create_color_tables:
    sRGB_to_LMS_table = np.load(os.path.join(color_tables_folder, "sRGB_to_LMS_table.npy"))
    LMS_to_master_table = np.load(os.path.join(color_tables_folder, "LMS_to_master_table.npy"))


# Color ranges go counterclockwise,
# # e.g., red starts at 5 and goes conterclockwise to 65 degrees.
# Color includes value most pointclockwise, and excludes point
# most counterclockwise. For example, red contains 5 but excludes 65. 65 belongs to green.
# WE ASSUME THAT 0 BELONGS TO MAGENTA OR RED.


# CIE illuminant tristimulus values
A_XYZ = XYZColor(0.95047, 1.00000, 0.35585)
B_XYZ = XYZColor(0.99072, 1.00000, 0.85223)
C_XYZ = XYZColor(0.98074, 1.00000, 1.18232)
D50_XYZ = XYZColor(0.96422, 1.00000, 0.82521)
D55_XYZ = XYZColor(0.95682, 1.00000, 0.92149)
D65_XYZ = XYZColor(0.9504, 1.00000, 1.08883)
D75_XYZ = XYZColor(0.94972, 1.00000, 1.22638)
E_XYZ = XYZColor(1.0, 1.0, 1.0)
F2_XYZ = XYZColor(0.99186, 1.00000, 0.67393)
F7_XYZ = XYZColor(0.95041, 1.00000, 1.08747)
F11_XYZ = XYZColor(1.00962, 1.00000, 0.64350)


# LMS ranges
L_min = 0.0
L_max = 1.0172
M_min = 0
M_max = 1.0172
S_min = 0
S_max = 1.0172


# COLORS FOR DRAWING
RGB_COLS = {'M':(255, 0, 255),
            'R':(255, 0, 0),
            'Y':(255, 255, 0),
            'G':(0, 255,0),
            'C':(0, 255, 255),
            'B':(0, 0, 255)}