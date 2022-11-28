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


from imageio import imread, imwrite
from imutils import rotate_bound
import cv2
import os
from parameters import file_extension

input_folder = "images/"
output_folder = "images/"
rotate = False

width = 1920
height = 1080

def get_files_in_folder(directory):
    return([name for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))])

def strecth_rotate_save(rotate_image = False):
    frames = get_files_in_folder(input_folder)
    for frame in frames:
        if os.path.splitext(frame)[1] == file_extension:
            full_input_img_file_name = os.path.join(input_folder, frame)
            full_output_img_file_name = os.path.join(output_folder, frame)
            img = imread(full_input_img_file_name)
            processed_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            if rotate_image:
                processed_img = rotate_bound(processed_img, 315)
            imwrite(full_output_img_file_name, processed_img)

strecth_rotate_save(rotate)