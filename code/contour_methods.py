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
import matplotlib.pyplot as plt

# The method makes two numpy arrays A1 and A2 (a 0-1 image) of size
# canvas_size_y (rows) by canvas_size_x (columns).
# It draws contour_1 on A1 and contour_2 on A2.
# It counts the number of pixels in the intersection of the contours.
# It also returns the number of pixels in the contours.
def compute_size_of_intersection_of_contours(contour_1, contour_2, canvas_size_x, canvas_size_y):
    A1 = np.zeros((canvas_size_y, canvas_size_x), dtype=np.uint8)
    A2 = np.zeros((canvas_size_y, canvas_size_x), dtype=np.uint8)
    cv2.drawContours(A1, [contour_1], 0, 1, -1)
    cv2.drawContours(A2, [contour_2], 0, 1, -1)

    I = np.logical_and(A1, A2)
    A1_size = np.count_nonzero(A1)
    A2_size = np.count_nonzero(A2)
    I_size = np.count_nonzero(I)
    return(A1_size, A2_size, I_size)