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


import matplotlib.pyplot as plt
from imageio import imread
import time
import numpy as np
import cv2
import imutils
from dynamic_color_master_table import convert_sRGB_to_normalized_CIEXYZ,\
    find_average_of_CIEXYZ_colors
from parameters import get_white_point_by_clicking_single,\
    get_white_point_by_clicking_multiple,\
    given_fixed_white_point,\
    D65_XYZ,\
    resize_height,\
    resize_width,\
    rotate_image


class VideoLoop():

    def __init__(self,
                 main_folder_name,
                 subfolder_name,
                 file_extension,
                 starting_frame_index,
                 subplot_layout,
                 process,
                 show_time_per_loop=False):
        self.main_folder_name = main_folder_name
        self.subfolder_name = subfolder_name
        self.file_extension = file_extension
        self.frame_name = ""
        self.frame_index = starting_frame_index
        self.process = process
        self.img = None
        self.show_time_per_loop = show_time_per_loop
        self.subplot_layout = subplot_layout
        self.subplot_number = subplot_layout[0] * subplot_layout[1]
        self.sRGB_sample_white = np.array([255, 255, 255])
        self.normalized_CIEXYZ_white_samples = []
        self.new_sRGB_white_samples = []
        self.chromatic_adaptation_lock = True
        self.resize_height = resize_height
        self.resize_width = resize_width

    def read_frame(self):
        self.frame_name = self.main_folder_name + "/" + self.subfolder_name + "/" + "img" + str(self.frame_index) + self.file_extension
        self.img = imread(self.frame_name)
        self.img = cv2.resize(self.img, (self.resize_width, self.resize_height), interpolation=cv2.INTER_LINEAR)

        if rotate_image:
            self.img = imutils.rotate_bound(self.img, 315)

    def press(self, event):
        if event.key == "]":
            self.frame_index = self.frame_index + 1
        elif event.key == "[" and self.frame_index > 0:
            self.frame_index = self.frame_index - 1
        elif event.key == "z":
            self.chromatic_adaptation_lock = not self.chromatic_adaptation_lock
            print("Adjust chromatic adaptation: " + str(self.chromatic_adaptation_lock))
        elif event.key == "w":
            self.CIEXYZ_white_point = D65_XYZ
            print("Chromatic adaptation reset to D65.")

        self.read_frame()

        if self.show_time_per_loop:
            t = time.time()

        images = self.process(self.img,
                              self.CIEXYZ_white_point,
                              self.frame_name)

        if self.show_time_per_loop:
            print("Time for image processing only method:")
            print(time.time() - t)

        for i in range(self.subplot_number):
            self.imgplot[i].set_data(images[i])

        if event.key != 'q':
            self.fig.canvas.draw()


    def onclick(self, event):

        if not self.chromatic_adaptation_lock:
            if event.inaxes in [self.axs[0,0]]:
                ix, iy = event.xdata, event.ydata
                ix = int(ix)
                iy = int(iy)
                # iy is row
                # ix is column
                if get_white_point_by_clicking_single:

                    self.sRGB_sample_white = self.img[iy, ix]
                    self.CIEXYZ_white_point = convert_sRGB_to_normalized_CIEXYZ(self.sRGB_sample_white)
                    print("New single white point: ")
                    print(self.sRGB_sample_white)
                    print("In CIEXYZ with Y normalized to 1:")
                    print(self.CIEXYZ_white_point)

                    images = self.process(self.img,
                                          self.CIEXYZ_white_point,
                                          self.frame_name)

                if get_white_point_by_clicking_multiple:
                    new_CIEXYZ_samples = []
                    new_sRGB_samples = []
                    for i in range(-1,2):
                        for j in range(-1,2):
                            sRGB_color = self.img[iy + j, ix + i]
                            new_sRGB_samples.append(sRGB_color)
                            normalized_XYZ_color = convert_sRGB_to_normalized_CIEXYZ(sRGB_color)
                            new_CIEXYZ_samples.append(normalized_XYZ_color)
                    self.normalized_CIEXYZ_white_samples = new_CIEXYZ_samples
                    self.new_sRGB_white_samples = new_sRGB_samples
                    self.CIEXYZ_white_point = find_average_of_CIEXYZ_colors(self.normalized_CIEXYZ_white_samples)
                    images = self.process(self.img,
                                          self.CIEXYZ_white_point,
                                          self.frame_name)
                    print("New average CIEXYZ white:")
                    print(self.CIEXYZ_white_point)
                    print("Produced from sRGB samples:")
                    print(self.new_sRGB_white_samples)

                if given_fixed_white_point:
                    print("White point is fixed:")
                    print(given_fixed_white_point)
                    images = self.process(self.img,
                                          given_fixed_white_point,
                                          self.frame_name)

                for i in range(self.subplot_number):
                    self.imgplot[i].set_data(images[i])
                self.fig.canvas.draw()

    def start(self):
        self.fig, self.axs = plt.subplots(self.subplot_layout[0], self.subplot_layout[1])
        self.read_frame()

        starting_white = D65_XYZ

        images = self.process(self.img,
                              starting_white,
                              self.frame_name)
        self.imgplot = []

        if self.subplot_layout[0] != 1 and self.subplot_layout[1] != 1:
            i = 0
            for row in range(self.subplot_layout[0]):
                for col in range(self.subplot_layout[1]):
                    self.imgplot.append(self.axs[row, col].imshow(images[i], cmap='gray'))
                    i = i + 1
        else:
            if self.subplot_layout[0] == 1:
                for entry in range(self.subplot_layout[1]):
                    if self.subplot_layout[1] != 1:
                        self.imgplot.append(self.axs[entry].imshow(images[entry], cmap='gray'))
                    else:
                        self.imgplot.append(self.axs.imshow(images[entry], cmap='gray'))
            else:
                for entry in range(self.subplot_layout[0]):
                    self.imgplot.append(self.axs[entry].imshow(images[entry], cmap='gray'))
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.draw()
        plt.show()