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
from parameters import file_extension,\
    images_folder,\
    sRGB_to_LMS_table,\
    LMS_to_master_table,\
    given_fixed_white_point

from dynamic_color_master_table import dynamic_convert_img_to_7_channels,\
    dynamic_get_adapted_sRGB_img,\
    compute_LMS_gain_coefficients_for_XYZ_white_point_to_D65

from disk_operations import *

import os

# We choose an arbitrary folder that contains frames
# experiment_folder = "D:/EXPERIMENTS/1m/2700_1m/"

# condition_folders = get_immediate_subdirectories(images_folder)
# a_condition_full_name = os.path.join(images_folder, condition_folders[0])
# all_frames_folder = os.path.join(a_condition_full_name, 'Random_Frames')

all_frames_folder = images_folder
frame_name_list = get_files_in_folder(all_frames_folder)

white_point_folder = os.path.join(images_folder, "sampled_white_point")

import os
if not os.path.exists(white_point_folder):
    os.makedirs(white_point_folder)


# Get video arbitrarily


# Main folder is experiment name.
# Each condition has a subfolder containing videos.
# We have to go over each of the
# conditions automatically.
# For white point, for each condition
# there is one, so we make a file with
# the name of the condition.

class VideoLoop_WhitePointSelector():

    def __init__(self,
                 experiment_folder,
                 all_frames_folder,
                 frame_name_list,
                 file_extension,
                 starting_file_index,
                 process):
        self.experiment_folder = experiment_folder
        self.all_frames_folder = all_frames_folder
        self.file_extension = file_extension
        self.file_index = starting_file_index
        self.frame_name_list = frame_name_list
        self.frame_name_list_length = len(self.frame_name_list)
        self.process = process
        self.img = None
        self.subplot_layout = np.array([3,1])
        self.subplot_number = 3
        self.CIEXYZ_white_point = D65_XYZ
        self.normalized_CIEXYZ_white_samples = []
        self.new_sRGB_white_samples = []
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.chromatic_adaptation_lock = False

    def read_frame(self):
        self.img = imread(os.path.join(self.all_frames_folder,
                                       self.frame_name_list[self.file_index]))
        self.img = cv2.resize(self.img, (self.resize_width, self.resize_height), interpolation=cv2.INTER_LINEAR)

    def press(self, event):
        if (event.key == "]") and (self.file_index < (self.frame_name_list_length - 1)):
            self.file_index = self.file_index + 1
        elif event.key == "[" and self.file_index > 0:
            self.file_index = self.file_index - 1
        elif event.key == "z":
            self.chromatic_adaptation_lock = not self.chromatic_adaptation_lock
            print("Adjust chromatic adaptation: " + str(self.chromatic_adaptation_lock))
        elif event.key == "w":
            self.CIEXYZ_white_point = D65_XYZ
            print("White is set D65.")
        elif event.key == "v":
            white_point = np.array(self.CIEXYZ_white_point.get_value_tuple())
            print("SAVING CIEXYZ WHITE POINT:")
            print(white_point)
            np.save(os.path.join(white_point_folder, "white_point.npy"), white_point)

        self.read_frame()

        images = self.process(self.img,
                              self.CIEXYZ_white_point)

        for i in range(self.subplot_number):
            self.imgplot[i].set_data(images[i])

        if event.key != 'q':
            self.fig.canvas.draw()


    def onclick(self, event):

        if not self.chromatic_adaptation_lock:
            if event.inaxes in [self.axs[0]]:
                ix, iy = event.xdata, event.ydata
                ix = int(ix)
                iy = int(iy)
                # iy is row
                # ix is column
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
                                      self.CIEXYZ_white_point)

                print("New average CIEXYZ white:")
                print(self.CIEXYZ_white_point)

                print("Produced from sRGB samples:")
                print(self.new_sRGB_white_samples)

                for i in range(self.subplot_number):
                    self.imgplot[i].set_data(images[i])
                self.fig.canvas.draw()

    def start(self):
        self.fig, self.axs = plt.subplots(self.subplot_layout[0], self.subplot_layout[1])
        self.read_frame()

        starting_white = D65_XYZ

        images = self.process(self.img,
                              starting_white)

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
                    self.imgplot.append(self.axs[entry].imshow(images[entry], cmap='gray'))
            else:
                for entry in range(self.subplot_layout[0]):
                    self.imgplot.append(self.axs[entry].imshow(images[entry], cmap='gray'))
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.draw()
        plt.show()


def process_wps(img, CIEXYZ_white_point):
    gain_coeff_L, \
    gain_coeff_M, \
    gain_coeff_S = compute_LMS_gain_coefficients_for_XYZ_white_point_to_D65(CIEXYZ_white_point)
    img_7_channels = dynamic_convert_img_to_7_channels(img,
                                                       sRGB_to_LMS_table,
                                                       LMS_to_master_table,
                                                       gain_coeff_L,
                                                       gain_coeff_M,
                                                       gain_coeff_S,
                                                       LMS_quantization_number)

    adapted_sRGB_img = img_7_channels[4]

    blobs_with_stats,\
    number_of_filtered_contours,\
    filtered_contours,\
    all_contours,\
    processed,\
    lab_chroma_img,\
    lab_L_img = find_blobs_with_stats(img_7_channels)
    img_marker_blobs = np.copy(img)
    draw_all_detected_blobs(img_marker_blobs,
                            blobs_with_stats,
                            filtered_contours)

    return(img, adapted_sRGB_img, img_marker_blobs)


VL_WPS = VideoLoop_WhitePointSelector(images_folder,
                                      all_frames_folder,
                                      frame_name_list,
                                      file_extension,
                                      0,
                                      process_wps)
VL_WPS.start()