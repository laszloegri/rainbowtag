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
import sys
import os
import csv
import numpy as np

class VideoReader():

    def __init__(self, video_path, show_frame_index = True):
        self.video_path = video_path
        self.open_video()
        self.show_frame_number = show_frame_index
        self.buffer_size = 1000
        if self.buffer_size < 0: self.buffer_size = float('inf')

        # Index in buffer. Current frame index is last_read_frame - back
        self.buffer_index = 0
        self.global_index = -1

    def open_video(self):
        # Read video
        self.video = cv2.VideoCapture(self.video_path)

        # Initialize buffer list
        self.buffer = []

        # self.last_read_frame is the maximum index of the frame read
        # We haven't read any frames yet
        self.last_read_frame = -1

        # Exit if video not opened.
        if not self.video.isOpened():
            print('Could not open video')
            sys.exit()

    # Updates instance to next and returns next frame
    def next(self):
        if self.buffer_index == 0:
            ok, frame = self.video.read()
            if ok:
                # Increase global index counter
                self.last_read_frame += 1
                self.global_index += 1

                # We write the frame numbers on the frames
                if self.show_frame_number:
                    cv2.putText(frame, "Frame index: " + str(int(self.last_read_frame)), (100, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2);

                # Add new frame to buffer
                self.buffer.append(frame)

                # If list exceeds allowed buffer size, remove oldest frame
                if len(self.buffer) == self.buffer_size + 1:
                    # Remove first element
                    self.buffer = self.buffer[1:]

                # return last element of buffer
                return(np.copy(self.buffer[-self.buffer_index-1]), self.global_index, True)
            else:
                print("End of video (or video read error).")
                return (np.copy(self.buffer[-self.buffer_index - 1]), self.global_index, False)

        # if self.buffer_index > 0
        else:
            self.buffer_index = self.buffer_index - 1
            self.global_index += 1
            return(np.copy(self.buffer[-self.buffer_index-1]), self.global_index, True)

    # Goes back a frame
    def previous(self):
        if (self.last_read_frame - self.buffer_index > 0) and self.buffer_index + 1 < self.buffer_size:
            self.buffer_index = self.buffer_index + 1
            self.global_index -= 1
            return(np.copy(self.buffer[-self.buffer_index - 1]), self.global_index, True)
        else:
            print("No more previous frames.")
            return(np.copy(self.buffer[-self.buffer_index - 1]), self.global_index, False)

    def current(self):
        return(np.copy(self.buffer[-self.buffer_index - 1]), self.global_index)

    # index is the index of the frame for which we want the index for the buffer
    def get_buffer_index(self, index):
        buffer_index = self.last_read_frame - index
        if (buffer_index < 0) or (buffer_index > self.buffer_size - 1) or (buffer_index > self.last_read_frame):
            return(None)

        return(buffer_index)

    # Goes to a frame with index if that frame is in the buffer. I.e.,
    #   - we cannot go to a future frame
    #   - we cannot go to a past frame outside the buffer
    def go_to(self, index):
        buff_ind = self.get_buffer_index(index)
        if buff_ind == None:
            print('Frame with index i is not readable.')
            sys.exit()
        else:
            self.buffer_index = buff_ind
            self.global_index = index


def write_frames_to_disk(video_path, frames_folder_name):
    VR = VideoReader(video_path, show_frame_index=False)
    i = 0
    while True:
        next_frame, _, ok = VR.next()
        if True:
            next_frame = cv2.resize(next_frame, (1920,1080))
        if False:
            next_frame = cv2.rotate(next_frame, cv2.ROTATE_90_CLOCKWISE)
        if not ok:
            break

        frame_name = frames_folder_name + "/img" + str(i) + ".bmp"
        cv2.imwrite(frame_name, next_frame)
        print(i)

        i = i + 1


def write_detections_to_csv(file_name, header, new_file_id, row_list):
    # Check if csv file exists
    csv_exists = os.path.exists(file_name)

    # If csv file does not exist, create it and put the labels for the first row
    writer = csv.writer(open(file_name, 'ab'))
    if not csv_exists:
        writer.writerow(header)

    writer.writerow(['PROCESSED FILE: ' + new_file_id])

    # Append to the end of the CSV file
    #writer.writerow(['NEW'])
    # Print.
    for i in range(len(row_list)):
        writer.writerow(row_list[i])
        print(str(i) + " " + str(row_list[i]))


def get_immediate_subdirectories(directory):
    return([name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))])


def get_files_in_folder(directory):
    return([name for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))])