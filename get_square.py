import numpy as np
from get_grid import get_grid, get_grid_mixed
import cv2
from project.SIV.get_corner_coordinates import get_corner_coordinates
from project.SIV.rotate_board import transform_frame
import math

DEBUG = False

from project.SIV.rotate_board import transform_frame
def get_square(frame, h_lines, v_lines, point):
    """
    :param frame: the frame of the video
    :param h_lines: all 9 horizontal lines detected
    :param v_lines: all 9 vertical lines detected
    :param point: the center of the base box of the piece
    Returns the square that holds the piece detected at (x, y) in the grid.
    """
    x, y = point
    if len(h_lines) != 9:
        raise ValueError("The number of horizontal lines detected is not 9.")
    if len(v_lines) != 9:
        raise ValueError("The number of vertical lines detected is not 9.")
    
    for i in range(9):
        pass

frames = []

if __name__ == '__main__':
    video_path = '/Users/amirgheser/SIV/project/test/video/IMG_0389.mov'
    # video_path = '/Users/amirgheser/SIV/project/test/video/video2.mp4'
    # video_path = '/Users/amirgheser/SIV/project/test/video/rotated_board.mp4'
    
    # I want to make a multithread application that does the following:
    # main thread should be responsible for reading the video and displaying the frames
    # second thread should be responsible for detecting the lines, I want the second thread to read 100 frames
    # and then process them and compute the average grid and then read the next 100 frames and so on
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    