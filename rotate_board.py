import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import sys
import math
import time
import copy
from get_corner_coordinates import get_corner_coordinates

def transform_frame(coords, frame):
    """
    :param coords: topL, topR, botR, botL coordinates of the corners of the chessboard
    :param frame: the frame of the video
    :return: the transformed frame
    """
    topL, topR, botR, botL = coords
    width = int(frame.shape[1])
    height = int(frame.shape[0])
    src_points = np.float32([topL, topR, botR, botL])
    dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M, mask = cv2.findHomography(src_points, dst_points)
    transformed_image = cv2.warpPerspective(frame, M, (width, height))
    return transformed_image

if __name__ == '__main__':
    # VIDEO_PATH = "/Users/amirgheser/SIV/project/test/video/IMG_0389.mov"
    VIDEO_PATH = "/Users/amirgheser/SIV/project/test/video/video2.mp4"
    cap = cv2.VideoCapture(VIDEO_PATH)

    ret, frame = cap.read()
    if ret:
        width = int(cap.get(3))
        height = int(cap.get(4))
        topL, topR, botR, botL = get_corner_coordinates(cap)
        src_points = np.float32([topL, topR, botR, botL])
        dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        M, mask = cv2.findHomography(src_points, dst_points)
        transformed_image = cv2.warpPerspective(frame, M, (width, height))
        cv2.imshow('Transformed Image', transformed_image)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
    cap.release()
    # Show video with fixed perspective
    print("starting video")
    cap = cv2.VideoCapture(VIDEO_PATH)
    while True:
        ret, frame = cap.read()
        if ret:
            transformed_image = cv2.warpPerspective(frame, M, (width, height))
            cv2.imshow('Transformed Image', transformed_image)
            # cv2.imshow('Video', frame)
            key = cv2.waitKey(1)
            if key & 0xFF==ord('q'):
                break
        else:
            break
    cap.release()