# First get the board to rotate to the correct coordinates by invoking get_corner_coordinates.py
# Then apply canny edge detection to the image and find the lines using hough transform
import cv2
import numpy as np
import math
import time
from get_corner_coordinates import get_corner_coordinates
from rotate_board import transform_frame
from canny import hough_image
video_path = '/Users/amirgheser/SIV/project/test/video/IMG_0389.mov'
cap = cv2.VideoCapture(video_path)
coords = get_corner_coordinates(cap)
while True:
    ret, frame = cap.read()
    if ret:
        transformed_image = transform_frame(coords, frame)
        cv2.imshow('Transformed Image', transformed_image)
        hough = hough_image(transformed_image)
        cv2.imshow("Hough Image", hough)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print("Error reading frame or no frames left")
        cap.release()
        break
#### this found some lines... now lets try to detect the board grid...

cap.release()