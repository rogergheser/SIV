import cv2
from get_grid import get_grid_mixed, draw_lines
from get_corner_coordinates import get_corner_coordinates
import numpy as np
import math
from rotate_board import transform_frame
from sklearn.cluster import k_means

DEBUG = 1
CORNERS = (643, 953),(1240, 955),(1161, 710),(745, 711)

def filter_lines(lines):
    """
    Filters out lines that are too close to each other.
    """
    lines = np.array(sorted(lines, key=lambda x: x[0][1]))
    indices_to_delete = []
    for i, (theta, rho) in enumerate(lines[:,  0]):
        for j, (theta2, rho2) in enumerate(lines[:,  0]):
            if i != j and abs(rho%math.pi - rho2%math.pi) <  0.3:
                pass
            # remove lines
    lines = np.delete(lines, indices_to_delete, axis=0)
    return lines


def houghlines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 90, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    print(lines)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    filtered_lines = filter_lines(lines)

    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 3000 * (-b))
        y1 = int(y0 + 3000 * (a))
        x2 = int(x0 - 3000 * (-b))
        y2 = int(y0 - 3000 * (a))
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for rho, theta in filtered_lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 3000 * (-b))
        y1 = int(y0 + 3000 * (a))
        x2 = int(x0 - 3000 * (-b))
        y2 = int(y0 - 3000 * (a))
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(edges, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame, edges

def get_crop(corners):
    """
    Returns top, right, bottom, left
    """
    top = int(min(corners, key=lambda x: x[1])[1])
    bottom = int(max(corners, key=lambda x: x[1])[1])
    left = int(min(corners, key=lambda x: x[0])[0])
    right = int(max(corners, key=lambda x: x[0])[0])
    
    return top, right, bottom, left

def crop_frame(frame, vals):
    top, right, bottom, left = vals
    return frame[top:bottom, left:right]

def insert_crop(original_frame, new_frame, vals):
    top, right, bottom, left = vals
    original_frame[top:bottom, left:right] = new_frame
    return original_frame

if __name__ == "__main__":
    # video_path = '/Users/amirgheser/SIV/project/test/video/IMG_0389.mov'
    video_path = '/Users/amirgheser/SIV/project/test/video/video2.mp4'
    # video_path = '/Users/amirgheser/SIV/project/test/video/rotated_board.mp4'
    cap = cv2.VideoCapture(video_path)
    if DEBUG:
        corners = CORNERS
    else: 
        corners = get_corner_coordinates(cap)
    vals = get_crop(corners)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = crop_frame(frame, vals)
        edges = frame.copy()
        hough_frame, hough_edges = houghlines(cropped_frame)
        insert_crop(frame, hough_frame, vals)
        insert_crop(edges, hough_edges, vals)
        frame = transform_frame(corners, frame)
        edges = transform_frame(corners, edges)
        cv2.imshow('frame', frame)
        cv2.imshow('Canny', edges)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        