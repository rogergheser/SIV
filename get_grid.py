# First get the board to rotate to the correct coordinates by invoking get_corner_coordinates.py
# Then apply canny edge detection to the image and find the lines using hough transform
import cv2
import numpy as np
import math
import time
from get_corner_coordinates import get_corner_coordinates
from rotate_board import transform_frame
from canny import hough_image
# video_path = '/Users/amirgheser/SIV/project/test/video/IMG_0389.mov'

def ver_dist(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    return np.abs(rho1*math.cos(theta1) - rho2*math.cos(theta2))

def filter_close_lines_ver(lines, min_delta_rho):
    filtered_lines = []

    # Sort the lines based on vertical distance
    if lines is not None:
        sorted_lines = sorted(lines, key=lambda x: x[0][0]*math.cos(x[0][1]))
        for i in range(len(sorted_lines)):
            rho, theta = sorted_lines[i][0]
            if i == 0 or ver_dist((rho, theta), filtered_lines[-1]) > min_delta_rho:
                filtered_lines.append((rho, theta))
    else:
        print("No lines found")


    return filtered_lines
def hor_dist(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    return np.abs(rho1*math.sin(theta1) - rho2*math.sin(theta2))

def filter_close_lines_hor(lines, min_delta_rho):
    filtered_lines = []

    # Sort the lines by rho
    if lines is not None:
        sorted_lines = sorted(lines, key=lambda x: x[0][0]*math.sin(x[0][1]))
        for i in range(len(sorted_lines)):
            rho, theta = sorted_lines[i][0]
            if i == 0 or hor_dist((rho, theta), (filtered_lines[-1])) > min_delta_rho:
                filtered_lines.append((rho, theta))
    else:
        print("No lines found")

    return filtered_lines

def find_horizontal_lines(transformed_image):
    # Apply hough transform to the image only to find horizontal lines
    # Return the lines that are horizontal with a maximum distorsion of 2 degrees
    # Return the lines in the form of a list of tuples (rho, theta)
    theta_delta = math.pi/12
    interline_delta = transformed_image.shape[1] // (8*2) # 8 squares in the board
    gray_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(gray_image, 0, 50, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100, 
                                  min_theta=(np.pi / 2) - theta_delta, 
                                  max_theta=(np.pi / 2) + theta_delta)
    hough_image = edges.copy()
    hough_image = cv2.cvtColor(hough_image, cv2.COLOR_GRAY2BGR)
    print(lines)

    # Pre-process lines to remove lines that are too close
    lines = filter_close_lines_hor(lines, interline_delta)

    if lines is not None:
        k_mul = transformed_image.shape[1] * 1.2
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Calculate start and end points (x1, y1) and (x2, y2)
            x1 = int(x0 + k_mul * (-b))
            y1 = int(y0 + k_mul * (a))
            x2 = int(x0 - k_mul * (-b))
            y2 = int(y0 - k_mul * (a))
            cv2.line(hough_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return hough_image

def find_vertical_lines(transformed_image):
    # Apply hough transform to the image only to find vertical lines
    # Return the lines that are vertical with a maximum distorsion of 15 degrees
    # Return the lines in the form of a list of tuples (rho, theta)
    interline_delta = transformed_image.shape[0] // (8*2) # 8 squares in the board
    vertical_theta_range = math.pi/30
    gray_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(gray_image, 0, 50, apertureSize=3, L2gradient=True)
    edges = cv2.GaussianBlur(edges, (3,3), 0)
    lines_vertical_0 = cv2.HoughLines(edges, 1, np.pi/180, 250, min_theta=0, max_theta=vertical_theta_range)
    lines_vertical_180 = cv2.HoughLines(edges, 1, np.pi/180, 250, min_theta=np.pi-vertical_theta_range, max_theta=np.pi)
    
    if lines_vertical_0 is not None:
        lines = np.concatenate((lines_vertical_0, lines_vertical_180), axis=0)
    else: 
        lines = lines_vertical_180
    hough_image = edges.copy()
    hough_image = cv2.cvtColor(hough_image, cv2.COLOR_GRAY2BGR)
    print(lines)

    # Pre-process lines to remove lines that are too close
    lines = filter_close_lines_ver(lines, interline_delta)

    if lines is not None:
        k_mul = transformed_image.shape[0] * 1.2
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = int(a * rho)
            y0 = int(b * rho)
            x1 = int(x0 + k_mul * (-b))
            y1 = int(y0 + k_mul * (a))
            x2 = int(x0 - k_mul * (-b))
            y2 = int(y0 - k_mul * (a))
            cv2.line(hough_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return hough_image


video_path = '/Users/amirgheser/SIV/project/test/video/video2.mp4'
cap = cv2.VideoCapture(video_path)
coords = get_corner_coordinates(cap)
while True:
    ret, frame = cap.read()
    if ret:
        transformed_image = transform_frame(coords, frame)
        cv2.imshow('Transformed Image', transformed_image)
        houghv = find_vertical_lines(transformed_image)
        houghh = find_horizontal_lines(transformed_image)
        hough = cv2.bitwise_or(houghv, houghh)
        cv2.imshow("Hough Image", hough)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print("Error reading frame or no frames left")
        cap.release()
        break
#### this found some lines... now lets try to detect the board grid...

cap.release()

