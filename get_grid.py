# First get the board to rotate to the correct coordinates by invoking get_corner_coordinates.py
# Then apply canny edge detection to the image and find the lines using hough transform
import cv2
import numpy as np
import math
import time
from get_corner_coordinates import get_corner_coordinates
from rotate_board import transform_frame
from canny import hough_image

DEBUG_CORNERS = (0, 0), (0,0), (0, 0), (0,0)
DEBUG = 0
LATEST_GRID = None
FRAME_SPEED = 1000

def ver_dist(line1, line2):
    """
    Calculates vertical distance between two lines
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    return np.abs(rho1*math.cos(theta1) - rho2*math.cos(theta2))

def filter_close_lines_ver(lines, min_delta_rho):
    """
    :param lines: list of lines in the form of tuples (rho, theta)
    :param min_delta_rho: minimum distance allowed between two lines
    :return: filtered list of lines
    """
    filtered_lines = []
    if lines is not None:
        # Sort the lines based on vertical distance
        sorted_lines = sorted(lines, key=lambda x: x[0][0]*math.cos(x[0][1]))
        
        for i in range(len(sorted_lines)):
            rho, theta = sorted_lines[i][0]
            if i == 0 or ver_dist((rho, theta), filtered_lines[-1]) > min_delta_rho:
                filtered_lines.append((rho, theta))
    else:
        print("No lines found")
    
    return filtered_lines

def hor_dist(line1, line2):
    """
    Calculates horizontal distance between two lines
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    return np.abs(rho1*math.sin(theta1) - rho2*math.sin(theta2))

def filter_close_lines_hor(lines, min_delta_rho):
    filtered_lines = []

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
    
    theta_delta = math.pi/240
    interline_delta = transformed_image.shape[1] // (8*1.7) # 8 squares in the board
    gray_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(gray_image, 0, 50, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100, 
                                  min_theta=(np.pi / 2) - theta_delta, 
                                  max_theta=(np.pi / 2) + theta_delta)
    hough_image = edges.copy()
    hough_image = cv2.cvtColor(hough_image, cv2.COLOR_GRAY2BGR)

    lines = filter_close_lines_hor(lines, interline_delta)

    if len(lines) != 9:
        print("Missing {} horizontal lines".format(9-len(lines)))
        lines = find_missing_hlines(lines, transformed_image)

    return np.array(lines)

def find_vertical_lines(transformed_image):
    # Apply hough transform to the image only to find vertical lines
    # Return the lines that are vertical with a maximum distorsion of 15 degrees
    # Return the lines in the form of a list of tuples (rho, theta
    interline_delta = transformed_image.shape[0] // (8*1.7) # 8 squares in the board
    vertical_theta_range = math.pi/120
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

    lines = filter_close_lines_ver(lines, interline_delta)

    if len(lines) != 9:
        print("Missing {} vertical lines".format(9-len(lines)))
        lines = find_missing_vlines(lines, transformed_image)

    return np.array(lines)

def find_missing_vlines(lines, transformed_image):
    target_delta = transformed_image.shape[1] // 8 # 8 squares in the board
    deltas = []
    for i in range(1, len(lines)):
        rho, theta = lines[i]
        rho_prev, theta_prev = lines[i-1]
        delta = rho*math.cos(theta) - rho_prev*math.cos(theta_prev)
        deltas.append(delta)
        deltas.append(delta/2)
    
    deltas = sorted(deltas, key=lambda x: abs(x-target_delta))
    print(f"Found delta {deltas[0]}, estimated delta {target_delta}")

    center = transformed_image.shape[1] // 2
    average_theta = np.mean([line[1] for line in lines])
    lines = []
    lines.append((center, 0))
    for i in range(4):
        lines.append((min(transformed_image.shape[1]-1, (center+(i+1)*abs(deltas[0]))), 0))
        lines.append((max(1, (center-(i+1)*abs(deltas[0]))), 0))
    
    return lines

def find_missing_hlines(lines, transformed_image):
    target_delta = transformed_image.shape[0] // 8 # 8 squares in the board
    deltas = []
    for i in range(1, len(lines)):
        rho, theta = lines[i]
        rho_prev, theta_prev = lines[i-1]
        delta = rho*math.sin(theta) - rho_prev*math.sin(theta_prev)
        deltas.append(delta)
        deltas.append(delta/2)
    
    deltas = sorted(deltas, key=lambda x: abs(x-target_delta))
    print(f"Found delta {deltas[0]}, estimated delta {target_delta}")

    center = transformed_image.shape[0] // 2
    average_theta = np.mean([line[0] for line in lines])
    lines = []
    lines.append((center, np.pi/2))
    for i in range(4):
        lines.append((min(transformed_image.shape[0]-1, (center+(i+1)*abs(deltas[0]))), np.pi/2))
        lines.append((max(1, (center-(i+1)*abs(deltas[0]))), np.pi/2))

    
    return lines
class GridLine:
    is_vertical = None
    rho = None
    theta = None
    coor = None
    def __init__(self, rho, theta):
        self.rho = rho
        self.theta = theta
        self.is_vertical = True if np.pi *0.8 < theta < np.pi*1.2 else False
        if self.is_vertical:
            self.coor = rho * math.cos(theta)
        else:
            self.coor = rho * math.sin(theta)

def get_grid_naive(frame):
    height = frame.shape[1]    
    width = frame.shape[0]

    vertical_lines = []
    horizontal_lines = []

    for i in range(0, width, width//8):
        vertical_lines.append((i, np.pi/2))
    
    for i in range(0, height, height//8):
        horizontal_lines.append((i, 0))

    return vertical_lines, horizontal_lines


def draw_lines(image, lines, color=(0, 0, 255)):
    k_mul = max(image.shape[0], image.shape[1]) * 1.2
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = int(a * rho)
        y0 = int(b * rho)
        x1 = int(x0 + k_mul * (-b))
        y1 = int(y0 + k_mul * (a))
        x2 = int(x0 - k_mul * (-b))
        y2 = int(y0 - k_mul * (a))
        cv2.line(image, (x1, y1), (x2, y2), color, 2)
    return image


def get_grid_lines(image):
    """
    :param image: image of the board
    :return: vertical lines, horizontal lines
    """
    v_lines = find_vertical_lines(image)
    h_lines = find_horizontal_lines(image)
    return v_lines, h_lines

def get_grid(image):
    v_lines, h_lines = get_grid_lines(image)
    hough = draw_lines(image, v_lines)
    hough = draw_lines(hough, h_lines, color=(0, 255, 0))
    return hough


if __name__ == '__main__':
    # video_path = '/Users/amirgheser/SIV/project/test/video/IMG_0389.mov'
    # video_path = '/Users/amirgheser/SIV/project/test/video/video2.mp4'
    video_path = '/Users/amirgheser/SIV/project/test/video/rotated_board.mp4'
    DEBUG_CORNERS = (768, 723), (1147, 721), (1208, 942), (679, 946)
    cap = cv2.VideoCapture(video_path)
    if DEBUG:
        coords = DEBUG_CORNERS
    else:
        coords = get_corner_coordinates(cap)
    v_avg_params = []
    h_avg_params = []
    new_fps = 3
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if frame_count % new_fps == 0:
            frame_count = 0
            if ret:
                transformed_image = transform_frame(coords, frame)
                v_lines, h_lines = get_grid_lines(transformed_image)

                if cv2.waitKey(FRAME_SPEED//new_fps) & 0xFF == ord('q'):
                    break
                if len(v_lines) == 9 and len(h_lines) == 9:
                    LATEST_GRID = (v_lines, h_lines)
                else:
                    if LATEST_GRID is not None:
                        v_lines, h_lines = LATEST_GRID
                hough = draw_lines(transformed_image, v_lines)
                hough = draw_lines(hough, h_lines, color=(0, 255, 0))
                cv2.imshow('Transformed Image', hough)
                if cv2.waitKey(FRAME_SPEED//new_fps) & 0xFF == ord('q'):
                    break
            else:
                print("Error reading frame or no frames left")
                cap.release()
                break
        frame_count += 1
    #### this found some lines... now lets try to detect the board grid...

    cap.release()