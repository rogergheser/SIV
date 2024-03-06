import cv2
from get_grid import get_grid_mixed, draw_lines
from get_corner_coordinates import get_corner_coordinates
import numpy as np
import math
from rotate_board import transform_frame
from sklearn.cluster import k_means

DEBUG = 0
CORNERS = (1161, 710),(745, 711),(643, 953),(1240, 955)

def filter_lines(lines, avg_dist):
    """
    Filters out lines that are too close to each other.
    """
    ret_lines = []
    clusters = []
    clusters.append([lines[0]])
    for i in range(1, len(lines)):
        if abs(lines[i][0][0] - lines[i-1][0][0]) > avg_dist*0.6:
            clusters.append([lines[i]])
        else:
            clusters[-1].append(lines[i])    
        
    for cluster in clusters:
        if len(cluster) > 1:
            np.mean(cluster, axis=0)
            ret_lines.append(np.mean(cluster, axis=0))
        else:
            ret_lines.append(cluster[0])
    return ret_lines
    
def get_y_intercept(rho, theta, x):
    """
    Returns the y-intercept of the line.
    """
    a = np.cos(theta)
    b = -np.sin(theta)
    return (rho - x * a) / b

def get_x_intercept(rho, theta, y):
    """
    Returns the x-intercept of the line.
    """
    a = np.cos(theta)
    b = -np.sin(theta)
    return (rho - y * b) / a

def houghlines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 70, 150, apertureSize=3)
    # horizontal lines
    hlines = np.array(cv2.HoughLines(edges, 1, np.pi / 180, 110, min_theta=math.pi/2-0.1, max_theta=math.pi/2+0.1))
    # vertical lines
    vlines0 = cv2.HoughLines(edges, 1, np.pi / 180, 90, min_theta=0, max_theta=0.7)
    vlines1 = cv2.HoughLines(edges, 1, np.pi / 180, 90, min_theta=math.pi-0.7, max_theta=math.pi)
    if vlines0 is not None and vlines1 is not None:
        vlines = np.concatenate((vlines0, vlines1), axis=0)
    elif vlines0 is not None:
        vlines = vlines0
    elif vlines1 is not None:
        vlines = vlines1
    else:
        vlines = None
    
    hlines = np.array(sorted(hlines, key=lambda x: x[0][0]))
    vlines = np.array(sorted(vlines, key=lambda x: x[0][0]))
    ### find a way to filter lines
    h_intercepts = [intersect_in((rho, theta), (frame.shape[0], 0)).astype(int) for rho, theta in hlines[:, 0]]
    v_intercepts = [intersect_in((rho, theta), (frame.shape[1], np.pi/2)).astype(int) for rho, theta in vlines[:, 0]]
    # ==============================
    avg_hrhos = np.median(np.diff(hlines[:, 0, 0]))
    avg_vrhos = np.median(np.diff(vlines[:, 0, 0]))

    hlines = filter_lines(hlines, avg_hrhos)
    vlines = filter_lines(vlines, avg_vrhos)

    hlines = np.concatenate(hlines, axis=0)  # flatten the list of arrays
    vlines = np.concatenate(vlines, axis=0)  # flatten the list of arrays
    lines = np.concatenate((hlines, vlines), axis=0)
    
    print(lines)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    for rho, theta in lines:
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

    points = get_lattice_points(vlines, hlines)
    frame = draw_points(frame, points)
    edges = draw_points(edges, points)

    return frame, edges, lines

def intersect_in(line1, line2):
    """
    :param line1: (rho1, theta1)
    :param line2: (rho2, theta2)
    Returns the intersection point of the two lines.
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])
    return np.linalg.solve(A, b)

def get_lattice_points(vlines, hlines):
    lattice_points = []
    for i in range(len(hlines)):
        for j in range(len(vlines)):
            point = intersect_in(hlines[i], vlines[j]).astype(int)
            lattice_points.append(point)

    return lattice_points

def draw_points(frame, points):
    for point in points:
        cv2.circle(frame, point, 5, (255, 0, 0), -1)
    return frame

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
        hough_frame, hough_edges, lines = houghlines(cropped_frame)
        insert_crop(frame, hough_frame, vals)
        insert_crop(edges, hough_edges, vals)
        # frame = transform_frame(corners, frame)
        edges = transform_frame(corners, edges)
        cv2.imshow('frame', frame)
        cv2.imshow('Canny', edges)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        