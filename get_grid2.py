import cv2
from get_grid import get_grid_mixed, draw_lines
from get_corner_coordinates import get_corner_coordinates
import numpy as np
import math
from rotate_board import transform_frame
from sklearn.cluster import k_means

DEBUG = 1
CORNERS = (1161, 710),(745, 711),(643, 953),(1240, 955)

def filter_lines(lines, avg_dist, axis=(0, np.pi/2)):
    """
    Filters out lines that are too close to each other.
    """
    ret_lines = []
    clusters = []
    clusters.append([lines[0]])
    for i in range(1, len(lines)):
        p1 = intersect_in(lines[i][0], axis)
        p0 = intersect_in(lines[i-1][0], axis)
        if abs(np.linalg.norm(p1-p0)) > avg_dist*.9:
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
    
    #### FIND A MORE SUITABLE CENTRAL V LINE

    vlines = np.array(sorted(vlines, key=lambda x: intersect_in(x[0], (0, np.pi/2))[0]))
    central_vline = vlines[len(vlines)//2][0]
    hlines = np.array(sorted(hlines, key=lambda x: intersect_in(x[0], central_vline)[1]))
    ### find a way to filter lines
    h_intercepts = [intersect_in((rho, theta), (frame.shape[0], 0)).astype(int) for rho, theta in hlines[:, 0]]
    v_intercepts = [intersect_in((rho, theta), (frame.shape[1], np.pi/2)).astype(int) for rho, theta in vlines[:, 0]]
    # ==============================
    avg_hrhos = np.median(np.diff(hlines[:, 0, 0]))
    avg_vrhos = np.median(np.diff(vlines[:, 0, 0]))

    vlines = filter_lines(vlines, avg_vrhos, hlines[len(hlines)//2][0])
    for vline in vlines:
        hlines = filter_lines(hlines, avg_hrhos, vline[0])
    

    hlines = np.concatenate(hlines, axis=0)  # flatten the list of arrays
    vlines = np.concatenate(vlines, axis=0)  # flatten the list of arrays
    lines = np.concatenate((hlines, vlines), axis=0)
    
    print(lines)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for line in lines:
        draw_line(edges, line, (0, 0, 255), 2)
        draw_line(frame, line, (0, 0, 255), 2)
    draw_line(edges, central_vline, (0, 255, 0), 2)   
    draw_line(frame, central_vline, (0, 255, 0), 2)

    points = get_lattice_points(vlines, hlines)
    frame = draw_points(frame, points)
    edges = draw_points(edges, points)

    return frame, edges, hlines, vlines, points

def draw_line(frame, line, color=(0,0,255), thickness=2):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 3000 * (-b))
    y1 = int(y0 + 3000 * (a))
    x2 = int(x0 - 3000 * (-b))
    y2 = int(y0 - 3000 * (a))
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

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

def draw_points(frame, points, color=(255, 0, 0), thickness=-1):
    for point in points:
        cv2.circle(frame, point, 5, color, thickness)
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
        points = frame.copy()
        
        
        hough_frame, hough_edges, hlines, vlines, points = houghlines(cropped_frame)
        lines = np.concatenate((hlines, vlines), axis=0)
        points_frame = np.zeros_like(cropped_frame)
        points_frame = draw_points(points_frame, points, (0, 255, 0))
        points_frame = insert_crop(frame.copy(), points_frame, vals)
        points_frame = transform_frame(corners, points_frame)        
        
        # frame = transform_frame(corners, frame)
        edges = transform_frame(corners, edges)
        cv2.imshow('frame', frame)
        cv2.imshow('Canny', edges)
        cv2.imshow('Points', points_frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        