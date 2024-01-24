## this is a test file for multiprocessing
import cv2
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import os
import sys
import math
from get_corner_coordinates import get_corner_coordinates
from rotate_board import transform_frame
from get_grid import draw_lines, get_average_grid, get_grid
from get_grid import get_grid_mixed
from multiprocessing import Value
GRID_AMT_AVG = 300
CORNERS = (766, 724), (1144, 717), (1208, 944), (677, 944)
DEBUG = 0
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

def get_lattice_points(v_lines, h_lines):
    lattice_points = []
    for i in range(9):
        for j in range(9):
            point = intersect_in(v_lines[i], h_lines[j])
            lattice_points.append(point)
    
    # lattice_points.sort(key=lambda point: (point[1], point[0]))  # sort by y, then by x
    return lattice_points

def get_average_grids(avg_grids, video_path, coords, stop_process):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    grid = None
    grids = []
    while True and not stop_process.value:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frame = transform_frame(coords, frame)
            grid = get_grid_mixed(frame)
            v_lines, h_lines = decompose_grid(grid)
            grids.append((v_lines, h_lines))
            if frame_count % GRID_AMT_AVG == 0:
                avg_grid = get_average_grid(grids)
                grids = []
                avg_grids.put(avg_grid)
                # Print in red color the following line
                print("\033[91mProcessed {}/{} frames\033[00m".format(frame_count, cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            frame_count += 1
    cap.release()


def decompose_grid(grid):
    v_lines1, v_lines2, h_lines1, h_lines2 = grid
    if len(v_lines1) == 0:
        v_lines = v_lines2
    elif len(v_lines2) == 0:
        v_lines = v_lines1
    else:
        v_lines = sorted(np.concatenate((v_lines1, v_lines2)), key=lambda x: x[0]*math.cos(x[1]))
    
    if len(h_lines1) == 0:
        h_lines = h_lines2
    elif len(h_lines2) == 0:
        h_lines = h_lines1
    else:
        h_lines = sorted(np.concatenate((h_lines1, h_lines2)), key=lambda x: x[0]*math.sin(x[1]))
    return v_lines, h_lines

def show_video(avg_grids, video_path, coords, stop_process):
    print("\033[92mStarting video\033[00m")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000/fps)
    frame_count = 0
    grid = None
    lattice_points = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            if frame_count % GRID_AMT_AVG == 0:
                grid = avg_grids.get()
                v_lines, h_lines = grid[:9], grid[9:]
                lattice_points = get_lattice_points(v_lines, h_lines)
                print(lattice_points)
            v_lines, h_lines = grid[:9], grid[9:]
            frame = transform_frame(coords, frame)
            frame = draw_lines(frame, v_lines, color=(0, 0, 255))
            frame = draw_lines(frame, h_lines, color=(0, 0, 255))
            for point in lattice_points:
                cv2.circle(frame, tuple(point.astype(int)), 7, (0, 255, 0), -1)

            for i in range(8):
                for j in range(8):
                    topL = lattice_points[i*9+j].astype(int)
                    botR = lattice_points[i*9+j+10].astype(int)
                    square = frame[topL[1]:botR[1], topL[0]:botR[0]]
                    print(topL, botR)
                    cv2.imshow("square", square)
                    cv2.waitKey(1)


            cv2.imshow("frame", frame)
            print("Frame: {}/{}".format(frame_count, cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_count += 1
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            stop_process.value = True
            break
    cap.release()

if __name__ == '__main__':
    VIDEO_PATH = "/Users/amirgheser/SIV/project/test/video/IMG_0389.mov"
    # VIDEO_PATH = "/Users/amirgheser/SIV/project/test/video/video2.mp4"
    # VIDEO_PATH = "/Users/amirgheser/SIV/project/test/video/rotated_board.mp4"
    avg_grids = mp.Queue()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if DEBUG:
        coords = CORNERS
    else:
        coords = get_corner_coordinates(cap)
    cap.release()
    stop_process = Value('i', False)
    process1 = mp.Process(target=show_video, args=(avg_grids, VIDEO_PATH, coords, stop_process))
    process2 = mp.Process(target=get_average_grids, args=(avg_grids, VIDEO_PATH, coords, stop_process))

    process1.start()
    process2.start()
    process1.join()
    process2.join()