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

GRID_AMT_AVG = 300

def get_average_grids(avg_grids, video_path, coords):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    grid = None
    grids = []
    while True:
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

def show_video(avg_grids, video_path, coords):
    print("\033[92mStarting video\033[00m")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000/fps)
    frame_count = 0
    grid = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            if frame_count % GRID_AMT_AVG == 0:
                grid = avg_grids.get()
            v_lines, h_lines = grid[:9], grid[9:]
            frame = transform_frame(coords, frame)
            frame = draw_lines(frame, v_lines, color=(0, 0, 255))
            frame = draw_lines(frame, h_lines, color=(0, 0, 255))
            cv2.imshow("frame", frame)
            print("Frame: {}/{}".format(frame_count, cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_count += 1
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    VIDEO_PATH = "/Users/amirgheser/SIV/project/test/video/IMG_0389.mov"
    # VIDEO_PATH = "/Users/amirgheser/SIV/project/test/video/video2.mp4"
    # VIDEO_PATH = "/Users/amirgheser/SIV/project/test/video/rotated_board.mp4"
    avg_grids = mp.Queue()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    coords = get_corner_coordinates(cap)
    cap.release()

    process1 = mp.Process(target=show_video, args=(avg_grids, VIDEO_PATH, coords))
    process2 = mp.Process(target=get_average_grids, args=(avg_grids, VIDEO_PATH, coords))

    process1.start()
    process2.start()
    process1.join()
    process2.join()