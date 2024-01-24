import cv2
import numpy as np
import matplotlib.pyplot as plt

from project.SIV.show_video import get_average_grid

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

def hasPiece():
    pass

if __name__ == '__main__':
    grid = get_average_grid()
    v_lines, h_lines = grid

    cap = cv2.VideoCapture('/Users/amirgheser/SIV/project/test/video/IMG_0389.mov')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        board = [[0 for _ in range(8)] for _ in range(8)]
        
        for i in range(8):
            for j in range(8):
                top_left = intersect_in(v_lines[i], h_lines[j])
                top_right = intersect_in(v_lines[i+1], h_lines[j])
                bottom_right = intersect_in(v_lines[i+1], h_lines[j+1])
                bottom_left = intersect_in(v_lines[i], h_lines[j+1])
                square = np.array([top_left, top_right, bottom_right, bottom_left])
                if hasPiece(frame, square):
                    board[i][j] = 1
        plt.imshow(frame)
    