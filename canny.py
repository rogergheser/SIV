import cv2
import numpy as np
import math
# VIDEO_PATH = "/Users/amirgheser/SIV/project/test/video/IMG_0389.mov"
# VIDEO_PATH = "/Users/amirgheser/SIV/project/test/video/video2.mp4"
VIDEO_PATH = '/Users/amirgheser/SIV/project/test/video/rotated_board.mp4'

cap = cv2.VideoCapture(VIDEO_PATH)

def hough_image(frame):
    hor_deg = 180
    hor_err = 2
    ver_deg = 90
    ver_err = 30

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, 0, 50, L2gradient=True)
    hough_image = edges.copy()
    lines = cv2.HoughLines(hough_image, 1, np.pi/180, 500, None, 0, 0) #math.radians(ver_deg-ver_err), math.radians(ver_deg+ver_err))
    # this can be improved since we know that the chessboard is centered
    # I will now define ranges in which the lines can be detected
    # Horizontal lines with a error delta of 2 degrees
    # Vertical lines with a error delta of 30 degrees
    if lines is not None:
        for rho, theta in lines[:,0]:
            # if theta > math.radians(hor_deg - hor_err) and theta < math.radians(hor_deg + hor_err) or \
            #     theta > math.radians(ver_deg - ver_err) and theta < math.radians(ver_deg + ver_err):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(hough_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    frame[hough_image != 0] = (0, 0, 255)
    return frame
    # return hough_image
if __name__ == '__main__':
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if ret:
            height , width = frame.shape[:2]
            start_row, start_col = int(height * .65), int(width * .25)
            frame = frame[start_row:height, start_col:int(width*0.8)]
            original_frame = frame
            frame = cv2.filter2D(frame, -1, kernel=cv2.getGaussianKernel(10, 0))
            frame = cv2.convertScaleAbs(frame, alpha=2, beta=0)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_frame, 50, 110, L2gradient=True)
            cv2.imshow("Edges", edges)


            hough_image = edges.copy()
            lines = cv2.HoughLines(hough_image, 2, np.pi/180, 300, None, 0, 0)
            hough_image = cv2.filter2D(hough_image, -1, kernel=cv2.getGaussianKernel(len(hough_image)//8, 0.1))

            if lines is not None:
                for i in range(0, len(lines)):
                    rho = lines[i][0][0]
                    theta = lines[i][0][1]
                    # I want to draw a circle at the intersection of the lines
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    # x1 stores the rounded off value of (r * cos(theta) - 1000 * sin(theta))
                    x1 = int(x0 + 1000 * (-b))
                    # y1 stores the rounded off value of (r * sin(theta)+ 1000 * cos(theta))
                    y1 = int(y0 + 1000 * (a))
                    # x2 stores the rounded off value of (r * cos(theta)+ 1000 * sin(theta))
                    x2 = int(x0 - 1000 * (-b))
                    # y2 stores the rounded off value of (r * sin(theta)- 1000 * cos(theta))
                    y2 = int(y0 - 1000 * (a))
                    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
                    # (0,0,255) denotes the colour of the line to be
                    # drawn. In this case, it is red.
                    cv2.line(hough_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # All the changes made in the input image are finally


            original_frame[hough_image != 0] = (0, 0, 255)
            cv2.imshow("Hough", original_frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
    cap.release()
    cv2.destroyAllWindows()
