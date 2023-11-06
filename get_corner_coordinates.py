import cv2

def mouse_callback(event, x, y, flags, params):
    '''
    :param: event - the type of mouse event
    :param: x - the x coordinate of the mouse click
    :param: y - the y coordinate of the mouse click
    :param: flags - any relevant flags passed by OpenCV
    :param: params - params = (frame, corners)
    '''
    frame , corners = params
    if len(corners) == 4: 
        print('You have already selected the four corners of the chessboard.')
        return
    if event == cv2.EVENT_LBUTTONUP:
        # Draw a circle at the location of the mouse click
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        # Add the coordinates of the circle to the list of circles
        corners.append((x, y))
        # Display the image to the user
        cv2.imshow('Chessboard', frame)

def get_corner_coordinates(cap):
    # Read the first frame
    ret, frame = cap.read()
    original_frame = frame.copy()
    corners = []
    cv2.namedWindow('Chessboard')
    cv2.setMouseCallback('Chessboard', mouse_callback, (frame, corners))
    # cv2.startWindowThread()

    # Display the image to the user
    cv2.imshow('Chessboard', frame)
    # Prompt the user to move the circles to the four corners of the chessboard
    print('Move the circles to the four corners of the chessboard and press any spacebar when done.')

    while True:
        if cv2.waitKey(0) & 0xFF == ord(' '):
            if len(corners) == 4:
                print("Successfully confirmed 4 corners")
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                break
            else:
                print("You must select 4 corners. Try again from scratch")
                corners = []
                frame = original_frame.copy()
                cv2.setMouseCallback('Chessboard', mouse_callback, (frame, corners))  
                cv2.waitKey(1)
                cv2.imshow('Chessboard', frame)

    for corner in corners:
        print(f"{corner[0]}, {corner[1]}")
    corners.sort(key=lambda x: x[0])
    # first two values are left corners
    if corners[0][1] < corners[1][1]:
        left_top = corners[0]
        left_bottom = corners[1]
    else:
        left_top = corners[1]
        left_bottom = corners[0]
    # last two values are right corners
    if corners[2][1] < corners[3][1]:
        right_top = corners[2]
        right_bottom = corners[3]
    else:
        right_top = corners[3]
        right_bottom = corners[2]
    
    cv2.waitKey(1)
    return left_top, right_top, right_bottom, left_bottom


if __name__ == '__main__':
    print("Starting get_corner_coordinates.py MAIN")
    video_path = '/Users/amirgheser/SIV/project/test/video/IMG_0389.mov'
    cap = cv2.VideoCapture(video_path)
    topL, topR, botR, botL = get_corner_coordinates(cap)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()    
    # draw 4 lines connecting the corners
    cv2.line(frame, topL, topR, (0, 0, 255), 2)
    cv2.line(frame, topR, botR, (0, 0, 255), 2)
    cv2.line(frame, botR, botL, (0, 0, 255), 2)
    cv2.line(frame, botL, topL, (0, 0, 255), 2)
    cv2.waitKey(1000)
    cv2.imshow('Chessboard', frame)
    cv2.waitKey(1000)
    cap.release()

