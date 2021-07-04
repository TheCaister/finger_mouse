import cv2
import numpy as np
import hand_tracking_module as htm
import time
import autopy

# Storing webcam dimensions in variables
width_cam, height_cam = 640, 480
frame_reduction = 100
smoothening = 5

# Setting up webcam
cap = cv2.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)

# For calculating FPS
previous_time = 0

# For smoothening mouse movements
previous_location_x, previous_location_y = 0, 0
current_location_x, current_location_y = 0, 0

# Making hand detector and it can only detect one hand
detector = htm.HandDetector(max_hands=1)

# Getting width and height of the monitor screen
width_screen, height_screen = autopy.screen.size()

while True:
    # Reading webcam
    success, img = cap.read()
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = detector.find_hands(img)

    # Getting landmarks and bounding box coordinates
    landmarks_list, bounding_box = detector.find_position(img)

    if len(landmarks_list) != 0:
        # Get coordinates of tip of index and middle finger
        x1, y1 = landmarks_list[8][1:]
        x2, y2 = landmarks_list[12][1:]

    # Get list of fingers that are up/down
    fingers = detector.fingers_up()

    # Drawing a region where you can move your fingers
    cv2.rectangle(img, (frame_reduction, frame_reduction),
                  (width_cam, frame_reduction, height_cam - frame_reduction),
                  (255, 0, 255), 2)

    # If index finger is up and middle finger is down
    if fingers[1] == 1 and fingers[2] == 0:
        cv2.rectangle(img, (frame_reduction, frame_reduction),
                      (width_cam, frame_reduction, height_cam - frame_reduction),
                      (255, 0, 255), 2)
        # Interpolate fingertip coordinates from webcam to monitor screen, then moving the mouse
        x3 = np.interp(x1, (frame_reduction, width_cam - frame_reduction), (0, width_screen))
        y3 = np.interp(x1, (frame_reduction, height_cam - frame_reduction), (0, height_screen))

        # Smoothening the mouse by
        # Adding the previous location to the difference between the interpolated location and
        # the previous location divided by the smoothening value
        # The cursor decelerates towards the intended position for a smooth effect
        current_location_x = previous_location_x + (x3 - previous_location_x) / smoothening
        current_location_y = previous_location_y + (y3 - previous_location_y) / smoothening

        # Inverting the x direction for more intuitive control
        autopy.mouse.move(width_screen - current_location_x, current_location_y)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        previous_location_x, previous_location_y = current_location_x, current_location_y

    # If index finger and middle finger are up
    if fingers[1] == 1 and fingers[2] == 1:
        # Getting distance between the 2 fingertips
        length, img, line_info = detector.find_distance(8, 12, img)

        # If the 2 tips are close enough, draw a green circle between then
        if length < 40:
            cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()

    # Displaying FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                3, (255, 0, 0), 3)
    # Displaying webcam
    cv2.imshow("Image", img)
    cv2.waitKey(1)