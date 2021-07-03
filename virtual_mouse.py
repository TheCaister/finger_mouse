import cv2
import numpy as np
import hand_tracking_module as htm
import time
import autopy

width_cam, height_cam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)

while True:
    success, img = cap.read()
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    

    cv2.imshow("Image", img)
    cv2.waitKey(1)