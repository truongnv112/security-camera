from detector.detector import *
import cv2
from ultralytics import YOLO

model = YOLO("/Users/macbook/Documents/DO_AN/1921050631_NguyenVanTruong/models/detector.pt")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
classNames = ['person']

cap = cv2.VideoCapture("/Users/macbook/Documents/DO_AN/1921050631_NguyenVanTruong/videos/people.mp4")
trackerobj = Tracker(model,tracker,classNames)

detect = False
while True:
    success,img = cap.read()
    img1 = img.copy()
    img = draw_polygon(img, points)
    if detect:
        detections = trackerobj.detect(img)
        trackerobj.track(img, detections, points,img1)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('d'):
        points.append(points[0])
        detect = True
    #img2 = cv2.flip(img, 1)
    cv2.imshow("Display", img)
    cv2.setMouseCallback('Display', handle_left_click, points)
    cv2.waitKey(1)