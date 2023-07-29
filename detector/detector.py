import math
import cv2
from sort import *
import telepot
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

points = []
def handle_left_click(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def draw_polygon (frame, points):
    for point in points:
        frame = cv2.circle( frame, (point[0], point[1]), 5, (0,0,255), -1)

    frame = cv2.polylines(frame, [np.int32(points)], False, (255,0, 0), thickness=2)
    return frame

class Tracker():
    def __init__(self, model, tracker, classNames, token="6028302841:AAGZvHAvCnP-aPwmMMCH1EYBoP8InYrd-jU", chatid="6022870988"):
        self.model = model
        self.tracker = tracker
        self.idList = []
        self.chatid = chatid
        self.token = token
        self.classNames = classNames
        self.tracker = tracker

    def sendPhoto(self, photoPath):
        try:
            bot = telepot.Bot(self.token)
            bot.sendPhoto(self.chatid,  photo=open(photoPath, 'rb'), caption="Warning!")
        except Exception as ex:
            print("Error",ex)
        else:
            print("success")

    def isInside(self, points, centroid):
        polygon = Polygon(points)
        centroid = Point(centroid)
        return polygon.contains(centroid)

    def detect(self, img, bbox=True, color=(255, 0, 0)):

        results = self.model(img)
        detections = np.empty((0, 5))
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                if bbox:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                currentClass = self.classNames[cls]

                if currentClass == "person" and conf >= 0.3:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        return detections

    def track(self, img, detections, points, img1,bbox=True, color=(0, 0, 255)):
        resultsTracker = self.tracker.update(detections)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            if bbox:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            centroid = (cx, cy)

            if self.isInside(points, centroid):
                cv2.putText(img, "warning", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
                crop_img = img1[y1:y1 + h, x1:x1 + w]

                if self.idList.count(id) == 0:
                    self.idList.append(id)
                    cv2.imwrite(f"/Users/macbook/Documents/DO_AN/1921050631_NguyenVanTruong/detected_images/intruder{str(int(id))}.png", crop_img)
                    self.sendPhoto(f"/Users/macbook/Documents/DO_AN/1921050631_NguyenVanTruong/detected_images/intruder{str(int(id))}.png")

