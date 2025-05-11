import cv2
import time
from ultralytics import YOLO
import math
import numpy as np


class YoloDetector:
    def __init__(self, device="cuda"):
        self.model = YOLO('/home/myseo/study/book_recognition/libro_server/libro_ai_server/src/book_recognition/book_recognition/models/yolo11s_best.pt').to(device)

    def get_angle(self, pt1, pt2):
        rad = math.atan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
        return rad, math.degrees(rad)

    def run_yolo(self, img):
        yolo_pts_list = []
        book_location_list = []
        book_gradient_list = []

        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        start = time.time()
        results = self.model(img, verbose=False)
        print(f"Yolo Inference Time: {time.time() - start:.4f} seconds")

        for result in results:
            if result.obb is not None:
                obb_polygons = [
                    polygon for polygon, cls in zip(result.obb.xyxyxyxy, result.obb.cls.tolist())
                    if int(cls) == 0
                ]
                for polygon in obb_polygons:
                    pts = polygon.cpu().numpy().astype(np.int32).reshape(4, 2)
                    yolo_pts_list.append(pts)
                    center = np.mean(pts, axis=0)
                    book_location_list.append((center[0]/2, center[1]/2))  # 확대한 좌표 원위치
                    sorted_pts = sorted(pts, key=lambda pt: self.get_angle(pt, center)[0])
                    _, deg = self.get_angle(sorted_pts[3], sorted_pts[0])
                    book_gradient_list.append(deg)
        return yolo_pts_list, book_location_list, book_gradient_list
