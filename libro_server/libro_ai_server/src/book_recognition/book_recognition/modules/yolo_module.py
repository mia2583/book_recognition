from book_recognition.config.config import yolo_model
import cv2
import numpy as np
import math
from pathlib import Path
import time

# ============================시각화를 위한 것. 최종 코드에서는 삭제=============================

def visualize_mask_yolo(img, idx): 
    save_path = Path(f"./yolo_test/mask_yolo{idx}.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, img)


def visualize_yolo(img, points, center, angle, idx):
    img_with_infos = img.copy()
    line_length = 200
    # 색 : 순서대로 빨, 초, 파, 보
    colors = [(0, 0, 255), (0, 255, 0),  (255, 0, 0), (255, 0, 255)] 

    # 책의 중심에서 90도 방향의 점
    x1 = int(center[0] + line_length * math.cos(math.radians(90)))
    y1 = int(center[1] + line_length * math.sin(math.radians(90)))

    # 책의 중심에서 책 기울기의 방향의 점
    x2 = int(center[0] + line_length * math.cos(math.radians(angle)))
    y2 = int(center[1] + line_length * math.sin(math.radians(angle)))

    # 책의 중심점 시각화 (초록색)
    cv2.circle(img_with_infos, (int(center[0]), int(center[1])), radius=5, color=(0, 255, 0), thickness=-1)
    # 책의 중심점에서 90도 방향을 나타내는 선 (빨간색)
    cv2.line(img_with_infos, (int(center[0]), int(center[1])), (x1, y1), (0, 0, 255), 2)
    # 책의 중심점에서 책의 방향을 나타내는 선 (파란색)
    cv2.line(img_with_infos, (int(center[0]), int(center[1])), (x2, y2), (255, 0, 0), 2)
    
    # yolo 사각형 점 (왼쪽 상단 -> 오른쪽 상단 -> 오른쪽 하단 -> 왼쪽 하단), 색상은 (빨, 초, 파, 보) 순서
    for i, pt in enumerate(points):
        color = colors[i % len(colors)]
        cv2.circle(img_with_infos, (int(pt[0]), int(pt[1])), radius=5, color=color, thickness=-1)
    
    save_path = Path(f'./yolo_test/yolo_with_info{idx}.jpg')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, img_with_infos)

# ============================시각화를 위한 것. 최종 코드에서는 삭제=============================


def get_angle_between_points(pt1, pt2):
    angle_rad = math.atan2(pt1[1] -pt2[1], pt1[0] - pt2[0])  # y/x의 아크탄젠트
    angle_deg = math.degrees(angle_rad) # 라디안을 도로 변환
    return angle_rad, angle_deg



def run_yolo(img_path):
    # yolo 처리 이후 이미지
    after_yolo_img_list = []
    # 책 위치 정보 리스트
    book_location_list = []
    # 책 기울기 정보 리스트
    book_gradient_list = []

    img = cv2.imread(img_path)

    # 추론 시작 시간 측정
    start_time = time.time()

    results = yolo_model(img_path)

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time:.4f} seconds")

    for result in results:
        if result.obb is not None:
            # cls가 0인 폴리곤만 필터링 [1][2]
            obb_polygons = [
                polygon
                for polygon, cls in zip(result.obb.xyxyxyxy, result.obb.cls.tolist())
                if int(cls) == 0  # 클래스 0만 선택
            ]

            # 필터링된 폴리곤 처리 [1][2]
            for i, polygon in enumerate(obb_polygons):
                # pts는 yolo가 적용된 사각형의 꼭짓점
                # rect는 pts를 감싸는 최소 사각형 바운딩
                pts = polygon.cpu().numpy().astype(np.int32).reshape(4,2)
                rect = cv2.boundingRect(pts)
                x, y, w, h = rect  # 여기서 x, y는 rect 사각형의 왼쪽 상단 (x,y)

                # 마스크 부분 외(배경화면)를 검게하고 원하는 이미지 부분만 잘라서 시각화 
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [pts], -1, 255, -1)
                cropped = img[y:y+h, x:x+w]
                mask_crop = mask[y:y+h, x:x+w]
                mask_yolo_img = cv2.bitwise_and(cropped, cropped, mask=mask_crop)
                after_yolo_img_list.append(mask_yolo_img)
            

                # 중심점 구하기
                center = np.mean(pts, axis=0)
                book_location_list.append((center[0], center[1]))

                # 꼭짓점 나열 순서를 각도를 기준으로 정렬 : 왼쪽 상단 -> 오른쪽 상단 -> 오른쪽 하단 -> 왼쪽 하단
                sorted_pts = sorted(pts, key=lambda pt: get_angle_between_points(pt, center)[0])

                # 책의 기울기 계산
                rad, deg = get_angle_between_points(sorted_pts[3], sorted_pts[0])
                book_gradient_list.append(deg)
                
                # ============================시각화를 위한 것. 최종 코드에서는 삭제=============================
                visualize_mask_yolo(mask_yolo_img, i)
                visualize_yolo(img, sorted_pts, center, deg, i)
                # ============================시각화를 위한 것. 최종 코드에서는 삭제=============================
   
    return after_yolo_img_list, book_location_list, book_gradient_list
