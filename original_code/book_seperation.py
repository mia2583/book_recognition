from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# 1. YOLO 모델 로드
model = YOLO("./yolo11s_best.pt")

# 2. 이미지 읽기
img_path = "./test_image/image6.jpg"
img = cv2.imread(img_path)

results = model(img_path)

for result in results:
    # print(result.names)
    if result.obb is not None:
        # cls가 0인 폴리곤만 필터링 [1][2]
        obb_polygons = [
            polygon
            for polygon, cls in zip(result.obb.xyxyxyxy, result.obb.cls.tolist())
            if int(cls) == 0  # 클래스 0만 선택
        ]

        # 필터링된 폴리곤 처리 [1][2]
        for i, polygon in enumerate(obb_polygons):
            pts = polygon.cpu().numpy().astype(np.int32)
            rect = cv2.boundingRect(pts)
            x, y, w, h = rect

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [pts], -1, 255, -1)
            cropped = img[y:y+h, x:x+w]
            mask_crop = mask[y:y+h, x:x+w]

            result_img = cv2.bitwise_and(cropped, cropped, mask=mask_crop)
            save_path = Path(f"./yolo_test_image1/{i}.png")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path, result_img)