import cv2
import numpy as np
import easyocr

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # 회전 변환 행렬 계산
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 회전 후 이미지 크기 계산
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    # 변환 행렬 조정
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    # 이미지 회전
    return cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255,255,255))

# 이미지 읽기
img = cv2.imread('test_image/image1.jpg')
if img is None:
    print("이미지 파일을 읽을 수 없습니다. 경로와 파일 존재 여부를 확인하세요.")
    exit()

# easyocr Reader 생성 (한국어+영어, 메시지 숨김)
reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)

angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 45도 간격 회전

for angle in angles:
    rotated_img = rotate_image(img, angle)
    results = reader.readtext(rotated_img)

    height, width, _ = rotated_img.shape
    new_width = width + 1000
    new_img = np.ones((height, new_width, 3), dtype=np.uint8) * 255
    new_img[:, :width] = rotated_img

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3  # 폰트 크기
    color = (10, 0, 0)
    thickness = 4
    line_height = 100
    start_y = 100

    texts = []
    for bbox, text, conf in results:
        texts.append(text)
        # 바운딩 박스 좌표 추출
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(new_img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    for idx, text in enumerate(texts, 1):
        y = start_y + (idx - 1) * line_height
        cv2.putText(new_img, f"{idx},{text}", (width + 10, y), font, font_scale, color, thickness)

    # 각도별로 결과 이미지 저장
    cv2.imwrite(f'ocr_result_with_text_easyocr_angle_{angle}.jpg', new_img)
    print(f"OCR 결과 저장: ocr_result_with_text_easyocr_angle_{angle}.jpg")

print("모든 각도에 대해 OCR이 완료되었습니다.")
