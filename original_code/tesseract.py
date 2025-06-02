import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Tesseract 경로 설정 (윈도우 사용자만 필요)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255,255,255))

# 이미지 읽기
img = cv2.imread('test_image/image1.jpg')
if img is None:
    print("이미지 파일을 읽을 수 없습니다.")
    exit()

angles = [0, 45, 90, 135, 180, 225, 270, 315]

for angle in angles:
    rotated_img = rotate_image(img, angle)
    
    # Tesseract OCR 수행 (한국어+영어)
    custom_config = r'--oem 3 --psm 6 -l kor+eng'
    data = pytesseract.image_to_data(rotated_img, config=custom_config, output_type=Output.DICT)
    
    # 결과 처리
    height, width, _ = rotated_img.shape
    new_width = width + 1000
    new_img = np.ones((height, new_width, 3), dtype=np.uint8) * 255
    new_img[:, :width] = rotated_img

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    color = (10, 0, 0)
    thickness = 4
    line_height = 100
    start_y = 100

    texts = []
    for i in range(len(data['text'])):
        if float(data['conf'][i]) > 60:  # 신뢰도 60% 이상만 처리
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            text = data['text'][i].strip()
            if text:
                texts.append(text)
                # 바운딩 박스 그리기 (사각형)
                cv2.rectangle(new_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # 오른쪽에 텍스트 나열
    for idx, text in enumerate(texts, 1):
        y_pos = start_y + (idx - 1) * line_height
        cv2.putText(new_img, f"{idx},{text}", (width + 10, y_pos), font, font_scale, color, thickness)

    cv2.imwrite(f'tesseract_result_angle_{angle}.jpg', new_img)
    print(f"Tesseract 결과 저장: tesseract_result_angle_{angle}.jpg")

print("모든 각도에 대해 OCR이 완료되었습니다.")
