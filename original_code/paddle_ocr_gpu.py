import cv2
from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt
import time
import paddle

# PaddleOCR 객체 생성 (한국어 지원, cls=True: 방향 분류 사용)

ocr = PaddleOCR(lang="korean", use_angle_cls=True, show_log=False)

gpu_available = paddle.device.is_compiled_with_cuda()
print(f'using gpu: ', gpu_available)


# 이미지 경로
image_path = './test_image/image7.jpg'

# OpenCV로 이미지 불러오기
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB로 변환 (PaddleOCR는 RGB 사용)

# OCR 수행
start_time = time.time()
results = ocr.ocr(img_rgb, cls=True)
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference Time: {inference_time:.4f} seconds")

# 텍스트 추출 및 출력
for line in results:
    for word_info in line:
        box, (text, score) = word_info
        print(f"인식된 텍스트: {text}, 좌표: {box[0]}, 신뢰도: {score:.2f}")
        print(f"위치: {box}")

# OCR 결과 이미지에 표시
boxes = [word_info[0] for line in results for word_info in line]
texts = [word_info[1][0] for line in results for word_info in line]
scores = [word_info[1][1] for line in results for word_info in line]

# 결과 이미지 시각화
image_with_boxes = draw_ocr(img_rgb, boxes, texts, scores, font_path='/home/addinedu/github/book_recognition/libro_server/libro_ai_server/src/book_recognition/book_recognition/resources/MaruBuri-Bold.ttf')

# OpenCV 이미지 출력
plt.figure(figsize=(10, 10))
plt.imshow(image_with_boxes)
plt.axis('off')
plt.title("OCR 결과")
plt.show()

