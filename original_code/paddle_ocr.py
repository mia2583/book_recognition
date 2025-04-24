import cv2
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# PaddleOCR 리더 생성(한글 'korean' 설정)
ocr = PaddleOCR(lang='korean')

# 이미지 로드 및 업스케일링
image_path = '/home/addinedu/test/yolo_test/0.png'
image = cv2.imread(image_path)
upscaled_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# PIL 형식으로 변환
upscaled_image_pil = Image.fromarray(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(upscaled_image_pil)
font = ImageFont.truetype("/home/addinedu/Downloads/maruburi/MaruBuriTTF/MaruBuri-Bold.ttf", 60)

# 텍스트 검출 및 박스 그리기
for line in ocr.ocr(np.array(upscaled_image_pil), cls=True):
    for word_info in line:
        box, (text, score) = word_info[0], word_info[1]
        top_left, bottom_right = tuple(map(int, box[0])), tuple(map(int, box[2]))
        bottom_left = (top_left[0], bottom_right[1])

        # 텍스트 및 확률 출력
        # print(f"Detected text: {text} (Probability: {score})")
        print(text)

        # 박스 그리기
#         draw.rectangle([top_left, bottom_right], outline=(250, 0, 0), width=2)
#         draw.text(bottom_left, text, font=font, fill=(0, 255, 0))

# # 결과 이미지 저장
# result_path = os.path.join('/home/addinedu/test/ocr_test', f'result.png')
# upscaled_image_pil.convert("RGB").save(result_path)