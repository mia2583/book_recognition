from book_recognition.config.config import pkg_path, ocr_model
import paddle

from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import numpy as np
from pathlib import Path
import time

# PaddleOCR 객체 생성 (한국어 지원, cls=True: 방향 분류 사용)


def run_ocr(image_list):
    # ===========gpu 확인 코드, 최종 코드에서는 삭제==================
    gpu_available = paddle.device.is_compiled_with_cuda()
    print(f'using gpu: ', gpu_available)
    #==========================================================
    
    search_keyword_list = []

    for img in image_list:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        start_time = time.time()
        results = ocr_model.ocr(img, cls=True)

        end_time = time.time()
        inference_time = end_time - start_time
        print(f"OCR Inference Time: {inference_time:.4f} seconds")

        # PIL 형식으로 변환
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(os.path.join(pkg_path, 'resources', 'MaruBuri-Bold.ttf'), 20)

        # ocr로 추출한 단어 저장
        extracted_words = ""

        # 텍스트 검출 및 박스 그리기
        for line in results:
            for word_info in line:
                box, (text, score) = word_info
                # print(f"인식된 텍스트: {text}, 좌표: {box}, 신뢰도: {score:.2f}")
                top_left, bottom_right = tuple(map(int, box[0])), tuple(map(int, box[2]))

                extracted_words += text

                # 박스 그리기
                draw.rectangle([top_left, bottom_right], outline=(250, 0, 0), width=2)
                draw.text(top_left, text, font=font, fill=(0, 0, 255))

        search_keyword_list.append(extracted_words)

        # ocr 결과 이미지 저장
        save_path = Path(f'./ocr_test/{extracted_words}.jpg')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)

    return search_keyword_list
