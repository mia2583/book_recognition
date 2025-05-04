import os
import requests
from urllib.parse import quote_plus
import paddle
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
import time
from ultralytics import YOLO
import torch
from paddleocr import PaddleOCR
import math
import numpy as np
import time


def run_ocr(img_path, ocr_model):
    """
    주어진 이미지 경로와 OCR 모델을 사용하여 이미지 내의 글자를 인식하는 함수

    입력: 
        img_path: 이미지 파일 경로
        ocr_model: OCR 모델
    반환:
        result: 인식한 글자의 좌표와 글자, 신뢰도. 이때 좌표는 이미지를 2배 확대했을 때의 좌표로 반환.
            result는 아래와 같은 형태로 되어 있다.
                [[   [   [[x, y], [좌표2], [좌표3], [좌표4]],   ('인식한_글자', 0.9981935620307922)   ], ...
    """

    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    start_time = time.time()
    results = ocr_model.ocr(img, cls=True)

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"OCR Inference Time: {inference_time:.4f} seconds")
    
    
    return results


def draw_ocr_coords_and_show_image(img_path, ocr_result):
    """
    주어진 이미지 경로와 OCR 결과를 바탕으로 이미지에 OCR의 박스와 텍스트를 그리는 함수

    입력: 
        img_path: 이미지 파일 경로
        ocr_result: OCR 결과
            ocr_result는 아래와 같은 형태로 되어 있다.
                [[   [   [[x, y], [좌표2], [좌표3], [좌표4]],   ('이', 0.9981935620307922)   ], ...
    """
    # 이미지 읽기
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 좌표 색상 설정
    font = ImageFont.truetype("/home/myseo/study/book_recognition/libro_server/libro_ai_server/src/book_recognition/book_recognition/resources/MaruBuri-Bold.ttf", 20)

    # OCR 좌표 그리기
    for line in ocr_result:
        # 인식된 글자가 없으면 이미지만 출력하도록 한다
        if line == None: 
            break
        for ocr_item in line:
            pts, text = ocr_item
            
            # 정확도를 제외하고 텍스트만 표시
            if isinstance(text, tuple):  # (텍스트, 정확도)로 저장되어 있을 때
                text = text[0]
            
            x1, y1 = pts[0]  # 좌측 상단
            x2, y2 = pts[2]  # 우측 하단
            draw.rectangle([x1, y1, x2, y2], outline=(250, 0, 0), width=2)  # 빨간색 사각형
            draw.text((x1, y1), text, font=font, fill=(255, 0, 0))  # 사각형의 왼쪽 상단에 빨간색 텍스트

    # 이미지 창
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imshow('Image with OCR Coordinates and Text', img)
    cv2.waitKey(0)  # 키 입력 대기
    cv2.destroyAllWindows()



def run_ocr_yolo_api():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_available = paddle.device.is_compiled_with_cuda()

    # 설정
    img_path = '/home/myseo/study/book_recognition/libro_server/libro_ai_server/src/book_recognition/book_recognition/resources/image0.jpg'
    ocr_model = PaddleOCR(lang="korean", use_gpu=gpu_available, use_angle_cls=True, show_log=False)
    yolo_model = YOLO('/home/myseo/study/book_recognition/libro_server/libro_ai_server/src/book_recognition/book_recognition/models/yolo11s_best.pt').to(device)

    # gpu 확인 코드
    print(f'gpu option: 1. yolo gpu: {yolo_model.device}, 2. ocr gpu: {paddle.get_device()}')

    ocr_result = run_ocr(img_path, ocr_model)  # ocr 처리
    draw_ocr_coords_and_show_image(img_path, ocr_result)
    print_ocr_result(ocr_result)

    return 

if __name__ == '__main__':
    run_ocr_yolo_api()