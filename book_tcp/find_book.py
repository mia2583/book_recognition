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


# yolo_model = YOLO(os.path.join(pkg_path, 'models', 'yolo11n-obb-only-book.pt'))
# yolo_model = YOLO(os.path.join(pkg_path, 'models', 'yolo11n-obb-book-and-label.pt'))


def run_ocr(img_path, ocr_model):
    search_keyword_list = []

    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    start_time = time.time()
    results = ocr_model.ocr(img, cls=True)

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"OCR Inference Time: {inference_time:.4f} seconds")
    
    return results


def get_angle_between_points(pt1, pt2):
    angle_rad = math.atan2(pt1[1] -pt2[1], pt1[0] - pt2[0])  # y/x의 아크탄젠트
    angle_deg = math.degrees(angle_rad) # 라디안을 도로 변환
    return angle_rad, angle_deg



def run_yolo(img_path, yolo_model):
    # yolo 처리 이후 이미지
    yolo_pts_list = []
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
    print(f"Yolo Inference Time: {inference_time:.4f} seconds")

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
                # rect = cv2.boundingRect(pts)
                # x, y, w, h = rect  # 여기서 x, y는 rect 사각형의 왼쪽 상단 (x,y)
                yolo_pts_list.append(pts)        

                # 중심점 구하기
                center = np.mean(pts, axis=0)
                book_location_list.append((center[0], center[1]))

                # 꼭짓점 나열 순서를 각도를 기준으로 정렬 : 왼쪽 상단 -> 오른쪽 상단 -> 오른쪽 하단 -> 왼쪽 하단
                sorted_pts = sorted(pts, key=lambda pt: get_angle_between_points(pt, center)[0])

                # 책의 기울기 계산
                rad, deg = get_angle_between_points(sorted_pts[3], sorted_pts[0])
                book_gradient_list.append(deg)
                
    return yolo_pts_list, book_location_list, book_gradient_list


def seperate_keyword_by_book(search_keyword_list, yolo_pts_list):
    



# API에 책 검색
def search_books_in_google(search_texts_list):
    api_key = ""  # Google Books API 키 (필요한 경우 입력)
    book_results = []

    for search_texts in search_texts_list:
        # 모든 추출된 텍스트를 기반으로 검색 시도
        found = False

        # 한글 제목을 URL 인코딩
        encoded_title = quote_plus(search_texts)

        # Google Books API 호출
        url = f"https://www.googleapis.com/books/v1/volumes?q={encoded_title}&langRestrict=ko"
        if api_key:
            url += f"&key={api_key}"

        try:
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                if "items" in data and len(data["items"]) > 0:
                    book_info = data["items"][0]["volumeInfo"]
                    result = {
                        "search_text": search_texts,
                        "title": book_info.get("title", "Unknown"),
                        "authors": book_info.get("authors", ["Unknown"]),
                        "publisher": book_info.get("publisher", "Unknown"),
                        "published_date": book_info.get("publishedDate", "Unknown"),
                        "description": book_info.get(
                            "description", "No description available"
                        ),
                        "thumbnail": book_info.get("imageLinks", {}).get(
                            "thumbnail", "No image"
                        ),
                    }
                    book_results.append(result)
                    found = True
                    # print(f"  - 검색 성공: {result['title']}")
        except Exception as e:
            # print(f"  - 검색 중 오류 발생: {str(e)}")
            continue

        # 검색 실패한 경우
        if not found:
            book_results.append(
                {
                    "search_text": search_texts,
                    "title": "검색 실패: No results found",
                }
            )
            # print(f"  - 검색 실패: 결과를 찾을 수 없습니다.")

    return book_results






def run_ocr_yolo_api():
    
    # gpu 확인 코드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_available = paddle.device.is_compiled_with_cuda()
    print(f'gpu option: 1. yolo {device}, 2. ocr gpu: {gpu_available}')

    # 설정
    img_path = 'libro_server/libro_ai_server/src/book_recognition/book_recognition/resources/image0.jpg'
    ocr_model = PaddleOCR(lang="korean", use_gpu=gpu_available, use_angle_cls=True, show_log=False)
    yolo_model = YOLO('./yolo11s_best.pt').to(device)


    search_keyword_list = run_ocr(img_path, ocr_model)

    yolo_pts_list, book_location_list, book_gradient_list = run_yolo(img_path, yolo_model)

    seperate_keyword_by_book(search_keyword_list, yolo_pts_list)


    
    # if len(after_yolo_img_list) == 0 :
    #     print('No book found')
    #     return 
    
    # search_keyword_list = run_ocr(after_yolo_img_list)

    # if len(search_keyword_list) == 0 :
    #     print('No word detected')
    #     return 

    # book_results_list = search_books_in_google(search_keyword_list)

    # return book_results_list, book_location_list, book_gradient_list




