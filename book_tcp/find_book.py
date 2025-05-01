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

    # PIL 형식으로 변환
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(img)
    
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
            draw.text(top_left, text, fill=(0, 0, 255))

    search_keyword_list.append(extracted_words)

    # ocr 결과 이미지 저장
    save_path = Path(f'./ocr_test/{extracted_words}.jpg')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)
    
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
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 추론 시작 시간 측정
    start_time = time.time()

    results = yolo_model(img)

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


def seperate_keyword_by_book(ocr_result, yolo_pts_list):
    # 결과 저장 리스트 (책 1개당 OCR 결과를 따로 저장)
    books_ocr_result = [[] for _ in range(len(yolo_pts_list))]
    books_ocr_text_only = ['' for _ in range(len(yolo_pts_list))]

    for line in ocr_result:
        for ocr_item in line:
            ocr_coords, (text, conf) = ocr_item

            # OCR 중심점 계산 (사각형 꼭짓점 평균)
            ocr_center = np.mean(ocr_coords, axis=0).astype(int)

            # 각 책 박스에 대해 포함 여부 검사
            for idx, yolo_pts in enumerate(yolo_pts_list):
                # YOLO 박스 내에 OCR 중심점이 포함되면 해당 책에 OCR 결과 추가
                # pointPolygonTest는: 내부 = 1, 외부 = -1, 경계 = 0 반환
                result = cv2.pointPolygonTest(yolo_pts, (int(ocr_center[0]), int(ocr_center[1])), False)
                if result >= 0:
                    books_ocr_result[idx].append({
                        'text': text,
                        'confidence': conf,
                        'coords': ocr_coords
                    })
                    books_ocr_text_only[idx] += text
                    break  # 하나의 박스에만 넣고 나머지는 스킵

    # # 디버깅용 출력
    # for i, book in enumerate(books_ocr_result):
    #     print(f"Book {i+1} OCR 결과:")
    #     if not book:
    #         print(" - 텍스트 없음")
    #     for item in book:
    #         print(f" - 텍스트: {item['text']}, 신뢰도: {item['confidence']:.3f}, 좌표: {item['coords']}")

    return books_ocr_result, books_ocr_text_only


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







import cv2
import numpy as np

def draw_yolo_points_on_image(img_path, yolo_pts_list):
    # 이미지 불러오기
    img = cv2.imread(img_path)

    # 점의 색깔과 크기 설정
    point_color = (0, 0, 255)  # 빨간색 (BGR)
    point_radius = 5  # 점의 반지름 크기

    # yolo_pts_list의 각 점을 이미지에 찍기
    for yolo_pts in yolo_pts_list:
        for pt in yolo_pts:
            # 각 꼭짓점에 점 찍기
            cv2.circle(img, tuple(pt), point_radius, point_color, -1)  # -1은 채우기

    # 결과 이미지 출력 (시각화)
    cv2.imshow('Image with YOLO Points', img)
    cv2.waitKey(0)  # 키 입력 대기
    cv2.destroyAllWindows()


def draw_ocr_coords_and_show_image(image_path, ocr_result):
    """
    주어진 이미지 경로와 OCR 결과를 바탕으로 좌표를 그려 이미지에 시각화하는 함수

    :param image_path: 이미지 파일 경로
    :param ocr_result: OCR 결과 (좌표와 텍스트가 있는 튜플)
    """
    # 이미지 읽기
    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 좌표 색상 설정
    coord_color = (0, 255, 0)  # 초록색 (BGR)
    point_radius = 5  # 점의 반지름
    line_thickness = 2  # 선의 두께

    # OCR 좌표 그리기
    for line in ocr_result:
        for ocr_item in line:
            pts, _ = ocr_item
            # 네 꼭짓점을 이어서 사각형 그리기
            pts = np.array(pts, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))  # OpenCV에서 사용하기 위한 형태로 변환
            cv2.polylines(img, [pts], isClosed=True, color=coord_color, thickness=line_thickness)


    # 이미지 시각화 (이미지 창에 띄우기)
    cv2.imshow('Image with OCR Coordinates', img)
    cv2.waitKey(0)  # 키 입력 대기
    cv2.destroyAllWindows()

import cv2

def draw_single_point_on_image(image_path, point_list):
    """
    이미지 위에 주어진 좌표 하나만 점으로 표시하는 함수
    """
    # 이미지 불러오기
    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 좌표를 int로 변환
    x, y = map(int, point_list[1])

    # 점 찍기
    cv2.circle(img, (x, y), color=(0, 0, 255), radius=8)

    # 이미지 출력
    cv2.imshow("Single Point on Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def run_ocr_yolo_api():
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_available = paddle.device.is_compiled_with_cuda()

    # 설정
    img_path = '/home/addinedu/github/book_recognition/libro_server/libro_ai_server/src/book_recognition/book_recognition/resources/image0.jpg'
    ocr_model = PaddleOCR(lang="korean", use_gpu=gpu_available, use_angle_cls=True, show_log=False)
    yolo_model = YOLO('/home/addinedu/github/book_recognition/libro_server/libro_ai_server/src/book_recognition/book_recognition/models/yolo11s_best.pt').to(device)

    # gpu 확인 코드
    print(f'gpu option: 1. yolo gpu: {yolo_model.device}, 2. ocr gpu: {paddle.get_device()}')

    # ocr_result는 ocr의 각 꼭짓점 좌표와 , 텍스트, 신뢰도를 가지고 있다.
    '''
        ocr_result는 아래와 같은 형태로 되어 있다.
        [[   [   [[x, y], [좌표2], [좌표3], [좌표4]],   ('이', 0.9981935620307922)   ], ...
    '''
    ocr_result = run_ocr(img_path, ocr_model)

    # yolo_pts_list는 책의 꼭짓점 좌표들을 가지고 있다.
    '''
        yolo_pts_list는 아래와 같은 형태로 되어 있다.
        [array([[123, 467],
            [261, 458],
            [235,  84],
            [ 98,  93]], dtype=int32), 
        array([[272, 458],
            [371, 450],
            [339,  52],
            [240,  60]], dtype=int32), 
        array([[390, 436],
            [540, 434],
            [536,  45],
            [386,  47]], dtype=int32)]
    '''
    yolo_pts_list, book_location_list, book_gradient_list = run_yolo(img_path, yolo_model)

    # draw_yolo_points_on_image(img_path, yolo_pts_list)
    # draw_ocr_coords_and_show_image(img_path, ocr_result)

    books_ocr_result, books_ocr_text_only = seperate_keyword_by_book(ocr_result, yolo_pts_list)

    print(books_ocr_text_only)

    a = search_books_in_google(books_ocr_text_only)
    print(a)

    print(book_location_list)
    # print()
    # print(yolo_pts_list)
    # draw_single_point_on_image(img_path, book_location_list)

    end_time = time.time()
    # 추론 시간 계산
    inference_time = end_time - start_time
    print(f"YOLO 모델 추론 시간: {inference_time:.4f} 초")

    return 



    
    # if len(after_yolo_img_list) == 0 :
    #     print('No book found')
    #     return 
    
    # search_keyword_list = run_ocr(after_yolo_img_list)

    # if len(search_keyword_list) == 0 :
    #     print('No word detected')
    #     return 

    # book_results_list = search_books_in_google(search_keyword_list)

    # return book_results_list, book_location_list, book_gradient_list



if __name__ == '__main__':
    run_ocr_yolo_api()