import requests
from urllib.parse import quote_plus
import paddle
import cv2
import time
from ultralytics import YOLO
import torch
from paddleocr import PaddleOCR
import math
import numpy as np
import time
import yaml
from visualization import draw_ocr_coords, print_ocr_result, draw_yolo_box, draw_single_point_on_image

class BookRecognizer:
    def __init__(self):
        print("모델 로딩 중...")
        self.yolo_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ocr_gpu_available = paddle.device.is_compiled_with_cuda()

        self.yolo_model = YOLO('/home/addinedu/github/book_recognition/libro_server/libro_ai_server/src/book_recognition/book_recognition/models/yolo11s_best.pt').to(self.yolo_device)
        self.ocr_model = PaddleOCR(lang="korean", use_gpu=self.ocr_gpu_available, use_angle_cls=True, show_log=False)
        self.img_path = '/home/addinedu/github/book_recognition/libro_server/libro_ai_server/src/book_recognition/book_recognition/resources/image0.jpg'
       
        # gpu 확인 코드
        print(f'gpu option: 1. yolo gpu: {self.yolo_model.device}, 2. ocr gpu: {paddle.get_device()}')
        print('모델 준비 완료')

    def infer(self, title):
        # ocr 처리
        start_time = time.time()
        print(f'Looking for book: {title}')
        ocr_result = self.run_ocr()

        # yolo 처리
        yolo_pts_list, book_location_list, book_gradient_list = self.run_yolo()

        # 책 별 ocr 분류
        books_text_result = self.seperate_keyword_by_book(ocr_result, yolo_pts_list)

        # 책 검색
        book_results_list = self.search_books_in_google(books_text_result)


        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Total Inference Time: {inference_time:.4f} seconds")

        for i, book in enumerate(book_results_list):
            if book['title'] == title:
                return f'Found book - location: {book_location_list[i][0]}, {book_location_list[i][1]}, theta: {book_gradient_list[i]}'
        
        return 'Book not found'


    def run_ocr(self):
        """
        주어진 이미지와 OCR 모델을 사용하여 이미지 내의 글자를 인식하는 함수

        입력: 
            img_path: 이미지 파일 경로
        반환:
            result: 인식한 글자의 좌표와 글자, 신뢰도. 이때 좌표는 이미지를 2배 확대했을 때의 좌표로 반환.
                result는 아래와 같은 형태로 되어 있다.
                    [[   [   [[x, y], [좌표2], [좌표3], [좌표4]],   ('인식한_글자', 0.9981935620307922)   ], ...
        """

        img = cv2.imread(self.img_path)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        start_time = time.time()
        results = self.ocr_model.ocr(img, cls=True)

        end_time = time.time()
        inference_time = end_time - start_time
        print(f"OCR Inference Time: {inference_time:.4f} seconds")
        
        return results


    def get_angle_between_points(self, pt1, pt2):
        """
        두 점을 잇는 선의 기울기

        입력: 
            pt1: x,y 좌표
            pt2: x,y 좌표
        반환:
            angle_rad: 기울기(라디안)
            angle_deg: 기울기(도)
        """

        angle_rad = math.atan2(pt1[1] -pt2[1], pt1[0] - pt2[0])  # y/x의 아크탄젠트
        angle_deg = math.degrees(angle_rad) # 라디안을 도로 변환
        return angle_rad, angle_deg


    def run_yolo(self):
        """
        주어진 이미지와 YOLO 모델을 사용하여 이미지 내의 책을 인식하는 함수

        입력: 
            img_path: 이미지 파일 경로
            yolo_model: YOLO 모델
        반환:
            좌표는 이미지를 2배로 확대했을 때의 좌표를 가진다.
            yolo_pts_list: yolo 박스들의 좌표를 담은 리스트
            book_location_list:  각 책 중앙점 좌표를 담은 리스트
            book_gradient_list: 각 책의 기울기를 담은 리스트
        """

        yolo_pts_list = []
        book_location_list = []
        book_gradient_list = []

        img = cv2.imread(self.img_path)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        start_time = time.time()

        results = self.yolo_model(img, verbose=False)

        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Yolo Inference Time: {inference_time:.4f} seconds")

        for result in results:
            if result.obb is not None:
                # 클래스가 0인 폴리곤만 필터링 [1][2]
                obb_polygons = [
                    polygon
                    for polygon, cls in zip(result.obb.xyxyxyxy, result.obb.cls.tolist())
                    if int(cls) == 0  # 클래스 0만 선택(클래스 0은 책을 의미, 다른 객체들은 다른 번호)
                ]

                # 필터링된 폴리곤 처리 [1][2]
                for i, polygon in enumerate(obb_polygons):
                    # pts는 yolo가 적용된 사각형의 꼭짓점
                    # rect는 pts를 감싸는 최소 사각형 바운딩
                    pts = polygon.cpu().numpy().astype(np.int32).reshape(4,2)
                    yolo_pts_list.append(pts)        

                    # 중심점 구하기
                    center = np.mean(pts, axis=0)
                    book_location_list.append((center[0], center[1]))

                    # 꼭짓점 나열 순서를 각도를 기준으로 정렬 : 왼쪽 상단 -> 오른쪽 상단 -> 오른쪽 하단 -> 왼쪽 하단
                    sorted_pts = sorted(pts, key=lambda pt: self.get_angle_between_points(pt, center)[0])

                    # 책의 기울기 계산
                    rad, deg = self.get_angle_between_points(sorted_pts[3], sorted_pts[0])
                    book_gradient_list.append(deg)
                    
        return yolo_pts_list, book_location_list, book_gradient_list


    def seperate_keyword_by_book(self, ocr_result, yolo_pts_list):
        """
        ocr의 글씨를 yolo 박스를 통해 책 별로 분류하는 함수

        입력: 
            ocr_result: OCR 결과
                ocr_result는 아래와 같은 형태로 되어 있다.
                [[   [   [[x, y], [좌표2], [좌표3], [좌표4]],   ('이', 0.9981935620307922)   ], ...
            yolo_pts_list: YOLO 박스 좌표를 담은 리스트
                yolo_pts_list는 아래와 같은 형태로 되어 있다.
                [array([[x, y], [좌표2], [좌표3], [좌표4]], dtype=int32), ...]
        반환:
            books_text_only: 책 별로 ocr 글자를 담은 리스트
        """

        # books_ocr_result = [[] for _ in range(len(yolo_pts_list))]
        books_text_only = ['' for _ in range(len(yolo_pts_list))]

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
                        # books_ocr_result[idx].append({
                        #     'text': text,
                        #     'confidence': conf,
                        #     'coords': ocr_coords
                        # })
                        books_text_only[idx] += text
                        break  # 하나의 박스에만 넣고 나머지는 스킵

        # return books_ocr_result, books_ocr_text_only
        return books_text_only


    # API에 책 검색
    def search_books_in_google(self, search_texts_list):
        """
        책 리스트들을 google api에 검색한 결과를 반환하는 함수

        입력: 
            search_texts_list: 검색어(책 제목)를 담은 리스트
        반환:
            book_results: 검색 결과를 담은 리스트
        """
        book_results = []

        for search_texts in search_texts_list:
            # 모든 추출된 텍스트를 기반으로 검색 시도
            found = False

            # 한글 제목을 URL 인코딩
            encoded_title = quote_plus(search_texts)

            # Google Books API 호출
            url = f"https://www.googleapis.com/books/v1/volumes?q={encoded_title}&langRestrict=ko"

            try:
                response = requests.get(url)

                if response.status_code == 200:
                    data = response.json()
                    # 검색 성공
                    if "items" in data and len(data["items"]) > 0:
                        book_info = data["items"][0]["volumeInfo"]
                        result = {
                            "search_text": search_texts,
                            "title": book_info.get("title", "Unknown"),
                        }
                        book_results.append(result)
                        found = True
                    else:
                        book_results.append(
                            {
                                "search_text": search_texts,
                                "title": "검색 실패: No results found",
                            }
                )
            except Exception as e:
                continue

        return book_results

def pixel_to_camera_coordinate(u, v):
    with open('./jetcobot.yaml', 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        K = data['K']

        # Z = d / depth_scale
        Z = 1
        X = (u - K[0][2]) * Z / K[0][0]
        Y = (v - K[1][2]) * Z / K[1][1]
    
    return X, Y

def main():
    recognizer = BookRecognizer()
    title = "ROS2 혼자공부하는 로봇SW 직접 만들고 코딩하자"
    ocr_result = recognizer.run_ocr()

    # yolo 처리
    yolo_pts_list, book_location_list, book_gradient_list = recognizer.run_yolo()

    # 책 별 ocr 분류
    books_text_result = recognizer.seperate_keyword_by_book(ocr_result, yolo_pts_list)

    # 책 검색
    book_results_list = recognizer.search_books_in_google(books_text_result)

    for i, book in enumerate(book_results_list):
        if book['title'] == title:
            print(f'Found book - location: {book_location_list[i][0]}, {book_location_list[i][1]}, theta: {book_gradient_list[i]}')
            return 

    print('Book not found')
        


if __name__ == '__main__':
    main()