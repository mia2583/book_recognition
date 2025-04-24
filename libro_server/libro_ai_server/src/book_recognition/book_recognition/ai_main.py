#!/home/addinedu/venv/ocr_venv/bin/python

from ultralytics import YOLO
from pathlib import Path
from paddleocr import PaddleOCR
import os
import cv2
import requests
from urllib.parse import quote_plus
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import math
from ament_index_python.packages import get_package_share_directory

pkg_path = get_package_share_directory('book_recognition')

# ex) img_path = "./test_image/image0.jpg"
# 책별로 crop한 이미지 리스트, 책 위치 정보 반환
def run_yolo(img_path):
    model = YOLO(os.path.join(pkg_path, 'models', 'yolo11s_best.pt'))
    # yolo 처리 이후에 이미지
    after_yolo_img_list = []
    # 책 위치 정보 리스트
    book_location_list = []
    # 책 기울기 정보 리스트
    book_gradient_list = []

    img = cv2.imread(img_path)
    img_with_dots = img.copy()
    # 이미지에 yolo 적용
    results = model(img_path)


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
                pts = polygon.cpu().numpy().astype(np.int32)
                rect = cv2.minAreaRect(pts)
                (center_x, center_y), (w, h), angle = rect
                book_location_list.append((center_x, center_y))

                # 각도 보정 (OpenCV는 -90 ~ 0도로 나옴)
                if w < h:
                    angle_corrected = angle + 90
                else:
                    angle_corrected = angle

                book_gradient_list.append(angle_corrected)

                print(angle_corrected)

                # 회전된 직사각형을 4개의 점으로 변환
                box_points = cv2.boxPoints(rect)
                box_points = box_points.astype(int) # 정수형으로 변환
                
                # 그 직사각형을 감싸는 최소한의 수평/수직 박스를 구하기
                x, y, w, h = cv2.boundingRect(box_points)

                x = max(x, 0)
                y = max(y, 0)
                x = min(h, x)
                y = min(w, y)

                # 이미지 자르기
                cropped = img[y:y+h, x:x+w]

                # 마스크 생성
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [box_points], -1, 255, -1)  # 회전된 박스에 대한 마스크

                # 마스크 적용하여 자른 이미지 얻기
                mask_crop = mask[y:y+h, x:x+w]
                result_img = cv2.bitwise_and(cropped, cropped, mask=mask_crop)

                print("이미지", x, ' ', y)

                after_yolo_img_list.append(result_img)
                save_path = Path(f"./yolo_test/{i}.png")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(save_path, result_img)
                
                # 중심 좌표 그리고 저장하기
                cv2.circle(img_with_dots, (int(center_x), int(center_y)), radius=20, color=(0, 255, 0), thickness=-1)

                # 선 길이
                line_length = 200


                # 선의 끝점 좌표 계산 (시계 방향 회전 고려)
                x2 = int(center_x + line_length * math.cos(math.radians(angle_corrected)))
                y2 = int(center_y + line_length * math.sin(math.radians(angle_corrected)))

                # 기울기 방향 선 그리기 (빨간색)
                cv2.line(img_with_dots, (int(center_x), int(center_y)), (x2, y2), (255, 0, 0), 4)

    # cv2.imwrite("result_with_dots.jpg", img_with_dots)
    print(len(after_yolo_img_list))
    print(len(book_location_list))
    print(len(book_gradient_list))
    print('here')
    return after_yolo_img_list, book_location_list, book_gradient_list


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



# ex) image_list = [image1, image2, ...]와 같은 형태. 각 이미지는 cv 형식
# 책들의 ocr 리스트를 생성
def run_ocr(image_list):
    # PaddleOCR 리더 생성(한글 'korean' 설정)
    ocr = PaddleOCR(lang='korean')
    search_keyword_list = []

    for img in image_list:
        upscaled_image = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # PIL 형식으로 변환
        upscaled_image_pil = Image.fromarray(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(upscaled_image_pil)
        font = ImageFont.truetype(os.path.join(pkg_path, 'resources', 'MaruBuri-Bold.ttf'), 20)

        # ocr로 추출한 단어 저장
        extracted_words = ""

        # 텍스트 검출 및 박스 그리기
        for line in ocr.ocr(np.array(upscaled_image_pil), cls=True):
            for word_info in line:
                box, (text, score) = word_info[0], word_info[1]
                top_left, bottom_right = tuple(map(int, box[0])), tuple(map(int, box[2]))
                bottom_left = (top_left[0], bottom_right[1])

                extracted_words += text

                # 박스 그리기
                draw.rectangle([top_left, bottom_right], outline=(250, 0, 0), width=2)
                draw.text(top_left, text, font=font, fill=(0, 0, 255))
                
        search_keyword_list.append(extracted_words)
        
        # ocr 결과 이미지 저장
        result_path = os.path.join('./', f'{extracted_words}.jpg')
        upscaled_image_pil.convert("RGB").save(result_path)

    return search_keyword_list


def ai_main():
    img_path = os.path.join(pkg_path, 'resources', 'image5.jpg')
    print('Yolo started')
    after_yolo_img_list, book_location_list, book_gradient_list = run_yolo(img_path)
    if len(after_yolo_img_list) == 0 :
        print('Yolo not detected')
        return 
    
    search_keyword_list = run_ocr(after_yolo_img_list)
    if len(search_keyword_list) == 0 :
        print('No word detected')
        return 

    book_results_list = search_books_in_google(search_keyword_list)

    # 책 별 중심 좌표
    # print(len(book_location_list))

    # # 책 별 제목
    # for book in book_results_list:
    #     print(book['search_text'])
    #     print(book['title'])
    #     if book['title'] != "검색 실패: No results found" :
    #         print(book['authors'])
    return book_results_list, book_location_list, book_gradient_list




