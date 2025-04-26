from pathlib import Path
from paddleocr import PaddleOCR
import os
import cv2
import requests
from urllib.parse import quote_plus
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

from book_recognition.modules.yolo_module import run_yolo
from pathlib import Path

from book_recognition.config.config import pkg_path

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
    img_path = os.path.join(pkg_path, 'resources', 'image0.jpg')
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

    return book_results_list, book_location_list, book_gradient_list




