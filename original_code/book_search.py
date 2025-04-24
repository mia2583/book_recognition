import cv2
import requests
from urllib.parse import quote_plus
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os


def search_books_in_google(book_details):
    print("4단계: Google Books API를 이용한 책 검색...")

    api_key = ""  # Google Books API 키 (필요한 경우 입력)
    book_results = []


    # 모든 추출된 텍스트를 기반으로 검색 시도
    found = False
    search_texts = book_details# 제목과 상위 3개 텍스트만 사용

    print(search_texts)


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
                print(f"  - 검색 성공: {result['title']}")
    except Exception as e:
        print(f"  - 검색 중 오류 발생: {str(e)}")

    # 검색 실패한 경우
    if not found:
        book_results.append(
            {
                "search_text": search_texts,
                "error": "No results found",
            }
        )
        print(f"  - 검색 실패: 결과를 찾을 수 없습니다.")

    return book_results



# PaddleOCR 리더 생성(한글 'korean' 설정)
ocr = PaddleOCR(lang='korean')

# 이미지 로드 및 업스케일링

image_path = './yolo_test_image1/1.png'
image = cv2.imread(image_path)
upscaled_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# PIL 형식으로 변환
upscaled_image_pil = Image.fromarray(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(upscaled_image_pil)
font = ImageFont.truetype("./MaruBuri-Bold.ttf", 60)

book_info=""

# 텍스트 검출 및 박스 그리기
for line in ocr.ocr(np.array(upscaled_image_pil), cls=True):
    for word_info in line:
        box, (text, score) = word_info[0], word_info[1]
        top_left, bottom_right = tuple(map(int, box[0])), tuple(map(int, box[2]))
        bottom_left = (top_left[0], bottom_right[1])

        book_info += text
        # # 텍스트 및 확률 출력
        print(f"Detected text: {text} (Probability: {score})")

        # 박스 그리기
        draw.rectangle([top_left, bottom_right], outline=(250, 0, 0), width=2)
        draw.text(bottom_left, text, font=font, fill=(0, 255, 0))
        

search_books_in_google(book_info)
# 결과 이미지 저장
result_path = os.path.join('./ocr_test_image1/', f'1.png')
upscaled_image_pil.convert("RGB").save(result_path)