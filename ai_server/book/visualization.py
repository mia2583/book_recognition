from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

def draw_ocr_coords(img_path, ocr_result):
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



def draw_yolo_box(img_path, yolo_pts_list):
    """
    주어진 이미지에 YOLO 박스를 그리는 함수

    입력: 
        img_path: 이미지 파일 경로
        yolo_pts_list: YOLO 박스 좌표를 담은 리스트
            yolo_pts_list는 아래와 같은 형태로 되어 있다.
               [array([[x, y], [좌표2], [좌표3], [좌표4]], dtype=int32), ...]
    """
    # 이미지 불러오기
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # YOLO 박스 그리기
    for yolo_pts in yolo_pts_list:
        pts = np.array(yolo_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2) # 초록색

    # 결과 이미지 출력 (시각화)
    cv2.imshow('Image with YOLO Boxes', img)
    cv2.waitKey(0)  # 키 입력 대기
    cv2.destroyAllWindows()



def print_ocr_result(ocr_result):
    for line in ocr_result:
        if line == None:
            print("이미지 내 인식된 글자 없음")
            break
        for word_info in line:
            box, (text, score) = word_info
            print(f"인식된 텍스트: {text}, 좌표: {box}, 신뢰도: {score:.2f}")
    


def print_ocr_per_book(books_ocr_result):
    for i, book in enumerate(books_ocr_result):
        print(f"Book {i+1} OCR 결과:")
        if not book:
            print(" - 텍스트 없음")
        for item in book:
            print(f" - 텍스트: {item['text']}, 신뢰도: {item['confidence']:.3f}, 좌표: {item['coords']}")



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


