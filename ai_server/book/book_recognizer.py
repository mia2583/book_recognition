import paddle
import cv2
import time
import torch
import numpy as np
import yaml
from ai_server.book.ocr_processor import OcrProcessor
from ai_server.book.yolo_detector import YoloDetector
from ai_server.book.book_searcher import BookSearcher
from ai_server.book.visualization import draw_ocr_coords, draw_yolo_box


class BookRecognizer:
    def __init__(self):
        self.title = ''
        self.img = None
        self.yolo_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ocr_gpu_available = paddle.device.is_compiled_with_cuda()

        self.ocr_processor = OcrProcessor(use_gpu=self.ocr_gpu_available)
        self.yolo_detector = YoloDetector(device=self.yolo_device)
        self.book_searcher = BookSearcher()

    def set_img(self, img):
        self.img = img

    def set_title(self, title):
        self.title = title

    def classify_texts(self, ocr_result, yolo_pts_list):
        if ocr_result == [None] or yolo_pts_list == []:
            return []
        
        books_text_only = ['' for _ in range(len(yolo_pts_list))]

        for line in ocr_result:
            if not line:
                return books_text_only
            for coords, (text, _) in line:
                center = np.mean(coords, axis=0).astype(int)
                for i, yolo_pts in enumerate(yolo_pts_list):
                    if cv2.pointPolygonTest(yolo_pts, (int(center[0]), int(center[1])), False) >= 0:
                        books_text_only[i] += text
                        break

        return books_text_only
    

    def infer(self, img, title):
        start = time.time()
        self.set_img(img)
        self.set_title(title)
        print(f"Looking for book: {self.title}")
        ocr_result = self.ocr_processor.run_ocr(self.img)
        yolo_pts_list, book_locations, book_angles = self.yolo_detector.run_yolo(self.img)
        books_text_result = self.classify_texts(ocr_result, yolo_pts_list)
        book_results = self.book_searcher.search_books_in_google(books_text_result)

        # draw_yolo_box(img, yolo_pts_list)
        # print(ocr_result)
        # draw_ocr_coords(img, ocr_result)

        print(f"Total Inference Time: {time.time() - start:.4f} seconds")

        # print(books_text_result)

        for i, book in enumerate(book_results):
            if book['title'] == self.title:
                response = {
                    "success": True,
                    "book_title": book['title'],
                    "center": book_locations[i],
                    "angle" : book_angles[i]
                }
                return response
                # return f"Found book - location: {book_locations[i]}, theta: {book_angles[i]}"
        
        response = {
            "success": False,
            "book_title": self.title,
            "center": None,
            "angle" : None
        }
        return response
        # return f"Not Found book - location: 0, theta: 0"


def main():
    recognizer = BookRecognizer()
    title="ROS2 혼자공부하는 로봇SW 직접 만들고 코딩하자"
    img_path="/home/addinedu/github/book_recognition/ai_server/resources/test_1.png"
    img = cv2.imread(img_path)

    recognizer.infer(img, title)


if __name__ == '__main__':
    main()