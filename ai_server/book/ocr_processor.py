import cv2
import time
from paddleocr import PaddleOCR


class OcrProcessor:
    def __init__(self, use_gpu=True):
        self.ocr_model = PaddleOCR(lang="korean", use_gpu=use_gpu, use_angle_cls=True, show_log=False)

    def run_ocr(self, img):
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        start = time.time()
        results = self.ocr_model.ocr(img, cls=True)
        print(f"OCR Inference Time: {time.time() - start:.4f} seconds")
        return results
