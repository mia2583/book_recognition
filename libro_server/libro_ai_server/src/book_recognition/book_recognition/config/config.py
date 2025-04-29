# book_recognition/config.py
# book_recognition 패키지에서 쓰이는 설정을 정의

from ament_index_python.packages import get_package_share_directory
import os
from ultralytics import YOLO
import torch
from paddleocr import PaddleOCR


pkg_path = get_package_share_directory('book_recognition')

# GPU 사용 가능한 pc인지 확인. 아니면 CPU 사용
device = "cuda" if torch.cuda.is_available() else "cpu"

yolo_model = YOLO(os.path.join(pkg_path, 'models', 'yolo11s_best.pt')).to(device)
# yolo_model = YOLO(os.path.join(pkg_path, 'models', 'yolo11n-obb-only-book.pt'))
# yolo_model = YOLO(os.path.join(pkg_path, 'models', 'yolo11n-obb-book-and-label.pt'))

ocr_model = PaddleOCR(lang="korean", use_gpu=False, use_angle_cls=True, show_log=False)
