# book_recognition/config.py
# book_recognition 패키지에서 쓰이는 설정을 정의

from ament_index_python.packages import get_package_share_directory
import os
from ultralytics import YOLO

pkg_path = get_package_share_directory('book_recognition')

yolo_model = YOLO(os.path.join(pkg_path, 'models', 'yolo11s_best.pt'))