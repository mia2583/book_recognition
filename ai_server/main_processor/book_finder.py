## udp로 지속적으로 받음
## tcp로 책 요청을 받고 결과를 전송

import socket
import threading
import time
import cv2
import numpy as np
from ai_server.book.book_recognizer import BookRecognizer
import struct
from collections import defaultdict
import os
import datetime
import yaml
import ai_server.main_processor.main_config as main_config
from ai_server.udp.udp_receiver import UdpStreamReceiver


# 메인 함수
def main():
    # UDP 영상 수신 스레드 시작

    receiver_instance = UdpStreamReceiver(
        calibration_file='./ai_server/resources/jetcobot.yaml'
    )

    udp_thread = threading.Thread(target=receiver_instance.start_receiving, daemon=True)
    udp_thread.start()

    try:
        # 메인 스레드는 계속 실행
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("서버를 종료합니다.")

    
    
if __name__ == '__main__':
    main()
