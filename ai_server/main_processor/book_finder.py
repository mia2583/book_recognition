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
from ai_server.tcp.tcp_server import TcpServer
import cv2
import numpy as np
import queue



def main():
    # 공유 프레임과 락 생성
    shared_frame = [None]  # 리스트 형태로 참조 유지
    shared_z = [None]
    frame_lock = threading.Lock()

    # UDP 수신기 초기화 및 실행
    udp_receiver = UdpStreamReceiver(
        calibration_file='./ai_server/resources/jetcobot.yaml'
    )

    def udp_runner():
        udp_receiver.start_receiving()
        frame_queue = udp_receiver.get_frame_queue()
        

        while True:
            try:
                encoded_frame = frame_queue.get(timeout=1.0)
                decoded_frame = cv2.imdecode(
                    np.frombuffer(encoded_frame, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if decoded_frame is not None:
                    with frame_lock:
                        shared_frame[0] = decoded_frame
                    
                    aruco_z = udp_receiver.get_latest_z_from_aruco()
                    shared_z[0] = aruco_z
            except queue.Empty:
                continue

    udp_thread = threading.Thread(target=udp_runner, daemon=True)
    udp_thread.start()

    # TCP 서버 초기화 및 실행
    tcp_server = TcpServer(frame_ref=shared_frame, frame_lock=frame_lock, aruco_z=shared_z)
    tcp_thread = threading.Thread(target=tcp_server.run, daemon=True)
    tcp_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] 메인 서버 종료 요청됨")
        udp_receiver.stop_receiving()


if __name__ == '__main__':
    main()
