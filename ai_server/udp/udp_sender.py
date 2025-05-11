# udp_sender.py

import os
import sys
import cv2
import numpy as np
import socket
import struct
import threading
import time
from camera_handler import CameraHandler
import udp_config


class UdpStreamSender:
    def __init__(self, target_ip, target_port, camera_index=udp_config.UDP_CAMERA_INDEX,
                 width=640, height=480, fps=30,
                 quality=udp_config.DEFAULT_JPEG_QUALITY,
                 calibration_file=None,
                 send_buffer_size=udp_config.SERVER_SEND_BUFFER):
        self.target_address = (target_ip, target_port)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality
        self.send_buffer_size = send_buffer_size
        self.running = False
        self.sock = None
        self.thread = None
        self.frame_id = 0
        self.cam_handler = None

        try:
            self.cam_handler = CameraHandler(
                index=self.camera_index,
                width=self.width,
                height=self.height,
                fps=self.fps,
                buffer_size=1,
                calibration_file=calibration_file
            )

            print("[INFO] UdpStreamSender: CameraHandler 초기화 완료.")
        except IOError as e:
            print(f"[ERROR] 카메라 초기화 실패: {e}")
            raise
        except Exception as e:
            print(f"[ERROR] CameraHandler 초기화 중 예기치 않은 ERROR: {e}")
            raise

        self._init_socket()

    def _init_socket(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.send_buffer_size)
            actual_buf_size = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            print(f"[INFO] UDP 송신 소켓 생성됨. 송신 버퍼 크기: {actual_buf_size}")
        except socket.error as e:
            print(f"[ERROR] UDP 소켓 생성 실패: {e}")
            self.sock = None
            raise
        except Exception as e:
            print(f"[ERROR] 소켓 설정 중 예외 발생: {e}")
            self.sock = None
            raise

    def _send_frame(self, frame: np.ndarray):
        if self.sock is None or frame is None:
            return False

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        result, encoded_bytes = cv2.imencode('.jpg', frame, encode_param)

        if not result:
            print("[WARN] 프레임 JPEG 인코딩 실패.")
            return False

        data = encoded_bytes.tobytes()
        data_size = len(data)

        total_chunks = (data_size + udp_config.CHUNK_PAYLOAD_SIZE - 1) // udp_config.CHUNK_PAYLOAD_SIZE
        if total_chunks == 0: total_chunks = 1

        success = True
        for chunk_id in range(total_chunks):
            start = chunk_id * udp_config.CHUNK_PAYLOAD_SIZE
            end = min(start + udp_config.CHUNK_PAYLOAD_SIZE, data_size)
            payload = data[start:end]

            try:
                header = struct.pack(udp_config.HEADER_FORMAT, self.frame_id, total_chunks, chunk_id)
                message = header + payload
                self.sock.sendto(message, self.target_address)
            except (socket.error, struct.error, Exception) as e:
                print(f"[ERROR] UDP 전송 실패/오류 (프레임 {self.frame_id}, 청크 {chunk_id}): {e}")
                success = False
                break

        if success:
            self.frame_id = (self.frame_id + 1) % (2 ** 32)

        return success

    def _streaming_loop(self):
        target_interval = 1.0 / self.fps if self.fps > 0 else 0
        last_capture_time = time.time()

        while self.running:
            current_time = time.time()
            elapsed = current_time - last_capture_time
            wait_time = target_interval - elapsed

            if target_interval > 0 and wait_time > 0.001:
                time.sleep(wait_time)

            last_capture_time = time.time()

            # 원본 프레임 캡처 (왜곡 보정 없이)
            frame_to_send = self.cam_handler.capture_frame(undistort=False)

            if frame_to_send is None:
                print("[WARN] UDP Sender: 프레임 캡처 실패, 건너뜁니다.")
                time.sleep(0.1)
                continue

            # 프레임 전송
            if not self._send_frame(frame_to_send):
                print("[WARN] 프레임 전송 실패, 계속 시도...")
                time.sleep(0.05)

        print("[INFO] 스트리밍 루프 종료됨.")

    def start_streaming(self, run_in_thread=False):
        if self.running:
            print("[WARN] 이미 스트리밍이 실행 중입니다.")
            return

        if self.sock is None or self.cam_handler is None:
            print("[ERROR] 소켓 또는 카메라 핸들러가 초기화되지 않았습니다.")
            return

        self.running = True
        print(f"[INFO] UDP 스트리밍 시작 -> {self.target_address[0]}:{self.target_address[1]}")

        if run_in_thread:
            self.thread = threading.Thread(target=self._streaming_loop, daemon=True)
            self.thread.start()
            print("[INFO] 송신 스레드가 백그라운드에서 시작되었습니다.")
        else:
            self._streaming_loop()

    def stop_streaming(self):
        if not self.running: return

        print("[INFO] 스트리밍 중지 요청 중...")
        self.running = False

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                print("[WARN] 송신 스레드가 시간 내에 깔끔하게 종료되지 않았습니다.")

        if self.cam_handler:
            self.cam_handler.release_camera()

        if self.sock:
            self.sock.close()
            print("[INFO] UDP 소켓 닫힘.")

        print("[INFO] 송신기 중지 완료.")