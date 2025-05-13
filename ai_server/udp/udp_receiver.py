# udp_receiver.py

import cv2
import socket
import queue
import threading
import time
import struct
import numpy as np
import ai_server.udp.udp_config as udp_config
from ai_server.udp.aruco_processor import ArucoProcessor
from ai_server.udp.calibration_utils import load_calibration_from_yaml


class UdpStreamReceiver:
    def __init__(self, listen_ip=udp_config.CLIENT_IP, listen_port=udp_config.PORT,
                 buffer_size=udp_config.CLIENT_RECV_BUFFER,
                 queue_size=udp_config.DEFAULT_QUEUE_MAX_SIZE,
                 calibration_file=udp_config.JETCOBOT_CALIBRATION_FILE):
        self.listen_address = (listen_ip, listen_port)
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.running_flag = threading.Event()
        self.thread = None
        self.sock = None
        self.reassembly_buffer = {}

        # 캘리브레이션 및 왜곡 보정 관련 변수
        self.K = None
        self.D = None
        self.mapx = None
        self.mapy = None
        self.aruco_processor = None

        self.latest_z = None  # Z축 값을 저장할 변수 추가

        # 캘리브레이션 파일 로드
        if calibration_file:
            print(f"[INFO] UdpReceiver: 캘리브레이션 파일 로드 시도: {calibration_file}")
            self.K, self.D, calib_img_width, calib_img_height = load_calibration_from_yaml(calibration_file)

            if self.K is not None and self.D is not None:
                # 왜곡 보정 맵 초기화
                img_size = (calib_img_width, calib_img_height)
                target_size = (640, 480)  # 예상 이미지 크기

                alpha = 0  # 유효 픽셀만 포함
                self.new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, img_size, alpha, target_size)
                self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                    self.K, self.D, None, self.new_K, target_size, cv2.CV_32FC1)

                # ArUco 프로세서 초기화 (왜곡 보정 후 사용할 파라미터)
                self.aruco_processor = ArucoProcessor(
                    camera_matrix=self.new_K,
                    dist_coeffs=None,  # 왜곡 보정 후에는 왜곡 계수 없음
                    aruco_dict_type=udp_config.ARUCO_DICT.get(udp_config.DEFAULT_ARUCO_TYPE, cv2.aruco.DICT_6X6_250),
                    marker_length=udp_config.JETCOBOT_ARUCO_LENGTH
                )
                print("[INFO] UdpReceiver: 왜곡 보정 맵 및 ArucoProcessor 초기화 완료.")
            else:
                print("[WARN] UdpReceiver: 캘리브레이션 파일에서 K 또는 D를 로드하지 못했습니다.")
        else:
            print("[WARN] UdpReceiver: 캘리브레이션 파일이 제공되지 않아 ArUco 처리가 제한될 수 있습니다.")

        self._init_socket()

    def _init_socket(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(1.0)
            self.sock.bind(self.listen_address)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size)
            actual_buf_size = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            print(
                f"[INFO] UDP 수신 소켓 생성됨 (수신 대기: {self.listen_address[0]}:{self.listen_address[1]}). 수신 버퍼: {actual_buf_size}")
        except (socket.error, Exception) as e:
            print(f"[ERROR] UDP 소켓 바인딩/설정 실패 또는 초기화 중 예외: {e}")
            self.sock = None
            raise

    def _cleanup_stale_frames(self, current_time):
        stale_ids = [fid for fid, data in self.reassembly_buffer.items()
                     if current_time - data['last_seen'] > udp_config.FRAME_REASSEMBLY_TIMEOUT]
        if stale_ids:
            # print(f"[WARN] 타임아웃 프레임 정리: ID {stale_ids}")
            for frame_id in stale_ids:
                del self.reassembly_buffer[frame_id]

    def _receiver_loop(self):
        if self.sock is None:
            print("[ERROR] 수신 루프 시작 불가: 소켓이 초기화되지 않음.")
            return

        last_cleanup_time = time.time()
        received_packets = 0
        processed_frames_count = 0
        last_processed_frame_id = -1

        while self.running_flag.is_set():
            try:
                msg, address = self.sock.recvfrom(self.buffer_size)
                current_time = time.time()
                received_packets += 1

                if len(msg) < udp_config.HEADER_SIZE: continue

                frame_id, total_chunks, chunk_id = struct.unpack(udp_config.HEADER_FORMAT, msg[:udp_config.HEADER_SIZE])
                payload = msg[udp_config.HEADER_SIZE:]

                # 간단한 랩어라운드 처리
                if last_processed_frame_id > 1000 and frame_id < (last_processed_frame_id - 1000): continue

                if frame_id not in self.reassembly_buffer:
                    if len(self.reassembly_buffer) > 50:
                        self._cleanup_stale_frames(current_time)
                        continue
                    self.reassembly_buffer[frame_id] = {'total': total_chunks, 'chunks': {}, 'last_seen': current_time}

                frame_entry = self.reassembly_buffer[frame_id]

                if chunk_id not in frame_entry['chunks']:
                    frame_entry['chunks'][chunk_id] = payload
                    frame_entry['last_seen'] = current_time

                if frame_entry['total'] != total_chunks: frame_entry['total'] = total_chunks

                if len(frame_entry['chunks']) == frame_entry['total']:
                    sorted_chunks = [frame_entry['chunks'][i] for i in range(frame_entry['total'])]
                    full_frame_data_bytes = b"".join(sorted_chunks)  # JPEG 바이트 데이터

                    # JPEG 바이트를 OpenCV 프레임으로 디코딩
                    cv_frame = cv2.imdecode(np.frombuffer(full_frame_data_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if cv_frame is None:
                        print(f"[WARN] 프레임 {frame_id} 디코딩 실패.")
                        del self.reassembly_buffer[frame_id]
                        continue

                    # 왜곡 보정 적용
                    undistorted_frame = None
                    if self.mapx is not None and self.mapy is not None:
                        try:
                            undistorted_frame = cv2.remap(cv_frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
                            # print(f"[INFO] 프레임 {frame_id} 왜곡 보정 완료.")
                        except Exception as e:
                            print(f"[ERROR] 왜곡 보정 중 오류: {e}")
                            undistorted_frame = cv_frame  # 오류 시 원본 사용
                    else:
                        undistorted_frame = cv_frame  # 왜곡 보정 맵이 없으면 원본 사용

                    # ArUco 마커 검출 및 처리
                    if self.aruco_processor:
                        corners, ids, rejected, rvecs, tvecs = self.aruco_processor.detect_markers(undistorted_frame)
                        aruco_data_list = self.aruco_processor.get_pose_data(corners, ids, rvecs, tvecs)

                        if aruco_data_list:  # 검출된 마커가 있다면
                            for marker_data in aruco_data_list:
                                tvec_m = marker_data.get('tvec')
                                # print(f"\n--- UDP 수신 (프레임 {tvecs}): 감지된 ArUco 마커 정보 ---")
                                # Z값을 최신 상태로 업데이트
                                self.latest_z = tvec_m[2][0] if tvec_m is not None else None
                                # print(f"\n--- UDP 수신 (프레임 {self.latest_z}): 감지된 ArUco 마커 정보 ---")
                                
                        
                        # else:
                        #     # print("[INFO] ArUco 마커가 감지되지 않았습니다.")
                        #     continue


                        # 프레임에 마커 정보 그리기
                        self.aruco_processor.draw_detected_markers(undistorted_frame, corners, ids, None)
                        if rvecs is not None and tvecs is not None and ids is not None:
                            for i in range(len(ids)):
                                self.aruco_processor.draw_axes(undistorted_frame, rvecs[i], tvecs[i])

                    # 시각화 정보가 포함된 프레임을 다시 JPEG으로 인코딩하여 큐에 넣음
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), udp_config.DEFAULT_JPEG_QUALITY]
                    result, final_encoded_bytes = cv2.imencode('.jpg', undistorted_frame, encode_param)

                    if result:
                        try:
                            if self.frame_queue.full():
                                try:
                                    self.frame_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            self.frame_queue.put(final_encoded_bytes.tobytes(), block=False)
                            processed_frames_count += 1
                            last_processed_frame_id = frame_id
                        except queue.Full:
                            print("[WARN] 프레임 큐가 계속 꽉 차 있습니다. 현재 프레임 버림.")
                            pass
                    else:
                        print(f"[WARN] 프레임 {frame_id} 재인코딩 실패.")

                    del self.reassembly_buffer[frame_id]

            except socket.timeout:
                current_time = time.time()
                if current_time - last_cleanup_time > udp_config.FRAME_REASSEMBLY_TIMEOUT / 2:
                    self._cleanup_stale_frames(current_time)
                    last_cleanup_time = current_time
                continue
            except struct.error as e:
                print(f"[WARN] 헤더 언패킹 실패: {e}")
                continue
            except Exception as e:
                print(f"[ERROR] UDP 수신 루프 ERROR: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

        print(f"[INFO] UDP 수신 스레드 종료됨. 총 수신 패킷: {received_packets}, 처리된 프레임: {processed_frames_count}")
        if self.sock:
            self.sock.close()
            print("[INFO] UDP 수신 소켓 닫힘.")

    def start_receiving(self):
        if self.thread is not None and self.thread.is_alive():
            print("[WARN] 이미 수신 스레드가 실행 중입니다.")
            return

        if self.sock is None:
            print("[ERROR] 수신 시작 불가: 소켓이 초기화되지 않음.")
            return

        self.running_flag.set()
        self.thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self.thread.start()
        print("[INFO] UDP 수신 스레드가 백그라운드에서 시작되었습니다.")

    def stop_receiving(self):
        if not self.running_flag.is_set(): return

        print("[INFO] 수신 스레드 중지 요청 중...")
        self.running_flag.clear()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                print("[WARN] 수신 스레드가 시간 내에 깔끔하게 종료되지 않았습니다.")

        print("[INFO] 수신기 중지 완료.")

    def get_frame_queue(self) -> queue.Queue:
        return self.frame_queue
    
    def get_latest_z_from_aruco(self):
        return self.latest_z  # _receiver_loop에서 계산된 Z값을 반환
