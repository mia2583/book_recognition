import threading
import socket
import os
import datetime
import cv2
import yaml
import ai_server.tcp.tcp_config as tcp_config
from ai_server.udp.aruco_processor import ArucoProcessor
import numpy as np

class TcpServer:
    def __init__(self, host_ip=tcp_config.HOST_IP, port=tcp_config.PORT, frame_ref=None, frame_lock=None, camera_matrix=None, dist_coeffs=None, aruco_z=None, book_recognizer=None):
        self.host = host_ip
        self.port = port
        self.frame_ref = frame_ref  # 외부에서 공유하는 frame (예: 글로벌 변수 참조)
        self.frame_lock = frame_lock
        self.camera_matrix = camera_matrix  # 카메라 매트릭스
        self.dist_coeffs = dist_coeffs  # 왜곡 계수
        self.aruco_z = aruco_z
        self.book_recognizer = book_recognizer
        self.sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_tcp.bind((self.host, self.port))
        self.sock_tcp.listen(5)
        print(f"[INFO] TCP 서버 시작: {self.host}:{self.port}")
        
        # ArUcoProcessor 객체 생성
        self.aruco_processor = ArucoProcessor(self.camera_matrix, self.dist_coeffs)

    def run(self):
        self._run()

    def _run(self):
        while True:
            try:
                conn, addr = self.sock_tcp.accept()
                print(f"TCP 클라이언트 연결됨: {addr}")
                threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()
            except Exception as e:
                print(f"[ERROR] TCP 서버 오류: {e}")

    
    def pixel_to_camera_coordinate(self, u, v):
        with open('./resources/jetcobot.yaml', 'r', encoding='utf-8') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            K = data['K']

            Z = self.aruco_z
            X = (u - K[0][2]) * Z / K[0][0]
            Y = (v - K[1][2]) * Z / K[1][1]
        
        return X, Y
    
    def numpy_to_native(self,obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def handle_client(self, conn, addr):
        try:
            while True:  # 클라이언트와 연결을 끊지 않고 계속 처리
                # 클라이언트로부터 받은 데이터 (책 제목)
                data = conn.recv(1024).decode('utf-8')
                if data.lower() == 'q':  # 종료 요청을 받으면 연결 종료
                    print(f"클라이언트 {addr}에서 종료 요청 받음. 연결 종료.")
                    break

                print(f"클라이언트로부터 받은 요청: 책 제목 '{data}' 찾는 중...")

                with self.frame_lock:
                    if self.frame_ref[0] is not None:
                        frame = self.frame_ref[0]

                        # 현재 프레임 저장
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"captured_frame_{timestamp}.jpg"
                        save_path = os.path.join("captured", filename)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(save_path, frame)
                        print(f"현재 프레임 저장됨: {save_path}")
                        book_search = self.book_recognizer.infer(frame, data)

                        # print("book search done")
                        
                        # print(f"아루코 z값: {self.numpy_to_native(self.aruco_z[0])}")

                        print(f"center: {book_search['center']}")
                        print(f"angle: {book_search['angle']}")
                        
                        book_x = self.pixel_to_camera_coordinate(book_search['center'][0]) if book_search['center'] is not None else None
                        book_y = self.pixel_to_camera_coordinate(book_search['center'][1]) if book_search['center'] is not None else None

                        if self.aruco_z:
                            response = {
                                "success": book_search['success'],
                                "book_title": data,
                                "book_x" : book_x,
                                "book_y" : book_y,
                                "book_z": self.numpy_to_native(self.aruco_z[0]),
                                "book_angle": book_search['angle']
                            }
                        else:
                            response = {
                                "success": False,
                                "book_title": data,
                                "book_x" : book_x,
                                "book_y" : book_y,
                                "book_z": None,
                                "book_angle": book_search['angle'],
                                "error": "ArUco 마커를 찾을 수 없습니다."
                            }
                    else:
                        response = {
                            "success": False,
                            "book_title": data,
                            "book_x" : None,
                            "book_y" : None,
                            "book_z": None,
                            "book_angle": None,
                            "error": "입력된 프레임이 없습니다."
                        }
                        print("사용 가능한 프레임 없음")

                # YAML 형식으로 응답 작성
                response_yaml = yaml.dump(response, allow_unicode=True)
                conn.send(response_yaml.encode('utf-8'))


        except Exception as e:
            print(f"[ERROR] TCP 처리 오류: {e}")
            response = {
                "success": False,
                "book_title": data,
                "aruco_z": None,
                "error": f"처리 오류: {e}"
            }
            response_yaml = yaml.dump(response, allow_unicode=True)
            conn.send(response_yaml.encode('utf-8'))

        finally:
            conn.close()
            print(f"TCP 클라이언트 연결 종료: {addr}")
