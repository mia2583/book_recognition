import cv2
import struct

SERVER_IP = "192.168.35.17"
CLIENT_IP = "0.0.0.0"
PORT = 9999

# 네트워크 설정
CHUNK_SIZE = 1500  # MTU 고려
SERVER_SEND_BUFFER = 65536
CLIENT_RECV_BUFFER = 262144

# 카메라 설정
UDP_CAMERA_INDEX = 0
USB_CAMERA_INDEX = 2
DEFAULT_JPEG_QUALITY = 90

CAMERA_DEFAULT_WIDTH = 640
CAMERA_DEFAULT_HEIGHT = 480


# --- 청킹 관련 설정 ---
# 헤더 형식: Frame ID (unsigned int, 4B), Total Chunks (unsigned short, 2B), Chunk ID (unsigned short, 2B)
HEADER_FORMAT = "!IHH" # 네트워크 바이트 순서(!), uint32, uint16, uint16
HEADER_SIZE = struct.calcsize(HEADER_FORMAT) # 헤더 크기 (8 바이트)
# 청크 데이터 페이로드 최대 크기 (네트워크 MTU 고려, 예: 이더넷 ~1472, 안전하게 1400)
CHUNK_PAYLOAD_SIZE = 1400
# 프레임 재조립 타임아웃 (초): 이 시간 동안 해당 프레임의 새 청크가 없으면 버림
FRAME_REASSEMBLY_TIMEOUT = 2.0

# 수신기 queue 설정
DEFAULT_QUEUE_MAX_SIZE = 10

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}
DEFAULT_ARUCO_TYPE = "DICT_6X6_250"
DEFAULT_ARUCO_LENGTH = 0.06
USB_ARUCO_LENGTH = 0.075

JETCOBOT_CALIBRATION_FILE = "camera_params/jetcobot.yaml"
GLOBAL_CALIBRATION_FILE = "camera_params/global.yaml"