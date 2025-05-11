# camera_handler.py
import cv2
import time
import numpy as np
from calibration_utils import load_calibration_from_yaml

class CameraHandler:
    def __init__(self, index, width, height, fps, buffer_size,
                 calibration_file=None):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise IOError(f"카메라 인덱스 {index}를 열 수 없습니다.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_buffer = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
        print(f"카메라 초기화 완료: 요청({width}x{height} @ {fps}FPS, 버퍼 {buffer_size}) -> "
              f"실제({self.actual_width}x{self.actual_height} @ {actual_fps}FPS, 버퍼 {actual_buffer})")

        self.K = None
        self.D = None
        self.new_K = None
        self.mapx = None
        self.mapy = None
        self.undistort_enabled = False
        self.calibration_image_width = None
        self.calibration_image_height = None

        if calibration_file:
            self._load_calibration(calibration_file)

    def _load_calibration(self, calibration_file):
        """캘리브레이션 파일을 로드하고 왜곡 보정 맵을 초기화합니다."""
        self.K, self.D, self.calibration_image_width, self.calibration_image_height = \
            load_calibration_from_yaml(calibration_file)

        if self.K is not None and self.D is not None:
            img_width_for_calib = self.calibration_image_width if self.calibration_image_width else self.actual_width
            img_height_for_calib = self.calibration_image_height if self.calibration_image_height else self.actual_height
            img_size = (img_width_for_calib, img_height_for_calib)
            target_size = (self.actual_width, self.actual_height)

            print(f"[INFO] 왜곡 보정 매핑 계산 중 (원본 크기: {img_size}, 목표 크기: {target_size})")
            try:
                alpha = 0  # Crop to valid pixels
                self.new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, img_size, alpha, target_size)
                self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                    self.K, self.D, None, self.new_K, target_size, cv2.CV_32FC1)

                if self.mapx is not None and self.mapy is not None:
                    self.undistort_enabled = True
                    print("[INFO] 카메라 왜곡 보정 맵 (mapx, mapy) 생성")
                else:
                    print("[WARN] 왜곡 보정 맵(mapx, mapy) 생성 실패. 보정 비활성화.")
            except Exception as e:
                print(f"[ERROR] 왜곡 보정 맵 계산 중 오류 발생: {e}. 보정 비활성화.")
                self.undistort_enabled = False
        else:
            print(f"[WARN] YAML 캘리브레이션 데이터 로드 실패 ({calibration_file}). 왜곡 보정 비활성화.")

    def capture_frame(self, undistort=True):
        """카메라에서 프레임을 캡처하여 반환합니다. undistort 플래그에 따라 왜곡 보정을 수행합니다."""
        ret, frame = self.cap.read()
        if not ret:
            print("[WARN] 카메라 프레임 읽기 실패.")
            time.sleep(0.1)
            return None

        if undistort and self.undistort_enabled:
            try:
                undistorted_frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
                # print("[INFO] 왜곡 보정된 프레임 전달")
                return undistorted_frame
            except Exception as e:
                print(f"[ERROR] 왜곡 보정 중 오류 발생: {e}. 원본 프레임 반환.")
                return frame
        else:
            print("[INFO] 원본 프레임 전달")
            return frame

    def get_camera_parameters(self):
        """
        카메라의 기본적인 파라미터를 반환합니다.
        왜곡 보정이 활성화된 경우, 보정된 새 카메라 매트릭스(new_K)와 0으로 채워진 왜곡 계수를 반환합니다.
        """
        if self.undistort_enabled and self.new_K is not None:
            corrected_D = np.zeros((5,), dtype=np.float32)
            return self.new_K, corrected_D
        elif self.K is not None:
            return self.K, self.D
        else:
            print("[WARN] get_camera_parameters: 카메라 매트릭스(K)가 로드되지 않았습니다.")
            return None, None

    def get_effective_camera_parameters(self):
        """
        실제로 ArUco 마커의 축을 그리거나 할 때 사용할 효과적인 카메라 파라미터를 반환합니다.
        """
        if self.undistort_enabled and self.new_K is not None:
            return self.new_K, np.zeros((5,), dtype=np.float32)
        elif self.K is not None:
            return self.K, self.D if self.D is not None else np.zeros((5,), dtype=np.float32)
        else:
            print("[WARN] get_effective_camera_parameters: 카메라 매트릭스(K)가 로드되지 않았습니다.")
            return None, None

    def release_camera(self):
        if self.cap.isOpened():
            self.cap.release()
            print("[INFO] 카메라 리소스 해제 완료.")