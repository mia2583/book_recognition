import socket
import argparse
import signal
import sys
import time

# 로컬 모듈 임포트
import udp_config
from udp_sender import UdpStreamSender # 정의된 클래스 임포트

# --- 종료 처리를 위한 전역 변수 및 핸들러 ---
sender_instance = None # 클래스 인스턴스 저장용
def signal_handler(sig, frame):
    print("\n[INFO] Ctrl+C 감지. 송신기 종료 중...")
    if sender_instance:
        sender_instance.stop_streaming()
    # sys.exit(0) # 필요시 강제 종료 (보통 stop_streaming 후 자동 종료됨)
signal.signal(signal.SIGINT, signal_handler)
# --- ---

def main():
    global sender_instance
    parser = argparse.ArgumentParser(description="UDP 비디오 스트림 송신기 (클래스 기반)")
    parser.add_argument("--target_ip", type=str, default=udp_config.SERVER_IP, help=f"수신자 IP 주소 (기본값: {udp_config.SERVER_IP})")
    parser.add_argument("--port", type=int, default=udp_config.PORT, help=f"UDP 포트 번호 (기본값: {udp_config.PORT})")
    parser.add_argument("--camera", type=int, default=udp_config.UDP_CAMERA_INDEX, help=f"카메라 인덱스 (기본값: {udp_config.UDP_CAMERA_INDEX})")
    parser.add_argument("--width", type=int, default=640, help="카메라 프레임 너비")
    parser.add_argument("--height", type=int, default=480, help="카메라 프레임 높이")
    parser.add_argument("--fps", type=int, default=30, help="카메라 및 전송 목표 FPS")
    parser.add_argument("--quality", type=int, default=udp_config.DEFAULT_JPEG_QUALITY, choices=range(0, 101), metavar="[0-100]", help=f"JPEG 품질 (기본값: {udp_config.DEFAULT_JPEG_QUALITY})")
    parser.add_argument("--calib", type=str, default=udp_config.JETCOBOT_CALIBRATION_FILE, help="카메라 캘리브레이션 YAML 파일 경로")
    # parser.add_argument("--no_undistort", action='store_true', help="캘리브레이션 파일 있어도 왜곡 보정 안 함")
    # parser.add_argument("--send_buffer", type=int, default=config.SERVER_SEND_BUFFER, help="UDP 송신 버퍼 크기") # 필요시 추가

    args = parser.parse_args()

    try:
        sender_instance = UdpStreamSender(
            target_ip=args.target_ip,
            target_port=args.port,
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            quality=args.quality,
            calibration_file=args.calib,
        )

        # 스트리밍 시작 (현재 스레드에서 실행, blocking)
        sender_instance.start_streaming(run_in_thread=False)
        # 또는 별도 스레드에서 실행 원할 시:
        # sender_instance.start_streaming(run_in_thread=True)
        # # 스레드로 실행 시 메인 스레드가 종료되지 않도록 대기 로직 필요
        # while sender_instance.running:
        #     try:
        #         time.sleep(1)
        #     except KeyboardInterrupt: # 여기서도 Ctrl+C 처리 가능
        #         signal_handler(None, None)
        #         break

    except (IOError, socket.error, Exception) as e:
        print(f"[ERROR] 송신기 초기화 또는 실행 중 오류 발생: {e}")
        if sender_instance:
            sender_instance.stop_streaming()
        sys.exit(1)

if __name__ == "__main__":
    main()
