import socket
from multiprocessing import Process, Queue
from yolo_ocr_api import main1
import subprocess

# AI 모델을 처리하는 프로세스 함수
def process_ocr(data, result_queue):
    print("Processing OCR data...")
    result = main1(data)  # main1 함수에서 OCR 모델 실행
    result_queue.put(result[0])  # 처리된 결과를 큐에 넣음

def main():
    # 서버의 IP와 포트 설정
    HOST = '192.168.0.153'  # 서버의 IP 주소
    PORT = 9999  # 서버의 포트 번호

    # 클라이언트 소켓 생성
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    print(f"Connected to server at {HOST}:{PORT}")

    while True:
        # 서버로부터 데이터 받기
        data = client_socket.recv(2**20).decode()  # 서버로부터 데이터 받음
        print("Received data from server:", data)

        if data == 'quit':
            break

        result = subprocess.check_output(['python3', 'yolo_ocr_api.py', data], text=True)
        
        print("Received result from AI model:", result)

        # 결과를 서버에 다시 전송 (필요시)
        client_socket.send(result.encode())  # 결과를 서버로 전송

    # 클라이언트 소켓 종료
    client_socket.close()

if __name__ == '__main__':
    main()
