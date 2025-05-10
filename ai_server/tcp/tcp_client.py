import socket
from book.BookRecognizer import BookRecognizer
import cv2


def main():
    # 서버의 IP와 포트 설정
    HOST = '192.168.35.17'  # 서버의 IP 주소
    PORT = 9999  # 서버의 포트 번호
    recognizer = BookRecognizer()

    # 클라이언트 소켓 생성
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    print(f"Connected to server at {HOST}:{PORT}")

    img_path="/home/myseo/study/book_recognition/libro_server/libro_ai_server/src/book_recognition/book_recognition/resources/image0.jpg"
    img = img = cv2.imread(img_path)

    while True:
        # 서버로부터 데이터 받기
        data = client_socket.recv(2**20).decode()  # 서버로부터 데이터 받음

        if data == 'quit':
            break
        
        result = recognizer.infer(img, data)

        print("Received result from AI model:", result)

        # 결과를 서버에 다시 전송 (필요시)
        client_socket.send(result.encode())  # 결과를 서버로 전송

    # 클라이언트 소켓 종료
    client_socket.close()

if __name__ == '__main__':
    main()