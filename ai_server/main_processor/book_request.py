import socket
import yaml

def request_book_z(server_ip='192.168.35.17', server_port=9999, book_title=''):
    try:
        # 소켓 생성 및 연결
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((server_ip, server_port))
        print(f"서버 {server_ip}:{server_port}에 연결되었습니다.")

        while True:  # 종료 전까지 계속 요청
            # 책 제목을 요청
            book_title = input("Z값을 요청할 책 제목을 입력하세요 (q 입력 시 종료): ")
            if book_title.lower() == 'q':
                client_socket.send(book_title.encode('utf-8'))  # 종료 요청
                break

            # 요청 메시지 전송
            client_socket.send(book_title.encode('utf-8'))

            # 응답 수신
            response = client_socket.recv(1024).decode('utf-8')
            response_data = yaml.unsafe_load(response)

            if response_data["success"]:
                print(f"책 제목: {response_data['book_title']}")
                print(f"ArUco Z 값: {response_data['aruco_z']}")
            else:
                print(f"오류: {response_data.get('error', '알 수 없는 오류')}")
            
    except Exception as e:
        print(f"클라이언트 오류 발생: {e}")
    finally:
        client_socket.close()

if __name__ == "__main__":
    request_book_z()
