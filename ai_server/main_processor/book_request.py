import socket
import yaml
import ai_server.tcp.tcp_config as tcp_config

def request_book_z(server_ip=tcp_config.SERVER_IP, server_port=tcp_config.PORT, book_title=''):
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

            # if response_data["success"]:
            print(f"검색 성공: {response_data['success']}")
            print(f"책 제목: {response_data['book_title']}")
            print(f"x: {response_data['book_x']}")
            print(f"y: {response_data['book_y']}")
            print(f"z: {response_data['book_z']}")
            print(f"angle(deg): {response_data['book_angle']}")
            # else:
                # print(f"검색 성공: {response_data['success']}")
                # print(f"오류 원인: {response_data['error']}")
            
    except Exception as e:
        print(f"클라이언트 오류 발생: {e}")
    finally:
        client_socket.close()

if __name__ == "__main__":
    request_book_z()
