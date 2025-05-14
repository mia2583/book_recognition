import socket
import yaml
import ai_server.tcp.tcp_config as tcp_config

import socket
import yaml
import ai_server.tcp.tcp_config as tcp_config

def request_book_z(server_ip=tcp_config.SERVER_IP, server_port=tcp_config.PORT):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((server_ip, server_port))
        print(f"[INFO] 서버에 연결됨: {server_ip}:{server_port}")
    except Exception as e:
        print(f"[ERROR] 서버에 연결할 수 없습니다: {e}")
        return

    try:
        while True:
            try:
                book_title = input("Z값을 요청할 책 제목을 입력하세요 (q 입력 시 종료): ")
                if book_title.lower() == 'q':
                    client_socket.send(book_title.encode('utf-8'))
                    break

                client_socket.send(book_title.encode('utf-8'))

                response = client_socket.recv(4096)  # 늘려줌
                if not response:
                    print("[INFO] 서버 연결이 끊어졌습니다.")
                    break

                response_data = yaml.safe_load(response.decode('utf-8'))

                print(f"\n[응답 수신]")
                print(f" - 검색 성공: {response_data.get('success')}")
                print(f" - 책 제목: {response_data.get('book_title')}")
                print(f" - x: {response_data.get('book_x')}")
                print(f" - y: {response_data.get('book_y')}")
                print(f" - z: {response_data.get('book_z')}")
                print(f" - angle(deg): {response_data.get('book_angle')}")
                if not response_data.get('success', False):
                    print(f" - 오류: {response_data.get('error')}")
                print()

            except (ConnectionResetError, BrokenPipeError):
                print("[INFO] 서버와의 연결이 끊어졌습니다.")
                break
            except Exception as e:
                print(f"[ERROR] 데이터 처리 중 오류 발생: {e}")
                break

    finally:
        client_socket.close()
        print("[INFO] 클라이언트 소켓 종료됨.")


if __name__ == "__main__":
    request_book_z()
