import socket
from _thread import *
import threading

client_sockets = []
## Server IP and Port ##
HOST = '0.0.0.0'
PORT = 9999

## 클라이언트 연결 처리 스레드 ##
def threaded(client_socket, addr):
    print('>> Connected by:', addr[0], ':', addr[1])
    print()

    while True:
        try:
            print("Enter book title")
            data = client_socket.recv(16384)
            if not data:
                print('>> Disconnected by', addr[0], ':', addr[1])
                break
            msg = data.decode()
            print(f">> Received from {addr[0]}:{addr[1]}: {msg}")
            print()
            broadcast(msg, sender=client_socket)
        except ConnectionResetError:
            print('>> Disconnected by', addr[0], ':', addr[1])
            break
    if client_socket in client_sockets:
        client_sockets.remove(client_socket)
        print('remove client list:', len(client_sockets))
    client_socket.close()

## 메시지 브로드캐스트 함수 ##
def broadcast(msg, sender=None):
    for client in client_sockets:
        if client != sender:
            try:
                client.send(msg.encode())
            except:
                pass

## 서버에서 직접 입력을 받아 메시지 전송하는 스레드 ##
def server_send():
    while True:
        msg = input()  # 서버 콘솔 입력
        broadcast(msg)

## 소켓 생성 및 바인딩 ##
print('>> Server Start with IP:', HOST)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()

## 서버 입력용 쓰레드 시작 ##
start_new_thread(server_send, ())

try:
    while True:
        print('>> Waiting client')
        client_socket, addr = server_socket.accept()
        client_sockets.append(client_socket)
        start_new_thread(threaded, (client_socket, addr))
except Exception as e:
    print('에러:', e)
finally:
    server_socket.close()