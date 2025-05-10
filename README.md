
## 0. 준비

### 코드 클론하기
```
git clone https://github.com/mia2583/book_recognition.git
```

### 필요한 패키지 설치

```bash
# 가상환경에 설치할 것
pip install 'empy==3.3.4' # 최신 버전인 4.2는 ros에서 에러남..
pip install catkin_pkg
pip install lark
pip install ultralytics
pip install paddleocr
pip install paddlepaddle
```

### 코드 수정
아래 파일에서 자신의 가상환경 주소로 변경하기
book_recognition/libro_server/libro_ai_server/src/book_recognition/book_recognition/setup.cfg
```
[build_scripts]
executable = 자신의 가상환경 주소 입력
```

### 코드 빌드하기
```bash
# 가상환경 활성화 후 코드를 실행할 것
cd book_recognition/libro_server/libro_ai_server
colcon build
. install/local_setup.bash
```

<br/><br/><br/>

## 1. 사진 내의 책들 정보 받아오기 (토픽)

### 동작 설명

`/run_libro_ai` 토픽으로 True가 전달되면 주어진 사진(테스트에서는 book_recognition/resources 아래 image0.jpg를 사용한다)내의 책들의 제목, 위치, 각도 정보가 BookInfo 메세지에 담겨 `/book_info` 토픽으로 퍼블리시된다.

### BookInfo.msg 정의

```
string title    # 책 제목
geometry_msgs/Point position  # x, y 좌표만 사용 (z 사용x)
float32 theta   # 회전 (단위: degree)
```

### 실행 방법

#### 터미널1

```bash
ros2 run book_recognition book_recognition_node
```

#### 터미널2

```bash
ros2 topic echo /book_info
```

#### 터미널3

```bash
ros2 topic pub /run_libro_ai std_msgs/msg/Bool 'data: true' 
```

### 출력 예시

터미널2에서 아래와 같이 출력된다.

```
title: ROS2 혼자공부하는 로봇SW 직접 만들고 코딩하자
position:
  x: 463.0
  y: 240.49998474121094
  z: 0.0
theta: 89.41085815429688
---
```

<br/><br/><br/>

## 2. 사진 내의 특정 책 정보 받아오기 (서비스)

### 동작 설명

`/find_book` 서비스로 책 제목(title)이 전달되면 주어진 사진(테스트에서는 book_recognition/resources 아래 image0.jpg를 사용한다)내의 책들중 해당 책을 찾아 책의 위치, 각도 정보를 전달한다.

### FindBook.srv 정의

```
string title    # 책 제목
---
bool success    # 책 검색 성공 여부
geometry_msgs/Point position  # x, y 좌표만 사용 (z 사용x)
float32 theta   # 회전 (단위: degree)
```

### 실행 방법

#### 터미널1

```bash
ros2 run book_recognition find_book_service_server 
```

#### 터미널2

```bash
ros2 service call /find_book book_msg/srv/FindBook "title: 'ROS2 혼자공부하는 로봇SW 직접 만들고 코딩하자'"
```

### 출력 예시

터미널2에서 아래와 같이 출력된다.

```
requester: making request: book_msg.srv.FindBook_Request(title='ROS2 혼자공부하는 로봇SW 직접 만들고 코딩하자')

response:
book_msg.srv.FindBook_Response(success=True, position=geometry_msgs.msg.Point(x=463.0, y=240.49998474121094, z=0.0), theta=89.41085815429688)
```


## 3. 사진 내의 특정 책 정보 받아오기 (TCP 통신)

### 동작 설명

서버측에서 책 제목을 전송하면, 클라이언트에서 주어진 사진에 해당 책이 있으면 책의 각도와 중앙 좌표를 전달한다.

### 실행 방법

#### 터미널1

```bash
cd ai_server
python3 tcp_server.py
```

#### 터미널2

```bash
cd ai_server
python3 tcp_client.py
```

### 출력 예시

터미널2를 실행한 후, 터미널 1에서 아래와 같이 연결 확인 메세지를 출력하고 책 제목 입력을 요청한다.

```
>> Connected by: 192.168.35.17 : 46660

Enter book title
```

책 제목을 입력하면 client쪽에서 책의 정보를 전달한다.

```
# cpu 예시
OCR Inference Time: 5.7084 seconds
Yolo Inference Time: 1.5950 seconds
Total Inference Time: 9.7400 seconds
Received result from AI model: Found book - location: 927.0, 482.25, theta: 89.41611417995652
```
