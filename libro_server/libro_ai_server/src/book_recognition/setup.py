from setuptools import find_packages, setup

package_name = 'book_recognition'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 리소스 파일 (폰트, 이미지 등)
        ('share/' + package_name + '/resources', [
            'book_recognition/resources/MaruBuri-Bold.ttf',
            'book_recognition/resources/image0.jpg',
            'book_recognition/resources/image1.jpg',
            'book_recognition/resources/image2.jpg',
            'book_recognition/resources/image3.jpg',
            'book_recognition/resources/image4.jpg',
            'book_recognition/resources/image5.jpg',
            'book_recognition/resources/image6.jpg',
            'book_recognition/resources/image7.jpg',
            'book_recognition/resources/image8.jpg',
            'book_recognition/resources/image9.jpg',
            'book_recognition/resources/rotate_image1.jpg',
            'book_recognition/resources/rotate_image2.jpg',
        ]),
        # 모델 파일
        ('share/' + package_name + '/models', [
            'book_recognition/models/yolo11s_best.pt',
            'book_recognition/models/yolo11n-obb-only-book.pt',
            'book_recognition/models/yolo11n-obb-book-and-label.pt',
        ]),
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='addinedu',
    maintainer_email='addinedu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'book_recognition_node = book_recognition.book_recognition_node:main',
            'ai_main = book_recognition.ai_main:run_yolo_ocr_api',
            'find_book_service_server = book_recognition.find_book_service_server:main',
        ],
    },
)
