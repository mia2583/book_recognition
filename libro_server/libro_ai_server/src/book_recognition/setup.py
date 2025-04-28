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
            'book_recognition/resources/image5.jpg',
        ]),
        # 모델 파일
        ('share/' + package_name + '/models', [
            'book_recognition/models/yolo11s_best.pt'
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
            'ai_main = book_recognition.ai_main:ai_main',
            'find_book_service_server = book_recognition.find_book_service_server:main',
        ],
    },
)
