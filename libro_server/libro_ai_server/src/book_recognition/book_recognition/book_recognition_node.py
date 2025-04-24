#!/home/addinedu/venv/ocr_venv/bin/python

import rclpy
from rclpy.node import Node

from book_msg.msg import BookInfo
from book_recognition.ai_main import ai_main

from std_msgs.msg import Bool


class BookRecognitionNode(Node):
    def __init__(self):
        super().__init__('book_recognition_node')
        self.subscription = self.create_subscription(
            Bool,
            '/run_libro_ai',
            self.run_libro_ai_callback,
            10
        )
        self.publisher_ = self.create_publisher(BookInfo, '/book_info', 10)

    def run_libro_ai_callback(self, msg):
        try:
            if msg.data == True :
                book_results_list, book_location_list, book_gradient_list = ai_main()

                for i in range(0, len(book_results_list)):
                    msg = BookInfo()
                    msg.title = book_results_list[i]['title']
                    msg.position.x, msg.position.y = book_location_list[i]
                    msg.theta = book_gradient_list[i]

                    self.publisher_.publish(msg)
        except Exception as e:
            self.get_logger().error(f"에러 발생: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = BookRecognitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
