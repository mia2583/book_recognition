from book_msg.srv import FindBook

import rclpy
from rclpy.node import Node

from book_recognition.ai_main import ai_main

class FindBookServiceServer(Node):
    def __init__(self):
        super().__init__('find_book')
        self.server = self.create_service(FindBook, 'find_book', self.find_book_callback)

    def find_book_callback(self, request, response):
        print('find book request')
        try:
            book_results_list, book_location_list, book_gradient_list = ai_main()

            for i, book in enumerate(book_results_list):
                print(book['title'])
                print(i)
                if book['title'] == request.title:
                    response.success = True
                    response.position.x, response.position.y = book_location_list[i]
                    response.theta = book_gradient_list[i]
                    return response
                
            response.success = False
            return response
                    
        except Exception as e:
            self.get_logger().error(f"에러 발생: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FindBookServiceServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()