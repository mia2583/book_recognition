import os

from book_recognition.modules.yolo_module import run_yolo
from book_recognition.modules.ocr_module2 import run_ocr
from book_recognition.modules.book_search_module import search_books_in_google

from book_recognition.config.config import pkg_path


def run_yolo_ocr_api():
    img_path = os.path.join(pkg_path, 'resources', 'image1.jpg')

    after_yolo_img_list, book_location_list, book_gradient_list = run_yolo(img_path)
    
    if len(after_yolo_img_list) == 0 :
        print('No book found')
        return 
    
    search_keyword_list = run_ocr(after_yolo_img_list)

    if len(search_keyword_list) == 0 :
        print('No word detected')
        return 

    book_results_list = search_books_in_google(search_keyword_list)

    return book_results_list, book_location_list, book_gradient_list




