import requests
from urllib.parse import quote_plus

# API에 책 검색
def search_books_in_google(search_texts_list):
    api_key = ""  # Google Books API 키 (필요한 경우 입력)
    book_results = []

    for search_texts in search_texts_list:
        # 모든 추출된 텍스트를 기반으로 검색 시도
        found = False

        # 한글 제목을 URL 인코딩
        encoded_title = quote_plus(search_texts)

        # Google Books API 호출
        url = f"https://www.googleapis.com/books/v1/volumes?q={encoded_title}&langRestrict=ko"
        if api_key:
            url += f"&key={api_key}"

        try:
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                if "items" in data and len(data["items"]) > 0:
                    book_info = data["items"][0]["volumeInfo"]
                    result = {
                        "search_text": search_texts,
                        "title": book_info.get("title", "Unknown"),
                        "authors": book_info.get("authors", ["Unknown"]),
                        "publisher": book_info.get("publisher", "Unknown"),
                        "published_date": book_info.get("publishedDate", "Unknown"),
                        "description": book_info.get(
                            "description", "No description available"
                        ),
                        "thumbnail": book_info.get("imageLinks", {}).get(
                            "thumbnail", "No image"
                        ),
                    }
                    book_results.append(result)
                    found = True
                    # print(f"  - 검색 성공: {result['title']}")
        except Exception as e:
            # print(f"  - 검색 중 오류 발생: {str(e)}")
            continue

        # 검색 실패한 경우
        if not found:
            book_results.append(
                {
                    "search_text": search_texts,
                    "title": "검색 실패: No results found",
                }
            )
            # print(f"  - 검색 실패: 결과를 찾을 수 없습니다.")

    return book_results

