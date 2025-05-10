import requests
from urllib.parse import quote_plus


class BookSearcher:
    def search_books_in_google(self, search_texts_list):
        results = []
        for text in search_texts_list:
            encoded_title = quote_plus(text)
            url = f"https://www.googleapis.com/books/v1/volumes?q={encoded_title}&langRestrict=ko"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if "items" in data and len(data["items"]) > 0:
                        book_info = data["items"][0]["volumeInfo"]
                        results.append({"search_text": text, "title": book_info.get("title", "Unknown")})
                    else:
                        results.append({"search_text": text, "title": "검색 실패: No results found"})
            except Exception:
                continue
        return results
