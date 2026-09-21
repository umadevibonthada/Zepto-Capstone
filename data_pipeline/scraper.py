import requests
from bs4 import BeautifulSoup
url = "https://books.toscrape.com/"
response=requests.get(url)
#print(response.status_code)
soup = BeautifulSoup(response.text, "html.parser")
print(soup.title.text)
book = soup.find("article", class_="product_pod")
#book = soup.find("article", class_="product_pod")

print(book.h3.a["title"])