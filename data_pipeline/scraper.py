import requests
from bs4 import BeautifulSoup
url = "https://books.toscrape.com/"
response=requests.get(url)
#print(response.status_code)
soup = BeautifulSoup(response.text, "html.parser")
print(soup.title.text)

#books=soup.find_all("article",class_="product_pod")
book = soup.find("article", class_="product_pod")

print(book.prettify())

#print(book.h3.a["title"])
#print(book.prettify())
#print("Number of books:", len(books))

for element in book.find_all():
    print(element.name, element.get("class"), element.get("href"))