import requests
from bs4 import BeautifulSoup

base_url = "https://www.publico.es/politica/pagina/"
start_page = 1
end_page = 200

enlaces = set()

for page in range(start_page, end_page + 1):
    url = f"{base_url}{page}#analytics-listado:paginacion"
    try:
        response = requests.get(url)
        content = response.content

        soup = BeautifulSoup(content, "html.parser")

        for enlace in soup.find_all("a", href=True):
            href = enlace["href"]
            if href.startswith("/politica") and not href.startswith("/politica/pagina"):
                enlaces.add("https://www.publico.es" + href)

    except requests.exceptions.ConnectionError as e:
        print(f"Connection error occurred for URL: {url}\n{e}")
        continue

# Save the results to a new file
with open("enlaces.txt", "w", encoding="utf-8") as file:
    for enlace in enlaces:
        file.write(f'{enlace}\n')

print(f"The links from the URL: {url} have been saved in the 'enlaces.txt' file.")
