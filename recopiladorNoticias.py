import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Leer los URLs desde el archivo de texto
with open("enlaces.txt", "r") as file:
    urls = [line.strip() for line in file]

# Guardar los resultados en un nuevo archivo
with open("resultados.txt", "a", encoding="utf-8") as file:
    for url in urls:
        response = requests.get(url)
        content = response.content

        soup = BeautifulSoup(content, "html.parser")

        try:
            titulo = soup.find("meta", {"property": "og:title"})["content"]
            descripcion = soup.find("meta", {"property": "og:description"})["content"]
            fecha_element = soup.find("time")

            if fecha_element is not None:
                fecha = fecha_element.get("datetime")
                fecha_obj = datetime.strptime(fecha, "%Y-%m-%dT%H:%M:%S%z")
                fecha_str = fecha_obj.strftime("%d/%m/%Y")

        except (KeyError, TypeError, ValueError):
            continue

        #cambia comillas dobles por simples para que no haya error a la hora de leer el formato más adelante
        titulo = titulo.replace('"', "'")
        titulo = titulo.replace(';', ",")
        descripcion = descripcion.replace('"', "'")
        descripcion = descripcion.replace(';', ",")

        file.write(f'ID;1;{titulo};{descripcion};{fecha_str}\n')

print("Los resultados se han guardado en el archivo 'resultados.txt'.")