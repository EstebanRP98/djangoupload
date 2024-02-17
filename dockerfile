# Utiliza una imagen oficial de Python como imagen base
FROM python:3.8

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /usr/src/app

# Copia el archivo de dependencias y lo instala
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia tu proyecto Django y el archivo .env y serviceAccountKey.json
COPY . .
COPY .env .
COPY serviceAccountKey.json .

# Expone el puerto en el que Django se ejecutará
EXPOSE 8000

# El comando para iniciar el servidor de Django
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
