# Imagen base ligera de Python
FROM python:3.10-slim

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema necesarias para OpenCV y Pillow
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar archivo requirements.txt al contenedor
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto al contenedor
COPY . .

# Definir volumen para guardar resultados de imágenes procesadas
VOLUME ["/app/data"]

# Comando por defecto -> ejecutar pipeline
CMD ["python", "src/data/integrator.py"]
