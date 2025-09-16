# Dockerfile para sistema de detección de neumonía (Versión corregida)
FROM python:3.9-slim

# Instalar dependencias del sistema necesarias para imágenes médicas
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    libfontconfig1 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Crear estructura de directorios necesaria
RUN mkdir -p /app/src /app/tests /app/modelo /app/data

# Copiar código fuente
COPY src/ ./src/
COPY tests/ ./tests/
COPY modelo/ ./modelo/
COPY *.py ./

# Configurar variables de entorno
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/modelo/conv_MLP_84.h5

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Comando por defecto (se puede sobrescribir al ejecutar)
CMD ["python", "--version"]