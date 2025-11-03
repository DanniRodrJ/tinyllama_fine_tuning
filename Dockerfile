FROM python:3.10-slim
# FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos y carpetas necesarios al contenedor
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY notebooks/ ./notebooks/
COPY main.py ./
COPY README.md ./

# Instala las dependencias necesarias (puedes agregar aquí las librerías)
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
