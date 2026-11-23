FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installation des librairies système requises par OpenCV et Matplotlib
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copier tout le projet
COPY . .

# Exposer le port
EXPOSE 8000

# Lancer l'API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]