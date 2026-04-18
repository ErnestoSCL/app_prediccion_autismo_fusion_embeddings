FROM python:3.10-slim

WORKDIR /app

# Copiamos los requirements primero (para aprovechar cache de Docker)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copiamos el código de la API y los pesos del modelo
COPY api/ ./api/
COPY models/ ./models/

# Puerto donde corre FastAPI
EXPOSE 8000

# Comando para arrancar la API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
