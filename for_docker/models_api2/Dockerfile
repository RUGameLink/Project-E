# for_docker/models_api2/Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

WORKDIR /app
COPY requirements.txt .           # если есть
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install "fastapi[all]" uvicorn[standard] joblib numpy pandas

COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
