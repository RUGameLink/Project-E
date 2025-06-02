# for_docker/visualization/Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

WORKDIR /app
COPY requirements.txt .           # если есть
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install streamlit requests

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.port", "8501", "--server.enableCORS", "false"]
