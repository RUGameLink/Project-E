# Docker-конфигурация проекта

Эта директория содержит файлы и конфигурации для запуска проекта в контейнерах Docker, что обеспечивает изолированную и воспроизводимую среду для работы с моделями нейронных сетей по прогнозированию выброса радона.

## Структура директории

```
for_docker/
├── model_save_preset/     # Копия директории моделей для Docker-контейнера
├── visualization/         # Копия визуализатора для Docker-контейнера
└── README.md              # Этот файл
```

## Файлы Docker

### Dockerfile

Основной файл для сборки Docker-образа:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "visualization/dashboard.py"]
```

### docker-compose.yml

Файл для управления многоконтейнерным приложением:

```yaml
version: '3'
services:
  visualization:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./model_save_preset:/app/model_save_preset
    environment:
      - MODEL_DIR=/app/model_save_preset/models
```

## Запуск проекта в Docker

### Сборка и запуск с помощью docker-compose

```bash
# Переходим в директорию проекта
cd path/to/project

# Запускаем сборку и запуск контейнеров
docker-compose up -d

# Для остановки контейнеров
docker-compose down
```

### Ручная сборка и запуск

```bash
# Сборка образа
docker build -t radon-model:latest .

# Запуск контейнера
docker run -p 8501:8501 -v $(pwd)/model_save_preset:/app/model_save_preset radon-model:latest
```

## Преимущества использования Docker

1. **Воспроизводимость** - одинаковая среда выполнения на любой машине
2. **Изоляция** - независимость от системных библиотек
3. **Переносимость** - проект работает на любой ОС с Docker
4. **Масштабируемость** - легкое развертывание в облачных сервисах

## Дополнительные замечания

- Каталоги `model_save_preset` и `visualization` в этой директории являются копиями соответствующих каталогов из корня проекта, адаптированными для работы в Docker-контейнере
- При обновлении основного проекта необходимо синхронизировать эти каталоги
- В Docker-контейнере используется порт 8501 для доступа к Streamlit-дашборду
