FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        supervisor \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

COPY fusion_service /app/fusion_service
COPY text_to_sentiment /app/text_to_sentiment
COPY voice_to_text /app/voice_to_text
COPY voice_to_sentiment /app/voice_to_sentiment
COPY disease_service /app/disease_service
COPY mental_fitness /app/mental_fitness
COPY age-gender-model /app/age-gender-model
COPY models /app/models
COPY .env /app/.env

# Sadece ana API (fusion_service) dışarı açılıyor
EXPOSE 8003

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

