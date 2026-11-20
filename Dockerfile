# KB Search Agent — HTTP API (POST /ask, GET /health).
# Образ содержит код, конфигурацию и данные public_v2 (5 обезличенных PDF и синтетический реестр).
# Полная база (реестры, КП, их индексы) в образ не входит — требование S8; профиль full подключается томом.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface \
    AGENT_PROFILE=public \
    AGENT_LLM_BASE_URL=http://ollama:11434 \
    LOG_FORMAT=json

WORKDIR /app

# torch только для CPU: без этого pip ставит сборку с CUDA на несколько гигабайт
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements-service.txt .
RUN pip install -r requirements-service.txt

RUN useradd --create-home --uid 1000 app
COPY --chown=app:app config ./config
COPY --chown=app:app src ./src
COPY --chown=app:app data/raw ./data/raw
COPY --chown=app:app data/raw_registry_public ./data/raw_registry_public
COPY --chown=app:app docker/entrypoint.sh ./docker/entrypoint.sh
RUN mkdir -p data/public data/escalations /home/app/.cache/huggingface \
    && chown -R app:app data /home/app/.cache \
    && chmod +x docker/entrypoint.sh

# сервис работает не от root
USER app
EXPOSE 8000

# 200 — модель и индекс доступны; 503 (Ollama недоступна) — контейнер помечается unhealthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["./docker/entrypoint.sh"]
