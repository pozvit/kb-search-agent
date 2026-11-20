#!/bin/sh
# Старт контейнера: при первом запуске собрать индекс public_v2, затем поднять HTTP API.
set -e

PROFILE="${AGENT_PROFILE:-public}"

if [ "$PROFILE" = "public" ] && [ ! -d data/public/qdrant_storage ]; then
    echo "Индекс public_v2 не найден — сборка. Первый запуск: модель эмбеддингов скачивается один раз в кэш."
    python src/build_index.py --profile public
fi

if [ "$PROFILE" = "full" ] && [ ! -d data/qdrant_storage ]; then
    echo "Профиль full: индекс полной базы подключается томом в /app/data, в образе его нет." >&2
    exit 1
fi

exec python src/serve.py --profile "$PROFILE" --host 0.0.0.0 --port "${PORT:-8000}"
