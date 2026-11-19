"""
Проверка состояния компонентов для GET /health.

  llm     — Ollama отвечает на GET /api/tags за 2 с, и нужная модель загружена (mock — всегда ok);
  index   — индекс открывается; число точек (Qdrant) или чанков (лексический индекс);
  catalog — каталог карточек реестров; без него сервис работает, но маршрут registry недоступен.

200 — модель и индекс доступны; 503 — недоступен хотя бы один из них.
Без каталога статус degraded и код 200.
"""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple


def check_llm(agent: Any, timeout: float = 2.0) -> Dict[str, Any]:
    cfg = agent.config.get("llm", {})
    model = cfg.get("model", "")
    if cfg.get("backend", "ollama") == "mock":
        return {"status": "ok", "model": "mock"}
    url = cfg.get("base_url", "http://localhost:11434").rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            names = {m.get("name") for m in json.load(resp).get("models", [])}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"status": "down", "model": model, "detail": f"Ollama недоступна: {type(exc).__name__}"}
    if model and model not in names and f"{model}:latest" not in names:
        return {"status": "down", "model": model, "detail": "модель не загружена в Ollama"}
    return {"status": "ok", "model": model}


def check_index(agent: Any) -> Dict[str, Any]:
    retriever, failure = agent.tools._get_retriever()
    if failure:
        return {"status": "down", "detail": "индекс недоступен"}
    points = None
    try:
        inner = getattr(retriever, "_retriever", None)
        if inner is not None and hasattr(inner, "client"):
            points = int(inner.client.count(inner.collection_name).count)
        elif hasattr(retriever, "chunks"):
            points = len(retriever.chunks)
        elif hasattr(retriever, "corpus"):
            points = len(retriever.corpus)
    except Exception:
        return {"status": "down", "detail": "индекс не отвечает"}
    if points == 0:
        return {"status": "down", "points": 0, "detail": "индекс пуст"}
    return {"status": "ok", "points": points}


def check_catalog(agent: Any) -> Dict[str, Any]:
    catalog = agent.tools.catalog()
    if catalog is None:
        return {"status": "absent", "detail": "каталог карточек не настроен или не найден"}
    return {"status": "ok", "cards": len(catalog.cards)}


def health(agent: Any) -> Tuple[int, Dict[str, Any]]:
    components = {"llm": check_llm(agent), "index": check_index(agent), "catalog": check_catalog(agent)}
    if components["llm"]["status"] != "ok" or components["index"]["status"] != "ok":
        return 503, {"status": "down", "components": components}
    status = "ok" if components["catalog"]["status"] == "ok" else "degraded"
    return 200, {"status": status, "components": components}
