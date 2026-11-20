"""
Загрузка config/agent.yaml, выбор профиля индекса (public | full) и точечные
переопределения вида {"llm.backend": "mock", "retrieval.collection_name": "x"}
(используются в сценариях для воспроизведения отказов).

Переменные окружения:
    AGENT_PROFILE        — public | full, перекрывает agent.profile
    AGENT_LLM_BACKEND    — ollama | mock
    AGENT_LLM_BASE_URL   — адрес Ollama (по умолчанию http://localhost:11434)
    AGENT_LLM_MODEL      — имя модели (по умолчанию qwen2.5:7b-instruct)
    AGENT_RETRIEVER      — qdrant | lexical
"""

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENT_CONFIG = "config/agent.yaml"


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _set_dotted(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    node = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_agent_config(path: str = DEFAULT_AGENT_CONFIG, profile: Optional[str] = None,
                      overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Возвращает словарь:
        agent, llm, tools, context, memory, guardrails, router, grounding, escalation, service — из agent.yaml
                  (+ overrides с префиксами agent./llm./...)
        rag       — конфиг retrieval выбранного профиля (+ overrides с префиксами retrieval./paths.)
        profile   — имя профиля
    """
    config = _read_yaml(path)
    config = copy.deepcopy(config)

    profile_name = profile or os.environ.get("AGENT_PROFILE") or config.get("agent", {}).get("profile", "public")
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Неизвестный профиль '{profile_name}'. Доступно: {', '.join(profiles)}.")

    rag_config = copy.deepcopy(_read_yaml(profiles[profile_name]["rag_config"]))

    # приоритет: agent.yaml < переменные окружения < overrides (CLI-флаги, сценарии)
    env_map = {
        "AGENT_LLM_BACKEND": "llm.backend",
        "AGENT_LLM_BASE_URL": "llm.base_url",
        "AGENT_LLM_MODEL": "llm.model",
        "AGENT_RETRIEVER": "retriever.backend",
    }
    for env_name, dotted in env_map.items():
        if os.environ.get(env_name):
            _set_dotted(config, dotted, os.environ[env_name])

    for key, value in (overrides or {}).items():
        head = key.split(".", 1)[0]
        if head in ("retrieval", "paths"):
            _set_dotted(rag_config, key, value)
        else:
            _set_dotted(config, key, value)

    retriever_cfg = config.setdefault("retriever", {})
    retriever_cfg.setdefault("chunks_jsonl", profiles[profile_name].get("chunks_jsonl"))

    config.setdefault("catalog_path", profiles[profile_name].get("cards_meta"))
    config["rag"] = rag_config
    config["profile"] = profile_name
    return config
