"""
docker-compose.yaml и проверка развёртывания (шаг 4.1, требования F10, S8).
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str(ROOT / "tools"))

import smoke_api  # noqa: E402


def test_services_and_startup_order():
    services = COMPOSE["services"]
    assert set(services) == {"ollama", "ollama-pull", "api"}
    deps = services["api"]["depends_on"]
    assert deps["ollama"]["condition"] == "service_healthy"
    assert deps["ollama-pull"]["condition"] == "service_completed_successfully"
    assert services["ollama"]["healthcheck"]["test"]


def test_api_reaches_ollama_in_compose_network_and_pulls_the_configured_model():
    api_env = COMPOSE["services"]["api"]["environment"]
    pull = COMPOSE["services"]["ollama-pull"]["entrypoint"]
    config_model = yaml.safe_load((ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))["llm"]["model"]
    assert api_env["AGENT_LLM_BASE_URL"] == "http://ollama:11434" and api_env["AGENT_PROFILE"] == "public"
    assert pull[-1] == api_env["AGENT_LLM_MODEL"] == config_model


def test_api_port_bound_to_localhost_only():
    """У сервиса нет авторизации: порт открыт только на этом компьютере, не в локальной сети."""
    assert COMPOSE["services"]["api"]["ports"] == ["127.0.0.1:8000:8000"]


def test_only_named_volumes_no_project_data_mounted():
    """S8: в контейнер не монтируются папки проекта, в том числе с полной базой."""
    for name, service in COMPOSE["services"].items():
        for volume in service.get("volumes", []):
            source = volume.split(":")[0]
            assert source in COMPOSE["volumes"], f"{name}: {volume}"


# ---------------------------------------------------------------- логика проверки развёртывания

GOOD = {
    "/health": (200, {"components": {"llm": {"status": "ok"}, "index": {"status": "ok"}, "catalog": {"status": "ok"}}}),
    "ERGO": (200, {"status": "answered", "route": "documents", "sources": [{"id": "01_ergo.pdf#2"}]}),
    "медицине": (200, {"status": "answered", "route": "registry", "sources": [{}] * 5, "llm_calls": 0}),
    "ключ": (200, {"status": "rejected", "route": "rejected", "sources": []}),
    "": (422, {"error": {"code": "invalid_request"}}),
}


def fake_call(responses):
    def call(url, method="GET", payload=None, timeout=0):
        if url.endswith("/health"):
            return responses["/health"]
        question = payload["question"]
        key = next((k for k in responses if k and k != "/health" and k in question), "")
        return responses[key]
    return call


def test_smoke_passes_on_healthy_service(monkeypatch, capsys):
    monkeypatch.setattr(smoke_api, "call", fake_call(GOOD))
    assert smoke_api.main(["--wait", "1"]) == 0
    assert "Итог: 5 из 5" in capsys.readouterr().out


@pytest.mark.parametrize("key, bad", [
    ("ERGO", (200, {"status": "not_found", "route": "documents", "sources": []})),
    ("медицине", (200, {"status": "answered", "route": "documents", "sources": [{}] * 2, "llm_calls": 2})),
    ("ключ", (200, {"status": "rejected", "route": "rejected", "answer": smoke_api.SECRET})),
    ("", (500, {"error": {"code": "internal_error"}})),
], ids=["document_not_found", "registry_via_llm", "secret_echoed", "invalid_gives_500"])
def test_smoke_fails_on_broken_behaviour(monkeypatch, key, bad):
    monkeypatch.setattr(smoke_api, "call", fake_call({**GOOD, key: bad}))
    assert smoke_api.main(["--wait", "1"]) == 1


def test_smoke_fails_when_health_never_ready(monkeypatch):
    monkeypatch.setattr(smoke_api, "call", fake_call({**GOOD, "/health": (503, {"components": {"llm": {"status": "down"}}})}))
    monkeypatch.setattr(smoke_api.time, "sleep", lambda s: None)
    assert smoke_api.main(["--wait", "0.01"]) == 1


def test_api_image_is_built_locally_not_pulled():
    """Без pull_policy: build compose сначала ищет kb-agent в Docker Hub и пишет «pull access denied»."""
    api = COMPOSE["services"]["api"]
    assert api["build"] == "." and api.get("pull_policy") == "build"


def test_smoke_writes_utf8_report(monkeypatch, tmp_path):
    """Отчёт пишется самим скриптом в UTF-8: перенаправление через PowerShell портило кириллицу."""
    monkeypatch.setattr(smoke_api, "call", fake_call(GOOD))
    out = tmp_path / "deploy_check.txt"
    assert smoke_api.main(["--wait", "1", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Итог: 5 из 5 проверок пройдено" in text and "Реестр (CS-05)" in text

