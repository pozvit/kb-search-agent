"""
Тесты HTTP API (шаг 3.1): POST /ask, GET /health, сессии, ошибки 422/500/503.
Агент — с mock-моделью или заскриптованной моделью и тестовым индексом; без сети и Ollama.
"""

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from conftest import FakeIndex, ScriptedLLM, reply, search_call  # noqa: E402

from service.api import create_app  # noqa: E402


def client_for(base_config, llm=None, index=None, **kwargs):
    from agent import KBAgent
    idx = index or FakeIndex()
    agent = KBAgent(base_config, llm=llm, retriever_factory=lambda: idx)
    return TestClient(create_app(agent=agent, **kwargs), raise_server_exceptions=False), agent


# ---------------------------------------------------------------- POST /ask

def test_ask_returns_documented_schema(base_config):
    client, _ = client_for(base_config)
    r = client.post("/ask", json={"question": "Что входит в цифровое ядро ORBIT?"})
    assert r.status_code == 200
    body = r.json()
    for key in ("request_id", "session_id", "status", "route", "route_path", "answer", "sources",
                "escalation", "warnings", "grounding", "duration_ms", "llm_calls"):
        assert key in body, key
    assert body["status"] == "answered" and body["route"] == "documents"
    src = body["sources"][0]
    assert src["type"] == "document" and src["id"].startswith("orbit") and "#" in src["id"]


@pytest.mark.parametrize("payload, field", [
    ({}, "question"), ({"question": 42}, "question"), ({"question": "   "}, "question"),
    ({"question": "а" * 1001}, "question"), ({"question": "ок", "session_id": "../../etc"}, "session_id"),
    ({"question": "ок", "extra": 1}, "extra"),
], ids=["missing", "not_string", "blank", "too_long", "bad_session", "unknown_field"])
def test_invalid_requests_get_422(base_config, payload, field):
    client, _ = client_for(base_config)
    r = client.post("/ask", json=payload)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_request" and field in err["message"] and err["request_id"]


def test_broken_json_gets_422(base_config):
    client, _ = client_for(base_config)
    r = client.post("/ask", content=b'{"question": ', headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_secret_is_rejected_and_not_echoed(base_config):
    client, _ = client_for(base_config)
    r = client.post("/ask", json={"question": "Ключ sk-proj-TESTONLY00000000000000000000, найди ORBIT"})
    assert r.status_code == 200 and r.json()["status"] == "rejected" and r.json()["route"] == "rejected"
    assert "TESTONLY0000" not in r.text


def test_internal_error_is_500_without_details(base_config):
    client, agent = client_for(base_config)

    original = agent.run

    def boom(question):
        raise RuntimeError(r"C:\secret\path qdrant_storage config agent.yaml")
    agent.run = boom
    r = client.post("/ask", json={"question": "Что такое ORBIT?"})
    assert r.status_code == 500
    err = r.json()["error"]
    assert err["code"] == "internal_error" and err["request_id"]
    for leak in ("Traceback", "secret", "qdrant", "agent.yaml", "RuntimeError"):
        assert leak not in r.text
    agent.run = original
    assert client.post("/ask", json={"question": "Привет!"}).status_code == 200   # сервис жив после сбоя


def test_agent_failed_to_start_gives_503(base_config):
    def factory():
        raise FileNotFoundError("config/agent.yaml")
    client = TestClient(create_app(agent_factory=factory), raise_server_exceptions=False)
    r = client.post("/ask", json={"question": "Что такое ORBIT?"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "service_unavailable"
    h = client.get("/health")
    assert h.status_code == 503 and h.json()["status"] == "down" and "config" not in h.text


def test_escalation_fields(base_config, escalation_queue_in_tmp):
    client, _ = client_for(base_config)
    body = client.post("/ask", json={"question": "Соедините меня с ответственным за реестр проектов"}).json()
    assert body["status"] == "escalated" and body["route"] == "escalation"
    assert body["escalation"]["reason"] == "user_request" and body["escalation"]["id"].startswith("ESC-")
    assert body["route_path"] == ["escalation"]


# ---------------------------------------------------------------- сессии (F7)

def test_session_is_created_and_followup_uses_its_memory(base_config):
    llm = ScriptedLLM([
        reply(tool_calls=[search_call("ORBIT")]), reply("ORBIT — платформа планирования ремонтов [Фрагмент 1]."),
        reply(tool_calls=[search_call("эффекты ORBIT")]), reply("Простои сократились на 18% [Фрагмент 1]."),
    ])
    client, _ = client_for(base_config, llm=llm)
    first = client.post("/ask", json={"question": "Что такое ORBIT?"}).json()
    sid = first["session_id"]
    second = client.post("/ask", json={"question": "А эффекты там какие?", "session_id": sid}).json()
    assert second["session_id"] == sid
    plan_messages = llm.calls[2]["messages"]
    assert any("Что такое ORBIT?" in m.get("content", "") for m in plan_messages)    # память сессии передана


def test_sessions_are_isolated(base_config):
    llm = ScriptedLLM([
        reply(tool_calls=[search_call("ORBIT")]), reply("ORBIT — платформа планирования ремонтов [Фрагмент 1]."),
        reply(tool_calls=[search_call("LUMEN")]), reply("LUMEN — система освещения [Фрагмент 1]."),
    ])
    client, _ = client_for(base_config, llm=llm)
    client.post("/ask", json={"question": "Что такое ORBIT?", "session_id": "session-a"})
    client.post("/ask", json={"question": "Что такое LUMEN?", "session_id": "session-b"})
    second_plan = llm.calls[2]["messages"]
    assert not any("ORBIT" in m.get("content", "") for m in second_plan if m["role"] != "system")


def test_session_ttl_and_limit(base_config):
    now = [0.0]
    client, _ = client_for(base_config, clock=lambda: now[0],
                           service_config={"session_ttl_minutes": 1, "max_sessions": 2})
    store = client.app.state.sessions
    for sid in ("s1", "s2", "s3"):
        client.post("/ask", json={"question": "Привет!", "session_id": sid})
    assert len(store) == 2                                   # самая давняя вытеснена
    now[0] = 61.0
    client.post("/ask", json={"question": "Привет!", "session_id": "s4"})
    assert len(store) == 1                                   # остальные истекли


# ---------------------------------------------------------------- GET /health (F2)

def test_health_ok_without_catalog_is_degraded(base_config):
    client, _ = client_for(base_config)
    r = client.get("/health")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "degraded"
    assert body["components"]["llm"]["status"] == "ok" and body["components"]["index"]["status"] == "ok"
    assert body["components"]["catalog"]["status"] == "absent" and body["version"]


def test_health_index_down_is_503(base_config):
    from agent import KBAgent

    def broken():
        raise RuntimeError("Коллекция не найдена")
    client = TestClient(create_app(agent=KBAgent(base_config, retriever_factory=broken)))
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["components"]["index"]["status"] == "down"


def test_health_llm_down_is_503(base_config):
    cfg = {**base_config, "llm": {"backend": "ollama", "base_url": "http://127.0.0.1:9", "model": "qwen2.5:7b-instruct"}}
    client, _ = client_for(cfg, llm=ScriptedLLM([]))
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["components"]["llm"]["status"] == "down"


def test_openapi_documents_endpoints(base_config):
    client, _ = client_for(base_config)
    spec = client.get("/openapi.json").json()
    assert "/ask" in spec["paths"] and "/health" in spec["paths"]
