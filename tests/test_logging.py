"""
Тесты структурированного журнала (шаг 3.2, требования S2, S3): одна строка JSON на событие,
обязательные события есть, а текста вопросов, ответов, имён файлов, фамилий, секретов и
сообщений исключений в журнале нет.
"""

import io
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from conftest import FakeIndex, ScriptedLLM, reply, search_call  # noqa: E402

from service.api import create_app  # noqa: E402

SECRET = "sk-proj-TESTONLY00000000000000000000"


def run_requests(base_config, requests, llm=None, crash=False):
    from agent import KBAgent
    stream = io.StringIO()
    agent = KBAgent(base_config, llm=llm, retriever_factory=lambda: FakeIndex())
    client = TestClient(create_app(agent=agent, log_stream=stream), raise_server_exceptions=False)
    responses = []
    for payload in requests:
        responses.append(client.post("/ask", json=payload))
    if crash:
        def boom(q):
            raise RuntimeError(r"C:\Users\Ivan.Petrov\KB\data\qdrant_storage: секретная причина")
        agent.run = boom
        responses.append(client.post("/ask", json={"question": "Что такое ORBIT?"}))
    lines = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    return lines, responses


QUESTIONS = [
    {"question": "Что входит в цифровое ядро ORBIT по словам Иванова Ивана?", "session_id": "my-private-session"},
    {"question": "Соедините меня с ответственным за реестр проектов"},
    {"question": f"Мой ключ {SECRET}, найди ORBIT"},
]


def test_every_line_is_json_with_event_and_request_id(base_config, escalation_queue_in_tmp):
    lines, _ = run_requests(base_config, QUESTIONS)
    assert lines and all({"ts", "level", "event", "request_id"} <= set(line) for line in lines)
    assert lines[0]["event"] == "service_started" and lines[0]["agent_ready"] is True     # видно, что сервис поднялся
    assert all(line["request_id"] for line in lines if line["event"] != "service_started")


def test_required_events_present(base_config, escalation_queue_in_tmp):
    lines, _ = run_requests(base_config, QUESTIONS)
    events = {line["event"] for line in lines}
    assert {"request_received", "guardrail_decision", "route_decided", "tool_call", "llm_call",
            "escalation_created", "response_sent"} <= events
    sent = [line for line in lines if line["event"] == "response_sent"]
    assert [s["status"] for s in sent] == ["answered", "escalated", "rejected"]
    secret_line = next(line for line in lines if line["event"] == "guardrail_decision" and not line["allowed"])
    assert secret_line["secret_kinds"] and secret_line["reason"]


def test_no_question_answer_files_names_or_secrets_in_log(base_config, escalation_queue_in_tmp):
    lines, responses = run_requests(base_config, QUESTIONS, crash=True)
    log = json.dumps(lines, ensure_ascii=False)
    answer = responses[0].json()["answer"]
    forbidden = [
        "цифровое ядро ORBIT", "Иванова", "Ивана",                   # текст вопроса и фамилия
        answer[:40],                                                  # текст ответа
        "orbit.pdf", "lumen", ".pdf", ".md",                          # имена файлов источников
        SECRET, "TESTONLY",                                           # секрет
        "my-private-session",                                         # сырой идентификатор сессии
        "секретная причина", "qdrant_storage", "Ivan.Petrov",         # сообщение исключения и путь
        "Соедините меня",                                             # текст вопроса эскалации
    ]
    leaked = [f for f in forbidden if f and f.lower() in log.lower()]
    assert not leaked, leaked


def test_grounding_and_regeneration_are_logged(base_config):
    llm = ScriptedLLM([reply(tool_calls=[search_call("эффекты ORBIT")]),
                       reply("Внеплановые простои оборудования сократились на 45% [Фрагмент 1]."),
                       reply("Внеплановые простои оборудования сократились на 18% [Фрагмент 1].")])
    lines, _ = run_requests(base_config, [{"question": "Какие эффекты ORBIT?"}], llm=llm)
    checks = [line for line in lines if line["event"] == "grounding_check"]
    assert [c["grounded"] for c in checks] == [False, True] and checks[-1]["regenerations"] == 1
    stages = [line["stage"] for line in lines if line["event"] == "llm_call"]
    assert stages == ["plan", "answer", "regenerate"]
    assert "45%" not in json.dumps(lines, ensure_ascii=False)        # неподкреплённое утверждение не в журнале


def test_internal_error_and_invalid_request_are_logged_without_details(base_config):
    lines, _ = run_requests(base_config, [{"question": ""}, {"question": 1}], crash=True)
    invalid = [line for line in lines if line["event"] == "invalid_request"]
    assert len(invalid) == 2 and invalid[0]["fields"] == ["question"] and invalid[0]["level"] == "WARNING"
    err = next(line for line in lines if line["event"] == "error")
    assert err == {**err, "code": "internal_error", "exception": "RuntimeError", "where": "/ask", "level": "ERROR"}


def test_session_is_hashed_consistently(base_config):
    lines, _ = run_requests(base_config, [{"question": "Привет!", "session_id": "abc"},
                                          {"question": "Привет!", "session_id": "abc"}])
    received = [line for line in lines if line["event"] == "request_received"]
    assert received[0]["session_hash"] == received[1]["session_hash"] != "abc"
    assert received[0]["new_session"] is True and received[1]["new_session"] is False


def test_text_format_for_humans(base_config, monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "text")
    lines_raw = io.StringIO()
    from agent import KBAgent
    client = TestClient(create_app(agent=KBAgent(base_config, retriever_factory=lambda: FakeIndex()), log_stream=lines_raw))
    client.post("/ask", json={"question": "Привет!"})
    assert "response_sent" in lines_raw.getvalue() and "Привет" not in lines_raw.getvalue()


def test_registry_tool_call_logs_card_count(base_config, tmp_path):
    from agent import KBAgent
    path = tmp_path / "cards_meta.jsonl"
    path.write_text(json.dumps({"card_type": "project", "card_id": "1", "card_title": "Клиники", "year": 2025,
                                "year_end": 2025, "industry": ["Pharma & HC"]}, ensure_ascii=False), encoding="utf-8")
    stream = io.StringIO()
    agent = KBAgent({**base_config, "catalog_path": str(path)}, retriever_factory=lambda: FakeIndex())
    TestClient(create_app(agent=agent, log_stream=stream)).post("/ask", json={"question": "Какие проекты были в медицине?"})
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    call = next(line for line in lines if line["event"] == "tool_call")
    assert call["name"] == "list_cards" and call["count"] == 1 and "Клиники" not in stream.getvalue()
