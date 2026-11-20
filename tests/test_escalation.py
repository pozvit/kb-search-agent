"""
Тесты эскалации к человеку (шаг 2.4.4): очередь обращений, инструмент escalate_to_human, правила E1–E4.
Синтетические данные; очередь пишется во временную папку (фикстура escalation_queue_in_tmp).
"""

import json

import pytest

from conftest import ScriptedLLM, reply

from agents.escalation import NO_DATA_INTENT, USER_REQUEST, EscalationQueue


def read_queue(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_queue_numbers_tickets_per_day(tmp_path, monkeypatch):
    monkeypatch.delenv("ESCALATION_QUEUE_PATH", raising=False)
    q = EscalationQueue(str(tmp_path / "q.jsonl"))
    a = q.create(reason="no_data", question="Сколько стоит?", route_path=["documents", "escalation"], evidence=[])
    b = q.create(reason="user_request", question="Соедините", route_path=["escalation"], evidence=[], topic="registry")
    assert a["id"].endswith("-0001") and b["id"].endswith("-0002") and a["queued"] and b["queued"]
    assert b["owner"] == "владелец реестра проектов"
    rows = read_queue(tmp_path / "q.jsonl")
    assert [r["reason"] for r in rows] == ["no_data", "user_request"] and rows[0]["status"] == "new"


def test_queue_write_failure_does_not_break_answer(make_agent, monkeypatch, tmp_path):
    blocked = tmp_path / "dir_instead_of_file"
    blocked.mkdir()
    monkeypatch.setenv("ESCALATION_QUEUE_PATH", str(blocked))      # записать в папку нельзя
    r = make_agent().run("Соедините меня с ответственным за реестр проектов")
    assert r.status == "escalated" and r.escalation["queued"] is False
    assert "не удалось записать" in r.answer


@pytest.mark.parametrize("question", [
    "Соедините меня с ответственным за реестр проектов",
    "Хочу поговорить с человеком",
    "Передайте вопрос специалисту по КП",
    "Нужно связаться с менеджером",
], ids=["connect", "talk", "pass_on", "contact"])
def test_user_request_phrases(question):
    assert USER_REQUEST.search(question)


@pytest.mark.parametrize("question", [
    "Свяжи эти данные с реестром", "Кто менеджер проекта ERGO?", "Передай список проектов в медицине",
], ids=["link_data", "who_is_manager", "pass_list"])
def test_not_user_request(question):
    assert not USER_REQUEST.search(question)


def test_e4_escalates_without_model_with_owner_by_topic(make_agent, escalation_queue_in_tmp):
    llm = ScriptedLLM([])
    r = make_agent(llm=llm).run("Соедините меня с ответственным за реестр проектов")
    assert r.status == "escalated" and r.escalation["reason"] == "user_request" and llm.calls == []
    assert r.escalation["owner"] == "владелец реестра проектов" and r.route_path == ["escalation"]
    assert r.tool_names == ["escalate_to_human"]
    r2 = make_agent(llm=ScriptedLLM([])).run("Передайте вопрос специалисту по коммерческим предложениям")
    assert r2.escalation["owner"] == "владелец базы пропозалов"


@pytest.mark.parametrize("question, expected", [
    ("Сколько стоит лицензия?", True), ("Какова стоимость внедрения?", True), ("Дайте телефон менеджера", True),
    ("Какие эффекты дал проект?", False), ("Сколько сегментов рынка в игре?", False),
], ids=["cost", "cost_noun", "phone", "effects", "how_many_segments"])
def test_no_data_intent(question, expected):
    assert bool(NO_DATA_INTENT.search(question)) == expected


def test_fabricated_price_is_e2_not_e1(make_agent):
    """Модель выдумала цену: проверка привязки её не пропустит, причина — нет данных в базе (E2)."""
    llm = ScriptedLLM([reply(tool_calls=[{"name": "search_knowledge_base", "arguments": {"query": "цена ORBIT"}}]),
                       reply("Лицензия ORBIT стоит 1 200 000 рублей в год."),
                       reply("Лицензия ORBIT стоит 1 200 000 рублей в год.")])
    r = make_agent(llm=llm).run("Сколько стоит лицензия ORBIT?")
    assert r.status == "escalated" and r.escalation["reason"] == "no_data" and "1 200 000" not in r.answer


def test_secret_never_reaches_queue(make_agent, escalation_queue_in_tmp):
    r = make_agent().run("Мой ключ sk-proj-TESTONLY00000000000000000000, соедините меня с человеком")
    assert r.status == "rejected" and not escalation_queue_in_tmp.exists()


def test_out_of_scope_is_not_escalated(base_config, escalation_queue_in_tmp):
    from agent import KBAgent
    cfg = {**base_config, "router": {"classifier": True, "products": ["ORBIT"]}}
    r = KBAgent(cfg, llm=ScriptedLLM([reply("out_of_scope")])).run("Какая погода будет завтра?")
    assert r.status == "smalltalk" and r.escalation is None and not escalation_queue_in_tmp.exists()


def test_tool_rejects_unknown_reason(make_agent):
    agent = make_agent()
    res, _ = agent.tools.execute("escalate_to_human", {"reason": "просто так", "question": "?"})
    assert res["status"] == "error" and res["error_code"] == "invalid_argument"
