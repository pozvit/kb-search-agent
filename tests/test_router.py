"""
Тесты маршрутизатора (шаг 2.4.1): правила по порядку, классификатор для неоднозначных вопросов,
буква «ё» в фамилиях. Синтетические данные, модель — заскриптованная или mock.
"""

import json

import pytest

from conftest import FakeIndex, ScriptedLLM, reply

from agents.router import Router, parse_label

CATALOG = [
    {"file_name": "projects_2025.md", "card_number": 1, "card_type": "project", "card_id": "1", "card_title": "Клиники",
     "year": 2025, "year_end": 2025, "industry": ["Pharma & HC"], "service": ["Стратегия"], "team": ["Белозёрова Б."]},
    {"file_name": "projects_2019.md", "card_number": 1, "card_type": "project", "card_id": "2", "card_title": "Завод",
     "year": 2019, "year_end": 2019, "industry": ["Industrial goods"], "service": ["Операционная эффективность"],
     "team": ["Ангарский А.", "Белозёрова Б."]},
]


@pytest.fixture
def router_config(base_config, tmp_path):
    path = tmp_path / "cards_meta.jsonl"
    path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in CATALOG), encoding="utf-8")
    return {**base_config, "catalog_path": str(path),
            "router": {"classifier": True, "products": ["ORBIT", "LUMEN"]}}


def run(config, question, replies=()):
    from agent import KBAgent
    llm = ScriptedLLM(list(replies))
    agent = KBAgent(config, llm=llm, retriever_factory=lambda: FakeIndex())
    r = agent.run(question)
    decided = next(d for d in r.trace.decisions() if d.get("point") == "route_decided")
    return r, decided, llm


def test_greeting_needs_no_model(router_config):
    r, d, llm = run(router_config, "Привет!")
    assert r.status == "smalltalk" and d["rule"] == "greeting" and llm.calls == []


@pytest.mark.parametrize("question, kind", [
    ("Какие проекты были в медицине?", "list"),
    ("Кто делал больше всего проектов в операционной эффективности?", "rank"),
    ("Проекты Белозёровой за последние 3 года", "list"),       # дефект baseline №8: «ё»
    ("Проекты Белозеровой", "list"),                            # та же фамилия, набрана через «е»
], ids=["list", "rank", "yo", "ye"])
def test_registry_questions_go_to_catalog_without_model(router_config, question, kind):
    r, d, llm = run(router_config, question)
    assert d["route"] == "registry" and d["kind"] == kind and llm.calls == []
    assert r.route.startswith("catalog") and r.status in ("answered", "not_found")


def test_person_with_yo_lists_her_cards(router_config):
    r, _, _ = run(router_config, "Проекты Белозёровой")
    assert {s["card"]["card_id"] for s in r.sources} == {"1", "2"}


@pytest.mark.parametrize("question, rule", [
    ("Что входит в цифровое ядро ORBIT?", "product_name"),
    ("Какие эффекты получены на химическом заводе?", "document_intent"),
], ids=["product", "intent"])
def test_document_questions_skip_classifier(router_config, question, rule):
    r, d, llm = run(router_config, question, [
        reply(tool_calls=[{"name": "search_knowledge_base", "arguments": {"query": question}}]),
        reply("Ответ [Фрагмент 1]."),
    ])
    assert d["route"] == "documents" and d["rule"] == rule and not d["classifier"]


def test_ambiguous_question_classified_as_registry(router_config):
    r, d, llm = run(router_config, "Что у нас есть по фарме?", [reply("registry")])
    assert d["route"] == "registry" and d["classifier"] and d["args"]["industry"] == ["Pharma & HC"]
    assert len(llm.calls) == 1 and llm.calls[0]["tools"] is False
    assert r.route == "catalog_list" and [s["card"]["card_id"] for s in r.sources] == ["1"]


def test_out_of_scope_question_gets_polite_refusal_without_search(router_config):
    r, d, llm = run(router_config, "Какая погода будет завтра в Москве?", [reply("out_of_scope")])
    assert d["route"] == "out_of_scope" and r.status == "smalltalk" and r.route == "smalltalk"
    assert r.tool_calls == [] and "базе знаний" in r.answer


@pytest.mark.parametrize("classifier_reply", ["что-то непонятное", LLMErrorMarker := "error"], ids=["garbage", "down"])
def test_classifier_failure_falls_back_to_documents(router_config, classifier_reply):
    from llm_client import LLMError
    first = LLMError("нет соединения") if classifier_reply == "error" else reply(classifier_reply)
    r, d, _ = run(router_config, "Расскажите что-нибудь полезное", [
        first,
        reply(tool_calls=[{"name": "search_knowledge_base", "arguments": {"query": "полезное"}}]),
        reply("Недостаточно информации в предоставленных документах."),
    ])
    assert d["route"] == "documents" and d["rule"] in ("classifier_unavailable",)


def test_followup_continues_with_documents(router_config):
    from agent import KBAgent
    llm = ScriptedLLM([
        reply(tool_calls=[{"name": "search_knowledge_base", "arguments": {"query": "ORBIT"}}]),
        reply("ORBIT — платформа планирования ремонтов [Фрагмент 1]."),
        reply(tool_calls=[{"name": "search_knowledge_base", "arguments": {"query": "эффекты ORBIT"}}]),
        reply("Простои сократились на 18% [Фрагмент 1]."),
    ])
    agent = KBAgent(router_config, llm=llm, retriever_factory=lambda: FakeIndex())
    agent.run("Что такое ORBIT?")
    r = agent.run("А эффекты там какие на химическом заводе?")
    decided = next(d for d in r.trace.decisions() if d.get("point") == "route_decided")
    assert decided["route"] == "documents" and not decided["classifier"]


def test_parse_label():
    assert parse_label("registry") == "registry"
    assert parse_label("Ответ: out_of_scope.") is None or parse_label("out_of_scope.") == "out_of_scope"
    assert parse_label("documents\nпотому что…") == "documents"
    assert parse_label("не знаю") is None


def test_mock_model_can_classify():
    from agent_llm import MockLLM
    from agent_prompts import CLASSIFIER_PROMPT
    m = MockLLM()
    ask = lambda q: m.chat([{"role": "system", "content": CLASSIFIER_PROMPT}, {"role": "user", "content": q}]).content
    assert ask("Что у нас есть по фарме?") == "registry"
    assert ask("Какие документы есть в базе знаний?") == "documents"
    assert ask("Какая погода будет завтра?") == "out_of_scope"


@pytest.mark.parametrize("question, expected", [
    ("Проекты Белозёровой", ["Белозёрова Б."]), ("проекты Белозеровой", ["Белозёрова Б."]),
    ("проекты Ангарского", ["Ангарский А."]), ("Что делала Камская", ["Камская К."]),
    ("проекты Камской", ["Камская К."]), ("Петровой", ["Петрова Н."]), ("Петровым", ["Петров П."]),
    ("Петрова", ["Петрова Н."]),
], ids=["fem_yo", "fem_ye", "adj_masc", "adj_fem_nom", "adj_fem", "namesake_fem", "namesake_masc", "exact"])
def test_surnames_in_any_case(tmp_path, question, expected):
    from agent_cards import CardCatalog
    rows = [{"card_type": "project", "card_id": str(i), "card_title": "x", "team": [n]}
            for i, n in enumerate(["Белозёрова Б.", "Ангарский А.", "Камская К.", "Петров П.", "Петрова Н."])]
    path = tmp_path / "c.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    assert CardCatalog(str(path)).find_person(question) == expected
