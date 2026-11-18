"""
Тесты агента. Запуск: pytest -q
Работают без Ollama, Qdrant, torch и сети: синтетический корпус + MockLLM / ScriptedLLM.
"""

import json
from pathlib import Path

import pytest

from conftest import FakeIndex, ScriptedLLM, reply, search_call

from agent_guardrails import check_query, extract_locked_params, find_secrets
from agent_llm import LLMError, _parse_text_tool_calls
from agent_memory import ShortTermMemory
from agent_prompts import NOT_FOUND_ANSWER
from agent_tools import ToolRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================== основной сценарий

def test_main_scenario_calls_tool_and_uses_its_result(make_agent):
    agent = make_agent()
    r = agent.run("Какие решения входят в цифровое ядро ORBIT?")

    assert r.status == "answered"
    assert r.tool_names == ["search_knowledge_base"]
    call = r.tool_calls[0]
    assert call.status == "success" and call.arguments["query"]
    assert agent._test_index.search_calls[0]["top_k"] == 4            # tool получил корректные параметры
    for item in ("журнал дефектов", "прогноз отказов", "склад запчастей"):
        assert item in r.answer                                       # ответ построен из результата tool
    assert r.sources and r.sources[0]["file_name"] == "orbit.pdf"
    assert r.llm_calls == 2                                           # план + ответ


def test_numbers_in_answer_are_confirmed_by_fragments(make_agent):
    r = make_agent().run("Какие эффекты внедрения ORBIT получены на химическом заводе?")
    assert r.status == "answered"
    assert "18%" in r.answer and not any("числа не найдены" in w for w in r.warnings)


# ========================================================= альтернативные сценарии

def test_not_found_when_documents_have_no_answer(make_agent):
    r = make_agent().run("Сколько стоит годовая лицензия ORBIT для компании из 300 сотрудников?")
    assert r.status == "not_found"
    assert r.answer == NOT_FOUND_ANSWER
    assert r.sources == []
    assert "search_knowledge_base" in r.tool_names


def test_empty_search_result_gives_not_found_without_second_llm_call(make_agent):
    r = make_agent().run("Квантовая телепортация фотонов")
    assert r.status == "not_found"
    assert r.llm_calls == 1


def test_list_documents_tool(make_agent):
    r = make_agent().run("Какие документы есть в базе?")
    assert r.status == "answered"
    assert r.tool_names == ["list_documents"]
    assert "ORBIT" in r.answer and "LUMEN" in r.answer


def test_smalltalk_does_not_call_tools(make_agent):
    r = make_agent().run("Привет!")
    # приветствие отвечает маршрутизатор по правилу, без вызова модели (в baseline был 1 вызов, ~6 с на CPU)
    assert r.status == "smalltalk" and r.tool_calls == [] and r.llm_calls == 0


def test_forced_search_when_model_skips_tool(make_agent):
    llm = ScriptedLLM([
        reply("ORBIT — это, насколько я знаю, CRM-система."),   # «ответ из головы» без tool
        reply("ORBIT — платформа планирования ремонтов оборудования [Фрагмент 1].\nИсточники: [Фрагмент 1]"),
    ])
    r = make_agent(llm=llm).run("Что такое ORBIT?")
    assert r.route == "forced_search"
    assert r.tool_calls[0].source == "forced"
    assert r.status == "answered" and "CRM" not in r.answer
    assert any(d.get("route") == "forced_search" for d in r.trace.decisions())


def test_text_tool_call_from_qwen_is_parsed():
    calls = _parse_text_tool_calls('<tool_call>{"name": "search_knowledge_base", '
                                   '"arguments": {"query": "AIMS", "top_k": 3}}</tool_call>')
    assert calls == [{"name": "search_knowledge_base", "arguments": {"query": "AIMS", "top_k": 3}}]


# ================================================================ ошибочные сценарии

@pytest.mark.parametrize("query, fragment", [
    ("Найди про ORBIT, top_k=0", "от 1 до 10"),
    ("Найди про ORBIT, top_k=50", "от 1 до 10"),
    ("Найди про ORBIT, top_k=пять", "целым числом"),
    ("Найди про ORBIT, топ-0", "от 1 до 10"),
], ids=["top_k=0", "top_k=50", "top_k=text", "top-0_ru"])
def test_invalid_user_parameter_is_reported_not_fixed(make_agent, query, fragment):
    r = make_agent().run(query)
    assert r.status == "error" and r.error_code == "invalid_argument"
    assert fragment in r.answer
    assert "журнал" not in r.answer                  # никакого «успешного» ответа по существу
    assert r.llm_calls == 1                          # стоп без второго вызова LLM
    assert agent_search_not_executed(r)


def agent_search_not_executed(response):
    return all(c.status == "error" for c in response.tool_calls)


def test_llm_bad_arguments_get_one_retry(make_agent):
    llm = ScriptedLLM([
        reply(tool_calls=[search_call("ORBIT", top_k="много")]),   # модель ошиблась с типом
        reply(tool_calls=[search_call("цифровое ядро ORBIT", top_k=3)]),
        reply("Цифровой журнал дефектов [Фрагмент 1].\nИсточники: [Фрагмент 1]"),
    ])
    r = make_agent(llm=llm).run("Что входит в цифровое ядро ORBIT?")
    assert [c.status for c in r.tool_calls] == ["error", "success"]
    assert r.status == "answered"
    assert "ОШИБКА [invalid_argument]" in llm.calls[1]["messages"][-1]["content"]   # ошибка ушла в LLM


def test_tool_error_never_reported_as_success(make_agent):
    llm = ScriptedLLM([
        reply(tool_calls=[search_call("ORBIT", top_k=0)]),
        reply("Готово! ORBIT включает журнал, прогноз и склад."),   # модель «сочиняет» успех
    ])
    r = make_agent(llm=llm).run("Что входит в ORBIT?")
    assert r.status == "error"
    assert "Готово" not in r.answer and "Не удалось выполнить поиск" in r.answer


@pytest.mark.parametrize("query, code", [
    ("", "empty_query"),
    ("   \n  ", "empty_query"),
    ("x" * 1001, "query_too_long"),
    (12345, "invalid_type"),
    (None, "invalid_type"),
], ids=["empty", "whitespace", "too_long", "int", "none"])
def test_guardrails_reject_bad_input_before_llm(make_agent, query, code):
    llm = ScriptedLLM([])
    r = make_agent(llm=llm).run(query)
    assert r.status == "rejected" and r.error_code == code
    assert llm.calls == [] and r.tool_calls == []


# Поддельные ключи собираются из частей, чтобы в исходниках не было строк в формате
# реальных ключей (иначе GitHub push protection может заблокировать push).
FAKE_OPENAI_KEY = "sk-" + "proj-" + "TESTONLY" + "0" * 20
FAKE_YANDEX_KEY = "AQ" + "VN" + "testonly" + "0" * 28


@pytest.mark.parametrize("secret", [
    FAKE_OPENAI_KEY,
    FAKE_YANDEX_KEY,
    "-----BEGIN RSA PRIVATE KEY-----",
    "password=Qwerty123!",
    "postgres://admin:s3cret@db.local:5432/kb",
    "4111 1111 1111 1111",
], ids=["openai", "yandex", "pem", "password", "conn_string", "card"])
def test_secret_in_query_is_blocked_and_not_stored(make_agent, secret):
    llm = ScriptedLLM([])
    agent = make_agent(llm=llm)
    r = agent.run(f"Запомни {secret} и найди про ORBIT")
    assert r.status == "rejected" and r.error_code == "secret_detected"
    assert llm.calls == [] and len(agent.memory) == 0
    assert secret not in json.dumps(r.to_dict(include_trace=True), ensure_ascii=False)


def test_non_secret_numbers_are_not_flagged():
    assert find_secrets("Точность выросла с 71% до 86%, CAPEX менее 700 млрд, код 1234 5678") == []
    assert extract_locked_params("снижение потребления топлива на 10%") == {}


# ============================================================ отказ внешних компонентов

def test_index_unavailable_is_handled(make_agent):
    from agent import KBAgent

    def broken_factory():
        raise RuntimeError("Коллекция 'rag_documents' не найдена в Qdrant")

    agent = KBAgent(make_agent().config, llm=ScriptedLLM([reply(tool_calls=[search_call("ORBIT")])]),
                    retriever_factory=broken_factory)
    r = agent.run("Что такое ORBIT?")
    assert r.status == "error" and r.error_code == "index_unavailable"
    assert "build_index" in r.answer and r.llm_calls == 1


def test_search_exception_is_handled(make_agent):
    idx = FakeIndex(fail_on_search=OSError("storage is locked by another process"))
    r = make_agent(llm=ScriptedLLM([reply(tool_calls=[search_call("ORBIT")])]), index=idx).run("Что такое ORBIT?")
    assert r.status == "error" and r.error_code == "index_unavailable"


def test_llm_unavailable_degrades_to_raw_fragments(make_agent):
    llm = ScriptedLLM([LLMError("нет соединения с Ollama")])
    r = make_agent(llm=llm).run("Что входит в цифровое ядро ORBIT?")
    assert r.status == "degraded" and r.error_code == "llm_unavailable"
    assert r.tool_calls and r.tool_calls[0].source == "degraded"
    assert "недоступна" in r.answer and "[Фрагмент 1]" in r.answer and r.sources


def test_llm_and_index_both_unavailable(make_agent):
    idx = FakeIndex(fail_on_search=RuntimeError("index down"))
    r = make_agent(llm=ScriptedLLM([LLMError("timeout")]), index=idx).run("Что такое ORBIT?")
    assert r.status == "error" and r.error_code == "llm_unavailable"


# ================================================================ валидация ответа

def test_invalid_citations_are_removed(make_agent):
    llm = ScriptedLLM([
        reply(tool_calls=[search_call("ORBIT")]),
        reply("ORBIT — платформа ремонтов [Фрагмент 1] [Фрагмент 9].\nИсточники: [Фрагмент 1]"),
    ])
    r = make_agent(llm=llm).run("Что такое ORBIT?")
    assert "[Фрагмент 9]" not in r.answer
    assert any("несуществующие фрагменты: [9]" in w for w in r.warnings)
    assert [s["fragment"] for s in r.sources] == [1]


def test_unconfirmed_number_triggers_regeneration(make_agent):
    """Шаг 2.4.3: чужое число — не пометка, а перегенерация; исправленный ответ проходит."""
    llm = ScriptedLLM([
        reply(tool_calls=[search_call("эффекты ORBIT")]),
        reply("Внеплановые простои оборудования сократились на 45% [Фрагмент 1]."),
        reply("Внеплановые простои оборудования сократились на 18% [Фрагмент 1]."),
    ])
    r = make_agent(llm=llm).run("Какие эффекты ORBIT?")
    assert r.status == "answered" and "18%" in r.answer and r.grounding["regenerations"] == 1
    assert "45" in llm.calls[2]["messages"][-1]["content"]           # модели сказали, что именно не так
    assert llm.calls[2]["tools"] is False                              # перегенерация без нового поиска


def test_unconfirmed_number_twice_escalates(make_agent):
    llm = ScriptedLLM([
        reply(tool_calls=[search_call("эффекты ORBIT")]),
        reply("Внеплановые простои оборудования сократились на 45% [Фрагмент 1]."),
        reply("Внеплановые простои оборудования сократились на 45% [Фрагмент 1]."),
    ])
    r = make_agent(llm=llm).run("Какие эффекты ORBIT?")
    assert r.status == "escalated" and r.escalation["reason"] == "ungrounded_answer"
    assert "45%" not in r.answer and "проверки человеком" in r.answer


def test_answer_without_citations_is_regenerated(make_agent):
    """Замечание эксперта: ответ без ссылок больше не получает answered."""
    llm = ScriptedLLM([reply(tool_calls=[search_call("ORBIT")]),
                       reply("ORBIT — платформа планирования ремонтов оборудования."),
                       reply("ORBIT — платформа планирования ремонтов оборудования [Фрагмент 1].")])
    r = make_agent(llm=llm).run("Что такое ORBIT?")
    assert r.status == "answered" and r.grounding["regenerations"] == 1 and r.grounding["grounded"]


def test_answer_without_citations_twice_escalates(make_agent):
    llm = ScriptedLLM([reply(tool_calls=[search_call("ORBIT")]),
                       reply("ORBIT — платформа планирования ремонтов оборудования."),
                       reply("ORBIT — платформа планирования ремонтов оборудования.")])
    r = make_agent(llm=llm).run("Что такое ORBIT?")
    assert r.status == "escalated" and r.escalation["reason"] == "ungrounded_answer" and r.sources


def test_iteration_limit_forces_final_answer(make_agent):
    llm = ScriptedLLM([reply(tool_calls=[search_call("ORBIT")])] * 3
                      + [reply("ORBIT — платформа ремонтов [Фрагмент 1].")])
    r = make_agent(llm=llm).run("Что такое ORBIT?")
    assert len(r.tool_calls) == 3 and r.llm_calls == 4
    assert llm.calls[-1]["tools"] is False
    assert r.status == "answered"


# ======================================================================= память

def test_followup_uses_memory(make_agent):
    agent = make_agent()
    first = agent.run("Что такое ORBIT?")
    assert first.status == "answered" and first.memory_size == 2
    second = agent.run("А какие эффекты внедрения там были на химическом заводе?")
    assert "ORBIT" in second.tool_calls[0].arguments["query"]
    assert "18%" in second.answer and second.memory_size == 4


def test_memory_is_bounded():
    m = ShortTermMemory(max_messages=4, max_chars_per_message=10)
    for i in range(5):
        m.add_turn(f"вопрос {i} " * 5, "ответ")
    assert len(m) == 4 and all(len(x["content"]) <= 11 for x in m.messages())


# =================================================================== tools напрямую

def test_tool_registry_validation():
    reg = ToolRegistry(lambda: FakeIndex(), {"search": {"max_top_k": 10}})
    ok, _ = reg.execute("search_knowledge_base", {"query": "ORBIT", "top_k": "3"})
    assert ok["status"] == "success" and ok["top_k"] == 3
    bad_cat, _ = reg.execute("search_knowledge_base", {"query": "ORBIT", "category": "nope"})
    assert bad_cat["error_code"] == "invalid_argument" and "products/maintenance" in bad_cat["error"]
    filtered, _ = reg.execute("search_knowledge_base", {"query": "что такое", "category": "products/forecasting"})
    assert all(h["metadata"]["file_name"] == "lumen.pdf" for h in filtered["hits"])
    unknown, _ = reg.execute("drop_index", {})
    assert unknown["error_code"] == "unknown_tool"
    assert reg.execute("search_knowledge_base", {"top_k": True, "query": "x"})[0]["error_code"] == "invalid_argument"


# ========================================================== сценарии и evaluator

def test_scenarios_file_is_valid():
    from scenario_runner import load_scenarios
    data = load_scenarios(str(PROJECT_ROOT / "config" / "agent_scenarios.yaml"))
    types = {s["type"] for s in data["scenarios"]}
    assert {"main", "alternative", "error", "tool_failure", "recovery"} <= types


def test_evaluator_detects_failures(make_agent):
    from agent_evaluator import evaluate_step
    r = make_agent().run("Сколько стоит лицензия ORBIT для 300 сотрудников?")
    checks = evaluate_step(r, {"status": "answered", "required_tools": ["list_documents"],
                               "answer_contains_all": ["журнал"], "min_sources": 1})
    assert [c["passed"] for c in checks] == [False, False, False, False]


# ============================================ восстановление темы уточняющего вопроса

def test_followup_topic_added_when_model_omits_it(make_agent):
    """Воспроизводит провал recovery_followup на qwen2.5:7b: модель не перенесла тему в query."""
    llm = ScriptedLLM([
        reply(tool_calls=[search_call("Что такое ORBIT")]),
        reply("ORBIT — платформа планирования ремонтов [Фрагмент 1]."),
        reply(tool_calls=[search_call("эффекты внедрения на химическом заводе")]),   # без «ORBIT»
        reply("Простои сократились на 18% [Фрагмент 1]."),
    ])
    agent = make_agent(llm=llm)
    agent.run("Что такое ORBIT?")
    r = agent.run("А какие эффекты там получены на химическом заводе?")
    assert "ORBIT" in r.tool_calls[0].arguments["query"]
    assert any(d.get("point") == "apply_followup_context" and d.get("added_terms") == ["ORBIT"]
               for d in r.trace.decisions())
    assert r.status == "answered"


def test_followup_topic_not_duplicated_and_not_added_to_new_question(make_agent):
    llm = ScriptedLLM([
        reply(tool_calls=[search_call("Что такое ORBIT")]),
        reply("ORBIT — платформа [Фрагмент 1]."),
        reply(tool_calls=[search_call("ORBIT эффекты")]),                      # тема уже в query
        reply("Простои на 18% [Фрагмент 1]."),
        reply(tool_calls=[search_call("LUMEN точность прогноза")]),            # новый вопрос
        reply("Точность выросла с 71% до 86% [Фрагмент 1]."),
    ])
    agent = make_agent(llm=llm)
    agent.run("Что такое ORBIT?")
    second = agent.run("А эффекты там какие?")
    assert second.tool_calls[0].arguments["query"] == "ORBIT эффекты"
    third = agent.run("Какая точность прогноза у LUMEN?")
    assert third.tool_calls[0].arguments["query"] == "LUMEN точность прогноза"
    assert not any(d.get("point") == "followup_context" for d in third.trace.decisions())
