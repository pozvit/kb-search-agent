"""
Тесты подагента «Документы» (шаг 2.4.2): схема без фильтров реестра, фильтр по документу ставит код
по названию продукта, запасной поиск без фильтра. Синтетические данные.
"""

import json

from conftest import FakeIndex, ScriptedLLM, reply

import agent_prompts as P

PRODUCTS = [
    {"text": "ERGO — бизнес-симулятор. В игре 5 сегментов рынка микроэлектроники: военная электроника и другие.",
     "metadata": {"document_name": "ERGO — бизнес-симулятор", "file_name": "01_ergo.pdf",
                  "category": "products/leadership_development", "chunk_index": 0, "chunk_id": "e0"}},
    {"text": "AIMS — система управления активами. Цифровое ядро: паспорт, пульс, биржа. Сегменты активов.",
     "metadata": {"document_name": "AIMS — управление активами", "file_name": "03_aims.pdf",
                  "category": "products/asset_management", "chunk_index": 0, "chunk_id": "a0"}},
]


def agent_with(config, replies, corpus=PRODUCTS, products=("ERGO", "AIMS")):
    from agent import KBAgent
    cfg = {**config, "router": {"classifier": False, "products": list(products)}}
    index = FakeIndex(corpus)
    llm = ScriptedLLM(list(replies))
    return KBAgent(cfg, llm=llm, retriever_factory=lambda: index), index, llm


def test_schema_has_no_registry_filters_and_no_list_cards(base_config):
    agent, _, llm = agent_with(base_config, [
        reply(tool_calls=[{"name": "search_knowledge_base", "arguments": {"query": "сегменты ERGO"}}]),
        reply("5 сегментов [Фрагмент 1]."),
    ])
    agent.run("Сколько сегментов рынка в ERGO?")
    names = [s["function"]["name"] for s in P.DOCUMENT_TOOL_SCHEMAS]
    props = P.DOCUMENT_TOOL_SCHEMAS[0]["function"]["parameters"]["properties"]
    assert names == ["search_knowledge_base", "list_documents"] and set(props) == {"query", "top_k"}
    assert llm.calls[0]["messages"][0]["content"] == P.DOCUMENTS_PROMPT


def test_main_ergo_regression_wrong_category_is_dropped_and_product_filter_applied(base_config):
    """Воспроизводит провал baseline CS-01: модель передала категорию AIMS и период на вопрос про ERGO."""
    agent, index, _ = agent_with(base_config, [
        reply(tool_calls=[{"name": "search_knowledge_base",
                           "arguments": {"query": "сегменты рынка ERGO", "category": "products/asset_management",
                                         "year_from": 2020, "year_to": 2023}}]),
        reply("В ERGO 5 сегментов рынка микроэлектроники, среди них военная электроника [Фрагмент 1]."),
    ])
    r = agent.run("Сколько сегментов рынка микроэлектроники моделируется в бизнес-симуляторе ERGO?")
    call = index.search_calls[-1]
    assert call["category"] is None and call["filters"] == {"files": ["01_ergo.pdf"]}
    assert r.status == "answered" and r.sources[0]["file_name"] == "01_ergo.pdf"
    decisions = {d.get("point") for d in r.trace.decisions()}
    assert {"document_filter", "drop_extra_arguments"} <= decisions


def test_question_without_product_searches_whole_base(base_config):
    agent, index, _ = agent_with(base_config, [
        reply(tool_calls=[{"name": "search_knowledge_base", "arguments": {"query": "цифровое ядро паспорт"}}]),
        reply("Паспорт, пульс, биржа [Фрагмент 1]."),
    ])
    agent.run("Как устроено цифровое ядро с паспортом активов?")
    assert not (index.search_calls[-1]["filters"] or {})


def test_product_filter_falls_back_to_whole_base_when_empty(base_config):
    class EmptyForFiles(FakeIndex):
        def search(self, query, top_k, category=None, filters=None):
            if filters and filters.get("files"):
                self.search_calls.append({"query": query, "filters": filters, "category": category, "top_k": top_k})
                return []
            return super().search(query, top_k, category, filters)

    from agent import KBAgent
    index = EmptyForFiles(PRODUCTS)
    cfg = {**base_config, "router": {"classifier": False, "products": ["ERGO", "AIMS"]}}
    llm = ScriptedLLM([reply(tool_calls=[{"name": "search_knowledge_base", "arguments": {"query": "сегменты"}}]),
                       reply("5 сегментов [Фрагмент 1].")])
    r = KBAgent(cfg, llm=llm, retriever_factory=lambda: index).run("Сколько сегментов рынка в ERGO?")
    assert [bool((c.get("filters") or {}).get("files")) for c in index.search_calls] == [True, False]
    assert r.status == "answered" and any("по всей базе" in w for w in r.warnings)


def test_forced_search_also_gets_product_filter(base_config):
    agent, index, _ = agent_with(base_config, [
        reply("ERGO — это, кажется, CRM."),                       # модель не вызвала поиск
        reply("В ERGO 5 сегментов [Фрагмент 1]."),
    ])
    r = agent.run("Сколько сегментов рынка в ERGO?")
    assert r.route == "forced_search" and index.search_calls[-1]["filters"] == {"files": ["01_ergo.pdf"]}


def test_documents_prompt_and_schema_are_much_smaller():
    """Прокси Q7: промпт и схема подагента «Документы» вдвое меньше baseline (3 000 + 819 символов)."""
    size = len(P.DOCUMENTS_PROMPT) + len(json.dumps(P.DOCUMENT_TOOL_SCHEMAS, ensure_ascii=False))
    assert size < (3000 + 819) / 2, size
    assert "только на русском" in P.DOCUMENTS_PROMPT


def test_documents_prompt_keeps_answer_on_the_asked_case():
    """Шаг 2.4.6: в CS-03 модель добавляла эффекты других проектов и пропускала пункт нужного случая."""
    assert "только на заданный вопрос" in P.DOCUMENTS_PROMPT
    assert "эффекты других проектов и общие диапазоны не добавляй" in P.DOCUMENTS_PROMPT
