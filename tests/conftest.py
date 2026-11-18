"""Общие фикстуры: синтетический корпус, фейковый индекс, сценарные LLM. Без Qdrant, torch и Ollama."""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from agent_llm import LLMReply  # noqa: E402

# Синтетический корпус (вымышленные продукты) — тесты не зависят от data/ и не содержат реальных данных.
CORPUS = [
    {"text": "Что такое ORBIT. ORBIT — платформа планирования ремонтов оборудования.\n"
             "Цифровое ядро ORBIT:\n• цифровой журнал дефектов;\n• цифровой прогноз отказов;\n• цифровой склад запчастей.",
     "metadata": {"document_name": "ORBIT — планирование ремонтов", "file_name": "orbit.pdf",
                  "category": "products/maintenance", "chunk_index": 0, "chunk_id": "c-orbit-0"}},
    {"text": "Эффекты внедрения ORBIT на химическом заводе: сокращение простоев на 18%, "
             "снижение запасов запчастей на 12%, срок проекта 7 месяцев.",
     "metadata": {"document_name": "ORBIT — планирование ремонтов", "file_name": "orbit.pdf",
                  "category": "products/maintenance", "chunk_index": 1, "chunk_id": "c-orbit-1"}},
    {"text": "Что такое LUMEN. LUMEN — система прогнозирования спроса для розничных сетей.\n"
             "В пилоте точность прогноза выросла с 71% до 86%.",
     "metadata": {"document_name": "LUMEN — прогноз спроса", "file_name": "lumen.pdf",
                  "category": "products/forecasting", "chunk_index": 0, "chunk_id": "c-lumen-0"}},
]


class FakeIndex:
    """Лексический индекс поверх CORPUS с тем же интерфейсом, что у QdrantRetrieverAdapter."""

    def __init__(self, corpus=None, fail_on_search: Exception = None):
        from agent_lexical import stems
        self._stems = stems
        self.corpus = corpus if corpus is not None else CORPUS
        self.fail_on_search = fail_on_search
        self.search_calls = []

    def search(self, query, top_k, category=None):
        self.search_calls.append({"query": query, "top_k": top_k, "category": category})
        if self.fail_on_search:
            raise self.fail_on_search
        q = set(self._stems(query))
        scored = []
        for i, c in enumerate(self.corpus):
            if category and c["metadata"]["category"] != category:
                continue
            overlap = len(q & set(self._stems(c["text"] + " " + c["metadata"]["document_name"])))
            if overlap:
                scored.append((overlap, -i, c))
        scored.sort(reverse=True)
        return [{"rank": r, "score": float(s), "chunk_id": c["metadata"]["chunk_id"], "document_id": "d",
                 "text": c["text"], "metadata": dict(c["metadata"])}
                for r, (s, _, c) in enumerate(scored[:top_k], start=1)]

    def list_documents(self):
        seen = {}
        for c in self.corpus:
            m = c["metadata"]
            seen.setdefault(m["file_name"], {"document_name": m["document_name"], "category": m["category"],
                                             "file_name": m["file_name"], "version": "n/a", "chunks": 0})
            seen[m["file_name"]]["chunks"] += 1
        return list(seen.values())

    def close(self):
        pass


class ScriptedLLM:
    """LLM, отдающая заранее заданную последовательность ответов (для точечных проверок логики агента)."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def describe(self):
        return "ScriptedLLM"

    def chat(self, messages, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": bool(tools)})
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def reply(content="", tool_calls=None):
    return LLMReply(content=content, tool_calls=tool_calls or [])


def search_call(query, **kwargs):
    return {"name": "search_knowledge_base", "arguments": {"query": query, **kwargs}}


@pytest.fixture
def base_config():
    return {
        "profile": "test",
        "agent": {"name": "KB Search Agent", "max_iterations": 3, "max_tool_calls_per_iteration": 3,
                  "force_search_if_no_tool": True},
        "llm": {"backend": "mock"},
        "retriever": {"backend": "lexical"},
        "tools": {"search": {"default_top_k": 4, "max_top_k": 10, "max_query_chars": 500}},
        "context": {"max_chars_per_chunk": 2000, "max_context_chars": 8000},
        "guardrails": {"max_query_chars": 1000, "block_secrets": True},
        "memory": {"max_messages": 6, "max_chars_per_message": 500},
        "validation": {"check_numbers": True},
        # продукты синтетического корпуса; классификатор выключен: эти тесты проверяют подагентов,
        # маршрутизация с классификатором — в tests/test_router.py
        "router": {"classifier": False, "products": ["ORBIT", "LUMEN"]},
        "rag": {},
    }


@pytest.fixture
def make_agent(base_config):
    from agent import KBAgent
    from agent_llm import MockLLM

    created = []

    def _make(llm=None, index=None, **cfg_updates):
        cfg = dict(base_config)
        cfg.update(cfg_updates)
        idx = index if index is not None else FakeIndex()
        agent = KBAgent(cfg, llm=llm or MockLLM(), retriever_factory=lambda: idx)
        agent._test_index = idx
        created.append(agent)
        return agent

    yield _make
    for a in created:
        a.close()
