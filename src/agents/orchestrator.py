"""
Оркестратор KB Search Agent: порядок шагов обработки запроса.

  1. guardrails: тип, длина, секреты; явные параметры пользователя (top_k)
  2. память сессии: уточняющий вопрос, карточки прошлого ответа
  3. маршрутизатор (agents/router.py): smalltalk | escalation (E4) | registry | documents | out_of_scope
  4. подагент:
       «Реестры»   — выборка из каталога и программный ответ (agents/registry_agent.py)
       «Документы» — цикл модели с поиском и проверкой привязки (agents/documents_agent.py)
  5. эскалация к человеку по правилам E1–E4 (agents/escalation.py)
  6. ответ, trace, память

Каждое решение записывается в trace: маршрут, правило, вызовы инструментов, проверки.
"""

import re
import time
from typing import Any, Callable, Dict, List, Optional

import agent_prompts as P
from agent_guardrails import check_query, extract_locked_params
from agent_lexical import stems
from agent_llm import create_backend
from agent_memory import ShortTermMemory
from agent_models import (STATUS_ANSWERED, STATUS_DEGRADED, STATUS_ERROR, STATUS_ESCALATED, STATUS_NOT_FOUND, STATUS_REJECTED,
                          STATUS_SMALLTALK, AgentResponse, AgentState, ToolCall, Trace)
from agent_tools import QdrantRetrieverAdapter, ToolRegistry
from agents.documents_agent import DocumentsAgent
from agents.escalation import EscalationQueue
from agents.registry_agent import card_sources, catalog_answer
from agents.router import Router, parse_label

# уточняющий вопрос: начинается с «а/и/ещё» или ссылается на предыдущую тему местоимением
_FOLLOWUP = re.compile(r"(?i)^\s*(а|и|ещё|еще)\s|(?<![а-яё])(там|тут|это|эти|этих|этим|этими|этот|эту|этом|этого|этой|"
                       r"их|они|нём|нем|него|неё|нее|них|ним|ними)(?![а-яё])")
# отсылка к тому, что было найдено в прошлом ответе: «эти проекты», «по ним», «перечисленные»
_ANAPHORA = re.compile(r"(?i)(?<![а-яё])(это|эти|этих|этим|этими|этот|эту|этого|этом|этой|их|они|них|ним|ними|"
                       r"перечисленн\w*|указанн\w*|найденн\w*|вышеуказанн\w*)(?![а-яё])")
_SHORT_FOLLOWUP = re.compile(r"(?i)^\s*(а|и|ещё|еще)\s")
# слова, которые задают новую область поиска, а не уточняют прошлый ответ
_SCOPE_WORDS = re.compile(r"(?i)(?<![а-яё])(проект\w*|пропозал\w*|предложени\w*|продукт\w*|учётн\w*|учетн\w*|"
                          r"кп|реестр\w*)(?![а-яё])|(19|20)\d{2}")
_TOPIC_WORD = re.compile(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9\-]*")
# правила реестров и маршрутизации — в agents/registry_agent.py и agents/router.py


class Orchestrator(DocumentsAgent):
    def __init__(self, config: Dict[str, Any], llm: Any = None,
                 retriever_factory: Optional[Callable[[], Any]] = None,
                 memory: Optional[ShortTermMemory] = None):
        self.config = config
        agent_cfg = config.get("agent", {})
        self.name = agent_cfg.get("name", "KB Search Agent")
        self.max_iterations = int(agent_cfg.get("max_iterations", 3))
        self.max_tool_calls_per_iteration = int(agent_cfg.get("max_tool_calls_per_iteration", 3))
        self.force_search = bool(agent_cfg.get("force_search_if_no_tool", True))

        guard_cfg = config.get("guardrails", {})
        self.max_query_chars = int(guard_cfg.get("max_query_chars", 1000))
        self.block_secrets = bool(guard_cfg.get("block_secrets", True))

        ctx_cfg = config.get("context", {})
        self.max_chars_per_chunk = int(ctx_cfg.get("max_chars_per_chunk", 2000))
        self.max_context_chars = int(ctx_cfg.get("max_context_chars", 8000))

        grounding_cfg = config.get("grounding", {})
        self.min_overlap = float(grounding_cfg.get("min_overlap", 0.5))
        self.min_terms = int(grounding_cfg.get("min_terms", 3))
        self.max_regenerations = int(grounding_cfg.get("max_regenerations", 1))

        self.llm = llm if llm is not None else create_backend(config.get("llm", {}))
        factory = retriever_factory or self._default_retriever_factory(config)
        escalation_cfg = config.get("escalation", {}) or {}
        self.escalation = EscalationQueue(escalation_cfg.get("queue_path", "data/escalations/escalations.jsonl"),
                                          escalation_cfg.get("owners"), profile=str(config.get("profile", "")))
        self.tools = ToolRegistry(factory, config.get("tools", {}), catalog_path=config.get("catalog_path"),
                                  escalation_queue=self.escalation)
        self.router = Router(self.tools.catalog, config.get("router", {}))


        mem_cfg = config.get("memory", {})
        self.memory = memory if memory is not None else ShortTermMemory(
            int(mem_cfg.get("max_messages", 6)), int(mem_cfg.get("max_chars_per_message", 500)))

    @staticmethod
    def _default_retriever_factory(config: Dict[str, Any]) -> Callable[[], Any]:
        backend = config.get("retriever", {}).get("backend", "qdrant")
        if backend == "lexical":
            chunks_path = config.get("retriever", {}).get("chunks_jsonl")
            from agent_lexical import LexicalRetrieverAdapter
            return lambda: LexicalRetrieverAdapter(chunks_path)
        if backend == "qdrant":
            rag_config = config.get("rag", {})
            return lambda: QdrantRetrieverAdapter(rag_config)
        raise ValueError(f"Неизвестный retriever.backend='{backend}'. Допустимо: qdrant, lexical.")

    def describe(self) -> str:
        backend = self.config.get("retriever", {}).get("backend", "qdrant")
        return (f"{self.name} | профиль {self.config.get('profile')} | retriever {backend} | "
                f"{self.llm.describe()}")

    def close(self) -> None:
        self.tools.close()

    def run(self, query: Any, request_id: Optional[str] = None) -> AgentResponse:
        started = time.monotonic()
        trace = Trace()

        # --- 2. GUARDRAILS -------------------------------------------------
        guard = check_query(query, self.max_query_chars, self.block_secrets)
        trace.add("state", stage="received", query=guard.safe_query, memory_size=len(self.memory))
        trace.add("decision", point="guardrails", allowed=guard.allowed, reason=guard.reason_code,
                  secret_kinds=guard.secret_kinds)
        state = AgentState(query=guard.safe_query, request_id=request_id)
        if not guard.allowed:
            state.route = "guardrail_block"
            return self._finish(state, trace, started, STATUS_REJECTED, guard.message,
                                error_code=guard.reason_code, remember=False)

        query = query.strip()
        state.locked_params = extract_locked_params(query)
        if state.locked_params:
            trace.add("decision", point="user_params", locked=state.locked_params,
                      rule="параметры пользователя подставляются в tool поверх значений LLM")

        state.followup_terms = self._followup_terms(query)
        state.followup_cards = self._previous_cards_for(query)
        if state.followup_cards:
            trace.add("decision", point="followup_cards", cards=len(state.followup_cards),
                      rule="вопрос ссылается на прошлый ответ: поиск ограничен карточками, которые в нём цитировались")
        if state.followup_terms:
            trace.add("decision", point="followup_context", terms=state.followup_terms,
                      rule="уточняющий вопрос: термины темы из памяти будут добавлены в query поиска, "
                           "если модель их не включила")

        is_followup = bool(state.followup_terms) or (len(self.memory) > 0 and bool(_FOLLOWUP.search(query)))
        decision = self.router.decide(query, state.followup_cards, is_followup,
                                      classify=lambda q: self._classify(q, state, trace))
        trace.add("decision", point="route_decided", **decision.trace_data())
        state.route_path.append("smalltalk" if decision.route == "out_of_scope" else decision.route)
        if decision.route == "escalation":
            state.route = "escalation"
            return self._escalate(state, trace, started, "user_request", topic=decision.details.get("topic"))
        if decision.route == "smalltalk":
            state.route = "smalltalk"
            return self._finish(state, trace, started, STATUS_SMALLTALK, P.SMALLTALK_ANSWER)
        if decision.route == "out_of_scope":
            state.route = "smalltalk"          # для внешних отчётов это тот же маршрут: ответ без поиска
            return self._finish(state, trace, started, STATUS_SMALLTALK, P.OUT_OF_SCOPE_ANSWER)
        if decision.route == "registry":
            return self._answer_registry(decision, state, trace, started)
        return self._run_documents(query, decision, state, trace, started)

    def _escalate(self, state: AgentState, trace: Trace, started: float, reason: str, topic: Optional[str] = None,
                  note: str = "", error_code: Optional[str] = None) -> AgentResponse:
        """Инструмент escalate_to_human: обращение в очередь и ответ пользователю с номером и ответственным."""
        if not state.route_path or state.route_path[-1] != "escalation":
            state.route_path.append("escalation")
        evidence = [{"type": "card", "id": f"{s['card'].get('card_type')}:{s['card'].get('card_id')}"} if s.get("card")
                    else {"type": "document", "id": f"{s.get('file_name')}#{s.get('chunk_index')}"}
                    for s in [f.source_dict() for f in state.fragments]]
        args = {"reason": reason, "question": state.query or "", "route_path": list(state.route_path),
                "request_id": state.request_id,
                "evidence": evidence, "topic": topic}
        call = ToolCall(name="escalate_to_human", iteration=state.iteration, source="rules",
                        arguments={"reason": reason, "topic": topic, "evidence": len(evidence)})
        trace.add("tool_call", name=call.name, arguments=call.arguments, source="rules", iteration=state.iteration)
        result, latency = self.tools.execute("escalate_to_human", args)
        call.latency_ms = latency
        state.tool_calls.append(call)
        ticket = result.get("ticket") if result.get("status") == "success" else None
        if ticket is None:
            call.status, call.error_code, call.error = "error", result.get("error_code"), result.get("error")
            ticket = {"id": None, "owner": self.escalation.owner_for(reason, topic), "queued": False}
        else:
            call.status = "success"
            call.result_summary = {"id": ticket["id"], "queued": ticket["queued"]}
        trace.add("tool_result", name=call.name, latency_ms=latency, status=call.status, ticket=ticket.get("id"),
                  queued=ticket.get("queued"))
        trace.add("decision", point="escalation", reason=reason, owner=ticket["owner"], ticket=ticket.get("id"),
                  rule={"ungrounded_answer": "E1", "no_data": "E2", "component_failure": "E3",
                        "user_request": "E4"}[reason])
        text = P.ESCALATION_TEMPLATES[reason].format(owner=ticket["owner"], id=ticket.get("id") or "—", note=note)
        if not ticket.get("queued"):
            text += P.ESCALATION_NOT_QUEUED
        escalation = {"id": ticket.get("id"), "reason": reason, "owner": ticket["owner"], "queued": bool(ticket.get("queued"))}
        return self._finish(state, trace, started, STATUS_ESCALATED, text,
                            sources=[f.source_dict() for f in state.fragments], remember=False,
                            escalation=escalation, error_code=error_code)

    def _tool_error_response(self, state, trace, started, err: ToolCall) -> AgentResponse:
        if err.error_code == "index_unavailable":
            # E3: без индекса или каталога агент ответить не может — сбой передаётся администратору
            hint = P.TOOL_ERROR_HINTS.get("index_unavailable", "")
            return self._escalate(state, trace, started, "component_failure", note="\n" + hint if hint else "",
                                  error_code=err.error_code)
        text = P.TOOL_ERROR_TEMPLATE.format(error=err.error)
        hint = P.TOOL_ERROR_HINTS.get(err.error_code or "")
        if hint:
            text += "\n" + hint
        return self._finish(state, trace, started, STATUS_ERROR, text, error_code=err.error_code, remember=False)

    def _classify(self, query: str, state: AgentState, trace: Trace) -> Optional[str]:
        """Короткий вызов модели без инструментов: registry | documents | out_of_scope; None — недоступна."""
        messages = [{"role": "system", "content": P.CLASSIFIER_PROMPT}, {"role": "user", "content": query}]
        reply = self._call_llm(messages, None, state, trace, stage="classify")
        if reply is None:
            state.llm_error = None          # недоступность модели обработает подагент «Документы»
            return None
        label = parse_label(reply.content)
        trace.add("decision", point="classifier", label=label, raw=(reply.content or "")[:40])
        return label

    def _answer_registry(self, decision, state: AgentState, trace: Trace, started: float) -> AgentResponse:
        """Подагент «Реестры»: выборка из каталога и программный ответ, без генерации."""
        state.route = {"rank": "catalog_people_rank", "people": "catalog_people"}.get(decision.kind, "catalog_list")
        messages: List[Dict[str, Any]] = []
        call = self._execute_and_append("list_cards", dict(decision.args), "router", state, trace, messages)
        if call.status != "success":
            return self._tool_error_response(state, trace, started, call)
        result = state.cards_result
        status = STATUS_ANSWERED if result["total"] else STATUS_NOT_FOUND
        return self._finish(state, trace, started, status, catalog_answer(result, decision.intro),
                            sources=card_sources(result["cards"]))

    def _previous_cards_for(self, query: str) -> List[Dict[str, str]]:
        """
        Карточки прошлого ответа, если текущий вопрос к ним относится:
        - есть отсылка («эти проекты», «их», «по ним», «перечисленные»);
        - или это короткое уточнение с «а/и/ещё» («А по людям?», «А сроки?»), в котором нет
          новой области поиска: отрасли из индекса, года, типа реестра.
        «А в энергетике?» или «А пропозалы?» — новый поиск, а не уточнение.
        """
        cards = self.memory.last_cards()
        if not cards:
            return []
        if _ANAPHORA.search(query):
            return cards
        if _SHORT_FOLLOWUP.search(query) and len(query.split()) <= 5 and not _SCOPE_WORDS.search(query):
            facets = self.tools.facets() or {}
            industry_stems = {st for name in (facets.get("industries") or {}) for st in stems(name)}
            if not set(stems(query)) & industry_stems:
                return cards
        return []

    def _followup_terms(self, query: str, max_terms: int = 6) -> List[str]:
        """
        Ключевые слова предыдущего вопроса, которых нет в текущем, — только для
        уточняющих вопросов и только при непустой памяти. Стоп-слова и короткие слова
        отбрасываются (та же токенизация, что в agent_lexical): «Что такое SIPS?» -> ["SIPS"].
        """
        if not len(self.memory) or not _FOLLOWUP.search(query):
            return []
        previous = self.memory.last_user_message()
        current = self._term_keys(query)
        words = _TOPIC_WORD.findall(previous)
        # названия (аббревиатуры, латиница, заглавная буква не в начале фразы) точнее описывают тему,
        # чем общие слова: «Какие эффекты даёт IBP?» -> ["IBP"], а не ["эффекты", "даёт", "IBP"]
        names = [w for i, w in enumerate(words)
                 if re.search(r"[A-Za-z]", w) or (sum(ch.isupper() for ch in w) >= 2)
                 or (i > 0 and w[:1].isupper())]
        candidates = names or words
        terms: List[str] = []
        seen = set()
        for word in candidates:
            key = self._term_key(word)
            if key is None or key in current or key in seen:
                continue
            seen.add(key)
            terms.append(word)
            if len(terms) >= max_terms:
                break
        return terms

    @staticmethod
    def _term_key(word: str) -> Optional[str]:
        """Ключ сравнения терминов: основа слова, а для аббревиатур (IBP, ERGO, ТОиР) — само слово."""
        word_stems = stems(word)
        if word_stems:
            return word_stems[0]
        letters = [ch for ch in word if ch.isalpha()]
        if len(letters) >= 2 and sum(ch.isupper() for ch in letters) >= 2:
            return word.lower()
        return None

    def _term_keys(self, text: str) -> set:
        return {k for k in (self._term_key(w) for w in _TOPIC_WORD.findall(text)) if k}

    def _finish(self, state: AgentState, trace: Trace, started: float, status: str, answer: str,
                error_code: Optional[str] = None, sources: Optional[List[Dict[str, Any]]] = None,
                remember: bool = True, escalation: Optional[Dict[str, Any]] = None) -> AgentResponse:
        if status == STATUS_ANSWERED:
            # документ продукта последнего ответа — для уточнений; ответ без фильтра сбрасывает тему
            self.memory.remember_documents(state.document_filter)
        if status == STATUS_ANSWERED and sources:
            cited_cards = [{"card_type": s["card"]["card_type"], "card_id": s["card"]["card_id"],
                            "card_title": s["card"].get("card_title", "")}
                           for s in sources if s.get("card", {}).get("card_type") and s["card"].get("card_id")]
            if cited_cards:
                self.memory.remember_cards(cited_cards)
                trace.add("memory", action="remember_cards", cards=len(cited_cards))
        if remember and status in (STATUS_ANSWERED, STATUS_NOT_FOUND, STATUS_SMALLTALK, STATUS_DEGRADED):
            before = len(self.memory)
            self.memory.add_turn(state.query, answer)
            trace.add("memory", action="add_turn", size_before=before, size_after=len(self.memory))
        trace.add("final", status=status, error_code=error_code, state=state.snapshot(),
                  warnings=list(state.warnings))
        return AgentResponse(
            status=status,
            answer=answer,
            route=state.route,
            sources=sources or [],
            tool_calls=list(state.tool_calls),
            llm_calls=state.llm_calls,
            warnings=list(state.warnings),
            error_code=error_code,
            trace=trace,
            memory_size=len(self.memory),
            duration_ms=int((time.monotonic() - started) * 1000),
            escalation=escalation,
            grounding=state.grounding,
            route_path=list(state.route_path),
        )
