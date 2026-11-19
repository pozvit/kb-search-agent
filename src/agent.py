"""
KBAgent — агент поиска по базе знаний.

Цепочка одного запуска (точки решений пишутся в trace как события "decision"):

  1. RECEIVE      запрос пользователя
  2. GUARDRAILS   тип / пустота / длина / секреты             -> rejected (без LLM и tools)
                  фиксация параметров пользователя (top_k=N)
  3. PLAN         LLM(system + память + вопрос, tools)          -> tool_calls | текст
                  нет tool_calls и это не приветствие          -> принудительный поиск (forced)
  4. ACT          выполнение tools, валидация аргументов в tool
                  index_unavailable / неверный параметр пользователя -> error (шаблон, без LLM)
                  0 фрагментов                                   -> not_found (без LLM)
  5. ANSWER       LLM(... + результаты tools) -> текст; не более max_iterations кругов
  6. VALIDATE     ссылки [Фрагмент N] ⊆ реально полученных; фраза «нет информации» -> not_found;
                  числа в ответе сверяются с текстом фрагментов
  7. RESPOND      статус, ответ, источники, вызовы tools, trace; обновление памяти

Недоступность LLM на любом шаге -> degraded: резервный поиск без LLM и честное
сообщение с найденными фрагментами. Ответ об успехе при ошибке tool невозможен:
в ветках ошибок текст берётся из шаблонов agent_prompts, а не от модели.
"""

import re
import time
from typing import Any, Callable, Dict, List, Optional

from llm_client import LLMError

import agent_prompts as P
from agent_guardrails import check_query, extract_locked_params
from agent_lexical import stems
from agent_llm import LLMReply, create_backend
from agent_memory import ShortTermMemory
from agent_models import (STATUS_ANSWERED, STATUS_DEGRADED, STATUS_ERROR, STATUS_ESCALATED, STATUS_NOT_FOUND,
                          STATUS_REJECTED, STATUS_SMALLTALK, AgentResponse, AgentState, ToolCall, Trace)
from agent_tools import (QdrantRetrieverAdapter, ToolRegistry, format_documents, format_error, format_fragments,
                         hits_to_fragments)
from agents.escalation import NO_DATA_INTENT, EscalationQueue
from agents.grounding import check_grounding
from agents.registry_agent import card_sources, catalog_answer
from agents.router import Router, parse_label

_SMALLTALK = re.compile(
    r"(?i)^\s*(привет|здравствуй|здравствуйте|добрый (день|вечер|утро)|спасибо|благодарю|пока)\b[\s!.,)]*$")
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

_CITATION = re.compile(r"(?i)фрагмент\w*\s*((?:\d+\s*(?:,|и|;)\s*)*\d+)")
_BRACKET_REF = re.compile(r"\[\s*Фрагмент\s+(\d+)\s*\]")
_NOT_FOUND_NORMALIZED = P.NOT_FOUND_ANSWER.lower().rstrip(".")

# ошибки, после которых повторять вызов бессмысленно: сразу шаблонный ответ
_FATAL_TOOL_ERRORS = {"index_unavailable", "internal_error"}


class KBAgent:
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

    # =====================================================================
    # Основной цикл
    # =====================================================================

    def run(self, query: Any) -> AgentResponse:
        started = time.monotonic()
        trace = Trace()

        # --- 2. GUARDRAILS -------------------------------------------------
        guard = check_query(query, self.max_query_chars, self.block_secrets)
        trace.add("state", stage="received", query=guard.safe_query, memory_size=len(self.memory))
        trace.add("decision", point="guardrails", allowed=guard.allowed, reason=guard.reason_code,
                  secret_kinds=guard.secret_kinds)
        state = AgentState(query=guard.safe_query)
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

        # --- подагент «Документы» ----------------------------------------------
        product = decision.details.get("product")
        if product:
            files = self.tools.files_for_product(product)
            if files:
                state.document_filter = files
                trace.add("decision", point="document_filter", product=product, files=files,
                          rule="фильтр по документу продукта из вопроса ставит код, а не модель")
        tool_schemas = P.DOCUMENT_TOOL_SCHEMAS

        messages: List[Dict[str, Any]] = [{"role": "system", "content": P.DOCUMENTS_PROMPT}]
        messages += self.memory.messages()
        messages.append({"role": "user", "content": query})

        final_text: Optional[str] = None
        pending_tools = False

        # --- 3–5. PLAN / ACT / ANSWER ---------------------------------------
        for iteration in range(1, self.max_iterations + 1):
            state.iteration = iteration
            reply = self._call_llm(messages, tool_schemas, state, trace,
                                   stage="plan" if not state.tool_calls else "answer")
            if reply is None:
                return self._degraded(state, trace, started, query)

            if reply.tool_calls:
                state.route = state.route or "tools"
                pending_tools = True
                calls = reply.tool_calls[: self.max_tool_calls_per_iteration]
                if len(reply.tool_calls) > len(calls):
                    state.warnings.append(f"LLM запросила {len(reply.tool_calls)} вызовов, "
                                          f"выполнено {len(calls)} (лимит на итерацию)")
                self._append_assistant_tool_calls(messages, reply.content, calls)
                for call in calls:
                    self._execute_and_append(call["name"], call.get("arguments") or {}, "llm",
                                             state, trace, messages)
                stop = self._check_stop_after_tools(state, trace, started)
                if stop is not None:
                    return stop
                continue

            # модель ответила текстом без вызова tools
            if not state.tool_calls:
                if _SMALLTALK.match(query) or not self.force_search:
                    state.route = "smalltalk"
                    trace.add("decision", point="route", route="smalltalk",
                              reason="приветствие" if _SMALLTALK.match(query) else "force_search отключён")
                    text = reply.content or P.SMALLTALK_ANSWER
                    return self._finish(state, trace, started, STATUS_SMALLTALK, text)
                state.route = "forced_search"
                trace.add("decision", point="route", route="forced_search",
                          reason="LLM не вызвала tool на вопрос по базе знаний — поиск выполняется принудительно",
                          llm_text_discarded=reply.content[:200])
                forced_call = {"name": "search_knowledge_base", "arguments": {"query": query}}
                self._append_assistant_tool_calls(messages, "", [forced_call])
                self._execute_and_append(forced_call["name"], forced_call["arguments"], "forced",
                                         state, trace, messages)
                pending_tools = True
                stop = self._check_stop_after_tools(state, trace, started)
                if stop is not None:
                    return stop
                continue

            final_text = reply.content
            pending_tools = False
            break

        if pending_tools:
            # лимит итераций исчерпан, модель всё ещё хотела вызывать tools: финальный ответ без tools
            trace.add("decision", point="iteration_limit", max_iterations=self.max_iterations,
                      action="финальный вызов LLM без tools")
            state.warnings.append(f"достигнут лимит итераций ({self.max_iterations})")
            reply = self._call_llm(messages, None, state, trace, stage="final_no_tools")
            if reply is None:
                return self._degraded(state, trace, started, query)
            final_text = reply.content

        return self._compose_final(state, trace, started, final_text or "", messages)

    # =====================================================================
    # Шаги
    # =====================================================================

    def _call_llm(self, messages, tools, state: AgentState, trace: Trace, stage: str) -> Optional[LLMReply]:
        state.llm_calls += 1
        try:
            reply = self.llm.chat(messages, tools)
        except LLMError as exc:
            state.llm_error = str(exc)
            trace.add("llm_call", stage=stage, iteration=state.iteration, ok=False, error=str(exc))
            trace.add("decision", point="llm_unavailable", action="degraded: резервный поиск без генерации")
            return None
        trace.add("llm_call", stage=stage, iteration=state.iteration, ok=True,
                  messages=len(messages), tools_offered=bool(tools), latency_ms=reply.latency_ms,
                  usage=reply.usage, tool_calls=reply.tool_calls,
                  tool_call_parsed_from_text=reply.raw_tool_call_in_text,
                  content_preview=(reply.content or "")[:200])
        return reply

    @staticmethod
    def _append_assistant_tool_calls(messages, content: str, calls: List[Dict[str, Any]]) -> None:
        messages.append({
            "role": "assistant",
            "content": content or "",
            "tool_calls": [{"function": {"name": c["name"], "arguments": c.get("arguments") or {}}}
                           for c in calls],
        })

    def _execute_and_append(self, name: str, arguments: Dict[str, Any], source: str,
                            state: AgentState, trace: Trace, messages: List[Dict[str, Any]]) -> ToolCall:
        args = dict(arguments) if isinstance(arguments, dict) else arguments
        if source == "llm" and isinstance(args, dict):
            allowed = P.DOCUMENT_TOOL_ARGS.get(name)
            if allowed is None:
                # инструмента нет в схеме подагента «Документы» (например, list_cards): не выполняем
                return self._reject_tool(name, args, state, trace, messages)
            extra = sorted(set(args) - allowed)
            if extra:
                for key in extra:
                    args.pop(key)
                trace.add("decision", point="drop_extra_arguments", tool=name, dropped=extra,
                          rule="аргументы вне схемы подагента отбрасываются до вызова инструмента")
                state.warnings.append(f"аргументы вне схемы отброшены: {', '.join(extra)}")
        if name == "search_knowledge_base" and isinstance(args, dict) and state.document_filter \
                and "files" not in args:
            args["files"] = list(state.document_filter)
        if name in ("search_knowledge_base", "list_cards") and isinstance(args, dict):
            self._apply_previous_cards(args, state, trace, bump_top_k=(name == "search_knowledge_base"))
        if name == "search_knowledge_base" and isinstance(args, dict) and "top_k" in state.locked_params:
            if args.get("top_k") != state.locked_params["top_k"]:
                trace.add("decision", point="apply_locked_param", param="top_k",
                          llm_value=args.get("top_k"), user_value=state.locked_params["top_k"])
            args["top_k"] = state.locked_params["top_k"]

        if (name == "search_knowledge_base" and state.followup_terms and isinstance(args, dict)
                and isinstance(args.get("query"), str) and args["query"].strip()):
            present = self._term_keys(args["query"])
            missing = [t for t in state.followup_terms if self._term_key(t) not in present]
            if missing:
                args["query"] = f"{args['query'].strip()} {' '.join(missing)}"
                trace.add("decision", point="apply_followup_context", added_terms=missing,
                          query=args["query"], source=source)

        call = ToolCall(name=name, arguments=args if isinstance(args, dict) else {"__raw__": args},
                        iteration=state.iteration, source=source)
        trace.add("tool_call", name=name, arguments=call.arguments, source=source, iteration=state.iteration)
        result, latency = self.tools.execute(name, args)
        call.latency_ms = latency
        state.tool_calls.append(call)

        if (result.get("status") != "success" and name == "search_knowledge_base" and source == "llm"
                and result.get("error_code") == "invalid_argument"
                and result.get("argument") in ("industry", "category", "year_from", "year_to")
                and isinstance(args, dict)):
            # фильтр придумала модель, а не пользователь: не обрываем ответ, ищем без него
            bad = result["argument"]
            trace.add("decision", point="drop_invalid_filter", argument=bad, value=args.get(bad),
                      reason=result.get("error"), action="повтор поиска без этого фильтра")
            state.warnings.append(f"фильтр {bad}={args.get(bad)!r} отклонён ({result.get('error')}); "
                                  f"поиск выполнен без него")
            retry_args = {k: v for k, v in args.items() if k != bad}
            call.arguments = retry_args
            trace.add("tool_call", name=name, arguments=retry_args, source="filter_dropped", iteration=state.iteration)
            result, latency = self.tools.execute(name, retry_args)
            call.latency_ms += latency

        if (result.get("status") == "success" and name == "search_knowledge_base" and not result.get("count")
                and isinstance(args, dict) and args.get("files")):
            # по документу продукта ничего не нашлось — ищем по всей базе, но говорим об этом
            retry_args = {k: v for k, v in args.items() if k != "files"}
            trace.add("decision", point="document_filter_fallback", files=args["files"],
                      action="повтор поиска без фильтра по документу")
            state.warnings.append("по документу продукта из вопроса ничего не найдено, поиск выполнен по всей базе")
            call.arguments = retry_args
            result, latency = self.tools.execute(name, retry_args)
            call.latency_ms += latency

        if result.get("status") != "success":
            call.status, call.error_code, call.error = "error", result.get("error_code"), result.get("error")
            state.last_tool_error = call
            trace.add("tool_error", name=name, error_code=call.error_code, error=call.error, latency_ms=latency)
            content = format_error(result)
        elif name == "search_knowledge_base":
            call.status = "success"
            fragments = hits_to_fragments(result["hits"], len(state.fragments),
                                          self.max_chars_per_chunk, self.max_context_chars)
            state.fragments.extend(fragments)
            call.result_summary = {
                "count": result["count"], "fragments_in_context": len(fragments),
                "top_k": result["top_k"], "category": result["category"],
                "numbers": [f.number for f in fragments],
                "top_score": fragments[0].score if fragments else None,
                "files": sorted({f.file_name for f in fragments}),
            }
            trace.add("tool_result", name=name, latency_ms=latency, **call.result_summary)
            content = format_fragments(fragments, result["query"])
        elif name == "list_cards":
            call.status = "success"
            state.cards_result = result
            call.result_summary = {"total": result["total"], "shown": result["shown"], "filters": result["filters"],
                                   "include_people": result["include_people"]}
            trace.add("tool_result", name=name, latency_ms=latency, **call.result_summary)
            content = f"Карточек по условиям: {result['total']} (показано {result['shown']}).\n{result['text']}"
        else:
            call.status = "success"
            state.documents_listed = True
            call.result_summary = {"count": result["count"], "truncated": result.get("truncated", False)}
            state.documents_result = result
            trace.add("tool_result", name=name, latency_ms=latency, **call.result_summary)
            content = format_documents(result)

        messages.append({"role": "tool", "content": content, "tool_name": name})
        return call

    def _ungrounded_response(self, state: AgentState, trace: Trace, started: float) -> AgentResponse:
        """
        E1: подтвердить ответ источниками не удалось — уверенный ответ не выдаётся.
        Если вопрос о данных, которых в базе не бывает (цена, контакты), первопричина — их отсутствие: E2.
        """
        reason = "no_data" if NO_DATA_INTENT.search(state.query or "") else "ungrounded_answer"
        previews = "\n\n".join(
            f"[Фрагмент {f.number}] {f.document_name} ({f.file_name})\n{f.text[:300]}{'…' if len(f.text) > 300 else ''}"
            for f in state.fragments[:4])
        note = f"\n\nНайденные фрагменты:\n\n{previews}" if previews and reason == "ungrounded_answer" else ""
        return self._escalate(state, trace, started, reason, note=note)

    def _not_found(self, state: AgentState, trace: Trace, started: float) -> AgentResponse:
        """
        Ответа в документах нет. Обычно это честное not_found. Но если вопрос о данных, которых в базе знаний
        не бывает (стоимость, контакты, планы, внутренние оценки), ответить может только человек — E2.
        """
        if NO_DATA_INTENT.search(state.query or ""):
            return self._escalate(state, trace, started, "no_data")
        return self._finish(state, trace, started, STATUS_NOT_FOUND, P.NOT_FOUND_ANSWER)

    def _escalate(self, state: AgentState, trace: Trace, started: float, reason: str, topic: Optional[str] = None,
                  note: str = "", error_code: Optional[str] = None) -> AgentResponse:
        """Инструмент escalate_to_human: обращение в очередь и ответ пользователю с номером и ответственным."""
        if not state.route_path or state.route_path[-1] != "escalation":
            state.route_path.append("escalation")
        evidence = [{"type": "card", "id": f"{s['card'].get('card_type')}:{s['card'].get('card_id')}"} if s.get("card")
                    else {"type": "document", "id": f"{s.get('file_name')}#{s.get('chunk_index')}"}
                    for s in [f.source_dict() for f in state.fragments]]
        args = {"reason": reason, "question": state.query or "", "route_path": list(state.route_path),
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


    def _reject_tool(self, name: str, args: Dict[str, Any], state: AgentState, trace: Trace,
                     messages: List[Dict[str, Any]]) -> ToolCall:
        call = ToolCall(name=name, arguments=args, iteration=state.iteration, source="llm",
                        status="error", error_code="unknown_tool",
                        error=f"инструмент {name} недоступен подагенту «Документы»")
        state.tool_calls.append(call)
        state.last_tool_error = call
        trace.add("tool_error", name=name, error_code=call.error_code, error=call.error, latency_ms=0)
        messages.append({"role": "tool", "content": f"ОШИБКА [unknown_tool]: {call.error}", "tool_name": name})
        return call

    def _check_stop_after_tools(self, state: AgentState, trace: Trace, started: float) -> Optional[AgentResponse]:
        err = state.last_tool_error
        if err is not None and err is state.tool_calls[-1]:
            if err.error_code in _FATAL_TOOL_ERRORS:
                trace.add("decision", point="tool_failed", error_code=err.error_code,
                          action="стоп: шаблонный ответ об ошибке, LLM не вызывается")
                return self._tool_error_response(state, trace, started, err)
            if (err.error_code == "invalid_argument" and err.name == "search_knowledge_base"
                    and "top_k" in state.locked_params and "top_k" in (err.error or "")):
                trace.add("decision", point="tool_failed", error_code=err.error_code,
                          action="стоп: некорректный параметр задан пользователем, молча исправлять нельзя")
                return self._tool_error_response(state, trace, started, err)
            trace.add("decision", point="tool_failed", error_code=err.error_code,
                      action="ошибка возвращена LLM для исправления аргументов")

        searches = state.successful_calls("search_knowledge_base")
        if searches and not state.fragments and not state.documents_listed:
            trace.add("decision", point="empty_results", action="not_found без вызова LLM")
            state.route = state.route or "tools"
            return self._not_found(state, trace, started)
        return None

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

    def _degraded(self, state: AgentState, trace: Trace, started: float, query: str) -> AgentResponse:
        state.route = "degraded"
        if not state.successful_calls("search_knowledge_base"):
            messages: List[Dict[str, Any]] = []
            self._execute_and_append("search_knowledge_base", {"query": query}, "degraded",
                                     state, trace, messages)
            if state.tool_calls[-1].status != "success":
                # E3: недоступны и модель, и поиск
                return self._escalate(state, trace, started, "component_failure", error_code="llm_unavailable")
        if not state.fragments:
            text = (f"Языковая модель недоступна ({state.llm_error}), а поиск не нашёл подходящих "
                    f"фрагментов. {P.NOT_FOUND_ANSWER}")
            return self._finish(state, trace, started, STATUS_DEGRADED, text, error_code="llm_unavailable")
        previews = "\n\n".join(
            f"[Фрагмент {f.number}] {f.document_name} ({f.file_name}, score {f.score:.4f})\n"
            f"{f.text[:400]}{'…' if len(f.text) > 400 else ''}" for f in state.fragments)
        text = P.LLM_UNAVAILABLE_TEMPLATE.format(error=state.llm_error, fragments=previews)
        sources = [f.source_dict() for f in state.fragments]
        return self._finish(state, trace, started, STATUS_DEGRADED, text,
                            error_code="llm_unavailable", sources=sources)

    # =====================================================================
    # Валидация ответа
    # =====================================================================

    def _compose_final(self, state: AgentState, trace: Trace, started: float, text: str,
                       messages: Optional[List[Dict[str, Any]]] = None, regenerations: int = 0) -> AgentResponse:
        text = (text or "").strip()
        has_search = bool(state.successful_calls("search_knowledge_base"))

        if not has_search and state.cards_result is not None:
            result = state.cards_result
            answer = text or catalog_answer(result, "")
            trace.add("validation", kind="catalog", ok=True)
            return self._finish(state, trace, started, STATUS_ANSWERED if result["total"] else STATUS_NOT_FOUND,
                                answer, sources=card_sources(result["cards"]))

        if not has_search and not state.documents_listed:
            # сюда попадаем, только если все вызовы tools завершились ошибкой и LLM ответила текстом
            if state.last_tool_error is not None:
                trace.add("decision", point="validation", action="успешных вызовов tools нет — текст LLM отброшен")
                return self._tool_error_response(state, trace, started, state.last_tool_error)
            return self._not_found(state, trace, started)

        if not has_search and state.documents_listed:
            docs = state.documents_result
            answer = text or (format_documents(docs) if docs else "")
            trace.add("validation", kind="documents", ok=True)
            return self._finish(state, trace, started, STATUS_ANSWERED, answer)

        if _NOT_FOUND_NORMALIZED in text.lower().replace("«", "").replace("»", "") or not text:
            if regenerations:
                # первый ответ был содержательным, но не подтвердился, а после перегенерации «нет информации»:
                # это не честное «данных нет», а непроверенный ответ — E1
                trace.add("validation", kind="not_found_after_regeneration", ok=False)
                return self._ungrounded_response(state, trace, started)
        if _NOT_FOUND_NORMALIZED in text.lower().replace("«", "").replace("»", ""):
            trace.add("validation", kind="not_found_phrase", ok=True)
            return self._not_found(state, trace, started)

        if not text:
            state.warnings.append("LLM вернула пустой ответ")
            trace.add("validation", kind="empty_answer", ok=False)
            return self._not_found(state, trace, started)

        valid_numbers = {f.number for f in state.fragments}
        cited = set()
        for group in _CITATION.findall(text):
            cited |= {int(n) for n in re.findall(r"\d+", group)}
        invalid = sorted(cited - valid_numbers)
        valid_cited = sorted(cited & valid_numbers)

        if invalid:
            text = _BRACKET_REF.sub(lambda m: "" if int(m.group(1)) in invalid else m.group(0), text)
            state.warnings.append(f"удалены ссылки на несуществующие фрагменты: {invalid}")
        trace.add("validation", kind="citations", cited=sorted(cited), valid=valid_cited, invalid=invalid)

        # проверка привязки: каждое утверждение — с процитированным фрагментом, где оно есть (Q1, E1)
        fragments = {f.number: f.text for f in state.fragments}
        check = check_grounding(text, fragments, self.min_overlap, self.min_terms)
        state.grounding = {**check.summary(), "regenerations": regenerations}
        trace.add("validation", kind="grounding", grounded=check.grounded, units=check.units,
                  unsupported=[u["why"] for u in check.unsupported][:5], regenerations=regenerations)
        if check.grounded:
            cited_numbers = set(check.cited) or set(valid_cited)
            sources = [f.source_dict() for f in state.fragments if f.number in cited_numbers] or \
                [f.source_dict() for f in state.fragments]
            return self._finish(state, trace, started, STATUS_ANSWERED, text, sources=sources)

        if regenerations < self.max_regenerations and messages is not None:
            problems = "\n".join(f"- «{u['unit']}» — {u['why']}" for u in check.unsupported[:6])
            feedback = P.REGENERATE_PROMPT.format(problems=problems, not_found=P.NOT_FOUND_ANSWER)
            retry_messages = messages + [{"role": "assistant", "content": text}, {"role": "user", "content": feedback}]
            trace.add("decision", point="regenerate", unsupported=len(check.unsupported),
                      rule="ответ не подкреплён источниками — одна перегенерация по тем же фрагментам, без нового поиска")
            reply = self._call_llm(retry_messages, None, state, trace, stage="regenerate")
            if reply is None:
                return self._ungrounded_response(state, trace, started)
            return self._compose_final(state, trace, started, reply.content, retry_messages, regenerations + 1)

        return self._ungrounded_response(state, trace, started)

    # ------------------------------------------------------------------ маршрутизация

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

    def _apply_previous_cards(self, args: Dict[str, Any], state: AgentState, trace: Trace,
                              bump_top_k: bool = True) -> None:
        """Подставляет в поиск фильтр по карточкам прошлого ответа (по правилу агента или по запросу модели)."""
        requested = args.pop("previous_results", None)
        cards = state.followup_cards
        if requested in (True, "true", "True", 1) and not cards:
            cards = self.memory.last_cards()
            trace.add("decision", point="previous_results", requested_by="llm", cards=len(cards),
                      applied=bool(cards))
        if not cards:
            return
        args["cards"] = cards
        if bump_top_k and "top_k" not in state.locked_params:
            try:
                current = int(args.get("top_k") or self.tools.default_top_k)
            except (TypeError, ValueError):
                return          # некорректный top_k от модели отклонит сам инструмент
            args["top_k"] = min(self.tools.max_top_k, max(current, 2 * len(cards)))

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

    # =====================================================================
    # Завершение
    # =====================================================================

    def _finish(self, state: AgentState, trace: Trace, started: float, status: str, answer: str,
                error_code: Optional[str] = None, sources: Optional[List[Dict[str, Any]]] = None,
                remember: bool = True, escalation: Optional[Dict[str, Any]] = None) -> AgentResponse:
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
