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
from agent_models import (STATUS_ANSWERED, STATUS_DEGRADED, STATUS_ERROR, STATUS_NOT_FOUND,
                          STATUS_REJECTED, STATUS_SMALLTALK, AgentResponse, AgentState, ToolCall, Trace)
from agent_tools import (QdrantRetrieverAdapter, ToolRegistry, detect_industries, detect_period, detect_services,
                         format_documents, format_error, format_fragments, hits_to_fragments)

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
# программный маршрутизатор: перечисления и вопросы о людях отвечаются по каталогу карточек, без поиска и без LLM
_ENUM = re.compile(r"(?i)^\s*(какие|каких|перечисли\w*|список|покажи|сколько|назови|выведи)(?![а-яё])|"
                   r"(?<![а-яё])все\s+(проект|пропозал)")
_REGISTRY_NOUN = re.compile(r"(?i)(?<![а-яё])(проект|пропозал|кейс|опыт)")
_PEOPLE = re.compile(r"(?i)(?<![а-яё])(кто|сотрудник\w*|команд\w*|участ\w*|люд\w*|человек\w*|исполнител\w*|"
                     r"делал\w*|выполнял\w*|работал\w*|партн[её]р\w*)(?![а-яё])")
# подсчёт: «кто сделал больше всего проектов в …», «у кого много пропозалов», «топ сотрудников по …»
_AGG = re.compile(r"(?i)(больше\s+всего|(?<![а-яё])много(?![а-яё])|чаще\w*|наибольш\w*|(?<![а-яё])топ(?![а-яё])|"
                  r"самы[йеех]\s+опытн\w*|лидер\w*|рейтинг\w*)")
_WHO = re.compile(r"(?i)(?<![а-яё])(кто|у\s+кого|сотрудник\w*|люди|людей|эксперт\w*)(?![а-яё])")
_PERSON_WORK = re.compile(r"(?i)(?<![а-яё])(делал\w*|работал\w*|участвовал\w*|вел[аи]?|выполнял\w*|опыт\w*)(?![а-яё])")
# слова, которые не задают тему вопроса-перечисления
_ROUTER_NEUTRAL = {"прое", "проп", "кейс", "опыт", "рабо", "отра", "сфер", "обла", "сект", "году", "года", "годы",
                   "пери", "реес", "спис", "пере", "пока", "назо", "выве", "скол", "дела", "выпо", "реал", "вели",
                   "наше", "наши", "всег", "посл", "нача", "ране", "позж", "пред", "мног", "боль", "чаще", "наиб",
                   "самы", "лиде", "рейт", "эксп", "сотр", "люди", "люде", "сдел", "учас", "кого", "делал",
                   "темы", "теме", "направ", "текущ", "этом", "прош", "лет", "года"}

_CITATION = re.compile(r"(?i)фрагмент\w*\s*((?:\d+\s*(?:,|и|;)\s*)*\d+)")
_BRACKET_REF = re.compile(r"\[\s*Фрагмент\s+(\d+)\s*\]")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?")
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

        self.check_numbers = bool(config.get("validation", {}).get("check_numbers", True))

        self.llm = llm if llm is not None else create_backend(config.get("llm", {}))
        factory = retriever_factory or self._default_retriever_factory(config)
        self.tools = ToolRegistry(factory, config.get("tools", {}), catalog_path=config.get("catalog_path"))

        self._schemas: Optional[List[Dict[str, Any]]] = None
        self._facets_summary: Optional[Dict[str, Any]] = None

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

        routed = self._route_catalog(query, state, trace, started)
        if routed is not None:
            return routed

        tool_schemas = self._tool_schemas(trace)

        messages: List[Dict[str, Any]] = [{"role": "system", "content": P.SYSTEM_PROMPT}]
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

        return self._compose_final(state, trace, started, final_text or "")

    # =====================================================================
    # Шаги
    # =====================================================================

    def _tool_schemas(self, trace: Trace) -> List[Dict[str, Any]]:
        """Схемы инструментов с допустимыми значениями фильтров из индекса; строятся один раз."""
        if self._schemas is None:
            facets = self.tools.facets()
            catalog = self.tools.catalog()
            self._schemas = P.build_tool_schemas(facets, with_catalog=catalog is not None,
                                                 services=catalog.services() if catalog else None)
            self._facets_summary = {
                "categories": len((facets or {}).get("categories") or {}),
                "industries": len((facets or {}).get("industries") or {}),
                "years": [(facets or {}).get("year_min"), (facets or {}).get("year_max")],
            } if facets else None
        trace.add("decision", point="tool_schema", facets=self._facets_summary,
                  rule="допустимые значения фильтров взяты из индекса" if self._facets_summary
                  else "фасеты недоступны: базовая схема инструментов")
        return self._schemas

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
            return self._finish(state, trace, started, STATUS_NOT_FOUND, P.NOT_FOUND_ANSWER)
        return None

    def _tool_error_response(self, state, trace, started, err: ToolCall) -> AgentResponse:
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
                text = P.LLM_AND_TOOL_UNAVAILABLE_TEMPLATE.format(
                    llm_error=state.llm_error, tool_error=state.tool_calls[-1].error)
                return self._finish(state, trace, started, STATUS_ERROR, text,
                                    error_code="llm_unavailable", remember=False)
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

    def _compose_final(self, state: AgentState, trace: Trace, started: float, text: str) -> AgentResponse:
        text = (text or "").strip()
        has_search = bool(state.successful_calls("search_knowledge_base"))

        if not has_search and state.cards_result is not None:
            result = state.cards_result
            answer = text or self._catalog_answer(result, "")
            trace.add("validation", kind="catalog", ok=True)
            return self._finish(state, trace, started, STATUS_ANSWERED if result["total"] else STATUS_NOT_FOUND,
                                answer, sources=self._card_sources(result["cards"]))

        if not has_search and not state.documents_listed:
            # сюда попадаем, только если все вызовы tools завершились ошибкой и LLM ответила текстом
            if state.last_tool_error is not None:
                trace.add("decision", point="validation", action="успешных вызовов tools нет — текст LLM отброшен")
                return self._tool_error_response(state, trace, started, state.last_tool_error)
            return self._finish(state, trace, started, STATUS_NOT_FOUND, P.NOT_FOUND_ANSWER)

        if not has_search and state.documents_listed:
            docs = state.documents_result
            answer = text or (format_documents(docs) if docs else "")
            trace.add("validation", kind="documents", ok=True)
            return self._finish(state, trace, started, STATUS_ANSWERED, answer)

        if _NOT_FOUND_NORMALIZED in text.lower().replace("«", "").replace("»", ""):
            trace.add("validation", kind="not_found_phrase", ok=True)
            return self._finish(state, trace, started, STATUS_NOT_FOUND, P.NOT_FOUND_ANSWER)

        if not text:
            state.warnings.append("LLM вернула пустой ответ")
            trace.add("validation", kind="empty_answer", ok=False)
            return self._finish(state, trace, started, STATUS_NOT_FOUND, P.NOT_FOUND_ANSWER)

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

        notes: List[str] = []
        if valid_cited:
            sources = [f.source_dict() for f in state.fragments if f.number in valid_cited]
        else:
            sources = [f.source_dict() for f in state.fragments]
            state.warnings.append("модель не сослалась ни на один фрагмент")
            notes.append("⚠️ Модель не указала, на каких фрагментах основан ответ — сверьте его с источниками.")

        if self.check_numbers:
            context_text = self._normalize_numbers(" ".join(f.text for f in state.fragments))
            unconfirmed = []
            for raw in _NUMBER.findall(self._strip_refs(text)):
                norm = self._normalize_numbers(raw)
                if len(norm.replace(".", "")) < 2:
                    continue          # одиночные цифры (нумерация списка) не проверяем
                if norm not in context_text and raw not in unconfirmed:
                    unconfirmed.append(raw)
            trace.add("validation", kind="numbers", unconfirmed=unconfirmed)
            if unconfirmed:
                state.warnings.append(f"числа не найдены во фрагментах: {unconfirmed}")
                notes.append("⚠️ Не подтверждены текстом фрагментов: " + ", ".join(unconfirmed))

        if notes:
            text = text.rstrip() + "\n\n" + "\n".join(notes)
        return self._finish(state, trace, started, STATUS_ANSWERED, text, sources=sources)

    # ------------------------------------------------------------------ маршрутизатор по каталогу

    def _route_catalog(self, query: str, state: AgentState, trace: Trace, started: float) -> Optional[AgentResponse]:
        """
        Вопросы, на которые нужен полный и точный список, отвечаются выборкой из каталога карточек
        без поиска и без LLM:
          - люди по карточкам прошлого ответа: «кто выполнял эти проекты», «какие сотрудники это делали»;
          - перечисление по отрасли, периоду или типу реестра: «какие проекты были в медицине»,
            «перечисли пропозалы за 2019–2021». Если в вопросе есть другая тема («по закупкам»),
            он уходит в обычный путь: поиск по смыслу с фильтром.
        """
        catalog = self.tools.catalog()
        if catalog is None:
            return None
        industries = detect_industries(query, catalog.industries())
        services = detect_services(query, catalog.services(), catalog.service_details())
        if industries and services:
            # одно слово и отрасль, и услуга («по логистике»): это тема услуги, а не отрасль клиента
            from agent_tools import INDUSTRY_SYNONYMS, SERVICE_SYNONYMS, covered_words, tag_keys
            service_phrases = [p for sv in services for p in SERVICE_SYNONYMS.get(sv, []) + tag_keys(sv) + [sv]]
            service_words = covered_words(query, service_phrases)
            industries = [ind for ind in industries
                          if covered_words(query, INDUSTRY_SYNONYMS.get(ind, []) + [ind]) - service_words]
        persons = catalog.find_person(query)
        year_from, year_to = detect_period(query)
        only_proposals = re.search(r"(?i)пропозал|предложени", query) and not re.search(r"(?i)проект", query)
        card_type = "proposal" if only_proposals else None
        scope = {k: v for k, v in (("industry", industries), ("service", services), ("person", persons),
                                   ("year_from", year_from), ("year_to", year_to), ("card_type", card_type)) if v}

        def has_other_topic() -> bool:
            residual = self._residual_topic(query, industries, services, persons)
            if residual:
                trace.add("decision", point="catalog_route", routed=False, residual=residual,
                          rule="в вопросе есть тема помимо отрасли, типа услуги и периода — нужен поиск по смыслу")
            return bool(residual)

        if _AGG.search(query) and (_WHO.search(query) or _PEOPLE.search(query)) and (
                _REGISTRY_NOUN.search(query) or scope):
            # проверяется первым: «кто делал много проектов в …» — не уточнение прошлого ответа, а подсчёт
            if has_other_topic():
                return None
            args = {**scope, "group_by": "people"}
            route, intro = "catalog_people_rank", "Кто чаще всего участвовал"
        elif state.followup_cards and _PEOPLE.search(query) and not scope:
            args = {"cards": state.followup_cards, "include_people": True}
            route, intro = "catalog_people", "Люди по карточкам из прошлого ответа"
        elif (_ENUM.search(query) and _REGISTRY_NOUN.search(query) and scope) or (
                persons and (_REGISTRY_NOUN.search(query) or _PERSON_WORK.search(query))):
            # «Проекты Сотрудник за последние 5 лет», «что делал Петров в 2020» — список карточек сотрудника
            if has_other_topic():
                return None
            args = {**scope, "include_people": bool(_PEOPLE.search(query)) and not persons}
            route, intro = "catalog_list", "В реестрах найдено"
        else:
            return None

        state.route = route
        trace.add("decision", point="catalog_route", routed=True, route=route,
                  args={k: v for k, v in args.items() if k != "cards"}, cards=len(args.get("cards") or []),
                  rule="полный список по каталогу карточек вместо top-k поиска; LLM не вызывается")
        messages: List[Dict[str, Any]] = []
        call = self._execute_and_append("list_cards", args, "router", state, trace, messages)
        if call.status != "success":
            return self._tool_error_response(state, trace, started, call)
        result = state.cards_result
        status = STATUS_ANSWERED if result["total"] else STATUS_NOT_FOUND
        return self._finish(state, trace, started, status, self._catalog_answer(result, intro),
                            sources=self._card_sources(result["cards"]))

    @staticmethod
    def _residual_topic(query: str, industries: List[str], services: Optional[List[str]] = None,
                        persons: Optional[List[str]] = None) -> List[str]:
        """Содержательные слова вопроса, кроме служебных, отраслевых, типа услуги, фамилий и периода."""
        from agent_lexical import STOPWORDS, stems
        from agent_tools import INDUSTRY_SYNONYMS, SERVICE_SYNONYMS, _matches_synonyms, covered_words
        phrases = [p for category in industries for p in INDUSTRY_SYNONYMS.get(category, []) + [category]]
        phrases += [p for service in services or [] for p in SERVICE_SYNONYMS.get(service, [])]
        from agent_tools import tag_keys
        service_stems = {st for service in services or [] for st in stems(service)}
        phrases += [w for service in services or [] for w in tag_keys(service)]
        surnames = {p.split()[0].lower() for p in persons or []}
        covered = covered_words(query, phrases) if phrases else set()
        residual = []
        for word in re.findall(r"[a-zа-яё]+", query.lower().replace("ё", "е")):
            if len(word) < 4 or word in STOPWORDS or word[:4] in _ROUTER_NEUTRAL:
                continue
            if word in covered or (phrases and _matches_synonyms(word, phrases)) or _PEOPLE.fullmatch(word):
                continue           # то же сопоставление, что и при определении отрасли
            if word[:4] in service_stems or any(word.startswith(sn) for sn in surnames):
                continue
            residual.append(word)
        return residual

    @staticmethod
    def _catalog_answer(result: Dict[str, Any], intro: str) -> str:
        from agent_cards import CARD_TYPE_RU, format_card_line
        filters = result.get("filters") or {}
        conditions = []
        if filters.get("industry"):
            conditions.append("отрасль: " + ", ".join(filters["industry"]))
        if filters.get("year_from") or filters.get("year_to"):
            conditions.append(f"период: {filters.get('year_from') or '…'}–{filters.get('year_to') or '…'}")
        if filters.get("card_type"):
            conditions.append("тип: " + CARD_TYPE_RU.get(filters["card_type"], filters["card_type"]))
        if filters.get("service"):
            conditions.append("тип услуги: " + ", ".join(filters["service"]))
        if filters.get("person"):
            conditions.append("сотрудник: " + ", ".join(filters["person"]))
        cond = f" ({'; '.join(conditions)})" if conditions else ""
        if not result["total"]:
            return f"По условиям{cond} карточек в реестрах нет."
        if result.get("group_by") == "people":
            by_type = result.get("by_type", {})
            head = (f"{intro}{cond}. Карточек: {result['total']} (проектов {by_type.get('project', 0)}, "
                    f"пропозалов {by_type.get('proposal', 0)}), люди указаны в {result['cards_with_people']}, "
                    f"разных людей: {result['people_total']}.")
            if not result.get("people"):
                return head + "\nВ этих карточках люди не указаны."
            return (head + "\n\n" + result["text"] + "\n\nСортировка по числу проектов: пропозал — это заявка, "
                    "а не выполненный проект. Учтены поля «Команда», «Партнёр / директор», «Имя сотрудника»; "
                    "человек считается один раз на карточку.\nИсточник: каталог карточек реестров.")
        lines = [f"{intro or 'Найдено'}{cond}: {result['total']}."]
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for card in result["cards"]:
            groups.setdefault(card.get("card_type", ""), []).append(card)
        titles = {"project": "Проекты", "proposal": "Пропозалы", "project_accounting": "Проекты учётной базы"}
        for card_type, cards in groups.items():
            lines.append(f"\n{titles.get(card_type, card_type)} ({len(cards)}):")
            lines += [format_card_line(i, c, result.get("include_people")) for i, c in enumerate(cards, start=1)]
        if result["shown"] < result["total"]:
            lines.append(f"\nПоказаны {result['shown']} из {result['total']}: уточните период, отрасль или тип реестра.")
        lines.append("\nИсточник: каталог карточек реестров (data/raw_cards/cards_meta.jsonl).")
        return "\n".join(lines)

    @staticmethod
    def _card_sources(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        keys = ("card_type", "card_id", "card_title", "year", "year_end", "industry")
        return [{"fragment": i, "document_name": c.get("card_title", ""), "file_name": c.get("file_name", ""),
                 "category": "", "chunk_index": "—", "score": 1.0,
                 "card": {k: c[k] for k in keys if c.get(k) not in (None, "", [])}}
                for i, c in enumerate(cards, start=1)]

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

    @staticmethod
    def _strip_refs(text: str) -> str:
        return re.sub(r"(?i)фрагмент\w*\s*((?:\d+\s*(?:,|и|;)\s*)*\d+)", " ", text)

    @staticmethod
    def _normalize_numbers(text: str) -> str:
        text = re.sub(r"(?<=\d)[ \u00a0](?=\d{3}\b)", "", text)
        return text.replace(",", ".")

    # =====================================================================
    # Завершение
    # =====================================================================

    def _finish(self, state: AgentState, trace: Trace, started: float, status: str, answer: str,
                error_code: Optional[str] = None, sources: Optional[List[Dict[str, Any]]] = None,
                remember: bool = True) -> AgentResponse:
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
        )
