"""
Подагент «Документы»: ответ по описаниям продуктов и коммерческим предложениям.

Цикл: план (модель выбирает инструмент) → поиск по документам → ответ модели → проверка привязки
к источникам → ответ, одна перегенерация или эскалация. Правила:
  - у модели в схеме только search_knowledge_base(query, top_k) и list_documents; аргументы вне схемы
    отбрасываются, фильтр по документу продукта ставит код (шаг 2.4.2);
  - модель не вызвала поиск на вопрос по базе — поиск выполняется принудительно;
  - модель недоступна — деградация: найденные фрагменты без генерации;
  - ответ не подтверждён источниками после перегенерации — эскалация E1 (шаг 2.4.3);
  - ответа нет, а вопрос о данных, которых в базе не бывает (стоимость, контакты) — эскалация E2 (шаг 2.4.4).

DocumentsAgent — примесь к оркестратору: использует его _finish, _escalate, _tool_error_response,
tools, memory и параметры из конфигурации.
"""

import re
from typing import Any, Dict, List, Optional

import agent_prompts as P
from agent_llm import LLMReply
from agent_models import (STATUS_ANSWERED, STATUS_DEGRADED, STATUS_NOT_FOUND, STATUS_SMALLTALK, AgentResponse,
                          AgentState, ToolCall, Trace)
from agent_tools import format_documents, format_error, format_fragments, hits_to_fragments
from agents.escalation import NO_DATA_INTENT
from agents.grounding import check_grounding
from agents.registry_agent import card_sources, catalog_answer
from llm_client import LLMError

_SMALLTALK = re.compile(
    r"(?i)^\s*(привет|здравствуй|здравствуйте|добрый (день|вечер|утро)|спасибо|благодарю|пока)\b[\s!.,)]*$")
# правила реестров и маршрутизации — в agents/registry_agent.py и agents/router.py

_CITATION = re.compile(r"(?i)фрагмент\w*\s*((?:\d+\s*(?:,|и|;)\s*)*\d+)")
_BRACKET_REF = re.compile(r"\[\s*Фрагмент\s+(\d+)\s*\]")
_NOT_FOUND_NORMALIZED = P.NOT_FOUND_ANSWER.lower().rstrip(".")

# ошибки, после которых повторять вызов бессмысленно: сразу шаблонный ответ
_FATAL_TOOL_ERRORS = {"index_unavailable", "internal_error"}


class DocumentsAgent:
    """Подагент «Документы» (примесь к Orchestrator)."""

    def _run_documents(self, query: str, decision: Any, state: AgentState, trace: Trace,
                       started: float) -> AgentResponse:
        """Цикл подагента «Документы»: план (LLM) → поиск → ответ (LLM) → проверка привязки → ответ или эскалация."""
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
