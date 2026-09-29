"""
Evaluator: проверяет ответ агента по декларативным критериям шага сценария.

Поддерживаемые проверки (ключи checks в config/agent_scenarios.yaml):
    status               ожидаемый статус (answered | not_found | rejected | error | degraded | smalltalk)
    route                ожидаемый маршрут (guardrail_block | tools | forced_search | smalltalk | degraded)
    error_code           ожидаемый код ошибки
    required_tools       tools, которые обязаны быть вызваны
    forbidden_tools      tools, которые вызываться не должны
    tool_calls           {min: N, max: M} — число вызовов tools
    max_llm_calls        верхняя граница вызовов LLM
    no_tool_errors       true — ни один вызов tool не завершился ошибкой
    answer_contains_all  все подстроки (без учёта регистра)
    answer_contains_any  хотя бы одна подстрока
    answer_not_contains  ни одной подстроки
    forbid_patterns      ни одного совпадения регулярного выражения
    sources_include      подстроки имён файлов, которые должны быть среди источников
    min_sources          минимальное число источников
    tool_args_contains   {tool: {arg: подстрока}} — аргументы хотя бы одного вызова
    memory_size          размер памяти после шага
    trace_not_contains   подстроки, которых не должно быть в сериализованном trace (утечка секретов)
"""

import json
import re
from typing import Any, Dict, List

from agent_models import AgentResponse


def _check(name: str, passed: bool, expected: Any, actual: Any) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "expected": expected, "actual": actual}


def evaluate_step(response: AgentResponse, checks: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    answer_lc = (response.answer or "").lower()
    tools_called = response.tool_names

    if "status" in checks:
        results.append(_check("status", response.status == checks["status"], checks["status"], response.status))
    if "route" in checks:
        results.append(_check("route", response.route == checks["route"], checks["route"], response.route))
    if "error_code" in checks:
        results.append(_check("error_code", response.error_code == checks["error_code"],
                              checks["error_code"], response.error_code))

    for tool in checks.get("required_tools", []) or []:
        results.append(_check(f"required_tool:{tool}", tool in tools_called, "вызван", tools_called))
    for tool in checks.get("forbidden_tools", []) or []:
        results.append(_check(f"forbidden_tool:{tool}", tool not in tools_called, "не вызван", tools_called))

    if "tool_calls" in checks:
        bounds = checks["tool_calls"] or {}
        n = len(response.tool_calls)
        lo, hi = bounds.get("min", 0), bounds.get("max", 10 ** 6)
        results.append(_check("tool_calls_count", lo <= n <= hi, f"{lo}..{hi}", n))
    if "max_llm_calls" in checks:
        results.append(_check("llm_calls_count", response.llm_calls <= checks["max_llm_calls"],
                              f"<= {checks['max_llm_calls']}", response.llm_calls))
    if checks.get("no_tool_errors"):
        errors = [c.error_code for c in response.tool_calls if c.status != "success"]
        results.append(_check("no_tool_errors", not errors, [], errors))

    if "answer_contains_all" in checks:
        missing = [s for s in checks["answer_contains_all"] if str(s).lower() not in answer_lc]
        results.append(_check("answer_contains_all", not missing, checks["answer_contains_all"],
                              {"missing": missing}))
    if "answer_contains_any" in checks:
        found = [s for s in checks["answer_contains_any"] if str(s).lower() in answer_lc]
        results.append(_check("answer_contains_any", bool(found), checks["answer_contains_any"], {"found": found}))
    if "answer_not_contains" in checks:
        present = [s for s in checks["answer_not_contains"] if str(s).lower() in answer_lc]
        results.append(_check("answer_not_contains", not present, checks["answer_not_contains"],
                              {"present": present}))
    if "forbid_patterns" in checks:
        hits = [p for p in checks["forbid_patterns"] if re.search(p, response.answer or "", re.I)]
        results.append(_check("forbid_patterns", not hits, checks["forbid_patterns"], {"matched": hits}))

    source_files = [s.get("file_name", "") for s in response.sources]
    for needle in checks.get("sources_include", []) or []:
        results.append(_check(f"sources_include:{needle}", any(needle in f for f in source_files),
                              needle, source_files))
    if "min_sources" in checks:
        results.append(_check("min_sources", len(response.sources) >= checks["min_sources"],
                              f">= {checks['min_sources']}", len(response.sources)))

    for tool, arg_checks in (checks.get("tool_args_contains") or {}).items():
        calls = [c for c in response.tool_calls if c.name == tool]
        for arg, needle in arg_checks.items():
            values = [str(c.arguments.get(arg, "")) for c in calls]
            ok = any(str(needle).lower() in v.lower() for v in values)
            results.append(_check(f"tool_args:{tool}.{arg}", ok, f"содержит «{needle}»", values))

    if "memory_size" in checks:
        results.append(_check("memory_size", response.memory_size == checks["memory_size"],
                              checks["memory_size"], response.memory_size))

    if "trace_not_contains" in checks:
        dumped = json.dumps(response.to_dict(include_trace=True), ensure_ascii=False)
        leaked = [s for s in checks["trace_not_contains"] if s in dumped]
        results.append(_check("trace_not_contains", not leaked, "секрет отсутствует в trace и ответе",
                              {"leaked_count": len(leaked)}))
    return results
