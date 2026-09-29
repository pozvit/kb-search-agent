"""
ScenarioRunner: загружает config/agent_scenarios.yaml, для каждого сценария создаёт
свежий экземпляр агента (с пустой памятью и overrides сценария), выполняет шаги,
передаёт ответы в evaluator и формирует два отчёта:

    data/agent/scenario_report_<llm>.json — полный: шаги, проверки, tool calls, trace
    data/test_results.json (mock) / data/test_results_ollama.json — краткий, в формате задания

Запуск — через CLI: python src/agent_cli.py --run-scenarios [...]
"""

import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from agent import KBAgent
from agent_config import load_agent_config
from agent_evaluator import evaluate_step

DEFAULT_SCENARIOS = "config/agent_scenarios.yaml"


def load_scenarios(path: str = DEFAULT_SCENARIOS) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    scenarios = data.get("scenarios") or []
    ids = [s.get("id") for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Повторяющиеся id сценариев в {path}")
    for s in scenarios:
        if not s.get("id") or not s.get("steps"):
            raise ValueError(f"Сценарий без id или steps: {s}")
        for step in s["steps"]:
            if "request" not in step or "checks" not in step:
                raise ValueError(f"Шаг сценария {s['id']} без request/checks")
    return data


class ScenarioRunner:
    def __init__(self, agent_config_path: str = "config/agent.yaml",
                 base_overrides: Optional[Dict[str, Any]] = None,
                 profile: str = "public", agent_factory=None):
        self.agent_config_path = agent_config_path
        self.base_overrides = dict(base_overrides or {})
        self.profile = profile
        self._agent_factory = agent_factory       # для тестов: (config) -> KBAgent

    def _make_agent(self, scenario: Dict[str, Any]) -> KBAgent:
        overrides = {**self.base_overrides, **(scenario.get("overrides") or {})}
        config = load_agent_config(self.agent_config_path, profile=self.profile, overrides=overrides)
        if self._agent_factory:
            return self._agent_factory(config)
        return KBAgent(config)

    def run_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        started = time.monotonic()
        steps_out: List[Dict[str, Any]] = []
        agent = None
        try:
            agent = self._make_agent(scenario)
            for index, step in enumerate(scenario["steps"], start=1):
                response = agent.run(step["request"])
                checks = evaluate_step(response, step.get("checks") or {})
                steps_out.append({
                    "step": index,
                    "request": response.trace.events[0].data.get("query") if response.trace.events else "",
                    "expected": step.get("expected", ""),
                    "passed": all(c["passed"] for c in checks),
                    "status": response.status,
                    "error_code": response.error_code,
                    "answer": response.answer,
                    "sources": response.sources,
                    "warnings": response.warnings,
                    "llm_calls": response.llm_calls,
                    "tool_calls": [c.to_dict() for c in response.tool_calls],
                    "checks": checks,
                    "duration_ms": response.duration_ms,
                    "trace": response.trace.to_list(),
                })
            agent_description = agent.describe()
        except Exception as exc:  # сценарий не должен ронять весь прогон
            steps_out.append({"step": len(steps_out) + 1, "passed": False,
                              "status": "runner_error", "answer": f"{type(exc).__name__}: {exc}",
                              "checks": [{"name": "runner", "passed": False, "expected": "без исключений",
                                          "actual": f"{type(exc).__name__}: {exc}"}],
                              "tool_calls": [], "llm_calls": 0, "trace": []})
            agent_description = "не создан"
        finally:
            if agent is not None:
                agent.close()

        return {
            "scenario_id": scenario["id"],
            "type": scenario.get("type"),
            "title": scenario.get("title"),
            "goal": scenario.get("goal"),
            "chain": scenario.get("chain", []),
            "decision_points": scenario.get("decision_points", []),
            "overrides": scenario.get("overrides") or {},
            "agent": agent_description,
            "passed": bool(steps_out) and all(s["passed"] for s in steps_out),
            "duration_ms": int((time.monotonic() - started) * 1000),
            "steps": steps_out,
        }

    def run(self, scenarios_path: str = DEFAULT_SCENARIOS,
            only: Optional[List[str]] = None, progress: bool = True) -> Dict[str, Any]:
        data = load_scenarios(scenarios_path)
        scenarios = data["scenarios"]
        if only:
            missing = sorted(set(only) - {s["id"] for s in scenarios})
            if missing:
                raise ValueError(f"Сценарии не найдены: {', '.join(missing)}")
            scenarios = [s for s in scenarios if s["id"] in only]

        results = []
        for scenario in scenarios:
            if progress:
                print(f"  > {scenario['id']:<28} ", end="", flush=True)
            result = self.run_scenario(scenario)
            results.append(result)
            if progress:
                mark = "PASS" if result["passed"] else "FAIL"
                print(f"{mark}  ({result['duration_ms'] / 1000:.1f} c)")
                if not result["passed"]:
                    for step in result["steps"]:
                        for check in step["checks"]:
                            if not check["passed"]:
                                print(f"      шаг {step['step']}: {check['name']}: ожидалось {check['expected']}, "
                                      f"получено {json.dumps(check['actual'], ensure_ascii=False)[:200]}")

        config = load_agent_config(self.agent_config_path, profile=self.profile, overrides=self.base_overrides)
        return {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "agent_name": data.get("agent_name", "KB Search Agent"),
            "environment": {
                "profile": self.profile,
                "llm_backend": config["llm"].get("backend"),
                "llm_model": config["llm"].get("model") if config["llm"].get("backend") == "ollama" else "mock",
                "retriever": config.get("retriever", {}).get("backend"),
                "embeddings": _embeddings_label(config),
                "python": platform.python_version(),
                "platform": platform.platform(terse=True),
            },
            "passed": all(r["passed"] for r in results),
            "summary": {
                "total": len(results),
                "passed": sum(r["passed"] for r in results),
                "failed": sum(not r["passed"] for r in results),
                "by_type": _count_by_type(results),
            },
            "results": results,
        }


def _embeddings_label(config: Dict[str, Any]) -> str:
    if config.get("retriever", {}).get("backend") != "qdrant":
        return "не используются (BM25 по chunks.jsonl)"
    if os.environ.get("RAG_FAKE_EMBEDDINGS") == "1":
        return "HashingModel (RAG_FAKE_EMBEDDINGS=1, смоук-режим, качество поиска нерелевантно)"
    return config.get("rag", {}).get("retrieval", {}).get("model_name", "n/a")


def _count_by_type(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for r in results:
        bucket = out.setdefault(r["type"] or "n/a", {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += int(r["passed"])
    return out


def to_test_results(report: Dict[str, Any]) -> Dict[str, Any]:
    """Краткий отчёт в формате задания (TC-MVP-NNN)."""
    tests = []
    for i, r in enumerate(report["results"], start=1):
        steps = r["steps"]
        tool_calls = [c["name"] for s in steps for c in s.get("tool_calls", [])]
        actual = " | ".join(f"[{s['status']}] {(s.get('answer') or '')[:300]}" for s in steps)
        tests.append({
            "id": f"TC-MVP-{i:03d}",
            "scenario_id": r["scenario_id"],
            "type": r["type"],
            "request": " → ".join(s.get("request", "") for s in steps),
            "expected": " → ".join(s.get("expected", "") for s in steps),
            "actual": actual,
            "tool_calls": tool_calls,
            "failed_checks": [f"шаг {s['step']}: {c['name']}" for s in steps for c in s["checks"] if not c["passed"]],
            "passed": r["passed"],
        })
    return {
        "agent_name": report["agent_name"],
        "created_at": report["created_at"],
        "environment": report["environment"],
        "summary": report["summary"],
        "tests": tests,
    }


def save_reports(report: Dict[str, Any], report_path: str, results_path: str) -> None:
    for path, payload in ((report_path, report), (results_path, to_test_results(report))):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
