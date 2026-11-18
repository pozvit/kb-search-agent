"""
Прогон контрольного набора (config/control_scenarios.yaml) на любой версии кода и подсчёт метрик Q1–Q8.

Принципы честного сравнения «до/после»:
  1. Код версии берётся из --version-dir (baseline — git worktree тега, итоговая — текущая папка),
     а данные (индекс Qdrant, чанки, каталог карточек) — из --data-dir по абсолютным путям.
     Обе версии отвечают по одному и тому же индексу public_v2.
  2. Эталоны и проверка привязки ответа к источникам считаются ЗДЕСЬ, независимым кодом:
     скрипт сам читает cards_meta.jsonl и chunks.jsonl и не использует ни каталог, ни валидатор
     проверяемой версии. Иначе итоговая версия проверяла бы сама себя.
  3. Сбои (недоступная модель, недоступный индекс, ответ модели без ссылок) подменяются одинаково.

Запуск:
    git worktree add ../kb_baseline baseline-2026-10-02
    python tools/run_control.py --version-dir ../kb_baseline --label baseline
    python tools/run_control.py --version-dir . --label final
    python tools/run_control.py --compare baseline final
    # без Ollama и модели эмбеддингов (проверка оркестрации, не качества модели):
    python tools/run_control.py --version-dir . --label final_mock --llm mock --retriever lexical

Отчёты: reports/control/<label>.json (всё: шаги, проверки, trace), <label>.md (сводка),
comparison.md (до/после). Профиль только public: в отчётах есть тексты ответов.
"""

import argparse
import datetime
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# ---------------------------------------------------------------- маршруты

ROUTE_EQUIVALENTS = {          # маршруты baseline -> маршруты итоговой версии
    "catalog_list": "registry", "catalog_people": "registry", "catalog_people_rank": "registry",
    "tools": "documents", "forced_search": "documents", "degraded": "documents",
    "guardrail_block": "rejected", "smalltalk": "smalltalk",
}
CANONICAL = {"registry", "documents", "escalation", "smalltalk", "rejected"}


def canonical_route(response: Any) -> Optional[str]:
    if getattr(response, "status", None) == "escalated":
        return "escalation"
    route = getattr(response, "route", None)
    if route in CANONICAL:
        return route
    return ROUTE_EQUIVALENTS.get(route, route)


# ---------------------------------------------------------------- независимая проверка привязки (Q1)

STOP = {"какие", "какой", "какая", "каких", "который", "которые", "также", "этого", "этой", "этом", "есть",
        "было", "были", "была", "будет", "между", "через", "после", "более", "менее", "очень", "может",
        "можно", "нужно", "этот", "такие", "такой", "всего", "всех", "данных", "данные", "документах",
        "фрагмент", "фрагмента", "фрагменте", "фрагменты", "источник", "источники", "базы", "знаний"}
CITE = re.compile(r"(?i)фрагмент\w*\s*((?:\d+\s*(?:,|и|;)\s*)*\d+)")
NUM = re.compile(r"(?<![\w.])\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?")
SKIP_LINE = re.compile(r"(?i)^\s*(источники|использованные|предупреждени|⚠)")


def stems(text: str) -> Set[str]:
    words = re.findall(r"[a-zа-яё0-9]+", text.lower().replace("ё", "е"))
    return {w[:4] for w in words if len(w) >= 4 and w not in STOP and not w.isdigit()}


def numbers(text: str) -> Set[str]:
    """
    Числа утверждения. Номер пункта списка в начале строки («1.», «2)») отбрасывается, а остальные
    числа проверяются все, включая однозначные: «5 сегментов» вместо «7» — это ошибка факта.
    (Валидатор baseline пропускал все однозначные числа и такую ошибку не видел.)
    """
    cleaned = re.sub(r"(?m)^\s*\d{1,2}[.)]\s+", " ", CITE.sub(" ", text))
    return {re.sub(r"[ \u00a0]", "", raw).replace(",", ".") for raw in NUM.findall(cleaned)}


def norm_numbers(text: str) -> str:
    return re.sub(r"(?<=\d)[ \u00a0](?=\d{3}\b)", "", text).replace(",", ".")


def cited(text: str) -> Set[int]:
    return {int(n) for group in CITE.findall(text) for n in re.findall(r"\d+", group)}


def check_grounding(answer: str, fragments: Dict[int, str], min_overlap: float = 0.5,
                    min_terms: int = 3) -> Dict[str, Any]:
    """
    Утверждение (предложение или пункт списка) с не менее чем min_terms значимыми словами подкреплено, если:
      - оно ссылается на фрагмент — само или ссылкой в конце своего абзаца;
      - фрагмент реально был получен;
      - доля значимых основ утверждения, найденных в процитированных фрагментах, >= min_overlap;
      - все его числа есть в процитированных фрагментах.
    Лексическая проверка: ловит утверждения без ссылки, ссылки «не туда» и чужие числа,
    но не смысловые ошибки при верных словах.
    """
    units, unsupported = 0, []
    for paragraph in re.split(r"\n\s*\n", answer or ""):
        lines = [l for l in paragraph.split("\n") if l.strip() and not SKIP_LINE.match(l)]
        if not lines:
            continue
        para_cites = cited("\n".join(lines))
        for line in lines:
            for unit in re.split(r"(?<=[.!?])\s+", line):
                terms = stems(CITE.sub(" ", unit))
                if len(terms) < min_terms:
                    continue
                units += 1
                refs = cited(unit) or para_cites
                texts = [fragments[r] for r in refs if r in fragments]
                if not texts:
                    unsupported.append({"unit": unit[:160], "why": "нет ссылки на полученный фрагмент"})
                    continue
                frag_terms = set().union(*(stems(t) for t in texts))
                overlap = len(terms & frag_terms) / len(terms)
                frag_numbers = {re.sub(r"[ \u00a0]", "", x).replace(",", ".") for x in NUM.findall(norm_numbers(" ".join(texts)))}
                missing = sorted(n for n in numbers(unit) if n not in frag_numbers)
                if overlap < min_overlap:
                    unsupported.append({"unit": unit[:160], "why": f"мало общих слов с фрагментом ({overlap:.2f})"})
                elif missing:
                    unsupported.append({"unit": unit[:160], "why": f"чисел нет во фрагменте: {missing}"})
    return {"units": units, "unsupported": unsupported, "grounded": units > 0 and not unsupported}


# ---------------------------------------------------------------- независимые эталоны по каталогу

PERSON = re.compile(r"([А-ЯЁA-Z][а-яёa-z]+(?:-[А-ЯЁA-Z][а-яёa-z]+)?)\s+([А-ЯЁA-Z])\.")


def people_of(card: Dict[str, Any]) -> Set[str]:
    out = set()
    for field in ("team", "partner", "employee"):
        value = card.get(field)
        text = "; ".join(value) if isinstance(value, list) else (value or "")
        out |= {f"{s} {i}." for s, i in PERSON.findall(text)}
    return out


class Oracle:
    def __init__(self, cards_meta: Path, today_year: int):
        self.year = today_year
        self.cards: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if cards_meta.exists():
            for line in cards_meta.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    c = json.loads(line)
                    self.cards[(c["card_type"], str(c["card_id"]))] = c

    def select(self, cond: Dict[str, Any]) -> Set[Tuple[str, str]]:
        y_from = y_to = None
        if cond.get("last_years"):
            y_from, y_to = self.year - int(cond["last_years"]) + 1, self.year
        out = set()
        for key, c in self.cards.items():
            if cond.get("industry") and not set(c.get("industry") or []) & set(cond["industry"]):
                continue
            if cond.get("service") and not (set(c.get("service") or []) | set(c.get("service_detail") or [])) \
                    & set(cond["service"]):
                continue
            if cond.get("person") and not people_of(c) & set(cond["person"]):
                continue
            if y_from and not (isinstance(c.get("year_end"), int) and c["year_end"] >= y_from):
                continue
            if y_to and not (isinstance(c.get("year"), int) and c["year"] <= y_to):
                continue
            out.add(key)
        return out

    def ranking(self, keys: Set[Tuple[str, str]]) -> List[Tuple[str, int]]:
        counts: Dict[str, int] = {}
        for key in keys:
            if key[0] != "project":
                continue
            for person in people_of(self.cards[key]):
                counts[person] = counts.get(person, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


# ---------------------------------------------------------------- подмена сбоев

class NoCitationsLLM:
    """Обёртка над моделью: из финального ответа удаляются все ссылки [Фрагмент N] (сценарий CS-13)."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.name = getattr(inner, "name", "wrapped")

    def describe(self) -> str:
        return f"{self.inner.describe()} + удаление ссылок из ответа"

    def chat(self, messages, tools=None):
        reply = self.inner.chat(messages, tools)
        if not reply.tool_calls and reply.content:
            reply.content = re.sub(r"\s*\[?\s*Фрагмент\w*\s*[\d,\sи;]+\]?", "", reply.content)
            reply.content = re.sub(r"(?im)^\s*Источники:.*$", "", reply.content).strip()
        return reply


def inject_overrides(inject: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if inject.get("llm") == "down":
        out.update({"llm.backend": "ollama", "llm.base_url": "http://127.0.0.1:9", "llm.timeout_seconds": 5})
    if inject.get("index") == "down":
        out.update({"retrieval.collection_name": "__missing_collection__",
                    "retriever.chunks_jsonl": "__missing__/chunks.jsonl"})
    return out


# ---------------------------------------------------------------- проверка шага

def card_keys(response: Any) -> Set[Tuple[str, str]]:
    return {(s["card"].get("card_type"), str(s["card"].get("card_id")))
            for s in getattr(response, "sources", []) or [] if s.get("card")}


def evaluate(step: Dict[str, Any], response: Any, oracle: Oracle, chunks: Dict[Tuple[str, Any], str],
             agent: Any, grounding_cfg: Dict[str, Any]) -> Dict[str, Any]:
    exp = step.get("expect", {})
    answer = getattr(response, "answer", "") or ""
    route = canonical_route(response)
    checks: List[Dict[str, Any]] = []

    def check(name, ok, expected, actual):
        checks.append({"name": name, "passed": bool(ok), "expected": expected, "actual": actual})

    if "route" in exp:
        check("route", route == exp["route"], exp["route"], route)
    if "status" in exp:
        check("status", response.status == exp["status"], exp["status"], response.status)
    files = [s.get("file_name", "") for s in response.sources or []]
    if "sources_any" in exp:
        check("sources_any", any(f in files for f in exp["sources_any"]), exp["sources_any"], files[:10])
    if "facts_all" in exp:
        missing = [f for f in exp["facts_all"] if f.lower() not in answer.lower()]
        check("facts_all", not missing, exp["facts_all"], {"missing": missing})
    for pattern in exp.get("forbid_patterns", []):
        check("forbid_pattern", not re.search(pattern, answer, re.I), pattern, "найдено" if re.search(pattern, answer, re.I) else "нет")
    if "tool_calls_max" in exp:
        n = len(getattr(response, "tool_calls", []) or [])
        check("tool_calls_max", n <= exp["tool_calls_max"], exp["tool_calls_max"], n)
    if "llm_calls_max" in exp:
        check("llm_calls_max", response.llm_calls <= exp["llm_calls_max"], exp["llm_calls_max"], response.llm_calls)
    if "warnings_any" in exp:
        text = json.dumps(getattr(response, "warnings", []), ensure_ascii=False).lower() + answer.lower()
        found = [w for w in exp["warnings_any"] if w.lower() in text]
        check("warnings_any", bool(found), exp["warnings_any"], found)

    completeness = None
    if "cards_oracle" in exp:
        expected = oracle.select(exp["cards_oracle"])
        got = card_keys(response)
        completeness = len(expected & got) / len(expected) if expected else None
        if exp.get("people_oracle") == "all_people_of_cards":
            people = set().union(*(people_of(oracle.cards[k]) for k in expected)) if expected else set()
            named = {p for p in people if p in answer}
            completeness = len(named) / len(people) if people else completeness
            check("people", named == people, sorted(people), sorted(named))
        elif exp.get("ranking_oracle") == "top3_by_projects":
            top = oracle.ranking(expected)[:3]
            positions = [answer.find(p) for p, _ in top]
            in_order = all(p >= 0 for p in positions) and positions == sorted(positions)
            counts_ok = all(re.search(re.escape(p) + r"[^\n]{0,40}?\b" + str(n) + r"\b", answer) for p, n in top)
            completeness = 1.0 if (in_order and counts_ok) else 0.0
            check("ranking_top3", in_order and counts_ok, top, {"order_ok": in_order, "counts_ok": counts_ok})
        if "completeness" in exp and completeness is not None:
            check("completeness", completeness >= exp["completeness"], exp["completeness"], round(completeness, 3))

    esc = getattr(response, "escalation", None) or {}
    escalated = response.status == "escalated"
    if "escalation" in exp:
        req = exp["escalation"].get("required", False)
        ok = escalated == req and (not req or esc.get("reason") == exp["escalation"].get("reason"))
        check("escalation", ok, exp["escalation"], {"escalated": escalated, "reason": esc.get("reason")})
    if "llm_regenerations" in exp:
        regen = (getattr(response, "grounding", None) or {}).get("regenerations",
                 1 if (getattr(response, "grounding", None) or {}).get("regenerated") else 0)
        check("llm_regenerations", regen == exp["llm_regenerations"], exp["llm_regenerations"], regen)
    if "secret_absent_in" in exp:
        secret = re.search(r"sk-[A-Za-z0-9_-]{10,}", step["question"]).group(0)
        places = {"response": json.dumps(response.to_dict(include_trace=True), ensure_ascii=False, default=str)}
        if agent is not None and hasattr(agent, "memory"):
            places["memory"] = json.dumps(agent.memory.messages(), ensure_ascii=False)
        leaked = [k for k, v in places.items() if secret in v]
        check("secret_absent", not leaked, "нигде", {"проверено": sorted(places), "утечка": leaked})

    # Q1: независимая проверка привязки для ответов маршрута documents
    grounding = None
    if route == "documents" and response.status == "answered":
        frags = {}
        for s in response.sources or []:
            text = chunks.get((s.get("file_name"), s.get("chunk_index")))
            if text and s.get("fragment") is not None:
                frags[int(s["fragment"])] = text
        grounding = check_grounding(answer, frags, grounding_cfg.get("min_overlap", 0.5),
                                    grounding_cfg.get("min_terms", 3))
        if exp.get("grounded"):
            check("grounded", grounding["grounded"], True, {"units": grounding["units"],
                                                            "unsupported": grounding["unsupported"][:3]})
    return {"route": route, "checks": checks, "passed": all(c["passed"] for c in checks),
            "completeness": completeness, "grounding": grounding, "escalated": escalated}


# ---------------------------------------------------------------- метрики

def trace_events(response: Any) -> List[Dict[str, Any]]:
    trace = getattr(response, "trace", None)
    return trace.to_list() if trace is not None and hasattr(trace, "to_list") else []


def aggregate(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    def ratio(a, b):
        return round(a / b, 3) if b else None

    docs_answered = [s for s in steps if s["eval"]["grounding"] is not None]
    ungrounded = [s for s in docs_answered if not s["eval"]["grounding"]["grounded"]]
    with_route = [s for s in steps if "route" in s["expect"]]
    must_esc = [s for s in steps if s["expect"].get("escalation", {}).get("required")]
    escalated = [s for s in steps if s["eval"]["escalated"]]
    esc_ok = [s for s in escalated if s["expect"].get("escalation", {}).get("required")]
    compl = [s["eval"]["completeness"] for s in steps if s["eval"]["completeness"] is not None]
    lat = {r: [s["wall_ms"] / 1000 for s in steps if s["eval"]["route"] == r and not s.get("inject")]
           for r in ("registry", "documents")}
    plan_tokens = [e["data"]["usage"]["prompt_tokens"] for s in steps if s["eval"]["route"] == "documents"
                   for e in s["trace"] if e["type"] == "llm_call" and e["data"].get("stage") == "plan"
                   and (e["data"].get("usage") or {}).get("prompt_tokens")]
    doc_steps = [s for s in steps if s["eval"]["route"] == "documents" and not s.get("inject")]
    forced = [s for s in doc_steps if any(e["type"] == "decision" and e["data"].get("route") == "forced_search"
                                          for e in s["trace"])]
    med = lambda xs: round(statistics.median(xs), 2) if xs else None
    return {
        "Q1_ungrounded_answered_rate": ratio(len(ungrounded), len(docs_answered)),
        "Q1_detail": {"answered_documents": len(docs_answered), "ungrounded": len(ungrounded)},
        "Q2_route_accuracy": ratio(sum(1 for s in with_route if s["eval"]["route"] == s["expect"]["route"]), len(with_route)),
        "Q3_escalation_recall": ratio(sum(1 for s in must_esc if s["eval"]["escalated"]), len(must_esc)),
        "Q3_escalation_precision": ratio(len(esc_ok), len(escalated)),
        "Q4_completeness_mean": round(sum(compl) / len(compl), 3) if compl else None,
        "Q6_latency_median_s": {r: med(v) for r, v in lat.items()},
        "Q7_plan_prompt_tokens_median": med(plan_tokens),
        "Q8_forced_search_rate": ratio(len(forced), len(doc_steps)),
        "steps_passed": f"{sum(1 for s in steps if s['eval']['passed'])}/{len(steps)}",
    }


# ---------------------------------------------------------------- прогон

def load_chunks(path: Path) -> Dict[Tuple[str, Any], str]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                m = c.get("metadata", {})
                out[(m.get("file_name"), m.get("chunk_index"))] = c.get("text", "")
    return out


def run(args) -> int:
    data_dir = Path(args.data_dir).resolve()
    version_dir = Path(args.version_dir).resolve()
    scenarios_path = Path(args.scenarios).resolve()
    out_dir = Path(args.out).resolve()
    paths = {
        "qdrant": data_dir / "data/public/qdrant_storage",
        "chunks": data_dir / "data/public/chunks/chunks.jsonl",
        "cards": data_dir / "data/raw_registry_public/cards_meta.jsonl",
    }
    spec = yaml.safe_load(scenarios_path.read_text(encoding="utf-8"))
    scenarios = [s for s in spec["scenarios"] if not args.only or s["id"] in args.only]
    oracle = Oracle(paths["cards"], args.today_year or datetime.date.today().year)
    chunks = load_chunks(paths["chunks"])

    os.chdir(version_dir)                       # конфиги версии читаются из её папки
    sys.path.insert(0, str(version_dir / "src"))
    from agent_config import load_agent_config  # noqa: E402  код проверяемой версии
    from agent import KBAgent                   # noqa: E402
    from agent_llm import create_backend        # noqa: E402

    base = {"llm.backend": args.llm, "retriever.backend": args.retriever,
            "paths.qdrant_storage": str(paths["qdrant"]), "retriever.chunks_jsonl": str(paths["chunks"]),
            "catalog_path": str(paths["cards"])}
    if args.llm_base_url:
        base["llm.base_url"] = args.llm_base_url

    def make_agent(inject: Dict[str, Any]):
        cfg = load_agent_config("config/agent.yaml", profile="public", overrides={**base, **inject_overrides(inject)})
        llm = None
        if inject.get("llm_reply_without_citations"):
            llm = NoCitationsLLM(create_backend(cfg["llm"]))
        return KBAgent(cfg, llm=llm), cfg

    print(f"Версия: {version_dir} ({describe_version(version_dir)}) | данные: {data_dir} | LLM: {args.llm} | retriever: {args.retriever}")
    if not args.no_warmup:
        agent, _ = make_agent({})
        t0 = time.perf_counter()
        agent.run("Что такое SIPS?")
        agent.close()
        print(f"  прогрев: {time.perf_counter() - t0:.1f} с")

    steps_out = []
    for sc in scenarios:
        inject = sc.get("inject") or {}
        agent, cfg = make_agent(inject)
        for n, step in enumerate(sc["steps"], start=1):
            t0 = time.perf_counter()
            try:
                response = agent.run(step["question"])
            except Exception as exc:              # падение версии — это провал шага, а не прогона
                print(f"  {sc['id']}.{n} ИСКЛЮЧЕНИЕ {type(exc).__name__}: {exc}")
                steps_out.append({"scenario": sc["id"], "step": n, "question": step["question"], "inject": inject,
                                  "expect": step.get("expect", {}), "wall_ms": 0, "trace": [],
                                  "eval": {"route": None, "checks": [{"name": "exception", "passed": False,
                                           "expected": "без исключений", "actual": f"{type(exc).__name__}: {exc}"}],
                                           "passed": False, "completeness": None, "grounding": None, "escalated": False}})
                continue
            wall = int((time.perf_counter() - t0) * 1000)
            ev = evaluate(step, response, oracle, chunks, agent, cfg.get("grounding", {}))
            steps_out.append({
                "scenario": sc["id"], "step": n, "question": step["question"], "inject": inject,
                "expect": step.get("expect", {}), "wall_ms": wall, "status": response.status,
                "route_raw": getattr(response, "route", None), "answer": response.answer,
                "sources": response.sources, "llm_calls": response.llm_calls,
                "warnings": getattr(response, "warnings", []), "escalation": getattr(response, "escalation", None),
                "trace": trace_events(response), "eval": ev,
            })
            mark = "PASS" if ev["passed"] else "FAIL"
            failed = ", ".join(c["name"] for c in ev["checks"] if not c["passed"])
            print(f"  {sc['id']}.{n} {mark}  {ev['route'] or '-':10} {response.status:10} {wall / 1000:6.1f} с"
                  + (f"  [{failed}]" if failed else ""))
        agent.close()

    metrics = aggregate(steps_out)
    report = {
        "label": args.label, "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "version": {"dir": str(version_dir), "describe": describe_version(version_dir)},
        "environment": {"llm": args.llm, "retriever": args.retriever, "python": platform.python_version(),
                        "platform": platform.platform(terse=True), "dataset": spec.get("dataset"),
                        "today_year": oracle.year},
        "metrics": metrics, "steps": steps_out,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.label}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / f"{args.label}.md").write_text(summary_md(report), encoding="utf-8")
    print(f"\nМетрики: {json.dumps(metrics, ensure_ascii=False)}\nОтчёт: {out_dir / (args.label + '.json')}")
    return 0


def describe_version(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "describe", "--tags", "--always", "--dirty"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "n/a"
    except Exception:
        return "n/a"


# ---------------------------------------------------------------- отчёты

METRIC_ROWS = [
    ("Q1", "Ответы `documents` без привязки к источникам", "Q1_ungrounded_answered_rate", "0"),
    ("Q2", "Точность маршрутизации", "Q2_route_accuracy", "≥ 0,90"),
    ("Q3", "Эскалации: полнота", "Q3_escalation_recall", "1,00"),
    ("Q3", "Эскалации: точность", "Q3_escalation_precision", "≥ 0,80"),
    ("Q4", "Полнота перечислений", "Q4_completeness_mean", "1,00"),
    ("Q6", "Время, маршрут registry, медиана, с", ("Q6_latency_median_s", "registry"), "< 1"),
    ("Q6", "Время, маршрут documents, медиана, с", ("Q6_latency_median_s", "documents"), "≤ 36"),
    ("Q7", "Промпт шага «план», медиана, токенов", "Q7_plan_prompt_tokens_median", "≤ 900"),
    ("Q8", "Доля принудительного поиска", "Q8_forced_search_rate", "не выше baseline"),
    ("—", "Шагов пройдено", "steps_passed", "17/17"),
]


def metric(report, key):
    m = report["metrics"]
    if isinstance(key, tuple):
        return (m.get(key[0]) or {}).get(key[1])
    return m.get(key)


def fmt(v):
    if v is None:
        return "—"
    return str(v).replace(".", ",") if isinstance(v, float) else str(v)


def summary_md(report) -> str:
    lines = [f"# Контрольный набор: {report['label']}", "",
             f"Версия `{report['version']['describe']}`, LLM `{report['environment']['llm']}`, "
             f"retriever `{report['environment']['retriever']}`, {report['created_at']}.", "",
             "| | Метрика | Значение | Цель |", "|---|---|---|---|"]
    for qid, name, key, target in METRIC_ROWS:
        lines.append(f"| {qid} | {name} | {fmt(metric(report, key))} | {target} |")
    lines += ["", "| Шаг | Маршрут | Статус | Время, с | Итог | Не пройдено |", "|---|---|---|---|---|---|"]
    for s in report["steps"]:
        failed = ", ".join(c["name"] for c in s["eval"]["checks"] if not c["passed"])
        lines.append(f"| {s['scenario']}.{s['step']} | {s['eval']['route'] or '—'} | {s.get('status', '—')} | "
                     f"{fmt(round(s['wall_ms'] / 1000, 1))} | {'✓' if s['eval']['passed'] else '✗'} | {failed} |")
    return "\n".join(lines) + "\n"


def compare(args) -> int:
    out_dir = Path(args.out)
    a, b = (json.loads((out_dir / f"{l}.json").read_text(encoding="utf-8")) for l in args.compare)
    lines = [f"# Сравнение: {a['label']} → {b['label']}", "",
             f"Одинаковые данные (`{a['environment']['dataset']}`) и сценарии. "
             f"{a['label']}: `{a['version']['describe']}`; {b['label']}: `{b['version']['describe']}`.", "",
             f"| | Метрика | {a['label']} | {b['label']} | Цель |", "|---|---|---|---|---|"]
    for qid, name, key, target in METRIC_ROWS:
        lines.append(f"| {qid} | {name} | {fmt(metric(a, key))} | {fmt(metric(b, key))} | {target} |")
    sb = {(s["scenario"], s["step"]): s for s in b["steps"]}
    lines += ["", f"| Шаг | {a['label']} | {b['label']} |", "|---|---|---|"]
    for s in a["steps"]:
        t = sb.get((s["scenario"], s["step"]))
        cell = lambda x: ("✓" if x["eval"]["passed"] else "✗") + f" {x['eval']['route'] or '—'} / {x.get('status', '—')}" if x else "—"
        lines.append(f"| {s['scenario']}.{s['step']} | {cell(s)} | {cell(t)} |")
    path = out_dir / "comparison.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))
    return 0


def force_utf8_when_redirected() -> None:
    """
    При выводе в конвейер или файл (| Tee-Object, > out.txt) Python на Windows кодирует текст
    в системной кодировке (cp1252/cp1251) и падает на кириллице. Перенаправленные потоки
    переводим в UTF-8; вывод прямо в консоль не трогаем. Действует и на модули проверяемой версии.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure") and not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def main(argv=None) -> int:
    force_utf8_when_redirected()
    p = argparse.ArgumentParser(description="Контрольный набор: прогон версии и сравнение")
    p.add_argument("--version-dir", default=".")
    p.add_argument("--label")
    p.add_argument("--data-dir", default=".")
    p.add_argument("--scenarios", default="config/control_scenarios.yaml")
    p.add_argument("--llm", choices=["ollama", "mock"], default="ollama")
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--retriever", choices=["qdrant", "lexical"], default="qdrant")
    p.add_argument("--only", action="append", metavar="CS-NN")
    p.add_argument("--today-year", type=int, default=None, help="год для относительных периодов (по умолчанию текущий)")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--out", default="reports/control")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = p.parse_args(argv)
    if args.compare:
        return compare(args)
    if not args.label:
        p.error("нужен --label")
    args.scenarios = str(Path(args.scenarios).resolve())
    args.out = str(Path(args.out).resolve())
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
