"""
Точка входа агента.

Диалог:
    python src/agent_cli.py                                 # интерактивный режим (qwen2.5 + Qdrant, профиль public)
    python src/agent_cli.py -q "Что входит в цифровое ядро AIMS?"
    python src/agent_cli.py -q "..." --show-trace --save-trace
    python src/agent_cli.py --profile full                  # полная база (только локально!)
    python src/agent_cli.py --llm mock --retriever lexical  # без Ollama, torch и Qdrant

Сценарии:
    python src/agent_cli.py --run-scenarios --llm mock --retriever lexical
    python src/agent_cli.py --run-scenarios                           # боевой прогон на Ollama
    python src/agent_cli.py --run-scenario main_aims --run-scenario recovery_followup
    python src/agent_cli.py --run-scenarios --scenario-report data/agent/my_report.json

Команды в диалоге: /reset — очистить память, /memory — показать память,
/trace — trace последнего ответа, /exit — выход.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).resolve().parent
PROJECT_ROOT = SRC.parent
sys.path.insert(0, str(SRC))

TRACES_DIR = Path("data/agent/traces")


def _overrides(args) -> dict:
    out = {}
    if args.llm:
        out["llm.backend"] = args.llm
    if args.retriever:
        out["retriever.backend"] = args.retriever
    return out


def _print_response(response, show_trace: bool = False) -> None:
    print()
    print(response.answer)
    if response.sources and response.status in ("answered", "degraded"):
        print("\nИсточники:")
        from agent_tools import card_label
        for s in response.sources:
            if s.get("card"):
                print(f"  [Фрагмент {s['fragment']}] {s['card'].get('card_title', '')} "
                      f"({card_label(s['card'])}) — {s['file_name']}, score {s['score']:.4f}")
            else:
                print(f"  [Фрагмент {s['fragment']}] {s['document_name']} — {s['file_name']}, "
                      f"чанк {s['chunk_index']}, score {s['score']:.4f}")
    if response.warnings:
        print("\nПредупреждения: " + "; ".join(response.warnings))
    tools = ", ".join(f"{c.name}({'ok' if c.status == 'success' else c.error_code})" for c in response.tool_calls)
    print(f"\n[{response.status}] маршрут: {response.route} | LLM: {response.llm_calls} | "
          f"tools: {tools or '—'} | {response.duration_ms / 1000:.1f} c | память: {response.memory_size}")
    if show_trace:
        print("\nTRACE:")
        for event in response.trace.to_list():
            print(f"  {event['t_ms']:>7} мс  {event['type']:<12} "
                  f"{json.dumps(event['data'], ensure_ascii=False)[:300]}")


def _save_trace(response, query_label: str) -> Path:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACES_DIR / f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"query": query_label, **response.to_dict(include_trace=True)}, f, ensure_ascii=False, indent=2)
    return path


def run_dialog(args) -> int:
    from agent import KBAgent
    from agent_config import load_agent_config

    try:
        config = load_agent_config(args.config, profile=args.profile, overrides=_overrides(args))
        agent = KBAgent(config)
    except Exception as exc:
        print(f"Не удалось инициализировать агента: {exc}", file=sys.stderr)
        return 2

    if config["profile"] == "full":
        print("ВНИМАНИЕ: профиль full — в ответах возможны ФИО и наименования клиентов. "
              "Не сохраняйте такие ответы и trace в репозиторий.\n")
    print(agent.describe())

    try:
        if args.query is not None:
            response = agent.run(args.query)
            if args.json:
                print(json.dumps(response.to_dict(include_trace=args.show_trace), ensure_ascii=False, indent=2))
            else:
                _print_response(response, args.show_trace)
            if args.save_trace:
                print(f"trace: {_save_trace(response, response.trace.events[0].data.get('query', ''))}")
            return 0 if response.status in ("answered", "not_found", "smalltalk") else 1

        print("Задайте вопрос по базе знаний. /reset — очистить память, /trace — trace, /exit — выход.")
        last = None
        while True:
            try:
                line = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            cmd = line.strip().lower()
            if cmd in ("/exit", "/quit", "exit", "quit"):
                break
            if cmd == "/reset":
                agent.memory.clear()
                print("Память очищена.")
                continue
            if cmd == "/memory":
                for m in agent.memory.messages():
                    print(f"  {m['role']}: {m['content'][:200]}")
                continue
            if cmd == "/trace":
                if last is None:
                    print("Ещё нет ответа.")
                else:
                    _print_response(last, show_trace=True)
                continue
            last = agent.run(line)
            _print_response(last, args.show_trace)
            if args.save_trace:
                print(f"trace: {_save_trace(last, last.trace.events[0].data.get('query', ''))}")
        return 0
    finally:
        agent.close()


def run_scenarios(args) -> int:
    from scenario_runner import ScenarioRunner, save_reports

    if args.profile == "full" and not args.allow_full_profile:
        print("Отчёты сценариев предназначены для репозитория, а профиль full содержит ФИО и клиентов. "
              "Если это осознанно — добавьте --allow-full-profile.", file=sys.stderr)
        return 2

    runner = ScenarioRunner(args.config, base_overrides=_overrides(args), profile=args.profile or "public")
    only = args.run_scenario or None
    try:
        report = runner.run(args.scenarios, only=only)
    except Exception as exc:
        print(f"Прогон сценариев прерван: {exc}", file=sys.stderr)
        return 2

    backend = report["environment"]["llm_backend"]
    report_path = args.scenario_report or f"data/agent/scenario_report_{backend}.json"
    results_path = args.results or ("data/test_results.json" if backend == "mock"
                                    else f"data/test_results_{backend}.json")
    save_reports(report, report_path, results_path)

    s = report["summary"]
    print(f"\nИтог: {s['passed']}/{s['total']} сценариев пройдено "
          f"({', '.join(f'{k}: {v['passed']}/{v['total']}' for k, v in s['by_type'].items())})")
    print(f"Окружение: {json.dumps(report['environment'], ensure_ascii=False)}")
    print(f"Полный отчёт: {report_path}\nКраткий отчёт: {results_path}")
    return 0 if report["passed"] else 1


def _force_utf8_when_redirected() -> None:
    """
    При выводе в конвейер или файл (| Tee-Object, > out.txt) Python на Windows кодирует
    текст в системной кодировке (cp1251/cp866) и падает на символах вне неё.
    Перенаправленные потоки переводим в UTF-8; вывод прямо в консоль не трогаем —
    там Python и так пишет через Unicode API Windows.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure") and not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="KB Search Agent — агент поиска по базе знаний")
    parser.add_argument("-q", "--query", help="один вопрос без интерактивного режима")
    parser.add_argument("--config", default="config/agent.yaml")
    parser.add_argument("--profile", choices=["public", "full"], default=None,
                        help="индекс: public (5 обезличенных PDF) или full (полная база)")
    parser.add_argument("--llm", choices=["ollama", "mock"], default=None)
    parser.add_argument("--retriever", choices=["qdrant", "lexical"], default=None)
    parser.add_argument("--show-trace", action="store_true")
    parser.add_argument("--save-trace", action="store_true", help=f"сохранить trace в {TRACES_DIR}/")
    parser.add_argument("--json", action="store_true", help="ответ в JSON (с -q)")

    parser.add_argument("--run-scenarios", action="store_true", help="прогнать все сценарии")
    parser.add_argument("--run-scenario", action="append", metavar="ID", help="прогнать сценарий по id (можно несколько)")
    parser.add_argument("--scenarios", default="config/agent_scenarios.yaml")
    parser.add_argument("--scenario-report", default=None, help="путь полного отчёта")
    parser.add_argument("--results", default=None, help="путь краткого отчёта")
    parser.add_argument("--allow-full-profile", action="store_true")
    args = parser.parse_args(argv)

    os.chdir(PROJECT_ROOT)
    _force_utf8_when_redirected()
    if args.run_scenarios or args.run_scenario:
        return run_scenarios(args)
    return run_dialog(args)


if __name__ == "__main__":
    sys.exit(main())
