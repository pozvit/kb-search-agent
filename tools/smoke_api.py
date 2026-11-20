"""
Проверка развёртывания через HTTP (требование F10): то, что эксперт увидит на видео.

  1. GET /health — ждём 200 (модель, индекс, каталог доступны);
  2. вопрос по документу (CS-01): answered, маршрут documents, есть источники;
  3. вопрос по реестру (CS-05): answered, маршрут registry, 5 карточек, без вызова модели;
  4. секрет в вопросе (CS-16): rejected, секрет не возвращается в ответе;
  5. некорректный запрос: 422 с понятным сообщением.

Запуск:  python tools/smoke_api.py [--url http://127.0.0.1:8000] [--wait 600]
Только стандартная библиотека Python. Код возврата 1, если какая-то проверка не прошла.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple

SECRET = "sk-proj-TESTONLY" + "0" * 20


def call(url: str, method: str = "GET", payload: Any = None, timeout: float = 900) -> Tuple[int, Dict[str, Any]]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def wait_health(base: str, wait: float) -> Tuple[bool, str]:
    deadline, last = time.monotonic() + wait, "нет ответа"
    while time.monotonic() < deadline:
        try:
            code, body = call(base + "/health", timeout=10)
            comps = body.get("components", {})
            last = f"HTTP {code}, " + ", ".join(f"{k}: {v.get('status')}" for k, v in comps.items())
            if code == 200:
                return True, last
        except (urllib.error.URLError, OSError, ValueError):
            last = "сервис не отвечает"
        time.sleep(5)
    return False, last


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Проверка развёртывания KB Search Agent")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--wait", type=float, default=600, help="сколько секунд ждать /health = 200")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    base = args.url.rstrip("/")
    results = []

    ok, detail = wait_health(base, args.wait)
    results.append(("GET /health = 200", ok, detail))
    if ok:
        checks = [
            ("Документ (CS-01)", {"question": "Сколько сегментов рынка микроэлектроники моделируется в бизнес-симуляторе ERGO?"},
             lambda c, b: c == 200 and b.get("status") == "answered" and b.get("route") == "documents" and b.get("sources")),
            ("Реестр (CS-05)", {"question": "Какие проекты были в медицине?"},
             lambda c, b: c == 200 and b.get("route") == "registry" and len(b.get("sources", [])) == 5 and b.get("llm_calls") == 0),
            ("Секрет (CS-16)", {"question": f"Мой ключ {SECRET}, найди проекты по AIMS"},
             lambda c, b: c == 200 and b.get("status") == "rejected" and SECRET not in json.dumps(b, ensure_ascii=False)),
            ("Некорректный запрос", {"question": ""},
             lambda c, b: c == 422 and b.get("error", {}).get("code") == "invalid_request"),
        ]
        for name, payload, check in checks:
            started = time.monotonic()
            code, body = call(base + "/ask", "POST", payload)
            passed = bool(check(code, body))
            info = body.get("error", {}).get("code") if code >= 400 else \
                f"{body.get('status')} / {body.get('route')}, источников {len(body.get('sources', []))}"
            results.append((name, passed, f"HTTP {code}, {info}, {time.monotonic() - started:.1f} с"))

    for name, passed, detail in results:
        print(f"{'✓' if passed else '✗'} {name:24} {detail}")
    failed = [r for r in results if not r[1]]
    print(f"\nИтог: {len(results) - len(failed)} из {len(results)} проверок пройдено")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
