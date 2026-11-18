"""
Тесты инструмента замера tools/run_control.py: если он ошибается, сравнение «до/после» теряет смысл.
Только синтетические данные.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_control as rc  # noqa: E402

FRAGS = {1: "Цифровое ядро AIMS: цифровой паспорт актива, цифровой пульс актива, цифровая биржа имущества.",
         2: "В игре 5 сегментов рынка микроэлектроники: военная электроника и другие."}


def test_grounded_answer_passes():
    ans = "Цифровое ядро AIMS включает цифровой паспорт актива, цифровой пульс актива и цифровую биржу [Фрагмент 1]."
    g = rc.check_grounding(ans, FRAGS)
    assert g["grounded"] and g["units"] == 1


@pytest.mark.parametrize("answer, why", [
    ("Цифровое ядро AIMS включает цифровой паспорт актива и цифровой пульс актива.", "нет ссылки"),
    ("Цифровое ядро AIMS включает цифровой паспорт актива и цифровой пульс [Фрагмент 7].", "нет ссылки"),
    ("Сегменты рынка микроэлектроники включают военную электронику и другие [Фрагмент 1].", "мало общих слов"),
    ("В игре 7 сегментов рынка микроэлектроники и военная электроника [Фрагмент 2].", "чисел нет"),
], ids=["no_citation", "unknown_fragment", "wrong_fragment", "wrong_number"])
def test_ungrounded_answers_are_caught(answer, why):
    g = rc.check_grounding(answer, FRAGS)
    assert not g["grounded"] and why in g["unsupported"][0]["why"]


def test_paragraph_level_citation_covers_list_items():
    ans = ("Цифровое ядро AIMS включает:\n• цифровой паспорт актива;\n• цифровой пульс актива;\n"
           "• цифровая биржа имущества [Фрагмент 1].\n\nИсточники: [Фрагмент 1]")
    assert rc.check_grounding(ans, FRAGS)["grounded"]


def test_oracle_selection_and_ranking(tmp_path):
    cards = [
        {"card_type": "project", "card_id": "1", "industry": ["Pharma & HC"], "service": ["Операционная эффективность"],
         "year": 2025, "year_end": 2025, "team": ["Иванов И.", "Петров П."]},
        {"card_type": "project", "card_id": "2", "industry": ["Financial services"], "service": ["Операционная эффективность"],
         "year": 2020, "year_end": 2020, "team": ["Иванов И."]},
        {"card_type": "proposal", "card_id": "1", "industry": ["Pharma & HC"], "service": ["Стратегия"],
         "service_detail": ["Закупки / управление поставщиками"], "year": 2026, "year_end": 2026, "partner": "Петров П."},
    ]
    path = tmp_path / "cards_meta.jsonl"
    path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cards), encoding="utf-8")
    o = rc.Oracle(path, today_year=2026)
    assert o.select({"industry": ["Pharma & HC"]}) == {("project", "1"), ("proposal", "1")}
    assert o.select({"service": ["Закупки / управление поставщиками"]}) == {("proposal", "1")}
    assert o.select({"person": ["Петров П."], "last_years": 2}) == {("project", "1"), ("proposal", "1")}
    oe = o.select({"service": ["Операционная эффективность"]})
    assert o.ranking(oe) == [("Иванов И.", 2), ("Петров П.", 1)]      # пропозалы в рейтинг проектов не входят


def test_route_equivalents():
    class R:
        def __init__(self, route, status="answered"):
            self.route, self.status = route, status
    assert rc.canonical_route(R("catalog_people_rank")) == "registry"
    assert rc.canonical_route(R("forced_search")) == "documents"
    assert rc.canonical_route(R("guardrail_block", "rejected")) == "rejected"
    assert rc.canonical_route(R("documents", "escalated")) == "escalation"


def test_no_citations_wrapper_strips_references():
    class Reply:
        def __init__(self, content, tool_calls=None):
            self.content, self.tool_calls = content, tool_calls or []

    class Inner:
        def describe(self):
            return "inner"

        def chat(self, messages, tools=None):
            return Reply("Паспорт актива [Фрагмент 1], пульс [Фрагменты 1, 2].\nИсточники: [Фрагмент 1]")

    out = rc.NoCitationsLLM(Inner()).chat([])
    assert "Фрагмент" not in out.content and "Источники" not in out.content


def test_scenarios_file_is_consistent():
    import yaml
    spec = yaml.safe_load((ROOT / "config" / "control_scenarios.yaml").read_text(encoding="utf-8"))
    ids = [s["id"] for s in spec["scenarios"]]
    assert len(ids) == len(set(ids)) == 16
    covered = {i for group in spec["coverage"].values() for i in group}
    assert covered <= set(ids)
    for key in ("tools", "rag", "memory", "no_information", "component_failure"):   # требование траектории
        assert spec["coverage"][key]


# ---------------------------------------------------------------- правила v2 (по итогам разбора ответов baseline)

def test_sources_line_counts_as_answer_level_citation():
    ans = ("Цифровое ядро AIMS включает три ключевых решения:\n\n"
           "1. Цифровой паспорт актива на всех стадиях.\n2. Цифровой пульс актива моделирует воздействия.\n\n"
           "Источники: [Фрагмент 1]")
    frags = {1: "Цифровой паспорт актива на всех стадиях жизненного цикла. Цифровой пульс актива моделирует воздействия."}
    g = rc.check_grounding(ans, frags)
    assert g["grounded"], g["unsupported"]


def test_answer_level_citation_still_requires_support():
    ans = "1. Компания открыла завод в Казани и наняла тысячу сотрудников.\n\nИсточники: [Фрагмент 1]"
    g = rc.check_grounding(ans, FRAGS)
    assert not g["grounded"] and "мало общих слов" in g["unsupported"][0]["why"]


def test_short_cited_items_without_long_units():
    assert rc.check_grounding("Сотрудники:\n1. Ветлугин В. [Фрагмент 1]", {1: "Команда: Ветлугин В."})["grounded"]
    assert not rc.check_grounding("Сотрудники:\n1. Ветлугин В.", {1: "Команда: Ветлугин В."})["grounded"]


def test_report_paths_do_not_contain_profile_name():
    assert rc.safe_path(r"C:\Users\Ivan.Petrov\kb_baseline") == r"C:\Users\~\kb_baseline"
    assert rc.safe_path("/home/ivan/kb") == "/home/~/kb"
    assert rc.safe_path(str(Path.home() / "kb_baseline")).startswith("~")
    assert rc.safe_path("../kb_baseline") == "../kb_baseline"
