"""
Тесты обработки реестров проектов и пропозалов и фильтров поиска по карточкам.
Только синтетические данные; без Ollama и torch. Тест с Qdrant пропускается, если qdrant-client не установлен.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

from conftest import ScriptedLLM, reply

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import csv_to_cards as cards  # noqa: E402
from agent_prompts import build_tool_schemas  # noqa: E402
from agent_tools import ToolRegistry, payload_matches  # noqa: E402
from card_metadata import card_meta_for_chunk, load_card_metadata  # noqa: E402
from chunker import chunk_document  # noqa: E402
from cleaner import clean_text, cleaning_for  # noqa: E402


# ------------------------------------------------------------------ синтетические CSV

def write_csv(path: Path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def registry_csv(tmp_path):
    src = tmp_path / "csv"
    src.mkdir()
    write_csv(src / "projects.csv",
              ["ID Проекта", "Наименование проекта", "Обезличенное наименование проекта",
               "Реальный клиент (краткое наименование)", "Обезличенное наименование клиента",
               "Год проекта", "Тип услуги", "Отрасль"],
              [["1", "Стратегия ремонтов", "Стратегия ТОиР", "Пример-Сталь", "Металлургический холдинг",
                "2019-2020", "Стратегия; 0", "Industrial goods"],
               ["2", "", "", "", "", "", "", ""],                          # пустая карточка
               ["3", "Программа закупок", "Закупки", "Пример-Транс", "Перевозчик", "2015", "Операционная эффективность",
                "Transportation"]])
    write_csv(src / "project_tags_industry.csv", ["ID проекта", "Вид тега", "Категория", "Тег"],
              [["1", "отрасль", "Верхний уровень", "Industrial goods"], ["1", "отрасль", "Industrial goods", "Металлургия"]])
    write_csv(src / "project_tags_service.csv", ["ID проекта", "Вид тега", "Категория", "Тег"],
              [["1", "услуга", "Стратегия", "Стратегия ТОиР"]])
    write_csv(src / "project_team.csv", ["ID проекта", "Сотрудник", "В штате"], [["1", "Иванов И.", "да"]])
    write_csv(src / "proposals.csv",
              ["ID", "Наименование пропозала", "Реальный клиент (краткое наименование)",
               "Обезличенное наименование клиента", "Год пропозала", "Тип услуги", "Отрасль"],
              [["10", "Цифровизация склада", "Пример-Ритейл", "Сеть магазинов", "2022", "Digital", "Consumer goods & retail"]])
    key_findings = "Итоги:\n1. Снижение простоев на 12%\n2. Рост выручки\n1.1. Детализация"
    write_csv(src / "projects_checked.csv",
              ["Код проекта", "Тип", "Дата начала", "Дата окончания - факт", "Год проекта", "Статус проекта ",
               "Клиент по договору (краткое наименование)", "Клиент по договору (полное наименование)",
               "Стоимость с НДС, руб.", "Название (юр.) проекта по Договору", "Обезличенное наименование проекта",
               "Тип услуги", "Отрасль", "Описание проекта", "Key findings", "Команда проекта"],
              [["21_001 ABC", "Проект", "2021-02-01", "2021-11-30", "2021", "завершен", "Пример-Энерго",
                "ООО «Пример-Энерго»", "1234567.5", "Оказание услуг по разработке стратегии",
                "Стратегия развития энергокомпании", "HR; Стратегия", "Топливно-энергетический комплекс\nЖКХ",
                "Разработка стратегии.", key_findings, "Иванов И.; Петров П."]])
    return src


def test_cards_structure_and_metadata(registry_csv, tmp_path):
    out = tmp_path / "cards"
    all_meta = []
    for kind in cards.SCHEMES:
        entries, meta, skipped = cards.build_documents(registry_csv, out, kind, anonymized=False)
        all_meta += meta
        if kind == "projects":
            assert skipped == 1                                     # пустая карточка не записана
    text = (out / "projects_2019-2020.md").read_text(encoding="utf-8")
    assert text.startswith("1. Стратегия ремонтов — Пример-Сталь (2019–2020)")   # без вводной строки файла
    assert "Клиент: Пример-Сталь" in text.splitlines()[1]
    assert "Тип услуги: Стратегия" in text and "; 0" not in text   # мусорное значение «0» убрано

    by_id = {m["card_id"]: m for m in all_meta}
    assert by_id["1"]["year"] == 2019 and by_id["1"]["year_end"] == 2020
    assert by_id["1"]["industry"] == ["Industrial goods"]
    assert by_id["10"]["card_type"] == "proposal"
    acc = by_id["21_001 ABC"]
    assert acc["card_type"] == "project_accounting" and acc["status"] == "Завершен"
    # русские отрасли сопоставлены категориям верхнего уровня, исходные значения сохранены
    assert acc["industry"] == ["Energy (O&G / chem / utilities)", "Топливно-энергетический комплекс", "ЖКХ"]
    assert acc["service"] == ["Организационное развитие / HR", "Стратегия"]

    acc_text = (out / "projects_accounting_2021.md").read_text(encoding="utf-8")
    assert "Стоимость с НДС: 1 234 567,50 руб." in acc_text
    assert "\n1) Снижение простоев" in acc_text and "\n1.1) Детализация" in acc_text
    # внутри карточки нет строк, которые чанкер принял бы за заголовок новой карточки
    body_lines = acc_text.splitlines()[1:]
    assert not any(line[:1].isdigit() and ". " in line[:6] for line in body_lines)


def test_anonymized_hides_money_and_people(registry_csv, tmp_path):
    out = tmp_path / "anon"
    cards.build_documents(registry_csv, out, "checked", anonymized=True)
    text = next(out.glob("*.md")).read_text(encoding="utf-8")
    for secret in ("Стоимость", "Иванов", "Пример-Энерго", "21_001"):
        assert secret not in text


def test_file_kind_and_partial_rebuild_keeps_other_metadata(tmp_path):
    assert cards.file_kind("projects_accounting_2021.md") == "checked"
    assert cards.file_kind("projects_2021.md") == "projects"
    meta_path = tmp_path / cards.CARD_META_FILE
    old = [{"file_name": "projects_accounting_2021.md", "card_number": 1},
           {"file_name": "projects_2019.md", "card_number": 1}]
    meta_path.write_text("\n".join(json.dumps(x) for x in old), encoding="utf-8")
    cards.write_card_meta(meta_path, [{"file_name": "projects_2020.md", "card_number": 1}], {"projects"})
    names = [json.loads(l)["file_name"] for l in meta_path.read_text(encoding="utf-8").splitlines()]
    assert names == ["projects_accounting_2021.md", "projects_2020.md"]


# ------------------------------------------------------------------ чанкинг и очистка

def test_chunks_get_card_metadata_including_continuations(registry_csv, tmp_path):
    out = tmp_path / "cards"
    _, meta, _ = cards.build_documents(registry_csv, out, "checked", anonymized=False)
    (out / cards.CARD_META_FILE).write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in meta), encoding="utf-8")
    index = load_card_metadata([str(out / cards.CARD_META_FILE)])
    text = (out / "projects_accounting_2021.md").read_text(encoding="utf-8")
    text = text.replace("Разработка стратегии.", "Разработка стратегии. " + "Длинное предложение описания. " * 60)
    chunks = chunk_document(text, "section", 500, 50, True, merge_short_sections=False)
    assert len(chunks) >= 2
    for c in chunks:
        assert c["text"].split("\n")[0].startswith("1. Стратегия развития энергокомпании")   # заголовок — отдельной строкой
        m = card_meta_for_chunk(index, "projects_accounting_2021.md", c["text"])
        assert m and m["card_type"] == "project_accounting" and m["year"] == 2021


def test_pdf_unwrap_only_for_pdf():
    cfg = {"unwrap_pdf_line_breaks": True}
    card = "1. Проект — Клиент (2020)\nКлиент: Пример\nТип услуги: Стратегия"
    assert clean_text(card, cleaning_for({"file_type": "txt"}, cfg)) == card
    assert "\n" not in clean_text("строка без точки\nпродолжение", cleaning_for({"file_type": "pdf"}, cfg))


# ------------------------------------------------------------------ фильтры поиска

REGISTRY = [
    {"text": "1. Стратегия ТОиР — Металлургический холдинг (2019–2020)\nТип услуги: Стратегия",
     "metadata": {"file_name": "projects_2019.md", "category": "internal/projects", "card_type": "project",
                  "card_title": "Стратегия ТОиР", "year": 2019, "year_end": 2020,
                  "industry": ["Industrial goods", "Металлургия и горнодобывающая отрасль"], "chunk_index": 0}},
    {"text": "1. Закупки — Перевозчик (2015)\nТип услуги: Операционная эффективность",
     "metadata": {"file_name": "projects_2015.md", "category": "internal/projects", "card_type": "project",
                  "card_title": "Закупки", "year": 2015, "year_end": 2015, "industry": ["Transportation"],
                  "chunk_index": 0}},
    {"text": "Коммерческое предложение по стратегии ТОиР",
     "metadata": {"file_name": "kp.pdf", "category": "internal/commercial_proposals", "chunk_index": 3}},
]


class RegistryIndex:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k, category=None, filters=None):
        self.calls.append({"category": category, "filters": filters})
        hits = [c for c in REGISTRY if payload_matches(c["metadata"], category, filters)]
        return [{"rank": i + 1, "score": 1.0 - i / 10, "chunk_id": str(i), "document_id": "d",
                 "text": c["text"], "metadata": dict(c["metadata"])} for i, c in enumerate(hits[:top_k])]

    def list_documents(self):
        return []

    def facets(self):
        from agent_tools import FacetAccumulator
        acc = FacetAccumulator()
        for c in REGISTRY:
            acc.add(c["metadata"])
        return acc.result()["facets"]


def registry_tools():
    return ToolRegistry(lambda: RegistryIndex(), {"search": {"max_top_k": 10}})


def test_year_filter_uses_period_overlap():
    tools = registry_tools()
    r, _ = tools.execute("search_knowledge_base", {"query": "стратегия", "year_from": 2020, "year_to": 2023})
    assert [h["metadata"]["card_title"] for h in r["hits"]] == ["Стратегия ТОиР"]   # 2019–2020 пересекается
    r, _ = tools.execute("search_knowledge_base", {"query": "стратегия", "year_to": "2016"})
    assert [h["metadata"]["card_title"] for h in r["hits"]] == ["Закупки"]


def test_industry_filter_is_case_insensitive_and_validated():
    tools = registry_tools()
    r, _ = tools.execute("search_knowledge_base", {"query": "стратегия", "industry": "металлургия и горнодобывающая отрасль"})
    assert r["status"] == "success" and r["filters"]["industry"] == ["Металлургия и горнодобывающая отрасль"]
    assert len(r["hits"]) == 1
    bad, _ = tools.execute("search_knowledge_base", {"query": "x", "industry": "Космос"})
    assert bad["error_code"] == "invalid_argument" and "Transportation" in bad["error"]


@pytest.mark.parametrize("args, fragment", [
    ({"year_from": 2023, "year_to": 2020}, "больше year_to"),
    ({"year_from": "позапрошлый"}, "должен быть годом"),
    ({"year_to": 1800}, "1900–2100"),
], ids=["reversed", "text", "range"])
def test_year_validation(args, fragment):
    r, _ = registry_tools().execute("search_knowledge_base", {"query": "x", **args})
    assert r["error_code"] == "invalid_argument" and fragment in r["error"]


def test_industry_filter_unavailable_without_facets(make_agent):
    agent = make_agent()                       # синтетический корпус без отраслевой разметки
    r, _ = agent.tools.execute("search_knowledge_base", {"query": "ORBIT", "industry": "Transportation"})
    assert r["error_code"] == "invalid_argument"


def test_tool_schema_lists_values_from_index():
    schemas = build_tool_schemas(RegistryIndex().facets())
    props = schemas[0]["function"]["parameters"]["properties"]
    assert "internal/projects — реестр проектов" in props["category"]["description"]
    assert "Transportation" in props["industry"]["description"]
    assert "2015–2020" in props["year_from"]["description"]
    assert "industry" not in build_tool_schemas(None)[0]["function"]["parameters"]["properties"]


def test_agent_passes_filters_and_shows_cards(base_config):
    from agent import KBAgent
    llm = ScriptedLLM([
        reply(tool_calls=[{"name": "search_knowledge_base",
                           "arguments": {"query": "стратегия ТОиР", "category": "internal/projects",
                                         "year_from": 2018, "industry": "Industrial goods"}}]),
        reply("Стратегия ТОиР для металлургического холдинга, 2019–2020 [Фрагмент 1]."),
    ])
    index = RegistryIndex()
    agent = KBAgent(base_config, llm=llm, retriever_factory=lambda: index)
    r = agent.run("Какие проекты по ТОиР в металлургии были после 2018 года?")
    assert r.status == "answered"
    assert index.calls[0]["filters"] == {"year_from": 2018, "industry": ["Industrial goods"]}
    assert r.sources[0]["card"]["card_title"] == "Стратегия ТОиР"
    tool_msg = llm.calls[1]["messages"][-1]["content"]
    assert "Карточка: Стратегия ТОиР (проект | 2019–2020 | Industrial goods" in tool_msg
    schema_props = llm.calls[0]["messages"] and agent._schemas[0]["function"]["parameters"]["properties"]
    assert "industry" in schema_props


# ------------------------------------------------------------------ фильтр Qdrant

def test_qdrant_filter_semantics():
    qc = pytest.importorskip("qdrant_client")
    from qdrant_client.http import models
    from agent_tools import build_qdrant_filter
    client = qc.QdrantClient(":memory:")
    client.create_collection("t", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    client.upsert("t", [models.PointStruct(id=i + 1, vector=[1.0, 0.1 * i], payload=c["metadata"])
                        for i, c in enumerate(REGISTRY)])

    def titles(category=None, **filters):
        res = client.query_points("t", query=[1.0, 0.0], limit=10,
                                  query_filter=build_qdrant_filter(category, filters)).points
        return sorted(p.payload.get("card_title", p.payload["file_name"]) for p in res)

    assert titles(year_from=2020) == ["Стратегия ТОиР"]
    assert titles(year_to=2016) == ["Закупки"]
    assert titles(industry=["Transportation"]) == ["Закупки"]
    assert titles(category="internal/commercial_proposals") == ["kp.pdf"]
    assert build_qdrant_filter(None, {}) is None
    client.close()
