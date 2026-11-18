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


def test_registry_filters_from_model_are_dropped_cards_still_shown(base_config):
    """Шаг 2.4.2: у подагента «Документы» нет фильтров реестра; если модель их передала — они отбрасываются."""
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
    assert not (index.calls[0]["filters"] or {}) and index.calls[0]["category"] is None
    assert any("category, industry, year_from" in w for w in r.warnings)
    assert r.sources[0]["card"]["card_title"] == "Стратегия ТОиР"
    assert "Карточка: Стратегия ТОиР (проект | 2019–2020 | Industrial goods" in llm.calls[1]["messages"][-1]["content"]
    offered = llm.calls[0]
    assert offered["tools"] is True


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


# ------------------------------------------------------------------ уточнения по карточкам прошлого ответа

MED = [
    {"text": "1. Бизнес-план курорта — Клиника (2011)\nОтрасль: Pharma & HC; Здравоохранение и фармацевтика\nКоманда: Иванов И.",
     "metadata": {"file_name": "projects_accounting_2011.md", "category": "internal/projects_accounting",
                  "card_type": "project_accounting", "card_id": "11_001 AAA", "card_title": "Бизнес-план курорта",
                  "year": 2011, "year_end": 2011, "industry": ["Pharma & HC", "Здравоохранение и фармацевтика"]}},
    {"text": "1. Аудит маркетинга — Медцентр (2018)\nОтрасль: Pharma & HC\nКоманда: Петров П.",
     "metadata": {"file_name": "projects_accounting_2018.md", "category": "internal/projects_accounting",
                  "card_type": "project_accounting", "card_id": "18_002 BBB", "card_title": "Аудит маркетинга",
                  "year": 2018, "year_end": 2018, "industry": ["Pharma & HC"]}},
    {"text": "Коммерческое предложение: команда проекта для нефтяной компании",
     "metadata": {"file_name": "kp.pdf", "category": "internal/commercial_proposals"}},
    {"text": "1. Энергоаудит — Генерация (2020)\nОтрасль: Energy (O&G / chem / utilities); Топливно-энергетический комплекс",
     "metadata": {"file_name": "projects_2020.md", "category": "internal/projects", "card_type": "project",
                  "card_id": "5", "card_title": "Энергоаудит", "year": 2020, "year_end": 2020,
                  "industry": ["Energy (O&G / chem / utilities)", "Топливно-энергетический комплекс"]}},
]


class MedIndex(RegistryIndex):
    def search(self, query, top_k, category=None, filters=None):
        self.calls.append({"query": query, "category": category, "filters": filters, "top_k": top_k})
        hits = [c for c in MED if payload_matches(c["metadata"], category, filters)]
        return [{"rank": i + 1, "score": 0.9 - i / 100, "chunk_id": str(i), "document_id": "d",
                 "text": c["text"], "metadata": dict(c["metadata"])} for i, c in enumerate(hits[:top_k])]

    def facets(self):
        from agent_tools import FacetAccumulator
        acc = FacetAccumulator()
        for c in MED:
            acc.add(c["metadata"])
        return acc.result()["facets"]


def med_agent(base_config, replies):
    from agent import KBAgent
    index = MedIndex()
    return KBAgent(base_config, llm=ScriptedLLM(replies), retriever_factory=lambda: index), index


FIRST = [reply(tool_calls=[{"name": "search_knowledge_base",
                            "arguments": {"query": "проекты медицина", "industry": "Pharma & HC"}}]),
         reply("Бизнес-план курорта [Фрагмент 1]; аудит маркетинга [Фрагмент 2].")]


def test_followup_restricts_search_to_cards_of_previous_answer(base_config):
    """Воспроизводит диалог: «кто делал эти проекты?» раньше искал по всей базе и находил чужие КП."""
    agent, index = med_agent(base_config, FIRST + [
        reply("Не знаю, нужно поискать."),                       # модель не вызвала поиск -> forced_search
        reply("Иванов И. [Фрагмент 1], Петров П. [Фрагмент 2]."),
    ])
    agent.run("Какие проекты были в медицине?")
    assert [c["card_id"] for c in agent.memory.last_cards()] == ["11_001 AAA", "18_002 BBB"]
    r = agent.run("кто делал эти проекты?")
    last = index.calls[-1]
    assert r.route == "forced_search" and len(last["filters"]["cards"]) == 2
    assert last["top_k"] >= 4                                    # на каждую карточку хватает фрагментов
    assert all(s["file_name"].startswith("projects_accounting") for s in r.sources)
    assert "Иванов" in r.answer and "нефтян" not in r.answer


def test_short_followup_without_new_scope_also_restricts(base_config):
    agent, index = med_agent(base_config, FIRST + [
        reply(tool_calls=[{"name": "search_knowledge_base", "arguments": {"query": "по людям"}}]),
        reply("Иванов И. [Фрагмент 1]."),
    ])
    agent.run("Какие проекты были в медицине?")
    agent.run("А по людям?")
    assert len(index.calls[-1]["filters"]["cards"]) == 2


@pytest.mark.parametrize("question", ["А в энергетике?", "А пропозалы?", "А в 2020 году?"],
                         ids=["industry", "registry", "year"])
def test_followup_with_new_scope_is_a_new_search(base_config, question):
    agent, index = med_agent(base_config, FIRST + [
        reply(tool_calls=[{"name": "search_knowledge_base", "arguments": {"query": question}}]),
        reply("Энергоаудит [Фрагмент 3]."),
    ])
    agent.run("Какие проекты были в медицине?")
    agent.run(question)
    assert "cards" not in (index.calls[-1]["filters"] or {})


def test_previous_results_argument_from_model_is_dropped(base_config):
    """Уточнения по прошлому ответу решают правила агента; аргумент previous_results от модели не принимается."""
    agent, index = med_agent(base_config, FIRST + [
        reply(tool_calls=[{"name": "search_knowledge_base",
                           "arguments": {"query": "команда", "previous_results": True}}]),
        reply("Иванов И. [Фрагмент 3]."),
    ])
    agent.run("Какие проекты были в медицине?")
    r = agent.run("Расскажи про команду")                        # без отсылки к прошлому ответу
    assert "cards" not in (index.calls[-1]["filters"] or {})
    assert any("previous_results" in w for w in r.warnings)


def test_reset_forgets_cards(base_config):
    agent, _ = med_agent(base_config, FIRST)
    agent.run("Какие проекты были в медицине?")
    agent.memory.clear()
    assert agent.memory.last_cards() == []
    assert agent._previous_cards_for("кто делал эти проекты?") == []


def test_qdrant_filter_by_cards():
    qc = pytest.importorskip("qdrant_client")
    from qdrant_client.http import models
    from agent_tools import build_qdrant_filter
    client = qc.QdrantClient(":memory:")
    client.create_collection("m", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    points = [dict(c["metadata"]) for c in MED] + [{"card_type": "proposal", "card_id": "11_001 AAA",
                                                    "card_title": "чужой реестр, тот же ID"}]
    client.upsert("m", [models.PointStruct(id=i + 1, vector=[1.0, 0.1 * i], payload=p) for i, p in enumerate(points)])
    cards = [{"card_type": "project_accounting", "card_id": "11_001 AAA"},
             {"card_type": "project_accounting", "card_id": "18_002 BBB"}]
    res = client.query_points("m", query=[1.0, 0.0], limit=10,
                              query_filter=build_qdrant_filter(None, {"cards": cards})).points
    assert sorted(p.payload["card_title"] for p in res) == ["Аудит маркетинга", "Бизнес-план курорта"]
    client.close()


def test_checked_sheet_excluded_by_default(registry_csv, tmp_path, monkeypatch):
    """Лист checked исключён из базы: обычная сборка убирает его карточки, метаданные и записи metadata.yaml."""
    out = tmp_path / "cards"
    monkeypatch.setattr(sys, "argv", ["csv_to_cards.py", "--src", str(registry_csv), "--out", str(out),
                                      "--include-checked"])
    cards.main()
    assert list(out.glob("projects_accounting_*.md"))
    monkeypatch.setattr(sys, "argv", ["csv_to_cards.py", "--src", str(registry_csv), "--out", str(out)])
    cards.main()
    assert not list(out.glob("projects_accounting_*.md"))
    meta = [json.loads(l) for l in (out / cards.CARD_META_FILE).read_text(encoding="utf-8").splitlines()]
    assert meta and not any(m["card_type"] == "project_accounting" for m in meta)
    assert "projects_accounting" not in (out / "metadata.yaml").read_text(encoding="utf-8")
    assert any(m["card_type"] == "project" for m in meta) and any(m["card_type"] == "proposal" for m in meta)


def test_people_fields_first_and_no_duplicate_participants(tmp_path):
    src = tmp_path / "csv"; src.mkdir()
    write_csv(src / "projects.csv",
              ["ID Проекта", "Наименование проекта", "Реальный клиент (краткое наименование)", "Год проекта",
               "Компания, где реализован проект", "Команда проекта (из числа действующих сотрудников)",
               "Имя сотрудника", "Тип услуги"],
              [["1", "Стратегия", "Клиент", "2013", "Другая компания", "Иванов И.; Петров П.", "Сидоров С.", "Стратегия"],
               ["2", "Закупки", "Клиент", "2014", "", "Иванов И.", "", "Закупки"]])
    write_csv(src / "project_team.csv", ["ID проекта", "Сотрудник", "В штате"],
              [["1", "Иванов И.", "да"], ["1", "Петров П.", "да"], ["2", "Иванов И.", "да"], ["2", "Кузнецов К.", "да"]])
    out = tmp_path / "cards"
    cards.build_documents(src, out, "projects", anonymized=False)
    first = (out / "projects_2013.md").read_text(encoding="utf-8")
    labels = [l.split(":")[0] for l in first.splitlines()[1:]]
    assert labels.index("Команда (действующие сотрудники)") < labels.index("Компания, где выполнен проект")
    assert "Имя сотрудника: Сидоров С." in first
    assert "Участники:" not in first                         # полностью совпадает с «Командой»
    assert "Участники: Иванов И.; Кузнецов К." in (out / "projects_2014.md").read_text(encoding="utf-8")
    anon = tmp_path / "anon"
    cards.build_documents(src, anon, "projects", anonymized=True)
    assert "Сидоров" not in (anon / "projects_2013_anon.md").read_text(encoding="utf-8")


def test_this_refers_to_previous_cards(base_config):
    agent, index = med_agent(base_config, FIRST + [
        reply("Нужно поискать."), reply("Иванов И. [Фрагмент 1]."),
    ])
    agent.run("Какие проекты были в медицине?")
    agent.run("Какие сотрудники это делали")
    assert len(index.calls[-1]["filters"]["cards"]) == 2


def test_close_industry_value_is_resolved():
    tools = registry_tools()
    r, _ = tools.execute("search_knowledge_base", {"query": "стратегия", "industry": "металлургия"})
    assert r["status"] == "success" and r["filters"]["industry"] == ["Металлургия и горнодобывающая отрасль"]
    bad, _ = tools.execute("search_knowledge_base", {"query": "x", "industry": "медицина"})
    assert bad["error_code"] == "invalid_argument"            # синоним без общего корня — пусть выберет модель


# ------------------------------------------------------------------ отрасли после исключения листа checked

EN_ONLY = [
    {"text": "1. Стратегия фармкомпании — Клиент (2021)\nОтрасль: Pharma & HC\nКоманда (действующие сотрудники): Иванов И.",
     "metadata": {"file_name": "projects_2021.md", "category": "internal/projects", "card_type": "project",
                  "card_id": "7", "card_title": "Стратегия фармкомпании", "year": 2021, "year_end": 2021,
                  "industry": ["Pharma & HC"]}},
    {"text": "1. Логистика — Перевозчик (2019)\nОтрасль: Transportation",
     "metadata": {"file_name": "projects_2019.md", "category": "internal/projects", "card_type": "project",
                  "card_id": "8", "card_title": "Логистика", "year": 2019, "year_end": 2019,
                  "industry": ["Transportation"]}},
]


class EnOnlyIndex(MedIndex):
    def search(self, query, top_k, category=None, filters=None):
        self.calls.append({"query": query, "category": category, "filters": filters, "top_k": top_k})
        hits = [c for c in EN_ONLY if payload_matches(c["metadata"], category, filters)]
        return [{"rank": i + 1, "score": 0.9, "chunk_id": str(i), "document_id": "d",
                 "text": c["text"], "metadata": dict(c["metadata"])} for i, c in enumerate(hits[:top_k])]

    def facets(self):
        from agent_tools import FacetAccumulator
        acc = FacetAccumulator()
        for c in EN_ONLY:
            acc.add(c["metadata"])
        return acc.result()["facets"]


@pytest.mark.parametrize("value", ["Здравоохранение и фармацевтика", "медицина", "фармацевтика"],
                         ids=["full_ru", "medicine", "pharma"])
def test_russian_industry_resolves_to_registry_category(value):
    """Инструмент сопоставляет русскую отрасль с категорией индекса (им пользуется подагент «Реестры»)."""
    tools = ToolRegistry(lambda: EnOnlyIndex(), {"search": {"max_top_k": 10}})
    r, _ = tools.execute("search_knowledge_base", {"query": "проекты", "industry": value})
    assert r["status"] == "success" and r["filters"]["industry"] == ["Pharma & HC"]


def test_unknown_industry_from_model_is_dropped_before_tool(base_config):
    from agent import KBAgent
    index = EnOnlyIndex()
    llm = ScriptedLLM([reply(tool_calls=[{"name": "search_knowledge_base",
                                          "arguments": {"query": "проекты", "industry": "Космос"}}]),
                       reply("Стратегия фармкомпании [Фрагмент 1].")])
    r = KBAgent(base_config, llm=llm, retriever_factory=lambda: index).run("Какие проекты были в космической отрасли")
    assert r.status == "answered" and r.llm_calls == 2
    assert "industry" not in (index.calls[-1]["filters"] or {})
    assert any("аргументы вне схемы" in w for w in r.warnings)
    assert any(d.get("point") == "drop_extra_arguments" for d in r.trace.decisions())


def test_industry_hints_in_tool_schema():
    props = build_tool_schemas(EnOnlyIndex().facets())[0]["function"]["parameters"]["properties"]
    assert "Pharma & HC (медицина" in props["industry"]["description"]


# ------------------------------------------------------------------ каталог карточек и программная маршрутизация

CATALOG = [
    {"file_name": "projects_2013.md", "card_number": 1, "card_type": "project", "card_id": "1", "card_title": "Курорты",
     "year": 2013, "year_end": 2013, "industry": ["Pharma & HC", "Construction & Real estate"], "client": "Клиент А",
     "team": ["Иванов И."], "company": "Компания Х"},
    {"file_name": "projects_2021.md", "card_number": 1, "card_type": "project", "card_id": "2", "card_title": "Фармстратегия",
     "year": 2021, "year_end": 2021, "industry": ["Pharma & HC"], "team": ["Петров П.", "Сидоров С."], "employee": "Козлов К."},
    {"file_name": "proposals_2019.md", "card_number": 1, "card_type": "proposal", "card_id": "1", "card_title": "Клиники",
     "year": 2019, "year_end": 2019, "industry": ["Pharma & HC"]},
    {"file_name": "projects_2020.md", "card_number": 1, "card_type": "project", "card_id": "3", "card_title": "Банк",
     "year": 2020, "year_end": 2020, "industry": ["Financial services"], "team": ["Орлов О."]},
]


@pytest.fixture
def catalog_config(base_config, tmp_path):
    path = tmp_path / "cards_meta.jsonl"
    path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in CATALOG), encoding="utf-8")
    return {**base_config, "catalog_path": str(path)}


def catalog_agent(config, replies=()):
    from agent import KBAgent
    llm = ScriptedLLM(list(replies))
    return KBAgent(config, llm=llm, retriever_factory=lambda: EnOnlyIndex()), llm


def test_enumeration_lists_all_cards_without_llm(catalog_config):
    """Воспроизводит жалобу: «какие проекты были в медицине» давало 5 из 18 — top-k поиска."""
    agent, llm = catalog_agent(catalog_config)
    r = agent.run("Какие проекты были в медицине")
    assert r.route == "catalog_list" and r.llm_calls == 0 and llm.calls == []
    assert r.status == "answered" and len(r.sources) == 3          # все карточки Pharma & HC, обоих типов
    assert "Проекты (2)" in r.answer and "Пропозалы (1)" in r.answer and "Банк" not in r.answer
    assert r.answer.index("Фармстратегия") < r.answer.index("Курорты")   # новые сверху


def test_people_followup_lists_people_of_all_previous_cards(catalog_config):
    agent, llm = catalog_agent(catalog_config)
    agent.run("Какие проекты были в медицине")
    r = agent.run("кто выполнял эти проекты?")
    assert r.route == "catalog_people" and r.llm_calls == 0
    assert "Петров П.; Сидоров С." in r.answer and "имя сотрудника: Козлов К." in r.answer
    assert "люди в реестре не указаны" in r.answer                  # у пропозала людей нет — так и сказано
    assert "Орлов" not in r.answer


def test_enumeration_with_period_and_type(catalog_config):
    agent, _ = catalog_agent(catalog_config)
    r = agent.run("Перечисли пропозалы за 2018–2020")
    assert r.route == "catalog_list" and [s["card"]["card_title"] for s in r.sources] == ["Клиники"]
    r2 = agent.run("Сколько проектов было в банках")
    assert r2.route == "catalog_list" and [s["card"]["card_title"] for s in r2.sources] == ["Банк"]


def test_enumeration_with_topic_goes_to_search(catalog_config):
    agent, llm = catalog_agent(catalog_config, [
        reply(tool_calls=[{"name": "search_knowledge_base", "arguments": {"query": "закупки медицина"}}]),
        reply("Стратегия фармкомпании [Фрагмент 1]."),
    ])
    r = agent.run("Какие проекты по закупкам были в медицине")
    assert r.route == "tools" and llm.calls
    decided = next(d for d in r.trace.decisions() if d.get("point") == "route_decided")
    assert decided["route"] == "documents" and "закупкам" in decided["residual"]


def test_empty_enumeration_is_not_found(catalog_config):
    agent, _ = catalog_agent(catalog_config)
    r = agent.run("Какие проекты были в медицине до 2010")
    assert r.status == "not_found" and "карточек в реестрах нет" in r.answer


def test_documents_subagent_cannot_call_list_cards(catalog_config):
    """list_cards — инструмент подагента «Реестры»; модели подагента «Документы» он не доступен."""
    agent, llm = catalog_agent(catalog_config, [
        reply(tool_calls=[{"name": "list_cards", "arguments": {"industry": "Pharma & HC", "card_type": "project"}}]),
        reply(tool_calls=[{"name": "search_knowledge_base", "arguments": {"query": "фарма"}}]),
        reply("Стратегия фармкомпании [Фрагмент 1]."),
    ])
    r = agent.run("Расскажи о наших материалах по фармацевтике")
    assert r.tool_calls[0].name == "list_cards" and r.tool_calls[0].error_code == "unknown_tool"
    assert r.tool_calls[1].status == "success" and r.status == "answered"
    assert "ОШИБКА [unknown_tool]" in llm.calls[1]["messages"][-1]["content"]


@pytest.mark.parametrize("args, fragment", [
    ({"card_type": "договор"}, "card_type"), ({"year_from": 2025, "year_to": 2020}, "больше"),
    ({"industry": "Космос"}, "неизвестная отрасль"),
], ids=["type", "years", "industry"])
def test_list_cards_validation(catalog_config, args, fragment):
    agent, _ = catalog_agent(catalog_config)
    r, _ = agent.tools.execute("list_cards", args)
    assert r["error_code"] == "invalid_argument" and fragment in r["error"]


def test_industry_detection_in_question():
    from agent_tools import detect_industries
    known = ["Pharma & HC", "Telecom & Media & IT", "Financial services", "Energy (O&G / chem / utilities)"]
    assert detect_industries("Какие проекты были в медицине", known) == ["Pharma & HC"]
    assert detect_industries("Сколько проектов было в банках", known) == ["Financial services"]
    assert detect_industries("проекты для газовых компаний", known) == ["Energy (O&G / chem / utilities)"]
    assert detect_industries("проекты по ИТ-стратегии", known) == []


def test_card_meta_contains_people(registry_csv, tmp_path):
    _, meta, _ = cards.build_documents(registry_csv, tmp_path / "c", "projects", anonymized=False)
    first = next(m for m in meta if m["card_id"] == "1")
    assert first["team"] == ["Иванов И."]
    _, anon, _ = cards.build_documents(registry_csv, tmp_path / "a", "projects", anonymized=True)
    assert "team" not in next(m for m in anon if m["card_id"] == "1")


# ------------------------------------------------------------------ подсчёт людей, типы и теги услуг

PEOPLE_CATALOG = [
    {"file_name": "projects_2019.md", "card_number": 1, "card_type": "project", "card_id": "1", "card_title": "ОЭ завода",
     "year": 2019, "year_end": 2019, "industry": ["Industrial goods"], "service": ["Операционная эффективность"],
     "service_detail": ["Закупки / управление поставщиками"], "team": ["Иванов И.", "Петров П."], "partner": "Иванов И."},
    {"file_name": "projects_2020.md", "card_number": 1, "card_type": "project", "card_id": "2", "card_title": "ОЭ банка",
     "year": 2020, "year_end": 2020, "industry": ["Financial services"], "service": ["Операционная эффективность"],
     "team": ["Иванов И."], "employee": "Иванов"},                     # фамилия без инициалов — тот же человек
    {"file_name": "proposals_2021.md", "card_number": 1, "card_type": "proposal", "card_id": "1", "card_title": "ОЭ сети",
     "year": 2021, "year_end": 2021, "industry": ["Consumer goods & retail"], "service": ["Операционная эффективность"],
     "team": ["Петров П.", "Сидоров С."]},
    {"file_name": "projects_2018.md", "card_number": 1, "card_type": "project", "card_id": "3", "card_title": "Цифровизация",
     "year": 2018, "year_end": 2018, "industry": ["Transportation"], "service": ["Digital"], "team": ["Сидоров С."]},
]


@pytest.fixture
def people_config(base_config, tmp_path):
    path = tmp_path / "cards_meta.jsonl"
    path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in PEOPLE_CATALOG), encoding="utf-8")
    return {**base_config, "catalog_path": str(path)}


def test_people_ranking_for_service(people_config):
    """Воспроизводит вопрос: «кто делал много проектов в операционной эффективности» — подсчёт, без LLM."""
    agent, llm = catalog_agent(people_config)
    r = agent.run("кто делал много проектов в операционной эффективности")
    assert r.route == "catalog_people_rank" and r.llm_calls == 0 and llm.calls == []
    lines = [l for l in r.answer.splitlines() if l[:2] in ("1.", "2.", "3.")]
    assert lines[0].startswith("1. Иванов И. — проектов 2, пропозалов 0")    # «Иванов» объединён с «Иванов И.»
    assert lines[1].startswith("2. Петров П. — проектов 1, пропозалов 1")
    assert "Карточек: 3 (проектов 2, пропозалов 1)" in r.answer
    assert "Сидоров С. — проектов 0, пропозалов 1" in r.answer               # Digital-проект не учтён


def test_people_ranking_by_detail_tag_and_person_filter(people_config):
    agent, _ = catalog_agent(people_config)
    r = agent.run("кто делал много проектов по закупкам")
    assert r.route == "catalog_people_rank" and "Закупки / управление поставщиками" in r.answer
    assert "Карточек: 1" in r.answer
    tools = agent.tools
    res, _ = tools.execute("list_cards", {"person": "Петрова"})               # падеж фамилии
    assert res["filters"]["person"] == ["Петров П."] and res["total"] == 2


def test_people_names_keep_initial_dot():
    from agent_cards import split_people
    assert split_people("Петров Б.; Иванов И.И.") == ["Петров Б.", "Иванов И.И."]
    assert split_people("Иванов И. Петров П.") == ["Иванов И.", "Петров П."]


def test_service_and_tag_detection():
    from agent_tools import detect_services
    types = ["Операционная эффективность", "Digital", "Стратегия"]
    tags = ["Закупки / управление поставщиками", "КПЭ / Управление эффективностью",
            "Управление проектами/программами / Проектный офис (PMO)", "Логистика / УЦП"]
    assert detect_services("кто делал много проектов в операционной эффективности", types, tags) == \
        ["Операционная эффективность"]                     # без тегов, которые объясняются тем же словом
    assert detect_services("кто делал много проектов по закупкам", types, tags) == ["Закупки / управление поставщиками"]
    assert detect_services("проекты по цифровой трансформации", types, tags) == ["Digital"]
    assert detect_services("Какие проекты были в медицине", types, tags) == []


def test_multiword_synonyms_and_long_keys():
    from agent_tools import _matches_synonyms, detect_industries
    assert not _matches_synonyms("на транспорте", ["цифровая трансформация"])   # нужны все слова фразы
    assert detect_industries("по цифровой трансформации", ["Transportation"]) == []   # «трансп» ≠ «трансф»
    assert detect_industries("транспортных компаний", ["Transportation"]) == ["Transportation"]
    assert detect_industries("таким образом", ["Public"]) == []


def test_logistics_is_service_not_industry(people_config):
    agent, _ = catalog_agent(people_config)
    catalog = agent.tools.catalog()
    catalog.cards.append({"card_type": "project", "card_id": "9", "card_title": "Склады", "year": 2022, "year_end": 2022,
                          "industry": ["Consumer goods & retail"], "service_detail": ["Логистика / УЦП"],
                          "_people": {"Орлов О.": ["team"]}})
    r = agent.run("кто чаще всего делал проекты по логистике")
    assert r.route == "catalog_people_rank" and "Орлов О." in r.answer and "отрасль" not in r.answer.splitlines()[0]


def test_people_ranking_by_tool_and_by_router(people_config):
    agent, llm = catalog_agent(people_config)
    res, _ = agent.tools.execute("list_cards", {"service": "операционная эффективность", "group_by": "people"})
    assert "Иванов И. — проектов 2" in res["text"]
    r = agent.run("Кто чаще всего делал проекты в операционной эффективности?")
    assert r.route == "catalog_people_rank" and r.llm_calls == 0 and "Иванов И. — проектов 2" in r.answer


def test_latin_spelling_merged_with_cyrillic(tmp_path):
    from agent_cards import CardCatalog, latin_key
    assert latin_key("Сидоренко") == latin_key("Sidorenko")
    path = tmp_path / "m.jsonl"
    rows = [{"card_type": "project", "card_id": "1", "card_title": "А", "team": ["Сидоренко А."]},
            {"card_type": "project", "card_id": "2", "card_title": "Б", "employee": "Sidorenko Alexey"},
            {"card_type": "project", "card_id": "3", "card_title": "В", "employee": "Sidorenko Boris"}]   # другой инициал
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    cat = CardCatalog(str(path))
    ranking = {r["person"]: r["projects"] for r in cat.aggregate_people(cat.cards)}
    assert ranking["Сидоренко А."] == 2 and ranking["Sidorenko Boris"] == 1


@pytest.mark.parametrize("text, expected", [
    ("за последние 5 лет", (2022, 2026)), ("за последние пять лет", (2022, 2026)),
    ("за последние 3 года", (2024, 2026)), ("в прошлом году", (2025, 2025)),
    ("в этом году", (2026, 2026)), ("за 2020–2023", (2020, 2023)), ("до 2015", (None, 2014)),
], ids=["5y", "5y_words", "3y", "last_year", "this_year", "range", "before"])
def test_relative_periods(text, expected):
    from agent_tools import detect_period
    assert detect_period(text, current_year=2026) == expected


def test_person_question_routes_to_catalog(people_config):
    agent, llm = catalog_agent(people_config)
    r = agent.run("Проекты Иванова за 2019–2020")
    assert r.route == "catalog_list" and r.llm_calls == 0 and llm.calls == []
    assert "сотрудник: Иванов И." in r.answer and len(r.sources) == 2
    r2 = agent.run("Что делал Сидоров в 2018 году")
    assert r2.route == "catalog_list" and [s["card"]["card_title"] for s in r2.sources] == ["Цифровизация"]
