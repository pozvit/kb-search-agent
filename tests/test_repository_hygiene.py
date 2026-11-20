"""
Гигиена репозитория (шаг 3.3, требование S1): нет секретов, ФИО вне синтетики, путей с именем профиля,
внутренних кодов и контактов — ни в файлах, ни в истории git. Чувствительные данные в .gitignore.

Образцы для проверки детекторов собираются из частей во время выполнения: этот файл тоже проверяется,
и целиком таких строк в нём быть не должно.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import hygiene as H  # noqa: E402

SENSITIVE_PATHS = [
    ".env", "data/raw_xlsb/x.xlsb", "data/raw_csv/x.csv", "data/raw_cards/x.md", "data/raw_office_pdf/x.pdf",
    "data/source_md_office/x.md", "data/qdrant_storage/x", "data/public/x", "data/chunks/x.jsonl",
    "data/agent/traces/x.json", "data/escalations/escalations.jsonl", "reports/control/run.log",
]


def test_working_tree_has_no_secrets_or_personal_data():
    found = H.scan_files(H.project_files(ROOT), ROOT)
    assert not found, "\n" + "\n".join(map(str, found))


@pytest.mark.skipif(not H.is_git_repo(ROOT), reason="не git-репозиторий (архив проекта)")
def test_git_history_has_no_secrets_or_personal_data():
    found = H.scan_history(ROOT)
    assert not found, "\n" + "\n".join(map(str, found))


def test_sensitive_paths_are_ignored():
    if H.is_git_repo(ROOT):
        not_ignored = [p for p in SENSITIVE_PATHS
                       if subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", p]).returncode != 0]
    else:
        prefixes = H.gitignore_prefixes(ROOT)
        not_ignored = [p for p in SENSITIVE_PATHS
                       if not (any(p == x or p.startswith(x + "/") for x in prefixes) or p.endswith(".log"))]
    assert not not_ignored, not_ignored


def test_env_file_is_not_in_project():
    assert not any(p.name == ".env" for p in H.project_files(ROOT))


# ---------------------------------------------------------------- детекторы на синтетических образцах

KEY = "sk-" + "proj-" + "A1b2C3d4" * 3
PEM = "-----BEGIN " + "PRIVATE KEY-----"
PERSON = "Смирн" + "ов А."
PERSON_FULL = "Кротов" + "а Е. В."
PATH = "C:" + "\\Users\\" + "petrov.s\\project"
CODE = "P" + "12345678"
ACC_CODE = "18_" + "123 XYZ"
MAIL = "ivan.petrov" + "@" + "company.ru"
PHONE = "+7 " + "(912) 345-67-89"


@pytest.mark.parametrize("text, kind", [
    (f"api_key = '{KEY}'", "secret"), (PEM, "secret"), (f"Команда: {PERSON}", "person"),
    (f"Руководитель {PERSON_FULL}", "person"), (f"путь {PATH}", "profile_path"), (f"код {CODE}", "internal_code"),
    (f"проект {ACC_CODE}", "internal_code"), (f"почта {MAIL}", "contact"), (f"тел. {PHONE}", "contact"),
], ids=["key", "pem", "person", "person_full", "profile_path", "code", "acc_code", "email", "phone"])
def test_detectors_find_samples(text, kind):
    found = H.scan_text(text, "x", set())
    assert [f.kind for f in found] == [kind]
    assert all(KEY not in f.sample and PERSON not in f.sample for f in found)     # вывод замаскирован


@pytest.mark.parametrize("text", [
    "Команда: Белозёрова Б.", "Команда: Иванов И.", "ключ sk-proj-TESTONLY" + "0" * 20,
    "путь C:" + "\\Users\\~\\kb_baseline", "код P00000000", "проект 21_001 ABC", "почта user@example.com",
    "Год: 2019. Тип услуги: Стратегия",
], ids=["synthetic", "placeholder", "fake_key", "masked_path", "zero_code", "synthetic_code", "example_mail", "plain"])
def test_synthetic_and_placeholders_are_allowed(text):
    assert H.scan_text(text, "x", {"Белозёрова Б."}) == []


def test_allowlist_by_fingerprint():
    value = "Смирн" + "ов А."
    assert H.scan_text(value, "x", set(), allowed={H.fingerprint(value)}) == []


@pytest.mark.skipif(shutil.which("git") is None, reason="git не установлен")
def test_history_scan_finds_data_removed_from_current_files(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (tmp_path / "notes.md").write_text(f"Ответственный: {PERSON}\n", encoding="utf-8")
    git("add", "notes.md")
    git("commit", "-q", "-m", "first")
    (tmp_path / "notes.md").write_text("Ответственный: роль из конфигурации\n", encoding="utf-8")
    git("commit", "-q", "-am", "second")
    assert H.scan_files(H.project_files(tmp_path), tmp_path) == []          # сейчас файлы чисты
    history = H.scan_history(tmp_path)
    assert [f.kind for f in history] == ["person"] and history[0].where.endswith("notes.md")


# ---------------------------------------------------------------- фамилии из полной базы (known_person)

def test_known_person_found_in_any_case_without_initials(tmp_path):
    """Фамилии из реестра ловятся в падеже и без инициалов — это пропускала проверка «Фамилия И.»."""
    import json
    cards = tmp_path / "cards_meta.jsonl"
    surname = "Тимоф" + "еева"
    cards.write_text(json.dumps({"team": [f"{surname} Т."], "partner": "Ангарский А."}, ensure_ascii=False),
                     encoding="utf-8")
    assert H.build_denylist([cards], root=tmp_path) == 1          # синтетика (Ангарский) не попадает
    deny = H.load_denylist(tmp_path)
    assert surname not in (tmp_path / H.DENYLIST_FILE).read_text(encoding="utf-8")   # хранятся отпечатки
    for text in (f"Проекты {surname[:-1]}ой за 2025 год", f"работал с {surname[:-1]}ой", f"Т. {surname}",
                 f"{surname} Татьяна"):
        found = H.scan_text(text, "x", set(), denylist=deny)
        assert [f.kind for f in found] == ["known_person"], text
    assert H.scan_text("Тимур и Тимофей работали над проектом", "x", set(), denylist=deny) == []


def test_without_denylist_known_person_is_skipped(tmp_path):
    assert H.load_denylist(tmp_path) == set()


def test_denylist_file_is_gitignored():
    assert ".hygiene_denylist" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_denylist_takes_only_strict_person_records(tmp_path):
    """В список попадает только «Фамилия И.»: заголовки и текст CSV («Data Science», «Опыт Работы») — нет."""
    csv = tmp_path / "project_team.csv"
    surname = "Тимоф" + "еева"
    csv.write_text("ID;Опыт Работы;Data Science;System Group\n"
                   f"P1;Процесс Перевозки грузов;{surname} Т.;Логистика Склада\n"
                   "P2;Строительство Объектов;Data Engineering;Опыт Продаж\n", encoding="utf-8")
    assert H.build_denylist([csv], root=tmp_path) == 1
    deny = H.load_denylist(tmp_path)
    text = "Опыт работы с системой, data science, логистика склада, процесс перевозки, System Group"
    assert H.scan_text(text, "x", set(), denylist=deny) == []

