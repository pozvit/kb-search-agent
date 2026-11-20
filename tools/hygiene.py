"""
Гигиена репозитория (требование S1): в коммитах не должно быть секретов и персональных данных.

Ищет:
  secret        ключи API и токены (sk-…, AQVN…, t1.…, ghp_…, AKIA…), приватные ключи PEM,
                пароли и строки подключения с учётными данными;
  person        «Фамилия И.» и «Фамилия И. О.» вне списка синтетических имён;
  profile_path  пути с именем профиля пользователя (C:\\Users\\<имя>, /home/<имя>) — в Windows
                имя профиля часто совпадает с ФИО;
  internal_code внутренние коды проектов (P + 8 цифр, коды учётной базы «18_123 ABC»);
  contact       адреса электронной почты и телефоны.

Синтетика разрешена: сотрудники синтетического реестра (config/control_scenarios.yaml, synthetic_people),
тестовые заглушки (ALLOWED_NAMES), заведомо поддельные ключи с TESTONLY или EXAMPLE. Остальные ложные
срабатывания (поддельные секреты во входных данных тестов, примеры форматов в комментариях) разрешаются
после ручной проверки в config/hygiene_allowlist.yaml — отпечатком значения, а не самим значением,
поэтому разрешение действует и для истории git.

Запуск:
    python tools/hygiene.py             # отслеживаемые git файлы (без git — файлы проекта вне .gitignore)
    python tools/hygiene.py --history   # плюс добавленные строки во всех коммитах истории
Код возврата 1, если есть находки. Находки выводятся замаскированными.
"""

import argparse
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set

import yaml

ROOT = Path(__file__).resolve().parents[1]

# тестовые заглушки и шаблоны документации — не реальные люди
ALLOWED_NAMES = {"Иванов И.", "Петров П.", "Петров Б.", "Петрова Н.", "Сидоров С.", "Сидоренко А.", "Орлов О.",
                 "Кузнецов К.", "Козлов К.", "Фамилия И."}
ALLOWED_CODE_LETTERS = {"ABC", "AAA", "BBB", "ADL"}       # синтетические коды в тестах и примере README
ALLOWED_EMAIL_DOMAINS = {"example.com", "example.org", "users.noreply.github.com"}

SECRET = re.compile(
    r"(sk-(?:proj-)?[A-Za-z0-9_-]{20,}|AQVN[A-Za-z0-9_-]{20,}|t1\.[A-Za-z0-9._-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"(?i:(?:password|passwd|пароль)\s*[:=]\s*['\"][^'\"\s]{6,}['\"])|"
    r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^:\s/]+:[^@\s]+@)")
PERSON = re.compile(r"(?<![А-Яа-яЁё])([А-ЯЁ][а-яё]{2,}(?:-[А-ЯЁ][а-яё]+)?)\s+([А-ЯЁ])\.(?:\s?([А-ЯЁ])\.)?")
PROFILE_PATH = re.compile(r"(?i)(?:[a-z]:)?[\\/]+(?:users|home)[\\/]+(?!~|public|default|runner|claude|ubuntu|user[\\/])"
                          r"([a-z][\w.\-]{2,})")
INTERNAL_CODE = re.compile(r"(?<![\w])(P\d{8}|\d{2}_\d{3}\s?([A-Z]{3}))(?![\w])")
EMAIL = re.compile(r"[\w.+-]+@([\w-]+\.)+[a-z]{2,}", re.I)
PHONE = re.compile(r"(?<![\d\w])(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?![\d])")

# синтетические пути в тестах safe_path, примеры в документации, служебный пользователь контейнера (Dockerfile)
ALLOWED_PROFILE_NAMES = {"ivan.petrov", "ivan", "имя", "user", "username", "you", "app"}


@dataclass
class Finding:
    kind: str
    where: str
    sample: str
    fingerprint: str = ""

    def __str__(self) -> str:
        return f"{self.kind:13} {self.where}  {self.sample}  [{self.fingerprint}]"


def fingerprint(value: str) -> str:
    """Отпечаток найденного значения: по нему разрешают ложные срабатывания, не храня само значение."""
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()[:16]


def allowed_fingerprints(root: Path = ROOT) -> Set[str]:
    path = root / "config" / "hygiene_allowlist.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return set()
    return {str(item["fingerprint"]) for item in data.get("allowed", []) if item.get("fingerprint")}


def mask(text: str) -> str:
    """Первые символы видны, остальное скрыто: проверка не должна печатать то, что ищет."""
    text = text.strip()
    return text[:3] + "*" * max(0, min(len(text) - 3, 12))


def synthetic_people(root: Path = ROOT) -> Set[str]:
    path = root / "config" / "control_scenarios.yaml"
    try:
        return set(yaml.safe_load(path.read_text(encoding="utf-8")).get("synthetic_people", []))
    except (OSError, AttributeError, yaml.YAMLError):
        return set()


def scan_text(text: str, where: str, allowed_people: Set[str], allowed: Optional[Set[str]] = None) -> List[Finding]:
    found = _scan_text(text, where, allowed_people)
    return [f for f in found if f.fingerprint not in (allowed or set())]


def _scan_text(text: str, where: str, allowed_people: Set[str]) -> List[Finding]:
    found: List[Finding] = []

    def hit(kind: str, loc: str, value: str) -> None:
        found.append(Finding(kind, loc, mask(value), fingerprint(value)))

    for lineno, line in enumerate(text.splitlines(), start=1):
        loc = f"{where}:{lineno}"
        for m in SECRET.finditer(line):
            if "TESTONLY" in m.group(0) or "EXAMPLE" in m.group(0).upper():
                continue
            hit("secret", loc, m.group(0))
        for m in PERSON.finditer(line):
            name = f"{m.group(1)} {m.group(2)}."
            if name not in allowed_people and name not in ALLOWED_NAMES:
                hit("person", loc, name)
        for m in PROFILE_PATH.finditer(line):
            if m.group(1).lower() not in ALLOWED_PROFILE_NAMES:
                hit("profile_path", loc, m.group(0))
        for m in INTERNAL_CODE.finditer(line):
            if m.group(1) == "P00000000" or (m.group(2) and m.group(2) in ALLOWED_CODE_LETTERS):
                continue
            hit("internal_code", loc, m.group(1))
        for m in EMAIL.finditer(line):
            domain = m.group(0).split("@")[-1].lower()
            if domain not in ALLOWED_EMAIL_DOMAINS and not domain.endswith(".example"):
                hit("contact", loc, m.group(0))
        for m in PHONE.finditer(line):
            hit("contact", loc, m.group(0))
    return found


def is_git_repo(root: Path = ROOT) -> bool:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"], capture_output=True,
                              text=True, timeout=10).stdout.strip() == "true"
    except (OSError, subprocess.SubprocessError):
        return False


def gitignore_prefixes(root: Path = ROOT) -> List[str]:
    try:
        lines = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [l.strip().rstrip("/") for l in lines if l.strip() and not l.startswith("#") and "*" not in l]


def project_files(root: Path = ROOT) -> List[Path]:
    """Отслеживаемые git файлы; без git — файлы проекта вне .gitignore."""
    if is_git_repo(root):
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, timeout=30).stdout
        return [root / p for p in out.decode("utf-8").split("\0") if p]
    prefixes = gitignore_prefixes(root) + [".venv", ".git", ".pytest_cache"]
    files = []
    for p in root.rglob("*"):
        rel = p.relative_to(root).as_posix()
        if p.is_file() and "__pycache__" not in p.parts and not p.suffix == ".pyc" \
                and not any(rel == x or rel.startswith(x + "/") or p.name == x for x in prefixes) \
                and not p.name.endswith((".log", ".env")) and p.name != ".env":
            files.append(p)
    return files


BINARY = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".zip", ".xlsx", ".xlsb", ".docx", ".pptx", ".pyc"}


def scan_files(files: Iterable[Path], root: Path = ROOT) -> List[Finding]:
    people, allowed = synthetic_people(root), allowed_fingerprints(root)
    found: List[Finding] = []
    for p in files:
        if p.suffix.lower() in BINARY or not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        found += scan_text(text, p.relative_to(root).as_posix(), people, allowed)
    return found


def scan_history(root: Path = ROOT) -> List[Finding]:
    """Добавленные строки во всех коммитах всех веток и тегов (git log -p)."""
    people, allowed = synthetic_people(root), allowed_fingerprints(root)
    log = subprocess.run(["git", "-C", str(root), "log", "--all", "-p", "--no-color", "--format=COMMIT %h",
                          "--", ".", ":(exclude)*.pdf"], capture_output=True, timeout=300).stdout.decode("utf-8", "ignore")
    found: List[Finding] = []
    commit, path = "?", "?"
    for line in log.splitlines():
        if line.startswith("COMMIT "):
            commit = line.split()[1]
        elif line.startswith("+++ b/"):
            path = line[6:]
        elif line.startswith("+") and not line.startswith("+++"):
            for f in scan_text(line[1:], f"{commit}:{path}", people, allowed):
                f.where = f.where.rsplit(":", 1)[0]
                found.append(f)
    # одна и та же строка в разных коммитах — одна находка
    unique, seen = [], set()
    for f in found:
        key = (f.kind, f.where.split(":", 1)[-1], f.sample)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Проверка репозитория на секреты и персональные данные (S1)")
    parser.add_argument("--history", action="store_true", help="проверить и всю историю git")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and not stream.isatty():
            stream.reconfigure(encoding="utf-8", errors="replace")
    files = project_files()
    found = scan_files(files)
    print(f"Файлы: проверено {len(files)}, находок {len(found)}")
    for f in found:
        print("  " + str(f))
    if args.history:
        if not is_git_repo():
            print("История: не git-репозиторий, пропущено")
        else:
            hist = scan_history()
            print(f"История git: находок {len(hist)}")
            for f in hist:
                print("  " + str(f))
            found += hist
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
