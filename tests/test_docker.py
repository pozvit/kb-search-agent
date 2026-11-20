"""
Docker-образ (шаг 3.5, требования F10, S8): статическая проверка Dockerfile, .dockerignore и стартового
скрипта. Сборка образа проверяется вручную (docker build), здесь — то, что можно проверить без Docker:
полная база не попадает в образ, сервис работает не от root, есть healthcheck, скрипт запустится в Linux.
"""

import fnmatch
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
ALLOWED_COPY = {"config", "src", "data/raw", "data/raw_registry_public", "docker/entrypoint.sh",
                "requirements-service.txt"}


def dockerignore_patterns():
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def in_build_context(path: str) -> bool:
    """Семантика .dockerignore: правила по порядку, последнее совпавшее решает, «!» — исключение из исключения."""
    parts = path.split("/")
    candidates = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
    included = True
    for pattern in dockerignore_patterns():
        negate = pattern.startswith("!")
        pat = pattern[1:] if negate else pattern
        pat = pat[3:] if pat.startswith("**/") else pat
        if any(fnmatch.fnmatch(c, pat) or fnmatch.fnmatch(c.split("/")[-1], pat) for c in candidates):
            included = negate
    return included


def test_copy_only_public_data_and_code():
    sources = set(re.findall(r"(?m)^COPY(?:\s+--\S+)*\s+(\S+)\s+\S+$", DOCKERFILE))
    assert sources <= ALLOWED_COPY, sources - ALLOWED_COPY
    assert "COPY . " not in DOCKERFILE


@pytest.mark.parametrize("path", [
    ".env", ".git/config", ".venv/lib/x.py", "data/raw_cards/projects_2019.md", "data/raw_csv/projects.csv",
    "data/raw_xlsb/база.xlsb", "data/raw_office_pdf/kp.pdf", "data/source_md_office/kp.md",
    "data/qdrant_storage/collection/x", "data/chunks/chunks.jsonl", "data/escalations/escalations.jsonl",
    "data/agent/traces/t.json", "reports/control/week2.json", "src/__pycache__/agent.cpython-312.pyc", "run.log",
])
def test_sensitive_files_not_in_build_context(path):
    """S8: реестры, КП, индексы полной базы, очередь обращений, trace и .env не уходят в контекст сборки."""
    assert not in_build_context(path)


@pytest.mark.parametrize("path", [
    "src/agent.py", "src/service/api.py", "config/agent.yaml", "config/public/rag.yaml",
    "data/raw/01_ergo_business_simulator.pdf", "data/raw_registry_public/cards_meta.jsonl",
    "docker/entrypoint.sh", "requirements-service.txt",
])
def test_needed_files_in_build_context(path):
    assert in_build_context(path)


def test_runs_as_non_root_with_healthcheck_and_port():
    user_at = DOCKERFILE.index("\nUSER app")
    assert user_at < DOCKERFILE.index("ENTRYPOINT")
    assert "HEALTHCHECK" in DOCKERFILE and "/health" in DOCKERFILE and "EXPOSE 8000" in DOCKERFILE


def test_torch_from_cpu_index():
    assert "download.pytorch.org/whl/cpu" in DOCKERFILE


def test_service_requirements_are_subset_without_dev_tools():
    def names(path):
        return {re.split(r"[<>=\s#]", line.strip())[0].lower() for line in (ROOT / path).read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")}
    service = names("requirements-service.txt")
    assert service <= names("requirements.txt")
    assert not service & {"pytest", "httpx", "reportlab", "pyxlsb", "python-docx", "python-pptx"}


def test_entrypoint_is_posix_sh_with_lf_and_starts_api():
    raw = (ROOT / "docker" / "entrypoint.sh").read_bytes()
    assert b"\r\n" not in raw, "CRLF в скрипте: в контейнере Linux он не запустится"
    text = raw.decode("utf-8")
    assert text.startswith("#!/bin/sh") and "exec python src/serve.py" in text and "0.0.0.0" in text


def test_gitattributes_keeps_lf_for_scripts():
    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert re.search(r"(?m)^\*\.sh\s+.*eol=lf", attrs) and re.search(r"(?m)^Dockerfile\s+.*eol=lf", attrs)


@pytest.mark.skipif(shutil.which("sh") is None, reason="sh недоступен")
def test_entrypoint_syntax():
    assert subprocess.run(["sh", "-n", str(ROOT / "docker" / "entrypoint.sh")]).returncode == 0
