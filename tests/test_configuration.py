"""
Конфигурация и переменные окружения (шаг 3.4, требование F9): изменяемые параметры — в config/agent.yaml
и переменных окружения, шаблон .env.example один и описывает каждую переменную, которую читает код.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ENV_READ = re.compile(r"(?:os\.environ\.get|os\.getenv|os\.environ\.setdefault)\(\s*[\"']([A-Z][A-Z0-9_]+)[\"']|"
                      r"os\.environ\[\s*[\"']([A-Z][A-Z0-9_]+)[\"']\s*\]|[\"']((?:AGENT|RAG|LOG|ESCALATION)_[A-Z0-9_]+)[\"']")
SECRET_NAME = re.compile(r"(KEY|TOKEN|PASSWORD|SECRET)$")


def env_vars_in_code():
    found = set()
    for folder in ("src", "tools"):
        for path in (ROOT / folder).rglob("*.py"):
            for m in ENV_READ.finditer(path.read_text(encoding="utf-8")):
                found.add(next(g for g in m.groups() if g))
    return found


def env_template():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    return {m.group(1): m.group(2).split("#")[0].strip()
            for m in re.finditer(r"(?m)^#?\s*([A-Z][A-Z0-9_]+)=(.*)$", text)}


def test_single_env_template():
    """Дефект baseline №4: актуальный шаблон лежал в env.example без точки рядом со старым .env.example."""
    assert (ROOT / ".env.example").exists()
    assert not (ROOT / "env.example").exists()
    others = [p.name for p in list(ROOT.glob("*env.example*")) + list((ROOT / "config").glob("*env*"))
              if p.name != ".env.example"]
    assert not others, others


def test_every_env_var_read_by_code_is_documented():
    missing = sorted(env_vars_in_code() - set(env_template()))
    assert not missing, f"не описаны в .env.example: {missing}"


def test_template_has_no_secret_values():
    filled = {k: v for k, v in env_template().items() if SECRET_NAME.search(k) and v}
    assert not filled, f"у секретных переменных в шаблоне есть значения: {sorted(filled)}"


def test_agent_yaml_covers_configurable_parameters():
    """F9: модель, temperature, top_k, лимиты, пороги, пути к данным, профиль, ответственные — в конфигурации."""
    c = yaml.safe_load((ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
    required = ["agent.profile", "llm.model", "llm.base_url", "llm.temperature", "llm.num_ctx",
                "tools.search.default_top_k", "tools.search.max_top_k", "guardrails.max_query_chars",
                "context.max_context_chars", "memory.max_messages", "router.products",
                "grounding.min_overlap", "grounding.max_regenerations", "escalation.queue_path",
                "escalation.owners.ungrounded_answer", "escalation.owners.no_data",
                "escalation.owners.component_failure", "escalation.owners.user_request",
                "service.port", "service.session_ttl_minutes", "service.max_sessions", "service.log_level",
                "profiles.public.rag_config", "profiles.public.cards_meta", "profiles.full.rag_config"]

    def get(path):
        node = c
        for part in path.split("."):
            node = node[part]
        return node
    missing = []
    for path in required:
        try:
            get(path)
        except (KeyError, TypeError):
            missing.append(path)
    assert not missing, missing


def test_owners_are_roles_not_people():
    owners = yaml.safe_load((ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))["escalation"]["owners"]
    assert all(not re.search(r"[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.", v) and "@" not in v for v in owners.values())


@pytest.mark.parametrize("profile, catalog", [("public", "data/raw_registry_public/cards_meta.jsonl"),
                                              ("full", "data/raw_cards/cards_meta.jsonl")])
def test_both_profiles_load(profile, catalog, monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from agent_config import load_agent_config
    monkeypatch.chdir(ROOT)
    for var in ("AGENT_PROFILE", "AGENT_LLM_MODEL", "AGENT_RETRIEVER", "AGENT_LLM_BASE_URL", "AGENT_LLM_BACKEND"):
        monkeypatch.delenv(var, raising=False)
    cfg = load_agent_config("config/agent.yaml", profile=profile)
    assert cfg["profile"] == profile and cfg["catalog_path"] == catalog


def test_environment_overrides_yaml(monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from agent_config import load_agent_config
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("AGENT_PROFILE", "full")
    monkeypatch.setenv("AGENT_LLM_MODEL", "other-model:1b")
    monkeypatch.setenv("AGENT_LLM_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("AGENT_RETRIEVER", "lexical")
    cfg = load_agent_config("config/agent.yaml")
    assert (cfg["profile"], cfg["llm"]["model"], cfg["llm"]["base_url"], cfg["retriever"]["backend"]) == \
        ("full", "other-model:1b", "http://ollama:11434", "lexical")


def test_escalation_queue_path_from_environment(monkeypatch, tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from agents.escalation import EscalationQueue
    monkeypatch.setenv("ESCALATION_QUEUE_PATH", str(tmp_path / "q.jsonl"))
    assert EscalationQueue("data/escalations/escalations.jsonl").path == tmp_path / "q.jsonl"
