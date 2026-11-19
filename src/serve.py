"""
Запуск HTTP API:  python src/serve.py [--profile public] [--host 127.0.0.1] [--port 8000]

Профиль, модель и адрес Ollama берутся из config/agent.yaml и переменных окружения
(AGENT_PROFILE, AGENT_LLM_MODEL, AGENT_LLM_BASE_URL — см. .env.example).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    import uvicorn
    from agent import KBAgent
    from agent_config import load_agent_config
    from service.api import create_app

    parser = argparse.ArgumentParser(description="HTTP API KB Search Agent")
    parser.add_argument("--config", default="config/agent.yaml")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = load_agent_config(args.config, profile=args.profile)
    service_cfg = {**config.get("service", {}), "profile": config.get("profile")}
    app = create_app(agent_factory=lambda: KBAgent(config), service_config=service_cfg)
    uvicorn.run(app, host=args.host or service_cfg.get("host", "127.0.0.1"),
                port=args.port or int(service_cfg.get("port", 8000)), log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
