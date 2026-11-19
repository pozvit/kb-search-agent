"""
Сессии: у каждой своя память диалога, агент — один на сервис.

Агент один, потому что локальный Qdrant открывается одним клиентом на процесс. Перед запросом
агенту подставляется память его сессии (см. api.py). Сессии живут ttl секунд с последнего
обращения; при превышении лимита вытесняется самая давняя.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass
class Session:
    memory: Any
    last_seen: float = field(default_factory=time.monotonic)


class SessionStore:
    def __init__(self, memory_factory: Callable[[], Any], ttl_seconds: float = 1800, max_sessions: int = 200,
                 clock: Callable[[], float] = time.monotonic):
        self._factory = memory_factory
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self._clock = clock
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: Optional[str]) -> Tuple[str, Session, bool]:
        now = self._clock()
        with self._lock:
            self._expire(now)
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_seen = now
                return session_id, session, False
            sid = session_id or str(uuid.uuid4())
            if len(self._sessions) >= self.max_sessions:
                oldest = min(self._sessions, key=lambda k: self._sessions[k].last_seen)
                del self._sessions[oldest]
            session = Session(memory=self._factory(), last_seen=now)
            self._sessions[sid] = session
            return sid, session, True

    def _expire(self, now: float) -> None:
        for sid in [k for k, s in self._sessions.items() if now - s.last_seen > self.ttl]:
            del self._sessions[sid]

    def __len__(self) -> int:
        return len(self._sessions)
