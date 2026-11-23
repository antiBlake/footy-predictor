"""Daily-digest subscribers, persisted as a small JSON file (atomic writes)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from footy.config import DATA_DIR
from footy.data.http import atomic_write

SUBSCRIBERS_PATH = DATA_DIR / "bot" / "subscribers.json"


class Subscribers:
    def __init__(self, path: Path = SUBSCRIBERS_PATH) -> None:
        self.path = path
        self._lock = threading.Lock()

    def all(self) -> set[int]:
        if not self.path.exists():
            return set()
        return {int(x) for x in json.loads(self.path.read_text() or "[]")}

    def _save(self, ids: set[int]) -> None:
        atomic_write(self.path, json.dumps(sorted(ids)).encode())

    def add(self, chat_id: int) -> bool:
        with self._lock:
            ids = self.all()
            new = chat_id not in ids
            ids.add(chat_id)
            self._save(ids)
            return new

    def remove(self, chat_id: int) -> bool:
        with self._lock:
            ids = self.all()
            present = chat_id in ids
            ids.discard(chat_id)
            self._save(ids)
            return present
