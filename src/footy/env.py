"""Minimal .env loader (no extra dependency). Real environment variables win."""

from __future__ import annotations

import os
from pathlib import Path

from footy.config import PROJECT_ROOT


def load_dotenv(path: Path | None = None) -> None:
    path = path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)
