"""Shared HTTP + raw-cache helpers for collectors."""

from __future__ import annotations

import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from footy.config import current_season

USER_AGENT = "footy-research/0.1 (+python-requests)"
# Current-season files change after every matchday; refetch if older than this.
CURRENT_SEASON_MAX_AGE_SECONDS = 6 * 3600


class SourceUnavailable(RuntimeError):
    """The requested data does not exist upstream, or the source is down."""


def make_session() -> requests.Session:
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def is_fresh(path: Path, season: int) -> bool:
    """Completed seasons are cached forever; the current season expires."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    if season < current_season():
        return True
    return (time.time() - path.stat().st_mtime) < CURRENT_SEASON_MAX_AGE_SECONDS


def atomic_write(path: Path, content: bytes) -> None:
    """Write via a temp file so a crash never leaves a half-written cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def fetch_cached(
    url: str,
    path: Path,
    season: int,
    *,
    refresh: bool = False,
    session: requests.Session | None = None,
) -> Path:
    """GET `url` into `path` unless a fresh cached copy exists."""
    if not refresh and is_fresh(path, season):
        return path
    session = session or make_session()
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        raise SourceUnavailable(f"{url} returned 404")
    resp.raise_for_status()
    if not resp.content.strip():
        raise SourceUnavailable(f"{url} returned an empty body")
    atomic_write(path, resp.content)
    return path
