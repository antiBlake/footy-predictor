"""Collector for Understat team-match xG via the soccerdata library.

Per match we keep: xG, non-penalty xG, PPDA (pressing), deep completions and
expected points for both sides. Player-level Understat data comes in Stage 7.

Robustness notes (learned the hard way):
  * When Understat is down it serves an HTML maintenance page with HTTP 200.
    soccerdata caches that page under a `.json` name and then fails with a
    JSONDecodeError on *every later run*, even after the site recovers.
    We (1) probe the site first and (2) purge any non-JSON cache files.
  * soccerdata's cache lives under data/raw/soccerdata (SOCCERDATA_DIR), so
    all raw data for the project sits in one place.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests

from footy.config import INTERIM_DIR, RAW_DIR, League, season_code
from footy.data.http import SourceUnavailable, make_session

log = logging.getLogger(__name__)

UNDERSTAT_URL = "https://understat.com"
SOCCERDATA_DIR = RAW_DIR / "soccerdata"
CACHE_DIR = SOCCERDATA_DIR / "data" / "Understat"
MAINTENANCE_MARKERS = ("We will come back soon",)

# soccerdata column -> our column
COLUMN_MAP = {
    "game_id": "understat_id",
    "date": "date",
    "home_team": "home_team",
    "away_team": "away_team",
    "home_goals": "home_goals",
    "away_goals": "away_goals",
    "home_xg": "home_xg",
    "away_xg": "away_xg",
    "home_np_xg": "home_np_xg",
    "away_np_xg": "away_np_xg",
    "home_ppda": "home_ppda",
    "away_ppda": "away_ppda",
    "home_deep_completions": "home_deep",
    "away_deep_completions": "away_deep",
    "home_expected_points": "home_xpts",
    "away_expected_points": "away_xpts",
}
UNDERSTAT_COLUMNS = ["league", "season", *COLUMN_MAP.values()]
FLOAT_COLUMNS = ["home_xg", "away_xg", "home_np_xg", "away_np_xg",
                 "home_ppda", "away_ppda", "home_xpts", "away_xpts"]


def check_available(session: requests.Session | None = None) -> None:
    """Raise SourceUnavailable if Understat is down or serving a maintenance page."""
    session = session or make_session()
    try:
        resp = session.get(UNDERSTAT_URL, timeout=20)
    except requests.RequestException as exc:
        raise SourceUnavailable(f"Understat unreachable: {exc}") from exc
    if resp.status_code != 200:
        raise SourceUnavailable(f"Understat returned HTTP {resp.status_code}")
    if any(marker in resp.text for marker in MAINTENANCE_MARKERS):
        raise SourceUnavailable("Understat is showing its maintenance page; try again later")


def purge_poisoned_cache(cache_dir: Path = CACHE_DIR) -> int:
    """Delete cached `.json` files that are not JSON (e.g. saved HTML error pages)."""
    removed = 0
    if not cache_dir.exists():
        return 0
    for path in cache_dir.rglob("*.json"):
        head = path.read_bytes()[:64].lstrip()
        if not head or head[:1] not in (b"{", b"["):
            log.warning("Removing non-JSON soccerdata cache file %s", path)
            path.unlink()
            removed += 1
    return removed


def conform(raw: pd.DataFrame, league: League) -> pd.DataFrame:
    """Turn soccerdata's read_team_match_stats() output into our Understat frame."""
    df = raw.reset_index()
    missing = [c for c in [*COLUMN_MAP, "season_id"] if c not in df.columns]
    if missing:
        raise ValueError(f"soccerdata output is missing columns {missing}; did its API change?")
    out = df[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    out.insert(0, "season", pd.to_numeric(df["season_id"]).astype("Int64"))
    out.insert(0, "league", league.key)
    out["date"] = pd.to_datetime(out["date"])
    for col in FLOAT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in ["understat_id", "home_goals", "away_goals", "home_deep", "away_deep"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in ["league", "home_team", "away_team"]:
        out[col] = out[col].astype("string")
    # Only played matches have team stats; drop anything without a score.
    out = out.dropna(subset=["home_goals", "away_goals"])
    return out[UNDERSTAT_COLUMNS].sort_values("date").reset_index(drop=True)


def collect_league(
    league: League,
    seasons: Iterable[int],
    *,
    refresh: bool = False,
    write: bool = True,
) -> pd.DataFrame:
    seasons = list(seasons)
    os.environ.setdefault("SOCCERDATA_DIR", str(SOCCERDATA_DIR))
    check_available()
    purge_poisoned_cache()

    import soccerdata as sd  # imported lazily: heavy, and SOCCERDATA_DIR must be set first

    reader = sd.Understat(
        leagues=league.soccerdata_id,
        seasons=[season_code(s) for s in seasons],
        no_cache=refresh,
    )
    try:
        raw = reader.read_team_match_stats()
    except json.JSONDecodeError as exc:
        raise SourceUnavailable(f"Understat returned non-JSON data: {exc}") from exc
    finally:
        purge_poisoned_cache()

    df = conform(raw, league)
    for season, n in df.groupby("season").size().items():
        log.info("%s %s: %d matches with xG", league.key, season, n)
    if write:
        path = INTERIM_DIR / "understat" / f"{league.key}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        log.info("Wrote %d matches to %s", len(df), path)
    return df
