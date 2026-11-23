"""Collector for Football-Data.co.uk historical results and odds.

Source: https://www.football-data.co.uk/mmz4281/{season_code}/{division}.csv
Notes on the files (see https://www.football-data.co.uk/notes.txt):
  * Dates are dd/mm/yy in older seasons and dd/mm/yyyy in newer ones.
  * Some older files have trailing commas / rows longer than the header.
  * Encoding is not always UTF-8 (Latin-1 accented names in older files).
  * Kick-off time (`Time`) and closing odds exist from 2019-20 onward.
  * Betbrain averages (`BbAvH`) were replaced by `AvgH` in 2019-20.

Design: raw CSVs are cached untouched under data/raw/football_data/<LEAGUE>/,
then parsed into the canonical schema (footy.data.schema) and written to
data/interim/football_data/<LEAGUE>.parquet.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests

from footy.config import INTERIM_DIR, RAW_DIR, League, season_code
from footy.data import schema
from footy.data.http import SourceUnavailable, fetch_cached, make_session

log = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{division}.csv"

# Source column -> canonical column. First match wins for lists (fallbacks).
COLUMN_MAP: dict[str, str | list[str]] = {
    "date": "Date",
    "kickoff": "Time",
    "home_team": "HomeTeam",
    "away_team": "AwayTeam",
    "home_goals": "FTHG",
    "away_goals": "FTAG",
    "result": "FTR",
    "ht_home_goals": "HTHG",
    "ht_away_goals": "HTAG",
    "referee": "Referee",
    "home_shots": "HS",
    "away_shots": "AS",
    "home_shots_on_target": "HST",
    "away_shots_on_target": "AST",
    "home_corners": "HC",
    "away_corners": "AC",
    "home_fouls": "HF",
    "away_fouls": "AF",
    "home_yellow": "HY",
    "away_yellow": "AY",
    "home_red": "HR",
    "away_red": "AR",
    "odds_b365_home": "B365H",
    "odds_b365_draw": "B365D",
    "odds_b365_away": "B365A",
    "odds_pin_home": "PSH",
    "odds_pin_draw": "PSD",
    "odds_pin_away": "PSA",
    "odds_avg_home": ["AvgH", "BbAvH"],
    "odds_avg_draw": ["AvgD", "BbAvD"],
    "odds_avg_away": ["AvgA", "BbAvA"],
    "odds_pin_close_home": "PSCH",
    "odds_pin_close_draw": "PSCD",
    "odds_pin_close_away": "PSCA",
    "odds_avg_close_home": "AvgCH",
    "odds_avg_close_draw": "AvgCD",
    "odds_avg_close_away": "AvgCA",
    "odds_avg_over25": ["Avg>2.5", "BbAv>2.5"],
    "odds_avg_under25": ["Avg<2.5", "BbAv<2.5"],
}


def raw_path(league: League, season: int) -> Path:
    return RAW_DIR / "football_data" / league.key / f"{season_code(season)}.csv"


def download_season(
    league: League,
    season: int,
    *,
    refresh: bool = False,
    session: requests.Session | None = None,
) -> Path:
    """Download one season's CSV into the raw cache. Returns the cached path."""
    url = BASE_URL.format(season=season_code(season), division=league.football_data_code)
    log.debug("Fetching %s (cached if fresh)", url)
    return fetch_cached(url, raw_path(league, season), season, refresh=refresh, session=session)


# --- Parsing -------------------------------------------------------------------


def _decode(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode CSV")  # latin-1 never fails; kept for clarity


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read a Football-Data CSV robustly (ragged rows, odd encodings, blank lines)."""
    text = _decode(path.read_bytes())
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return pd.DataFrame()
    header = [h.strip() for h in rows[0]]
    # Some files have trailing empty header cells; drop them.
    while header and header[-1] == "":
        header.pop()
    width = len(header)
    body = [
        (r + [""] * width)[:width]
        for r in rows[1:]
        if any(cell.strip() for cell in r)
    ]
    df = pd.DataFrame(body, columns=header)
    return df.replace({"": pd.NA})


def _parse_dates(s: pd.Series) -> pd.Series:
    """Football-Data mixes dd/mm/yy and dd/mm/yyyy; parse each explicitly."""
    s = s.astype("string").str.strip()
    long = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return long.fillna(short)


def _pick(df: pd.DataFrame, source: str | list[str]) -> pd.Series | None:
    candidates = [source] if isinstance(source, str) else source
    for col in candidates:
        if col in df.columns:
            return df[col]
    return None


def make_match_id(league: str, date: pd.Series, home: pd.Series, away: pd.Series) -> pd.Series:
    """Stable, source-independent ID. Uses raw names for now; Stage 2 will
    regenerate IDs from canonical team names so sources can be joined."""
    keys = league + "|" + date.dt.strftime("%Y-%m-%d") + "|" + home + "|" + away
    return keys.map(lambda k: hashlib.sha1(k.encode()).hexdigest()[:16])


def map_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename source columns to canonical ones (results and fixtures files alike)."""
    out = pd.DataFrame(index=raw.index)
    for canonical, source in COLUMN_MAP.items():
        col = _pick(raw, source)
        if col is not None:
            out[canonical] = col
    return out


def parse_season(raw: pd.DataFrame, league: League, season: int) -> pd.DataFrame:
    """Map a raw Football-Data frame onto the canonical schema (not yet validated)."""
    out = map_columns(raw)

    # Drop rows with no teams (footer junk) or no final score (postponed / not played).
    before = len(out)
    out = out.dropna(subset=["home_team", "away_team"])
    out = out.dropna(subset=["home_goals", "away_goals"])
    if len(out) < before:
        log.info("%s %s: dropped %d rows without teams/score", league.key, season, before - len(out))

    out["home_team"] = out["home_team"].str.strip()
    out["away_team"] = out["away_team"].str.strip()
    out["date"] = _parse_dates(out["date"])
    bad_dates = out["date"].isna()
    if bad_dates.any():
        log.warning("%s %s: dropping %d rows with unparseable dates", league.key, season,
                    bad_dates.sum())
        out = out[~bad_dates]

    out["league"] = league.key
    out["season"] = season
    out["match_id"] = make_match_id(league.key, out["date"], out["home_team"], out["away_team"])
    return out


FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
FIXTURES_PATH = RAW_DIR / "football_data" / "fixtures.csv"


def fetch_fixtures(
    leagues: Iterable[League],
    *,
    refresh: bool = False,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Upcoming fixtures (~next week) with pre-match odds, in canonical columns.

    Football-Data's fixtures file covers many leagues and uses the same team
    names as its results files, so no alias mapping is needed. Goals/result
    are NA. Refetched when older than 6 hours.
    """
    from footy.config import current_season

    path = fetch_cached(FIXTURES_URL, FIXTURES_PATH, current_season(),
                        refresh=refresh, session=session)
    raw = read_raw_csv(path)
    frames = []
    for league in leagues:
        rows = raw[raw["Div"] == league.football_data_code]
        out = map_columns(rows)
        out = out.dropna(subset=["home_team", "away_team", "date"])
        if out.empty:
            continue
        for col in ("home_team", "away_team"):
            out[col] = out[col].str.strip()
        out["date"] = _parse_dates(out["date"])
        out["league"] = league.key
        out["season"] = out["date"].map(lambda d: current_season(d.date()))
        out["match_id"] = make_match_id(league.key, out["date"], out["home_team"], out["away_team"])
        for col in ("home_goals", "away_goals", "result"):
            out[col] = pd.NA
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=schema.ALL_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    for col in schema.OPTIONAL_COLUMNS:
        if col not in df:
            df[col] = pd.NA
    df = df[schema.ALL_COLUMNS]
    df["date"] = pd.to_datetime(df["date"])
    for col in schema.INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in [c for c in schema.ALL_COLUMNS if c.startswith("odds_")]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in schema.STRING_COLUMNS:
        df[col] = df[col].astype("string")
    return df


def collect_league(
    league: League,
    seasons: Iterable[int],
    *,
    refresh: bool = False,
    write: bool = True,
) -> pd.DataFrame:
    """Download (with caching), parse and validate several seasons for one league.

    Seasons that are unavailable upstream are skipped with a warning; any other
    failure is raised so problems are never silently ignored.
    """
    session = make_session()
    frames: list[pd.DataFrame] = []
    for season in seasons:
        try:
            path = download_season(league, season, refresh=refresh, session=session)
        except SourceUnavailable as exc:
            log.warning("Skipping %s %s: %s", league.key, season, exc)
            continue
        parsed = parse_season(read_raw_csv(path), league, season)
        log.info("%s %s: %d matches", league.key, season, len(parsed))
        frames.append(parsed)

    if not frames:
        raise SourceUnavailable(f"No data collected for {league.key}")

    df = schema.validate(schema.conform(pd.concat(frames, ignore_index=True)))

    if write:
        out_path = INTERIM_DIR / "football_data" / f"{league.key}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        log.info("Wrote %d matches to %s", len(df), out_path)
    return df
