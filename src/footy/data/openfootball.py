"""Collector for openfootball/football.json (fixtures + results, public domain).

Source: https://github.com/openfootball/football.json/<season>/<code>.json

Why we use it:
  * Second, independently maintained source of results: lets us build and
    verify the team-name mapping against Football-Data.
  * Contains *future* fixtures (no score yet), needed for inference (Stage 6).

Format quirks handled:
  * `score` is a dict {"ft": [h, a], "ht": [h, a]} OR a bare list [h, a].
  * Unplayed fixtures have no `score` and often no `time`.
  * `status: "cancelled"` marks voided matches (e.g. Ligue 1 2019-20).
  * The current season can lag a few matchdays behind real results.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from footy.config import INTERIM_DIR, RAW_DIR, League, season_label
from footy.data.http import SourceUnavailable, fetch_cached, make_session

log = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{code}.json"

FIXTURE_COLUMNS = [
    "league", "season", "date", "kickoff", "round", "home_team", "away_team",
    "home_goals", "away_goals", "ht_home_goals", "ht_away_goals", "status",
]


def raw_path(league: League, season: int) -> Path:
    return RAW_DIR / "openfootball" / league.key / f"{season_label(season)}.json"


def _scores(score: Any) -> tuple[Any, Any, Any, Any]:
    """Return (ft_home, ft_away, ht_home, ht_away), NA where absent."""
    na = pd.NA
    if isinstance(score, list) and len(score) == 2:
        return score[0], score[1], na, na
    if isinstance(score, dict):
        ft = score.get("ft") or [na, na]
        ht = score.get("ht") or [na, na]
        return ft[0], ft[1], ht[0], ht[1]
    return na, na, na, na


def _fix_wrong_year(dates: pd.Series, league: League, season: int) -> pd.Series:
    """Known openfootball error: EPL 2019-20 post-COVID games (July 2020) are dated
    July 2019. No Big-5 season since 2015 started before August, so a date
    before 1 August of the start year must belong to the following year."""
    wrong = dates < pd.Timestamp(season, 8, 1)
    if wrong.any():
        log.warning("%s %s: %d fixtures dated before the season start; shifting +1 year",
                    league.key, season, wrong.sum())
        dates = dates.where(~wrong, dates + pd.DateOffset(years=1))
    return dates


def parse_season(payload: dict[str, Any], league: League, season: int) -> pd.DataFrame:
    records = []
    for m in payload.get("matches", []):
        ft_h, ft_a, ht_h, ht_a = _scores(m.get("score"))
        if m.get("status") == "cancelled":
            status = "cancelled"
        elif pd.isna(ft_h) or pd.isna(ft_a):
            status = "scheduled"
        else:
            status = "played"
        records.append({
            "league": league.key,
            "season": season,
            "date": m.get("date"),
            "kickoff": m.get("time"),
            "round": m.get("round"),
            "home_team": (m.get("team1") or "").strip() or pd.NA,
            "away_team": (m.get("team2") or "").strip() or pd.NA,
            "home_goals": ft_h,
            "away_goals": ft_a,
            "ht_home_goals": ht_h,
            "ht_away_goals": ht_a,
            "status": status,
        })
    df = pd.DataFrame.from_records(records, columns=FIXTURE_COLUMNS)
    df = df.dropna(subset=["home_team", "away_team", "date"])
    df["date"] = _fix_wrong_year(
        pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce"), league, season
    )
    for col in ["home_goals", "away_goals", "ht_home_goals", "ht_away_goals"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["league", "kickoff", "round", "home_team", "away_team", "status"]:
        df[col] = df[col].astype("string")
    df["season"] = df["season"].astype("Int64")
    return df


def collect_league(
    league: League,
    seasons: Iterable[int],
    *,
    refresh: bool = False,
    write: bool = True,
) -> pd.DataFrame:
    session = make_session()
    frames = []
    for season in seasons:
        url = BASE_URL.format(season=season_label(season), code=league.openfootball_code)
        try:
            path = fetch_cached(url, raw_path(league, season), season,
                                refresh=refresh, session=session)
        except SourceUnavailable as exc:
            log.warning("Skipping %s %s: %s", league.key, season, exc)
            continue
        df = parse_season(json.loads(path.read_text(encoding="utf-8")), league, season)
        counts = df["status"].value_counts().to_dict()
        log.info("%s %s: %s", league.key, season, counts)
        frames.append(df)

    if not frames:
        raise SourceUnavailable(f"No openfootball data for {league.key}")
    out = pd.concat(frames, ignore_index=True).sort_values(["date", "home_team"])
    out = out.reset_index(drop=True)
    if write:
        path = INTERIM_DIR / "openfootball" / f"{league.key}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        log.info("Wrote %d fixtures to %s", len(out), path)
    return out
