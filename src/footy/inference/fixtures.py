"""Gather upcoming fixtures for one league.

Sources, in priority order:
  1. Football-Data fixtures.csv: about the next week, WITH odds, canonical names.
  2. openfootball: the rest of the season, no odds, mapped via team aliases.
     Its results lag by weeks, so it is used only for fixtures, and any pairing
     already played (per Football-Data) or already in source 1 is dropped.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from footy.config import INTERIM_DIR, League, current_season
from footy.data import teams
from footy.data.football_data import make_match_id

log = logging.getLogger(__name__)

PAIR = ["season", "home_team", "away_team"]


def _not_in(df: pd.DataFrame, pairs: set[tuple]) -> pd.Series:
    """Boolean mask: rows whose (season, home, away) is not in `pairs`.
    (A plain list mask breaks on empty frames: df[[]] selects zero columns.)"""
    return pd.Series([tuple(r) not in pairs for r in df[PAIR].itertuples(index=False)],
                     index=df.index, dtype=bool)


def _openfootball_fixtures(league: League) -> pd.DataFrame:
    path = INTERIM_DIR / "openfootball" / f"{league.key}.parquet"
    if not path.exists():
        log.warning("%s: no openfootball data; only fixtures.csv fixtures available", league.key)
        return pd.DataFrame(columns=["date", *PAIR, "kickoff"])
    of = pd.read_parquet(path)
    of = of[(of["status"] == "scheduled") & (of["season"] == current_season())]
    if of.empty:
        return of
    return teams.to_canonical(of, "openfootball", league.key)


def upcoming_fixtures(
    league: League,
    history: pd.DataFrame,
    fd_fixtures: pd.DataFrame,
    *,
    days: int = 7,
    today: date | None = None,
) -> pd.DataFrame:
    today_ts = pd.Timestamp(today or date.today())
    end = today_ts + pd.Timedelta(days=days)
    played = set(map(tuple, history[PAIR].itertuples(index=False)))

    fd_all = fd_fixtures[fd_fixtures["league"] == league.key]
    # Every Football-Data fixture (even outside the window, e.g. last night's game
    # with no result yet) blocks openfootball's copy, whose dates can be wrong.
    fd_pairs = set(map(tuple, fd_all[PAIR].itertuples(index=False)))
    fd = fd_all[fd_all["date"].between(today_ts, end)].copy()
    fd = fd[_not_in(fd, played)]
    fd["fixture_source"] = "football_data"

    of = _openfootball_fixtures(league)
    taken = played | fd_pairs
    stale = of[(of["date"] < today_ts) & _not_in(of, taken)]
    if len(stale):
        log.warning("%s: %d openfootball fixtures dated before today have no result "
                    "(postponed, or results not refreshed: run `footy update`)",
                    league.key, len(stale))

    of = of[of["date"].between(today_ts, end)]
    of = of[_not_in(of, taken)]
    extra = pd.DataFrame({
        "league": league.key, "season": of["season"].astype(int).to_numpy(),
        "date": of["date"].to_numpy(), "kickoff": of["kickoff"].to_numpy(),
        "home_team": of["home_team"].to_numpy(), "away_team": of["away_team"].to_numpy(),
        "fixture_source": "openfootball",
    })
    if not extra.empty:
        extra["match_id"] = make_match_id(league.key, extra["date"], extra["home_team"],
                                          extra["away_team"])
    out = pd.concat([fd, extra], ignore_index=True)
    return out.sort_values(["date", "kickoff", "home_team"], na_position="last").reset_index(drop=True)
