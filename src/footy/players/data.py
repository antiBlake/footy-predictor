"""Collect Understat per-player match stats (via soccerdata).

One Understat request per match, so the first full run is slow (soccerdata
throttles requests; later runs read its cache). Start with a few recent
seasons: `footy players collect --from 2021`.

Output: data/interim/understat_players/<LEAGUE>.parquet, one row per player
per match, with canonical team names and the match date.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable

import pandas as pd

from footy.config import INTERIM_DIR, League, season_code
from footy.data import teams
from footy.data.http import SourceUnavailable
from footy.data.understat import SOCCERDATA_DIR, check_available, purge_poisoned_cache

log = logging.getLogger(__name__)

PLAYER_COLUMNS = ["league", "season", "game_id", "date", "team", "player_id", "player",
                  "position", "minutes", "goals", "own_goals", "shots", "xg", "xa",
                  "assists", "key_passes"]
INT_COLS = ["game_id", "player_id", "minutes", "goals", "own_goals", "shots", "assists",
            "key_passes"]


def players_path(league: League):
    return INTERIM_DIR / "understat_players" / f"{league.key}.parquet"


def conform(raw: pd.DataFrame, schedule: pd.DataFrame, league: League,
            aliases: teams.AliasTable | None = None) -> pd.DataFrame:
    """soccerdata player stats + schedule -> canonical player-match rows."""
    df = raw.reset_index()
    missing = [c for c in ["game_id", "team", "player_id", "player", "position", "minutes",
                           "goals", "shots", "xg", "xa", "assists", "key_passes", "season_id"]
               if c not in df.columns]
    if missing:
        raise ValueError(f"soccerdata player output missing {missing}; did its API change?")
    dates = schedule.reset_index()[["game_id", "date"]].drop_duplicates("game_id")
    df = df.merge(dates, on="game_id", how="left")
    if df["date"].isna().any():
        raise ValueError(f"{df['date'].isna().sum()} player rows have no match date")

    table = (aliases if aliases is not None else teams.load_aliases()).get("understat", {}) \
        .get(league.key, {})
    unknown = sorted(set(df["team"]) - set(table))
    if unknown:
        raise teams.UnmappedTeamError(
            f"Understat team names not in team_aliases.yaml for {league.key}: {unknown}. "
            "Run `footy teams propose --source understat` first.")
    out = pd.DataFrame({
        "league": league.key,
        "season": pd.to_numeric(df["season_id"]).astype("Int64"),
        "game_id": df["game_id"], "date": pd.to_datetime(df["date"]).dt.normalize(),
        "team": df["team"].map(table), "player_id": df["player_id"], "player": df["player"],
        "position": df["position"], "minutes": df["minutes"], "goals": df["goals"],
        "own_goals": df.get("own_goals", 0), "shots": df["shots"], "xg": df["xg"],
        "xa": df["xa"], "assists": df["assists"], "key_passes": df["key_passes"],
    })
    for c in INT_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int64")
    for c in ["xg", "xa"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype("float64")
    for c in ["league", "team", "player", "position"]:
        out[c] = out[c].astype("string")
    return out[PLAYER_COLUMNS].sort_values(["date", "team", "player_id"]).reset_index(drop=True)


def collect_league(league: League, seasons: Iterable[int], *, refresh: bool = False,
                   write: bool = True) -> pd.DataFrame:
    seasons = list(seasons)
    os.environ.setdefault("SOCCERDATA_DIR", str(SOCCERDATA_DIR))
    check_available()
    purge_poisoned_cache()
    import soccerdata as sd

    reader = sd.Understat(leagues=league.soccerdata_id, seasons=[season_code(s) for s in seasons],
                          no_cache=refresh)
    try:
        schedule = reader.read_schedule()
        raw = reader.read_player_match_stats()
    except json.JSONDecodeError as exc:
        raise SourceUnavailable(f"Understat returned non-JSON data: {exc}") from exc
    finally:
        purge_poisoned_cache()
    df = conform(raw, schedule, league)
    log.info("%s: %d player-match rows, %d players", league.key, len(df), df["player_id"].nunique())
    if write:
        path = players_path(league)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        log.info("Wrote %s", path)
    return df


def load_players(league: League) -> pd.DataFrame | None:
    path = players_path(league)
    return pd.read_parquet(path) if path.exists() else None
