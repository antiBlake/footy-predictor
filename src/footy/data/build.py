"""Join per-source interim tables into one processed match table per league.

Base: Football-Data (results + odds), already in canonical names.
Joined: Understat xG/PPDA/deep/xPts, mapped to canonical names.

Join key is (league, season, home_team, away_team). In a double round-robin
league every home/away pair occurs exactly once per season, so this key is
unique and immune to timezone/postponement date differences. Dates are then
checked (must be within MAX_DAY_GAP) and scores compared as a guard against
mapping mistakes.

If Understat data is missing (e.g. the site is down), the table is still built
with NA xG columns and a warning, so downstream stages can proceed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from footy.config import CONFIG_DIR, INTERIM_DIR, PROCESSED_DIR, League
from footy.data import teams
from footy.data.understat import UNDERSTAT_COLUMNS

log = logging.getLogger(__name__)

EXCLUSIONS_PATH = CONFIG_DIR / "match_exclusions.yaml"
MAX_DAY_GAP = 3
JOIN_KEY = ["league", "season", "home_team", "away_team"]
UNDERSTAT_FEATURES = [c for c in UNDERSTAT_COLUMNS
                      if c not in {"league", "season", "date", "home_team", "away_team",
                                   "home_goals", "away_goals"}]


class BuildError(RuntimeError):
    pass


def _attach_understat(base: pd.DataFrame, league: League) -> pd.DataFrame:
    path = INTERIM_DIR / "understat" / f"{league.key}.parquet"
    if not path.exists():
        log.warning("%s: no Understat data at %s; xG columns will be NA", league.key, path)
        return base.assign(**{c: pd.NA for c in UNDERSTAT_FEATURES})

    us = teams.to_canonical(pd.read_parquet(path), "understat", league.key)
    us = us[JOIN_KEY + ["date", "home_goals", "away_goals", *UNDERSTAT_FEATURES]]
    if us.duplicated(JOIN_KEY).any():
        raise BuildError(f"{league.key}: duplicate Understat fixtures after mapping")

    merged = base.merge(us, on=JOIN_KEY, how="left", suffixes=("", "_us"), validate="1:1")
    found = merged["understat_id"].notna()

    gap = (merged["date"] - merged["date_us"].dt.normalize()).abs().dt.days
    bad_date = found & (gap > MAX_DAY_GAP)
    if bad_date.any():
        raise BuildError(f"{league.key}: {bad_date.sum()} joined matches differ by > "
                         f"{MAX_DAY_GAP} days; check team mapping")
    bad_score = found & ((merged["home_goals"] != merged["home_goals_us"])
                         | (merged["away_goals"] != merged["away_goals_us"]))
    if bad_score.any():
        log.warning("%s: %d matches where Understat score differs from Football-Data "
                    "(keeping Football-Data)", league.key, bad_score.sum())

    coverage = found.groupby(merged["season"]).mean().mul(100).round(1)
    log.info("%s: Understat coverage by season (%%): %s", league.key, coverage.to_dict())
    return merged.drop(columns=["date_us", "home_goals_us", "away_goals_us"])


def _apply_exclusions(df: pd.DataFrame, league: League,
                      path: Path = EXCLUSIONS_PATH) -> pd.DataFrame:
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else []
    out = df.assign(exclude_reason=pd.Series(pd.NA, index=df.index, dtype="string"))
    for e in entries or []:
        if e["league"] != league.key:
            continue
        hit = ((out["season"] == e["season"]) & (out["home_team"] == e["home_team"])
               & (out["away_team"] == e["away_team"]))
        if hit.sum() != 1:
            raise BuildError(f"Exclusion {e} matched {hit.sum()} rows (expected 1)")
        out.loc[hit, "exclude_reason"] = e["reason"]
        log.info("%s: excluding %s v %s (%s): %s", league.key, e["home_team"],
                 e["away_team"], e["season"], e["reason"])
    return out


def build_matches(league: League, *, write: bool = True) -> pd.DataFrame:
    fd_path = INTERIM_DIR / "football_data" / f"{league.key}.parquet"
    if not fd_path.exists():
        raise BuildError(f"Missing {fd_path}; run `footy collect --league {league.key}` first")
    base = pd.read_parquet(fd_path)
    df = _apply_exclusions(_attach_understat(base, league), league)
    df = df.sort_values(["date", "kickoff", "home_team"], na_position="first").reset_index(drop=True)
    if write:
        out = PROCESSED_DIR / "matches" / f"{league.key}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        log.info("Wrote %d matches to %s", len(df), out)
    return df
