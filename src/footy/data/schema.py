"""Canonical match schema shared by every data source.

Every collector maps its source columns onto these names, so downstream code
(team mapping, features, models) never sees source-specific column names.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

# Identity / core result columns: must exist and be non-null for played matches.
REQUIRED_COLUMNS: list[str] = [
    "match_id",
    "league",
    "season",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
]

# Optional columns: kept if the source provides them, otherwise filled with NA.
OPTIONAL_COLUMNS: list[str] = [
    "kickoff",
    "referee",
    "ht_home_goals",
    "ht_away_goals",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
    "home_yellow",
    "away_yellow",
    "home_red",
    "away_red",
    # Pre-match odds (decimal). "avg" = market average across bookmakers.
    "odds_b365_home", "odds_b365_draw", "odds_b365_away",
    "odds_pin_home", "odds_pin_draw", "odds_pin_away",
    "odds_avg_home", "odds_avg_draw", "odds_avg_away",
    # Closing odds (only from 2019-20 onward in Football-Data).
    "odds_pin_close_home", "odds_pin_close_draw", "odds_pin_close_away",
    "odds_avg_close_home", "odds_avg_close_draw", "odds_avg_close_away",
    # Over/under 2.5 market average
    "odds_avg_over25", "odds_avg_under25",
]

ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

INT_COLUMNS = [
    "season", "home_goals", "away_goals", "ht_home_goals", "ht_away_goals",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners", "home_fouls", "away_fouls",
    "home_yellow", "away_yellow", "home_red", "away_red",
]
STRING_COLUMNS = ["match_id", "league", "home_team", "away_team", "result", "kickoff", "referee"]


class SchemaError(ValueError):
    """Raised when data cannot be coerced into the canonical schema."""


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing optional columns, order columns and coerce dtypes."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")

    out = df.copy()
    for col in OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[ALL_COLUMNS]

    out["date"] = pd.to_datetime(out["date"])
    for col in INT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in STRING_COLUMNS:
        out[col] = out[col].astype("string")
    odds_cols = [c for c in ALL_COLUMNS if c.startswith("odds_")]
    for col in odds_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a conformed frame. Raises on hard errors, logs and fixes soft ones.

    Hard errors (raise): nulls in required columns, negative goals, bad result codes.
    Soft issues (warn + fix): duplicate fixtures, result inconsistent with score,
    implausible odds (<= 1.0) which are set to NA.
    """
    nulls = df[REQUIRED_COLUMNS].isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        raise SchemaError(f"Nulls in required columns: {nulls.to_dict()}")

    if (df["home_goals"] < 0).any() or (df["away_goals"] < 0).any():
        raise SchemaError("Negative goal counts found")

    bad_results = ~df["result"].isin(["H", "D", "A"])
    if bad_results.any():
        raise SchemaError(f"Invalid result codes: {df.loc[bad_results, 'result'].unique()}")

    out = df

    expected = pd.Series("D", index=out.index, dtype="string")
    expected[out["home_goals"] > out["away_goals"]] = "H"
    expected[out["home_goals"] < out["away_goals"]] = "A"
    mismatch = expected != out["result"]
    if mismatch.any():
        log.warning("%d rows where result disagrees with score; using score", mismatch.sum())
        out = out.assign(result=expected)

    dup_key = ["league", "season", "home_team", "away_team"]
    dups = out.duplicated(subset=dup_key, keep="last")
    if dups.any():
        log.warning("Dropping %d duplicate fixtures (same teams twice in a season)", dups.sum())
        out = out[~dups]

    odds_cols = [c for c in out.columns if c.startswith("odds_")]
    bad_odds = out[odds_cols] <= 1.0
    if bad_odds.to_numpy().any():
        log.warning("Setting %d implausible odds values (<= 1.0) to NA", bad_odds.to_numpy().sum())
        out = out.copy()
        out[odds_cols] = out[odds_cols].mask(bad_odds)

    return out.sort_values(["date", "kickoff", "home_team"], na_position="first").reset_index(
        drop=True
    )
