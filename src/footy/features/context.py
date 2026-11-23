"""Match context features that are not team form."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from footy.config import CONFIG_DIR

CROWD_PATH = CONFIG_DIR / "crowd_restrictions.yaml"


def crowd_restricted(df: pd.DataFrame, league: str, path: Path = CROWD_PATH) -> pd.Series:
    periods = (yaml.safe_load(path.read_text()) or {}).get(league, [])
    flag = pd.Series(False, index=df.index)
    for p in periods:
        flag |= df["date"].between(pd.Timestamp(p["start"]), pd.Timestamp(p["end"]))
    return flag.astype(float).rename("crowd_restricted")


def season_progress(df: pd.DataFrame, h_match_no: pd.Series) -> pd.Series:
    """Fraction of the season played (0..1), from the home team's match number."""
    n_teams = df.groupby("season")["home_team"].transform("nunique")
    return (h_match_no / (2 * (n_teams - 1))).clip(upper=1.0).rename("season_progress")
