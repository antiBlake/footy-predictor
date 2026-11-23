"""Rolling team form from a team-perspective ("long") table.

Each match becomes two rows, one per team, holding that team's stats *for* and
*against*. Per team we take an exponentially weighted mean of previous matches
(`shift(1)` first, so a match never sees itself), at two half-lives:
    short (3 matches): current form
    long (10 matches): underlying level
Half-lives count matches, not days.

"Spells": a team relegated and promoted again later would otherwise carry
form from years ago. A spell is a run of consecutive top-flight seasons, and
form resets at the start of each spell. Promoted teams therefore start with
no form (NaN) plus an `is_promoted` flag; models must handle the NaNs (LightGBM
does natively; the linear models impute).

Unplayed / excluded matches have NA stats: they are skipped by the EWM and do
not count in `n_prev`, so future fixtures get the same features as if they
were the next match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# feature stem -> (home-side column, away-side column) in the match table
FORM_STATS: dict[str, tuple[str, str]] = {
    "goals": ("home_goals", "away_goals"),
    "shots": ("home_shots", "away_shots"),
    "sot": ("home_shots_on_target", "away_shots_on_target"),
    "corners": ("home_corners", "away_corners"),
    "xg": ("home_xg", "away_xg"),
    "npxg": ("home_np_xg", "away_np_xg"),
    "ppda": ("home_ppda", "away_ppda"),
    "deep": ("home_deep", "away_deep"),
}
HALF_LIVES = {"s": 3, "l": 10}
REST_CAP_DAYS = 30


def _as_float(df: pd.DataFrame, col: str) -> np.ndarray:
    """Column as float64 with NaN for missing values; all-NaN if the column is absent
    (e.g. xG before Understat is collected)."""
    if col not in df:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match. `row` is the match's position in df."""
    sides = []
    for side, opp in (("home", "away"), ("away", "home")):
        part = pd.DataFrame({
            "row": np.arange(len(df)),
            "season": df["season"].to_numpy(),
            "date": df["date"].to_numpy(),
            "team": df[f"{side}_team"].to_numpy(),
            "is_home": side == "home",
        })
        for stem, (h_col, a_col) in FORM_STATS.items():
            mine, theirs = (h_col, a_col) if side == "home" else (a_col, h_col)
            part[f"{stem}_for"] = _as_float(df, mine)
            part[f"{stem}_against"] = _as_float(df, theirs)
        gf, ga = part["goals_for"], part["goals_against"]
        part["points"] = np.select([gf > ga, gf == ga], [3.0, 1.0], 0.0)
        part.loc[gf.isna() | ga.isna(), "points"] = np.nan
        sides.append(part)
    long = pd.concat(sides, ignore_index=True)
    return long.sort_values(["team", "date", "row"]).reset_index(drop=True)


def _spells(long: pd.DataFrame) -> pd.Series:
    """Spell id per (team, season): increments when a team skips a season."""
    ts = long[["team", "season"]].drop_duplicates().sort_values(["team", "season"])
    new_spell = ts.groupby("team")["season"].diff().ne(1)
    ts["spell"] = new_spell.groupby(ts["team"]).cumsum()
    return long.merge(ts, on=["team", "season"], how="left")["spell"]


def team_form(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-match form features with h_ / a_ prefixes, indexed like df."""
    long = to_long(df)
    long["spell"] = _spells(long).to_numpy()
    grp_keys = [long["team"], long["spell"]]

    value_cols = [c for c in long.columns if c.endswith(("_for", "_against"))] + ["points"]
    feats = pd.DataFrame(index=long.index)
    shifted = long.groupby(grp_keys)[value_cols].shift(1)
    for tag, hl in HALF_LIVES.items():
        ewm = (shifted.groupby([long["team"], long["spell"]])
               .transform(lambda s, hl=hl: s.ewm(halflife=hl, ignore_na=True, min_periods=1).mean()))
        feats[[f"{c}_ewm{tag}" for c in value_cols]] = ewm.to_numpy()

    played = long["points"].notna().astype(float)
    feats["n_prev"] = played.groupby(grp_keys).cumsum() - played  # strictly previous
    prev_date = long.groupby(grp_keys)["date"].shift(1)
    feats["rest_days"] = (long["date"] - prev_date).dt.days.clip(upper=REST_CAP_DAYS)
    feats["season_match_no"] = long.groupby(["team", "season"]).cumcount() + 1

    first_season = df["season"].min()
    spell_start = long.groupby(grp_keys)["season"].transform("min")
    promoted = (long["season"] == spell_start) & (long["season"] > first_season)
    feats["is_promoted"] = promoted.astype(float).where(long["season"] > first_season)

    feats["row"] = long["row"]
    feats["is_home"] = long["is_home"]
    home = feats[feats["is_home"]].drop(columns="is_home").set_index("row").add_prefix("h_")
    away = feats[~feats["is_home"]].drop(columns="is_home").set_index("row").add_prefix("a_")
    out = home.join(away).sort_index()
    out.index = df.index
    return out
