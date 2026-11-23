"""Goal-difference Elo ratings, one rating pool per league.

Update rule (World Football Elo style):
    expected_home = 1 / (1 + 10 ** (-(R_home + HFA - R_away) / 400))
    R += K * G * (actual - expected), G = goal-difference multiplier
Between seasons ratings regress towards the league mean, and promoted teams
start at the average rating of the teams relegated the previous season
(they are replacing them; this keeps the pool mean stable).

The ratings stored for each match are *pre-match*: they are recorded before
that match updates them, so they never contain the match's own result.
Matches without a score (future fixtures, excluded matches) get ratings but
do not update them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    home_advantage: float = 60.0
    initial: float = 1500.0
    season_carryover: float = 0.8  # 1.0 = no regression, 0.0 = everyone resets to mean
    default_promoted_gap: float = 100.0  # used when no relegated teams are known yet


def goal_multiplier(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


def expected_home(r_home: float, r_away: float, hfa: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(r_home + hfa - r_away) / 400))


def compute_elo(df: pd.DataFrame, params: EloParams = EloParams()) -> pd.DataFrame:
    """Return pre-match elo_home, elo_away, elo_diff, elo_p_home for each row of df.

    df must be one league, sorted by date, with columns season, home_team,
    away_team, home_goals, away_goals (NA = not played / not usable).
    """
    ratings: dict[str, float] = {}
    out = np.full((len(df), 2), np.nan)
    current_season: int | None = None
    prev_teams: set[str] = set()
    season_teams: set[str] = set()

    seasons = df["season"].to_numpy()
    homes = df["home_team"].to_numpy()
    aways = df["away_team"].to_numpy()
    hg = df["home_goals"].to_numpy(dtype="float64", na_value=np.nan)
    ag = df["away_goals"].to_numpy(dtype="float64", na_value=np.nan)

    for i in range(len(df)):
        season = int(seasons[i])
        if season != current_season:
            if current_season is not None:
                prev_teams = season_teams
            this_season = set(df.loc[df["season"] == season, "home_team"])
            ratings = _new_season(ratings, prev_teams, this_season, params)
            current_season, season_teams = season, this_season

        h, a = homes[i], aways[i]
        rh, ra = ratings[h], ratings[a]
        out[i] = (rh, ra)
        if np.isnan(hg[i]) or np.isnan(ag[i]):
            continue
        exp_h = expected_home(rh, ra, params.home_advantage)
        actual = 1.0 if hg[i] > ag[i] else 0.5 if hg[i] == ag[i] else 0.0
        delta = params.k * goal_multiplier(int(hg[i] - ag[i])) * (actual - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

    res = pd.DataFrame(out, columns=["elo_home", "elo_away"], index=df.index)
    res["elo_diff"] = res["elo_home"] - res["elo_away"]
    res["elo_p_home"] = 1.0 / (1.0 + 10 ** (-(res["elo_diff"] + params.home_advantage) / 400))
    return res


def _new_season(
    ratings: dict[str, float],
    prev_teams: set[str],
    teams: set[str],
    params: EloParams,
) -> dict[str, float]:
    if not ratings:  # very first season: everyone equal
        return {t: params.initial for t in teams}

    mean = float(np.mean([ratings[t] for t in prev_teams]))
    regressed = {t: mean + params.season_carryover * (r - mean) for t, r in ratings.items()}
    relegated = prev_teams - teams
    promoted_start = (
        float(np.mean([regressed[t] for t in relegated]))
        if relegated else mean - params.default_promoted_gap
    )
    new = {}
    for t in teams:
        # A team returning after years away keeps no stale rating: treat as promoted.
        new[t] = regressed[t] if t in prev_teams else promoted_start
    # keep relegated teams' ratings so a quick return can be inspected later
    for t in relegated:
        new.setdefault(t, regressed[t])
    return new
