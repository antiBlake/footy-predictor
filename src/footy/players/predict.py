"""Player tables for upcoming fixtures, driven by the match predictions."""

from __future__ import annotations

import logging

import pandas as pd

from footy.config import League
from footy.players.data import load_players
from footy.players.model import PlayerParams, predict_team_players

log = logging.getLogger(__name__)

MIN_P_APPEAR = 0.15  # don't list fringe players


def team_match_dates(history: pd.DataFrame) -> dict[str, list[pd.Timestamp]]:
    """Every date each team played (from the match table, canonical names)."""
    played = history.dropna(subset=["home_goals", "away_goals"])
    long = pd.concat([played[["date", "home_team"]].set_axis(["date", "team"], axis=1),
                      played[["date", "away_team"]].set_axis(["date", "team"], axis=1)])
    return {t: sorted(g["date"]) for t, g in long.groupby("team")}


def predict_fixture_players(league: League, match_preds: pd.DataFrame, history: pd.DataFrame,
                            params: PlayerParams = PlayerParams()) -> pd.DataFrame:
    """One row per (fixture, likely player). Empty if no player data is collected."""
    pm = load_players(league)
    if pm is None:
        return pd.DataFrame()
    dates = team_match_dates(history)
    frames = []
    for _, m in match_preds.iterrows():
        for side, lam in (("home", m["exp_home_goals"]), ("away", m["exp_away_goals"])):
            team = m[f"{side}_team"]
            t = predict_team_players(pm, team, pd.Timestamp(m["date"]), dates.get(team, []),
                                     float(lam), params)
            if t.empty:
                log.warning("%s: no recent player data for %s", league.key, team)
                continue
            t = t[t["p_appear"] >= MIN_P_APPEAR]
            frames.append(t.assign(match_id=m["match_id"], side=side, league=league.key,
                                   date=m["date"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
