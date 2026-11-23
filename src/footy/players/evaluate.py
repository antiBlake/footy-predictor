"""Backtest the player model: anytime-scorer probabilities vs what happened.

For each team-match in the test period, predict the team's players as of
the match date, using the walk-forward (out-of-sample) expected goals of
the match model as team_lambda. Score p_score against whether the player
scored. Every player who *actually played* is scored, including those the
model didn't list (counted at p = 0, clipped), so missing a scorer is penalised.

Baseline: player's goals per appearance over his previous 38 appearances
(a typical "form" heuristic), so we can see whether the model adds anything.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from footy.players.model import PlayerParams, player_positions, position_priors, predict_team_players

log = logging.getLogger(__name__)
EPS = 1e-4


def binary_scores(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = np.clip(p, EPS, 1 - EPS)
    return {
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "mean_p": float(p.mean()), "rate": float(y.mean()), "n": int(len(y)),
    }


def team_lambdas_from_predictions(preds: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """(date, team) -> expected goals, from match-level predictions with exp_*_goals."""
    m = preds[["match_id", "exp_home_goals", "exp_away_goals"]].dropna().merge(
        matches[["match_id", "date", "home_team", "away_team"]], on="match_id")
    home = m[["date", "home_team", "exp_home_goals"]].set_axis(["date", "team", "lam"], axis=1)
    away = m[["date", "away_team", "exp_away_goals"]].set_axis(["date", "team", "lam"], axis=1)
    return pd.concat([home, away], ignore_index=True)


def backtest(pm: pd.DataFrame, lambdas: pd.DataFrame, team_dates: dict[str, list],
             start: pd.Timestamp, end: pd.Timestamp,
             params: PlayerParams = PlayerParams()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-player predictions with outcomes, summary table)."""
    test = lambdas[(lambdas["date"] >= start) & (lambdas["date"] <= end)]
    hist_before = pm[pm["date"] < start]
    positions = player_positions(hist_before)
    priors = position_priors(hist_before, positions)  # frozen at test start: no leakage
    rows = []
    for _, r in test.iterrows():
        pred = predict_team_players(pm, r["team"], r["date"], team_dates.get(r["team"], []),
                                    float(r["lam"]), params, priors, positions)
        actual = pm[(pm["date"] == r["date"]) & (pm["team"] == r["team"]) & (pm["minutes"] > 0)]
        merged = actual[["player_id", "player", "goals", "minutes"]].merge(
            pred[["player_id", "p_score", "exp_goals"]] if len(pred) else
            pd.DataFrame(columns=["player_id", "p_score", "exp_goals"]),
            on="player_id", how="left")
        merged["listed"] = merged["p_score"].notna()
        merged["p_score"] = merged["p_score"].fillna(0.0)
        # baseline: goals per appearance over previous 38 appearances
        prev = pm[(pm["date"] < r["date"]) & (pm["minutes"] > 0)
                  & pm["player_id"].isin(merged["player_id"])]
        base = (prev.sort_values("date").groupby("player_id").tail(38)
                .groupby("player_id")["goals"].agg(lambda g: (g > 0).mean()))
        merged["p_baseline"] = merged["player_id"].map(base).fillna((pm["goals"] > 0).mean())
        merged["date"], merged["team"] = r["date"], r["team"]
        rows.append(merged)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    res = pd.concat(rows, ignore_index=True)
    y = (res["goals"] > 0).astype(float).to_numpy()
    summary = pd.DataFrame({
        "player_model": binary_scores(res["p_score"].to_numpy(), y),
        "baseline": binary_scores(res["p_baseline"].to_numpy(), y),
    }).T
    summary["listed_share"] = res["listed"].mean()
    summary["goals_by_unlisted"] = float(res.loc[~res["listed"], "goals"].sum() / max(res["goals"].sum(), 1))
    return res, summary
