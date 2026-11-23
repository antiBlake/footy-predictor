"""Predict upcoming fixtures.

For each fixture:
  * features come from the exact code path used in training (build_features
    with `fixtures=`), on top of the full results history;
  * 1X2 = average of the calibrated Poisson-DC and LightGBM probabilities,
    using the 'with_market' models when the fixture has odds, else 'no_market';
  * goals outputs (expected goals, over 2.5, BTTS, likely scorelines) come from
    the Poisson-DC member of the same feature set. They are not separately
    calibrated, so they can differ slightly from the ensemble's 1X2.
"""

from __future__ import annotations

import json
import logging
from datetime import date

import numpy as np
import pandas as pd

from footy.config import PROCESSED_DIR, League
from footy.data.football_data import fetch_fixtures
from footy.evaluation.metrics import PROB_COLS
from footy.features.pipeline import build_features
from footy.inference.fixtures import upcoming_fixtures
from footy.inference.production import load_production
from footy.models.poisson import MAX_GOALS

log = logging.getLogger(__name__)

PREDICTIONS_DIR = PROCESSED_DIR.parent / "predictions"
LATEST_PATH = PREDICTIONS_DIR / "upcoming_latest.parquet"
PLAYERS_LATEST_PATH = PREDICTIONS_DIR / "upcoming_players_latest.parquet"
MARKET_COLS = ["mkt_p_home", "mkt_p_draw", "mkt_p_away"]
OUTPUT_COLUMNS = [
    "league", "date", "kickoff", "home_team", "away_team", "model", "fixture_source",
    "p_home", "p_draw", "p_away", "fair_odds_home", "fair_odds_draw", "fair_odds_away",
    "mkt_p_home", "mkt_p_draw", "mkt_p_away",
    "exp_home_goals", "exp_away_goals", "p_over25", "p_btts", "top_scores", "top_scores_json",
    "match_id",
]


def _top_scores(grid: np.ndarray, k: int = 5) -> list[list[dict]]:
    """Most likely scorelines per match: [{"home": 1, "away": 1, "p": 0.12}, ...]."""
    out = []
    for g in grid:
        flat = g.ravel()
        idx = np.argsort(flat)[::-1][:k]
        out.append([{"home": int(i // (MAX_GOALS + 1)), "away": int(i % (MAX_GOALS + 1)),
                     "p": round(float(flat[i]), 4)} for i in idx])
    return out


def _predict_set(models: dict, rows: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    poisson, gbm = models["poisson_dc"], models["lightgbm"]
    p_pois = poisson.predict(rows)
    p_gbm = gbm.predict(rows)
    out = pd.DataFrame(index=rows.index)
    out[PROB_COLS] = (p_pois[PROB_COLS].to_numpy() + p_gbm[PROB_COLS].to_numpy()) / 2
    for col in ("exp_home_goals", "exp_away_goals", "p_over25", "p_btts"):
        out[col] = p_pois[col].to_numpy()
    top = _top_scores(poisson.predict_scorelines(rows))
    out["top_scores"] = [", ".join(f"{t['home']}-{t['away']} {t['p']:.0%}" for t in ts[:3])
                         for ts in top]
    out["top_scores_json"] = [json.dumps(ts) for ts in top]
    out["model"] = f"ensemble[{feature_set}]"
    return out


def load_history(league: League) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "matches" / f"{league.key}.parquet")


def predict_league(league: League, bundle: dict, fd_fixtures: pd.DataFrame,
                   *, days: int, today: date | None) -> pd.DataFrame:
    history = load_history(league)
    fx = upcoming_fixtures(league, history, fd_fixtures, days=days, today=today)
    if fx.empty:
        log.info("%s: no fixtures in the next %d days", league.key, days)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return predict_rows(league, history, fx, bundle)


def predict_rows(league: League, history: pd.DataFrame, fx: pd.DataFrame,
                 bundle: dict) -> pd.DataFrame:
    """Predict given fixtures (canonical columns + `fixture_source`) for one league.
    Fixtures with all three odds use the with-market ensemble."""
    feats = build_features(history, league, fixtures=fx.drop(columns="fixture_source"))
    rows = feats[feats["match_id"].isin(fx["match_id"])].copy()
    rows = rows.merge(fx[["match_id", "fixture_source"]], on="match_id", how="left")
    rows.index = range(len(rows))

    has_odds = rows[MARKET_COLS].notna().all(axis=1)
    parts = []
    if has_odds.any():
        parts.append(_predict_set(bundle["with_market"], rows[has_odds], "with_market"))
    if (~has_odds).any():
        parts.append(_predict_set(bundle["no_market"], rows[~has_odds], "no_market"))
    pred = pd.concat(parts).sort_index()
    out = pd.concat([rows[["league", "date", "kickoff", "home_team", "away_team", "match_id",
                           "fixture_source", *MARKET_COLS]], pred], axis=1)
    for side, col in zip(("home", "draw", "away"), PROB_COLS):
        out[f"fair_odds_{side}"] = 1 / out[col]
    return out[OUTPUT_COLUMNS]


def predict_upcoming(leagues: list[League], *, days: int = 7, today: date | None = None,
                     refresh: bool = False, save: bool = True) -> pd.DataFrame:
    bundle, meta = load_production()
    log.info("Production models trained %s on seasons %s (last match %s)",
             meta["trained_at"], meta["seasons"], meta["last_match_date"])
    fd_fixtures = fetch_fixtures(leagues, refresh=refresh)
    frames = [predict_league(lg, bundle, fd_fixtures, days=days, today=today) for lg in leagues]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    out = pd.concat(frames, ignore_index=True).sort_values(["date", "kickoff", "league"])
    out = out.reset_index(drop=True)
    if save:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = (today or date.today()).isoformat()
        out.to_parquet(PREDICTIONS_DIR / f"upcoming_{stamp}.parquet", index=False)
        out.to_csv(PREDICTIONS_DIR / f"upcoming_{stamp}.csv", index=False)
        # stable name for the API: always the most recent run
        out.to_parquet(LATEST_PATH, index=False)
        _save_players(leagues, out)
        log.info("Wrote %d predictions to %s", len(out), PREDICTIONS_DIR / f"upcoming_{stamp}.csv")
    return out


def _save_players(leagues: list[League], match_preds: pd.DataFrame) -> None:
    """Player tables for the same fixtures (only if Understat player data exists)."""
    from footy.players.predict import predict_fixture_players

    frames = []
    for lg in leagues:
        sub = match_preds[match_preds["league"] == lg.key]
        if not sub.empty:
            frames.append(predict_fixture_players(lg, sub, load_history(lg)))
    frames = [f for f in frames if not f.empty]
    if not frames:
        log.info("No player data collected yet (`footy players collect`); skipping player tables")
        PLAYERS_LATEST_PATH.unlink(missing_ok=True)
        return
    players = pd.concat(frames, ignore_index=True)
    players.to_parquet(PLAYERS_LATEST_PATH, index=False)
    log.info("Wrote %d player rows to %s", len(players), PLAYERS_LATEST_PATH)
