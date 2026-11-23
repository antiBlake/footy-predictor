"""Stage 4 holdout experiment: train every model on seasons before `test_from`,
score all of them on the same later matches.

This is a single time-based split, to compare models quickly. Stage 5
replaces it with a walk-forward backtest (refit each season) plus calibration
diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from footy.config import PROCESSED_DIR, PROJECT_ROOT
from footy.evaluation.metrics import PROB_COLS, encode, score
from footy.models.base import FEATURE_SETS, FIRST_TRAIN_SEASON, load_features, select_features
from footy.models.fitting import MatchModel
from footy.models.outcome import GBMOutcomeModel, LogisticOutcomeModel
from footy.models.poisson import PoissonGoalsModel

log = logging.getLogger(__name__)
MODELS_DIR = PROJECT_ROOT / "models"
MARKET_COLS = ["mkt_p_home", "mkt_p_draw", "mkt_p_away"]


def make_models(feature_set: str, df: pd.DataFrame) -> list[MatchModel]:
    feats = select_features(df, feature_set)
    models: list[MatchModel] = [
        PoissonGoalsModel(feats, dixon_coles=False, calibrate=False),
        PoissonGoalsModel(feats, dixon_coles=True, calibrate=True),
        LogisticOutcomeModel(feats),
        GBMOutcomeModel(feats),
    ]
    for m in models:
        if isinstance(m, PoissonGoalsModel) and not m.calibrate:
            m.name = "poisson_raw"  # pure independent Poisson, no DC, no calibration
        m.name = f"{m.name}[{feature_set}]"
    return models


@dataclass
class HoldoutResult:
    scores: pd.DataFrame
    by_season: pd.DataFrame
    goals: pd.DataFrame
    predictions: pd.DataFrame
    chosen: pd.DataFrame


def _goal_scores(pred: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    """Expected-goals quality: Poisson deviance-style log-likelihood + MAE."""
    from scipy.stats import poisson
    out = {}
    for side in ("home", "away"):
        lam, k = pred[f"exp_{side}_goals"].to_numpy(), test[f"{side}_goals"].astype(int).to_numpy()
        out[f"{side}_nll"] = float(-poisson.logpmf(k, lam).mean())
        out[f"{side}_mae"] = float(np.abs(lam - k).mean())
        out[f"{side}_mean_pred"] = float(lam.mean())
        out[f"{side}_mean_actual"] = float(k.mean())
    y = (test["home_goals"] + test["away_goals"] > 2).astype(int).to_numpy()
    p = np.clip(pred["p_over25"].to_numpy(), 1e-12, 1 - 1e-12)
    out["over25_log_loss"] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    return out


def run_holdout(test_from: int = 2024, *, save: bool = True) -> HoldoutResult:
    df = load_features()
    df = df[df["is_usable"] & (df["season"] >= FIRST_TRAIN_SEASON)]
    # Compare everything on matches where the market benchmark exists
    df = df[df[MARKET_COLS].notna().all(axis=1)]
    train, test = df[df["season"] < test_from], df[df["season"] >= test_from]
    log.info("Train %d matches (%d-%d), test %d matches (%d+)", len(train),
             train["season"].min(), train["season"].max(), len(test), test_from)
    y = encode(test["result"])

    preds = {"market": test[MARKET_COLS].set_axis(PROB_COLS, axis=1)}
    base_rate = np.bincount(encode(train["result"]), minlength=3) / len(train)
    preds["base_rate"] = pd.DataFrame(np.tile(base_rate, (len(test), 1)), columns=PROB_COLS,
                                      index=test.index)
    goal_rows = {}
    chosen = {}
    if test["mkt_p_over25"].notna().all():
        mkt_goals = pd.DataFrame({"p_over25": test["mkt_p_over25"]})
        yo = (test["home_goals"] + test["away_goals"] > 2).astype(int)
        p = mkt_goals["p_over25"].clip(1e-12, 1 - 1e-12)
        goal_rows["market"] = {"over25_log_loss": float(-(yo * np.log(p) + (1 - yo) * np.log(1 - p)).mean())}

    for fs in FEATURE_SETS:
        for model in make_models(fs, df):
            log.info("Fitting %s", model.name)
            model.fit(train)
            pred = model.predict(test)
            chosen[model.name] = {
                "params": {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in (model.params_ or {}).items()},
                "calibrator": model.calibrator_.name if model.calibrator_ else "off",
                **{f"cal_{k}": v for k, v in model.calibration_scores_.items()},
            }
            preds[model.name] = pred
            if "exp_home_goals" in pred:
                goal_rows[model.name] = _goal_scores(pred, test)
            if save:
                MODELS_DIR.mkdir(exist_ok=True)
                joblib.dump(model, MODELS_DIR / f"{model.name}.joblib")

    rows, season_rows = [], []
    for name, pred in preds.items():
        p = pred[PROB_COLS].to_numpy()
        rows.append({"model": name, **score(p, y)})
        for season, idx in test.groupby("season").indices.items():
            season_rows.append({"model": name, "season": season, **score(p[idx], y[idx])})
    scores = pd.DataFrame(rows).set_index("model").sort_values("log_loss")
    by_season = pd.DataFrame(season_rows).pivot(index="model", columns="season", values="log_loss")
    goals = pd.DataFrame(goal_rows).T

    long = pd.concat(
        [pred[PROB_COLS].assign(model=name, match_id=test["match_id"].to_numpy())
         for name, pred in preds.items()], ignore_index=True)
    if save:
        out = PROCESSED_DIR / "predictions" / f"holdout_{test_from}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        long.to_parquet(out, index=False)
        log.info("Wrote %s", out)
    return HoldoutResult(scores, by_season.loc[scores.index], goals, long,
                         pd.DataFrame(chosen).T)
