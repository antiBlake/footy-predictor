"""Walk-forward backtest: for each test season S, fit every model on all
usable seasons before S and predict S. Predictions are therefore always
out-of-sample and time-ordered, exactly as they would have been made live
(models refit once per season; features such as Elo and form update after
every match regardless).

The first test season must leave >= 4 training seasons (3 are used inside
fit() for tuning/calibration), so with data from 2016-17 the first test
season is 2020-21.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from footy.config import PROCESSED_DIR, current_season
from footy.evaluation.metrics import PROB_COLS, encode
from footy.models.base import FEATURE_SETS, FIRST_TRAIN_SEASON, load_features, select_features
from footy.models.fitting import MatchModel
from footy.models.outcome import GBMOutcomeModel, LogisticOutcomeModel
from footy.models.poisson import PoissonGoalsModel

log = logging.getLogger(__name__)

PREDICTIONS_PATH = PROCESSED_DIR / "predictions" / "walkforward.parquet"
MARKET_COLS = ["mkt_p_home", "mkt_p_draw", "mkt_p_away"]
ID_COLS = ["match_id", "league", "season", "date", "home_team", "away_team", "result",
           "home_goals", "away_goals"]
EXTRA_PRED_COLS = ["exp_home_goals", "exp_away_goals", "p_over25", "p_btts"]

# Equal-weight probability averages. Members are chosen for being different
# kinds of model (goals-based vs direct classifier), not by test performance.
ENSEMBLES = {f"ensemble[{fs}]": [f"poisson_dc[{fs}]", f"lightgbm[{fs}]"] for fs in FEATURE_SETS}


def default_models(feature_set: str, df: pd.DataFrame) -> list[MatchModel]:
    feats = select_features(df, feature_set)
    models: list[MatchModel] = [
        PoissonGoalsModel(feats, dixon_coles=True),
        LogisticOutcomeModel(feats),
        GBMOutcomeModel(feats),
    ]
    for m in models:
        m.name = f"{m.name}[{feature_set}]"
    return models


def walk_forward(
    first_test: int = 2020,
    last_test: int | None = None,
    make_models: Callable[[str, pd.DataFrame], list[MatchModel]] = default_models,
    *,
    save: bool = True,
) -> pd.DataFrame:
    """Return long-format out-of-sample predictions: one row per (match, model)."""
    last_test = last_test if last_test is not None else current_season()
    df = load_features()
    df = df[df["is_usable"] & (df["season"] >= FIRST_TRAIN_SEASON)]
    frames = []
    for season in range(first_test, last_test + 1):
        train, test = df[df["season"] < season], df[df["season"] == season]
        if test.empty:
            log.warning("No usable matches in %s; skipping", season)
            continue
        log.info("Season %s: train %d matches, test %d", season, len(train), len(test))
        ids = test[ID_COLS].reset_index(drop=True)

        frames.append(ids.assign(model="market",
                                 **dict(zip(PROB_COLS, test[MARKET_COLS].to_numpy().T))))
        rate = np.bincount(encode(train["result"]), minlength=3) / len(train)
        frames.append(ids.assign(model="base_rate", **dict(zip(PROB_COLS, rate))))

        by_name = {}
        for fs in FEATURE_SETS:
            for model in make_models(fs, df):
                model.fit(train)
                pred = model.predict(test).reset_index(drop=True)
                keep = [c for c in PROB_COLS + EXTRA_PRED_COLS if c in pred]
                by_name[model.name] = pred[keep]
                frames.append(pd.concat([ids, pred[keep]], axis=1).assign(
                    model=model.name, calibrator=model.calibrator_.name if model.calibrator_ else "off"))
        for name, members in ENSEMBLES.items():
            if all(m in by_name for m in members):
                avg = sum(by_name[m][PROB_COLS].to_numpy() for m in members) / len(members)
                frames.append(ids.assign(model=name, **dict(zip(PROB_COLS, avg.T))))

    preds = pd.concat(frames, ignore_index=True)
    if save:
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(PREDICTIONS_PATH, index=False)
        log.info("Wrote %d predictions to %s", len(preds), PREDICTIONS_PATH)
    return preds
