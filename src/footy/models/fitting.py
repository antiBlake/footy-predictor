"""The fit procedure shared by every model (so comparisons are fair).

Given training seasons S_1..S_k and n_valid = V (default 3):
  1. For each of the last V seasons v, and each hyper-parameter setting:
     fit on seasons < v, predict season v (out-of-sample, time-ordered).
  2. Keep the setting with the best mean log loss across the V seasons
     (one season alone is too noisy: see calibration.py for what went wrong).
  3. Choose and fit the calibrator on the pooled out-of-sample predictions
     of the chosen setting (leave-one-season-out; "none" is a candidate).
  4. Refit the base model on S_1..S_k with the chosen setting.
Prediction = calibrator(base model trained on S_1..S_k).

No future match ever influences a prediction.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from footy.evaluation.metrics import PROB_COLS, encode, log_loss
from footy.models.calibration import Calibrator, select_calibrator

log = logging.getLogger(__name__)


class MatchModel(ABC):
    """Subclasses implement _fit_base / _predict_base, a param_grid and,
    optionally, _combine_params (how fold-specific params merge, e.g. the
    number of boosting rounds)."""

    name: str = "model"
    param_grid: list[dict[str, Any]] = [{}]

    def __init__(self, features: list[str], calibrate: bool = True, n_valid: int = 3) -> None:
        self.features = features
        self.calibrate = calibrate
        self.n_valid = n_valid
        self.params_: dict[str, Any] | None = None
        self.base_: Any = None
        self.calibrator_: Calibrator | None = None
        self.calibration_scores_: dict[str, float] = {}
        self.tuning_: pd.DataFrame | None = None

    # -- to implement ------------------------------------------------------------
    @abstractmethod
    def _fit_base(self, train: pd.DataFrame, params: dict[str, Any],
                  valid: pd.DataFrame | None = None) -> tuple[Any, dict[str, Any]]:
        """Fit and return (fitted base, params actually used e.g. best iteration)."""

    @abstractmethod
    def _predict_base(self, base: Any, df: pd.DataFrame) -> pd.DataFrame:
        """Return a frame with at least PROB_COLS (uncalibrated)."""

    def _combine_params(self, fold_params: list[dict[str, Any]]) -> dict[str, Any]:
        return fold_params[-1]

    # -- shared procedure ---------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> MatchModel:
        seasons = sorted(train["season"].unique())
        if len(seasons) < self.n_valid + 1:
            raise ValueError(f"need >= {self.n_valid + 1} training seasons, got {len(seasons)}")
        valid_seasons = seasons[-self.n_valid:]

        rows, oos = [], {}
        for gi, params in enumerate(self.param_grid):
            preds, used_all = [], []
            for v in valid_seasons:
                early, late = train[train["season"] < v], train[train["season"] == v]
                base, used = self._fit_base(early, params, valid=late)
                p = self._predict_base(base, late)[PROB_COLS].to_numpy()
                preds.append((p, encode(late["result"]), np.full(len(late), v)))
                used_all.append(used)
                rows.append({"grid": gi, "season": v, **used,
                             "log_loss": log_loss(p, encode(late["result"]))})
            oos[gi] = (preds, used_all)
        self.tuning_ = pd.DataFrame(rows)
        mean_ll = self.tuning_.groupby("grid")["log_loss"].mean()
        best = int(mean_ll.idxmin())
        preds, used_all = oos[best]
        self.params_ = self._combine_params(used_all)
        log.info("%s: params %s (mean valid log loss %.4f over %s)", self.name, self.params_,
                 mean_ll[best], valid_seasons)

        if self.calibrate:
            p = np.vstack([x[0] for x in preds])
            y = np.concatenate([x[1] for x in preds])
            s = np.concatenate([x[2] for x in preds])
            self.calibrator_, self.calibration_scores_ = select_calibrator(p, y, s)
            log.info("%s: calibrator=%s %s", self.name, self.calibrator_.name,
                     {k: round(v, 4) for k, v in self.calibration_scores_.items()})
        self.base_, _ = self._fit_base(train, self.params_)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.base_ is None:
            raise RuntimeError(f"{self.name} not fitted")
        out = self._predict_base(self.base_, df)
        if self.calibrator_ is not None:
            out[PROB_COLS] = self.calibrator_.transform(out[PROB_COLS].to_numpy())
        out.index = df.index
        return out
