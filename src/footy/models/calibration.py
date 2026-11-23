"""Post-hoc calibration for 3-way probabilities, chosen by out-of-season validation.

Candidates:
  * none: the model is already calibrated (common for likelihood-trained
    models such as Poisson and logistic regression)
  * temperature: one parameter T, p_k ∝ p_k^(1/T); fixes over/under-confidence
    without moving the average H/D/A mix
  * dirichlet: multinomial LR on log-probabilities (Kull et al. 2019), a 3x3
    matrix + bias; also fixes class-specific bias but can learn one season's
    quirks

Lesson from the first Stage 4 run: a Dirichlet calibrator fitted on ONE
season (2023-24, draw rate 26.4%) shifted every later prediction towards
draws and made test log loss worse for every model. So the calibrator is
chosen by leave-one-season-out over several held-out seasons, and "none"
wins unless calibrating actually helps on seasons it did not see.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from footy.evaluation.metrics import log_loss
from footy.models.base import normalise

log = logging.getLogger(__name__)


class Calibrator:
    name = "none"

    def fit(self, p: np.ndarray, y: np.ndarray) -> Calibrator:
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return normalise(p)


class TemperatureCalibrator(Calibrator):
    name = "temperature"

    def __init__(self) -> None:
        self.T_ = 1.0

    @staticmethod
    def _apply(p: np.ndarray, T: float) -> np.ndarray:
        return normalise(np.exp(np.log(normalise(p)) / T))

    def fit(self, p, y):
        res = minimize_scalar(lambda T: log_loss(self._apply(p, T), y),
                              bounds=(0.5, 2.0), method="bounded")
        self.T_ = float(res.x)
        return self

    def transform(self, p):
        return self._apply(p, self.T_)


class DirichletCalibrator(Calibrator):
    name = "dirichlet"

    def __init__(self, C: float = 1.0) -> None:
        self.C = C
        self.lr_: LogisticRegression | None = None

    def fit(self, p, y):
        self.lr_ = LogisticRegression(C=self.C, max_iter=1000)
        self.lr_.fit(np.log(normalise(p)), y)
        if list(self.lr_.classes_) != [0, 1, 2]:
            raise ValueError("calibration data must contain all three outcomes")
        return self

    def transform(self, p):
        if self.lr_ is None:
            raise RuntimeError("calibrator not fitted")
        return self.lr_.predict_proba(np.log(normalise(p)))


CANDIDATES = (Calibrator, TemperatureCalibrator, DirichletCalibrator)


def select_calibrator(
    p: np.ndarray, y: np.ndarray, season: np.ndarray
) -> tuple[Calibrator, dict[str, float]]:
    """Leave-one-season-out choice among CANDIDATES, then refit the winner on all.

    p, y, season: pooled out-of-sample predictions from several held-out seasons.
    Returns (fitted calibrator, mean held-out log loss per candidate).
    """
    seasons = np.unique(season)
    if len(seasons) < 2:
        raise ValueError("need predictions from at least two seasons to choose a calibrator")
    scores: dict[str, float] = {}
    for cls in CANDIDATES:
        fold_ll = []
        for s in seasons:
            held = season == s
            cal = cls().fit(p[~held], y[~held])
            fold_ll.append(log_loss(cal.transform(p[held]), y[held]))
        scores[cls.name] = float(np.mean(fold_ll))
    best = min(scores, key=scores.get)
    winner = next(c for c in CANDIDATES if c.name == best)
    return winner().fit(p, y), scores
