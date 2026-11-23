"""Independent Poisson goals model (+ optional Dixon-Coles low-score correction).

    home_goals ~ Poisson(lambda_h),  log lambda_h = w_h . x
    away_goals ~ Poisson(lambda_a),  log lambda_a = w_a . x
Two L2-regularised Poisson GLMs on the same pre-match features (sklearn
PoissonRegressor). The feature vector already includes both teams' Elo and
form, so the GLM learns how attack/defence signals combine.

From (lambda_h, lambda_a) we build the full scoreline grid, which gives
1X2, expected goals, over/under 2.5 and both-teams-to-score consistently.

Independent Poisson under-predicts 0-0 and 1-1 draws. Dixon-Coles (1997)
multiplies the four low-score cells by tau(rho); rho is fitted by maximum
likelihood on the training matches. `dixon_coles=False` gives the pure model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from footy.models.base import Preprocessor
from footy.models.fitting import MatchModel

MAX_GOALS = 10  # P(a side scores > 10) is negligible


def dc_tau(i: np.ndarray, j: np.ndarray, lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    tau = np.ones_like(lh, dtype=float)
    tau = np.where((i == 0) & (j == 0), 1 - lh * la * rho, tau)
    tau = np.where((i == 0) & (j == 1), 1 + lh * rho, tau)
    tau = np.where((i == 1) & (j == 0), 1 + la * rho, tau)
    tau = np.where((i == 1) & (j == 1), 1 - rho, tau)
    return tau


def fit_rho(lh: np.ndarray, la: np.ndarray, hg: np.ndarray, ag: np.ndarray) -> float:
    """MLE of rho given fitted means (only the tau term depends on rho)."""
    def nll(rho: float) -> float:
        tau = dc_tau(hg, ag, lh, la, rho)
        return np.inf if (tau <= 0).any() else -np.log(tau).sum()
    return float(minimize_scalar(nll, bounds=(-0.25, 0.25), method="bounded").x)


def score_grid(lh: np.ndarray, la: np.ndarray, rho: float = 0.0) -> np.ndarray:
    """(n, MAX_GOALS+1, MAX_GOALS+1) scoreline probabilities, rows=home goals."""
    goals = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(goals[None, :], lh[:, None])
    pa = poisson.pmf(goals[None, :], la[:, None])
    grid = ph[:, :, None] * pa[:, None, :]
    if rho:
        for i in (0, 1):
            for j in (0, 1):
                grid[:, i, j] *= dc_tau(np.array(i), np.array(j), lh, la, rho)
    return grid / grid.sum(axis=(1, 2), keepdims=True)


def grid_markets(grid: np.ndarray) -> pd.DataFrame:
    n = MAX_GOALS + 1
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return pd.DataFrame({
        "p_home": (grid * (i > j)).sum(axis=(1, 2)),
        "p_draw": (grid * (i == j)).sum(axis=(1, 2)),
        "p_away": (grid * (i < j)).sum(axis=(1, 2)),
        "p_over25": (grid * (i + j > 2)).sum(axis=(1, 2)),
        "p_btts": (grid * ((i > 0) & (j > 0))).sum(axis=(1, 2)),
    })


class PoissonGoalsModel(MatchModel):
    param_grid = [{"alpha": a} for a in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)]

    def __init__(self, features: list[str], dixon_coles: bool = True, calibrate: bool = True):
        super().__init__(features, calibrate)
        self.dixon_coles = dixon_coles
        self.name = "poisson_dc" if dixon_coles else "poisson"

    def _fit_base(self, train, params, valid=None):
        prep = Preprocessor(self.features, scale=True).fit(train)
        X = prep.transform(train)
        hg = train["home_goals"].astype(float).to_numpy()
        ag = train["away_goals"].astype(float).to_numpy()
        home = PoissonRegressor(alpha=params["alpha"], max_iter=1000).fit(X, hg)
        away = PoissonRegressor(alpha=params["alpha"], max_iter=1000).fit(X, ag)
        rho = 0.0
        if self.dixon_coles:
            rho = fit_rho(home.predict(X), away.predict(X), hg.astype(int), ag.astype(int))
        return {"prep": prep, "home": home, "away": away, "rho": rho}, {**params, "rho": rho}

    def _predict_base(self, base: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        X = base["prep"].transform(df)
        lh, la = base["home"].predict(X), base["away"].predict(X)
        out = grid_markets(score_grid(lh, la, base["rho"]))
        out["exp_home_goals"], out["exp_away_goals"] = lh, la
        return out

    def predict_scorelines(self, df: pd.DataFrame) -> np.ndarray:
        """Full scoreline grid (uncalibrated), for correct-score style outputs."""
        X = self.base_["prep"].transform(df)
        return score_grid(self.base_["home"].predict(X), self.base_["away"].predict(X),
                          self.base_["rho"])
