"""Probabilistic scoring rules for 1X2 forecasts. Lower is better for all three.

Outcomes are ordered Home(0) < Draw(1) < Away(2). That order matters for RPS,
which rewards putting probability *near* the true outcome (a home win
predicted as a draw is a smaller miss than one predicted as an away win).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ["H", "D", "A"]
PROB_COLS = ["p_home", "p_draw", "p_away"]
EPS = 1e-15


def encode(result: pd.Series) -> np.ndarray:
    """'H'/'D'/'A' -> 0/1/2."""
    codes = pd.Categorical(result, categories=OUTCOMES).codes
    if (codes < 0).any():
        raise ValueError("result contains values other than H/D/A")
    return codes.astype(int)


def one_hot(y: np.ndarray) -> np.ndarray:
    return np.eye(3)[y]


def _check(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected (n, 3) probabilities, got {p.shape}")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to 1")
    return p


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = _check(p)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0)).mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score (sum over the three outcomes; range 0-2)."""
    p = _check(p)
    return float(((p - one_hot(y)) ** 2).sum(axis=1).mean())


def rps(p: np.ndarray, y: np.ndarray) -> float:
    """Ranked Probability Score (range 0-1)."""
    p = _check(p)
    cum_p = np.cumsum(p, axis=1)[:, :-1]
    cum_y = np.cumsum(one_hot(y), axis=1)[:, :-1]
    return float(((cum_p - cum_y) ** 2).sum(axis=1).mean() / (p.shape[1] - 1))


def score(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"log_loss": log_loss(p, y), "brier": brier(p, y), "rps": rps(p, y), "n": len(y)}
