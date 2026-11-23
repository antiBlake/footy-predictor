"""Research-only value-betting backtest. Not betting advice.

Strategy: for each match take the outcome with the largest model edge
    edge = p_model * odds - 1
and stake 1 unit if edge > threshold. Prices are Bet365 pre-match odds
(collected by Football-Data before kick-off, so they could actually have been
taken). All thresholds are reported, so no single lucky one is cherry-picked.

Two measures:
  * ROI: realised profit per unit staked. Very noisy: at average odds of ~3,
    hundreds of bets are needed before ROI means anything.
  * CLV (closing-line value): expected return of the bet if the *closing*
    market (margin removed) is the truth:  CLV = q_close * odds_taken - 1.
    Closing prices are the sharpest public estimate, so consistently positive
    CLV is much stronger evidence of a real edge than ROI over a few seasons.

Only meaningful for models that did not see market features (`no_market`);
a model fed the odds will mostly just find bookmaker noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footy.evaluation.metrics import OUTCOMES, PROB_COLS

SIDES = ["home", "draw", "away"]
THRESHOLDS = (0.0, 0.03, 0.05, 0.10)


def _matrix(df: pd.DataFrame, template: str) -> np.ndarray:
    return df[[template.format(s) for s in SIDES]].to_numpy(dtype=float)


def value_bets(preds: pd.DataFrame, matches: pd.DataFrame, model: str) -> pd.DataFrame:
    """One row per match: best-edge outcome, its odds, edge, result and CLV."""
    g = preds[preds["model"] == model].merge(
        matches[["match_id", *[f"odds_b365_{s}" for s in SIDES],
                 *[f"odds_avg_close_{s}" for s in SIDES]]],
        on="match_id", how="inner")
    p = g[PROB_COLS].to_numpy()
    odds = _matrix(g, "odds_b365_{}")
    close_inv = 1.0 / _matrix(g, "odds_avg_close_{}")
    q_close = close_inv / close_inv.sum(axis=1, keepdims=True)

    edge = p * odds - 1
    valid = np.isfinite(edge).all(axis=1) & np.isfinite(q_close).all(axis=1)
    g, p, odds, q_close, edge = g[valid], p[valid], odds[valid], q_close[valid], edge[valid]
    pick = edge.argmax(axis=1)
    rows = np.arange(len(g))
    won = np.asarray(g["result"]) == np.array(OUTCOMES)[pick]
    taken = odds[rows, pick]
    return pd.DataFrame({
        "match_id": g["match_id"].to_numpy(), "season": g["season"].to_numpy(),
        "date": g["date"].to_numpy(), "pick": np.array(OUTCOMES)[pick],
        "odds": taken, "edge": edge[rows, pick], "won": won,
        "profit": np.where(won, taken - 1, -1.0),
        "clv": q_close[rows, pick] * taken - 1,
    })


def summarise(bets: pd.DataFrame, thresholds=THRESHOLDS, n_boot: int = 2000,
              seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in thresholds:
        b = bets[bets["edge"] > t]
        if b.empty:
            rows.append({"threshold": t, "bets": 0})
            continue
        boot = rng.choice(b["profit"].to_numpy(), size=(n_boot, len(b))).mean(axis=1)
        rows.append({
            "threshold": t, "bets": len(b), "avg_odds": b["odds"].mean(),
            "hit_rate": b["won"].mean(), "roi": b["profit"].mean(),
            "roi_lo": np.quantile(boot, 0.025), "roi_hi": np.quantile(boot, 0.975),
            "mean_clv": b["clv"].mean(), "clv_positive": (b["clv"] > 0).mean(),
        })
    return pd.DataFrame(rows).set_index("threshold")
