"""Bookmaker-implied probabilities.

We use the *pre-match market average* (`odds_avg_*`), not closing odds:
  * full coverage in every season (Pinnacle stops mid-2025-26);
  * an average is less noisy than one bookmaker;
  * closing odds are only known at kick-off, so using them would overstate
    accuracy for predictions made earlier.
Bet365 is the fallback when the average is missing.

Margin removal: proportional (divide each 1/odds by their sum). It is simple
and well understood; Shin's method is a possible later refinement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def implied_1x2(df: pd.DataFrame) -> pd.DataFrame:
    def pick(side: str) -> pd.Series:
        return df[f"odds_avg_{side}"].fillna(df[f"odds_b365_{side}"])

    inv = pd.DataFrame({s: 1.0 / pick(s) for s in ("home", "draw", "away")})
    overround = inv.sum(axis=1, min_count=3)
    return pd.DataFrame({
        "mkt_p_home": inv["home"] / overround,
        "mkt_p_draw": inv["draw"] / overround,
        "mkt_p_away": inv["away"] / overround,
        "mkt_overround": overround - 1.0,
    }, index=df.index)


def implied_over25(df: pd.DataFrame) -> pd.Series:
    if not {"odds_avg_over25", "odds_avg_under25"} <= set(df.columns):
        return pd.Series(np.nan, index=df.index, name="mkt_p_over25")
    inv_over = 1.0 / df["odds_avg_over25"]
    inv_under = 1.0 / df["odds_avg_under25"]
    return (inv_over / (inv_over + inv_under)).rename("mkt_p_over25")


def market_features(df: pd.DataFrame) -> pd.DataFrame:
    out = implied_1x2(df).join(implied_over25(df))
    # log-odds of home vs away is a handy single "strength gap" number
    out["mkt_log_ratio"] = np.log(out["mkt_p_home"] / out["mkt_p_away"])
    return out
