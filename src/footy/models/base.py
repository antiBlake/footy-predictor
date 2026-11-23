"""Shared model plumbing: dataset loading, feature sets, preprocessing.

All five leagues are pooled into one training set with a league indicator:
~4x more data per model than per-league training, and league-specific
effects (e.g. Bundesliga's higher scoring) are still learnable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from footy.config import PROCESSED_DIR, load_leagues
from footy.features.pipeline import feature_columns

FEATURE_SETS: dict[str, list[str]] = {
    # The model forms its own view: lets you compare it with the market later.
    "no_market": ["elo", "form", "context"],
    # Market + everything: the strongest pure-forecast setup.
    "with_market": ["market", "elo", "form", "context"],
}
FIRST_TRAIN_SEASON = 2016  # 2015-16 is the Elo / form warm-up season


def load_features(leagues: list[str] | None = None) -> pd.DataFrame:
    keys = leagues or list(load_leagues())
    frames = []
    for key in keys:
        path = PROCESSED_DIR / "features" / f"{key}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `footy features --league {key}`")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True).sort_values(["date", "league"]).reset_index(drop=True)


def select_features(df: pd.DataFrame, feature_set: str) -> list[str]:
    groups = feature_columns(df)
    cols = [c for g in FEATURE_SETS[feature_set] for c in groups[g]]
    return [c for c in cols if c != "elo_warmup"]  # constant once warm-up seasons are dropped


@dataclass
class Preprocessor:
    """Turns feature frames into model matrices, with statistics learned on train only.

    * drops columns that are entirely NA in training (e.g. xG before Understat)
    * linear models: median imputation + missing-indicator columns + standardisation
    * tree models: raw values (NaN kept; LightGBM handles it natively)
    * league one-hot (fixed order from config, so unseen leagues fail loudly)
    """

    columns: list[str]
    scale: bool = True
    leagues: list[str] = field(default_factory=lambda: list(load_leagues()))
    keep_: list[str] = field(default_factory=list, init=False)
    medians_: pd.Series | None = field(default=None, init=False)
    indicator_cols_: list[str] = field(default_factory=list, init=False)
    mean_: pd.Series | None = field(default=None, init=False)
    std_: pd.Series | None = field(default=None, init=False)

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        X = df[self.columns].astype("float64")
        self.keep_ = [c for c in self.columns if X[c].notna().any()]
        X = X[self.keep_]
        self.medians_ = X.median()
        self.indicator_cols_ = [c for c in self.keep_ if X[c].isna().any()]
        if self.scale:
            filled = self._impute(X)
            self.mean_ = filled.mean()
            self.std_ = filled.std().replace(0, 1.0)
        return self

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.medians_ is not None
        return X.fillna(self.medians_)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        unknown = set(df["league"]) - set(self.leagues)
        if unknown:
            raise ValueError(f"Unknown leagues {unknown}")
        X = df[self.keep_].astype("float64")
        league = pd.get_dummies(pd.Categorical(df["league"], categories=self.leagues),
                                prefix="lg", dtype="float64").set_index(df.index)
        if not self.scale:
            return pd.concat([X, league], axis=1)
        assert self.mean_ is not None and self.std_ is not None
        miss = X[self.indicator_cols_].isna().astype("float64").add_suffix("_missing")
        Xs = (self._impute(X) - self.mean_) / self.std_
        return pd.concat([Xs, miss, league], axis=1)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def normalise(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, None)
    return p / p.sum(axis=1, keepdims=True)
