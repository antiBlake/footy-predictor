"""Direct 1X2 classifiers: multinomial logistic regression and LightGBM.

LightGBM needs the OpenMP runtime (macOS: `brew install libomp`). If it cannot
be imported we fall back to scikit-learn's HistGradientBoostingClassifier,
the same family of algorithm, and log a warning.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from footy.evaluation.metrics import PROB_COLS, encode
from footy.models.base import Preprocessor
from footy.models.fitting import MatchModel

log = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAVE_LGBM = True
except (ImportError, OSError) as exc:  # OSError: libomp missing
    log.warning("LightGBM unavailable (%s); using HistGradientBoosting instead", exc)
    HAVE_LGBM = False


class LogisticOutcomeModel(MatchModel):
    name = "logistic"
    param_grid = [{"C": c} for c in (0.0003, 0.001, 0.003, 0.01, 0.03)]

    def _fit_base(self, train, params, valid=None):
        prep = Preprocessor(self.features, scale=True).fit(train)
        lr = LogisticRegression(C=params["C"], max_iter=2000)
        lr.fit(prep.transform(train), encode(train["result"]))
        return {"prep": prep, "lr": lr}, params

    def _predict_base(self, base, df):
        p = base["lr"].predict_proba(base["prep"].transform(df))
        return pd.DataFrame(p, columns=PROB_COLS)


class GBMOutcomeModel(MatchModel):
    """Gradient boosting, deliberately shallow and slow-learning: football
    outcomes are very noisy, so deep trees overfit quickly."""

    name = "lightgbm"
    param_grid = [
        {"num_leaves": 7, "min_child_samples": 200},
        {"num_leaves": 15, "min_child_samples": 100},
    ]
    learning_rate = 0.02
    max_rounds = 3000

    def _fit_base(self, train, params, valid=None):
        prep = Preprocessor(self.features, scale=False).fit(train)
        X, y = prep.transform(train), encode(train["result"])
        if not HAVE_LGBM:
            return self._fit_hgb(prep, X, y, params, valid)

        lgb_params = dict(
            objective="multiclass", num_class=3, learning_rate=self.learning_rate,
            num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
            verbose=-1, seed=42, num_threads=4,
        )
        dtrain = lgb.Dataset(X, y)
        if valid is not None:  # tuning run: early-stop on the held-out season
            dvalid = lgb.Dataset(prep.transform(valid), encode(valid["result"]), reference=dtrain)
            booster = lgb.train(lgb_params, dtrain, self.max_rounds, valid_sets=[dvalid],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
            rounds = booster.best_iteration
        else:  # final refit on all seasons: reuse the tuned number of rounds
            rounds = params["rounds"]
            booster = lgb.train(lgb_params, dtrain, rounds)
        return {"prep": prep, "booster": booster, "rounds": rounds}, {**params, "rounds": rounds}

    def _combine_params(self, fold_params):
        # each tuning fold early-stopped at its own round; use the average
        rounds = int(np.mean([fp["rounds"] for fp in fold_params if "rounds" in fp] or [0]))
        return {**fold_params[-1], "rounds": rounds} if rounds else fold_params[-1]

    def _fit_hgb(self, prep, X, y, params, valid):
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_leaf_nodes=params["num_leaves"],
            min_samples_leaf=params["min_child_samples"], l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=42,
        ).fit(X, y)
        return {"prep": prep, "hgb": clf}, params

    def _predict_base(self, base, df):
        X = base["prep"].transform(df)
        if "booster" in base:
            p = base["booster"].predict(X, num_iteration=base["rounds"])
        else:
            p = base["hgb"].predict_proba(X)
        return pd.DataFrame(np.asarray(p), columns=PROB_COLS)

    def feature_importance(self) -> pd.Series:
        b = self.base_["booster"]
        return pd.Series(b.feature_importance("gain"), index=b.feature_name()).sort_values(ascending=False)
