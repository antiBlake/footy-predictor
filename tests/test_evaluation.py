import numpy as np
import pandas as pd
import pytest

from footy.evaluation import betting, report
from footy.evaluation.backtest import walk_forward
from footy.evaluation.metrics import PROB_COLS
from footy.models.fitting import MatchModel


def _preds(n=600, seed=0):
    """Two 'models' on the same matches: truth-ish and a noisier one."""
    rng = np.random.default_rng(seed)
    true_p = rng.dirichlet([4, 3, 3], size=n)
    result = np.array(["H", "D", "A"])[[rng.choice(3, p=r) for r in true_p]]
    noisy = 0.6 * true_p + 0.4 * rng.dirichlet([1, 1, 1], size=n)
    base = pd.DataFrame({
        "match_id": [f"m{i}" for i in range(n)],
        "date": pd.Timestamp("2022-08-06") + pd.to_timedelta(np.arange(n) // 10 * 7, unit="D"),
        "season": 2022, "league": "EPL", "result": result,
    })
    good = base.assign(model="market", **dict(zip(PROB_COLS, true_p.T)))
    bad = base.assign(model="noisy", **dict(zip(PROB_COLS, noisy.T)))
    return pd.concat([good, bad], ignore_index=True), true_p, result


def test_bootstrap_detects_worse_model():
    preds, _, _ = _preds()
    d = report.bootstrap_diff(preds, "noisy", "market", n_boot=500)
    assert d.mean > 0 and d.lo < d.mean < d.hi
    assert d.share_better < 0.05


def test_common_rows_drops_partial_matches():
    preds, _, _ = _preds(n=20)
    preds = preds[~((preds["model"] == "noisy") & (preds["match_id"] == "m0"))]
    assert "m0" not in set(report.common_rows(preds)["match_id"])


def test_reliability_of_true_probabilities_is_good():
    preds, _, _ = _preds(n=6000, seed=1)
    rel = report.reliability(preds, "market")
    assert set(rel["outcome"]) == {"H", "D", "A"}
    assert (report.ece(rel) < 0.03).all()
    assert report.ece(rel).max() < report.ece(report.reliability(preds, "noisy")).max()


def test_value_bets_profit_and_clv():
    preds = pd.DataFrame({
        "match_id": ["a", "b"], "model": "m", "season": 2022,
        "date": pd.to_datetime(["2022-08-06", "2022-08-06"]), "result": ["H", "A"],
        "p_home": [0.6, 0.3], "p_draw": [0.2, 0.3], "p_away": [0.2, 0.4],
    })
    matches = pd.DataFrame({
        "match_id": ["a", "b"],
        "odds_b365_home": [2.0, 2.0], "odds_b365_draw": [4.0, 3.2], "odds_b365_away": [4.0, 3.0],
        "odds_avg_close_home": [1.8, 2.0], "odds_avg_close_draw": [4.0, 3.5], "odds_avg_close_away": [5.0, 3.5],
    })
    bets = betting.value_bets(preds, matches, "m").set_index("match_id")
    assert bets.loc["a", "pick"] == "H" and bets.loc["a", "edge"] == pytest.approx(0.2)
    assert bets.loc["a", "profit"] == pytest.approx(1.0)
    assert bets.loc["b", "pick"] == "A" and bets.loc["b", "profit"] == pytest.approx(2.0)
    # a: closing home price shortened 2.0 -> 1.8, so we beat the close
    assert bets.loc["a", "clv"] > 0
    s = betting.summarise(bets.reset_index(), thresholds=(0.0, 0.5), n_boot=50)
    assert s.loc[0.0, "bets"] == 2 and s.loc[0.5, "bets"] == 0


class RateModel(MatchModel):
    name = "rate"

    def __init__(self):
        super().__init__([], calibrate=False, n_valid=1)
        self.train_seasons: list[list[int]] = []

    def _fit_base(self, train, params, valid=None):
        if valid is None:
            self.train_seasons.append(sorted(train["season"].unique()))
        return "fitted", params

    def _predict_base(self, base, df):
        return pd.DataFrame(np.full((len(df), 3), 1 / 3), columns=PROB_COLS)


def test_walk_forward_trains_only_on_past(monkeypatch):
    rng = np.random.default_rng(2)
    rows = []
    for season in range(2016, 2022):
        for i in range(50):
            rows.append({"match_id": f"{season}-{i}", "league": "EPL", "season": season,
                         "date": pd.Timestamp(season, 9, 1) + pd.Timedelta(days=i),
                         "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0,
                         "result": rng.choice(["H", "D", "A"]), "is_usable": True,
                         "mkt_p_home": 0.5, "mkt_p_draw": 0.25, "mkt_p_away": 0.25})
    monkeypatch.setattr("footy.evaluation.backtest.load_features", lambda: pd.DataFrame(rows))
    seen = []

    def make(fs, df):
        m = RateModel()
        m.name = f"rate[{fs}]"
        seen.append(m)
        return [m]

    preds = walk_forward(2019, 2021, make_models=make, save=False)
    assert set(preds["season"]) == {2019, 2020, 2021}
    for m in seen:
        assert max(m.train_seasons[0]) < max(preds["season"])  # sanity
    finals = sorted({max(m.train_seasons[0]) for m in seen})
    assert finals == [2018, 2019, 2020]
    assert {"market", "base_rate", "rate[no_market]"} <= set(preds["model"])
