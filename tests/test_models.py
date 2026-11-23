import numpy as np
import pandas as pd
import pytest

from footy.evaluation import metrics
from footy.models.calibration import (
    Calibrator, DirichletCalibrator, TemperatureCalibrator, select_calibrator,
)
from footy.models.fitting import MatchModel
from footy.models.poisson import PoissonGoalsModel, dc_tau, fit_rho, grid_markets, score_grid


# --- metrics ------------------------------------------------------------------------

def test_metrics_known_values():
    p = np.array([[1.0, 0.0, 0.0], [0.5, 0.3, 0.2]])
    y = np.array([0, 2])
    assert metrics.log_loss(p, y) == pytest.approx(-np.log(0.2) / 2)
    assert metrics.brier(p, y) == pytest.approx((0 + (0.25 + 0.09 + 0.64)) / 2)
    # RPS row 2: cum p = (0.5, 0.8), cum y = (0, 0) -> (0.25 + 0.64) / 2
    assert metrics.rps(p, y) == pytest.approx((0 + 0.445) / 2)


def test_rps_rewards_near_misses():
    y = np.array([0])
    near = np.array([[0.2, 0.7, 0.1]])
    far = np.array([[0.2, 0.1, 0.7]])
    assert metrics.rps(near, y) < metrics.rps(far, y)
    assert metrics.log_loss(near, y) == metrics.log_loss(far, y)  # log loss can't tell


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        metrics.log_loss(np.array([[0.5, 0.5, 0.5]]), np.array([0]))
    with pytest.raises(ValueError):
        metrics.encode(pd.Series(["H", "X"]))


# --- calibration --------------------------------------------------------------------

def _sample(n, rng, true_p):
    return np.array([rng.choice(3, p=row) for row in true_p])


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_p = rng.dirichlet([4, 3, 3], size=6000)
    y = _sample(6000, rng, true_p)
    overconfident = TemperatureCalibrator._apply(true_p, 0.6)  # sharpened
    cal = TemperatureCalibrator().fit(overconfident, y)
    assert cal.T_ == pytest.approx(1 / 0.6, rel=0.1)
    assert metrics.log_loss(cal.transform(overconfident), y) < metrics.log_loss(overconfident, y)


def test_selection_prefers_none_for_calibrated_input():
    rng = np.random.default_rng(1)
    true_p = rng.dirichlet([4, 3, 3], size=4500)
    y = _sample(4500, rng, true_p)
    season = np.repeat([2021, 2022, 2023], 1500)
    cal, scores = select_calibrator(true_p, y, season)
    assert set(scores) == {"none", "temperature", "dirichlet"}
    assert scores["none"] <= scores["dirichlet"]


def test_dirichlet_fixes_class_bias():
    rng = np.random.default_rng(2)
    true_p = rng.dirichlet([4, 3, 3], size=6000)
    y = _sample(6000, rng, true_p)
    biased = true_p * np.array([1.0, 0.6, 1.0])
    biased /= biased.sum(1, keepdims=True)  # under-predicts draws
    cal = DirichletCalibrator().fit(biased, y)
    assert abs(cal.transform(biased)[:, 1].mean() - (y == 1).mean()) < 0.01
    assert isinstance(Calibrator().transform(biased), np.ndarray)


# --- Poisson ------------------------------------------------------------------------

def test_score_grid_sums_and_markets_consistent():
    lh, la = np.array([1.5, 0.8]), np.array([1.1, 2.0])
    grid = score_grid(lh, la, rho=-0.05)
    assert np.allclose(grid.sum(axis=(1, 2)), 1)
    m = grid_markets(grid)
    assert np.allclose(m[["p_home", "p_draw", "p_away"]].sum(axis=1), 1)
    assert m.loc[0, "p_home"] > m.loc[0, "p_away"] and m.loc[1, "p_away"] > m.loc[1, "p_home"]


def test_dixon_coles_negative_rho_adds_draws():
    lh, la = np.array([1.3]), np.array([1.1])
    plain = grid_markets(score_grid(lh, la, 0.0))
    dc = grid_markets(score_grid(lh, la, -0.1))
    assert dc.loc[0, "p_draw"] > plain.loc[0, "p_draw"]


def test_fit_rho_recovers_sign():
    rng = np.random.default_rng(3)
    n = 20000
    lh, la = np.full(n, 1.4), np.full(n, 1.1)
    grid = score_grid(lh[:1], la[:1], rho=-0.1)[0].ravel()
    cells = rng.choice(grid.size, size=n, p=grid)
    hg, ag = np.divmod(cells, 11)
    assert fit_rho(lh, la, hg, ag) == pytest.approx(-0.1, abs=0.04)
    assert dc_tau(np.array(1), np.array(1), lh[:1], la[:1], -0.1)[0] == pytest.approx(1.1)


# --- shared fitting procedure ---------------------------------------------------------

class SpyModel(MatchModel):
    """Records which seasons each fit saw; predicts the training base rate."""
    name = "spy"
    param_grid = [{"k": 1}, {"k": 2}]

    def __init__(self):
        super().__init__(features=[], calibrate=True, n_valid=3)
        self.seen: list[tuple[list[int], list[int] | None]] = []

    def _fit_base(self, train, params, valid=None):
        self.seen.append((sorted(train["season"].unique()),
                          None if valid is None else sorted(valid["season"].unique())))
        rate = np.bincount(metrics.encode(train["result"]), minlength=3) / len(train)
        return rate, params

    def _predict_base(self, base, df):
        return pd.DataFrame(np.tile(base, (len(df), 1)), columns=metrics.PROB_COLS)


def _toy_matches(seasons=range(2016, 2022), n=300, seed=4):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "season": np.repeat(list(seasons), n),
        "result": rng.choice(["H", "D", "A"], size=n * len(seasons), p=[0.45, 0.25, 0.30]),
    })


def test_fit_never_trains_on_its_validation_season_or_later():
    m = SpyModel().fit(_toy_matches())
    tuning = [s for s in m.seen if s[1] is not None]
    assert len(tuning) == 2 * 3  # grid x validation seasons
    for train_seasons, valid_seasons in tuning:
        assert max(train_seasons) < min(valid_seasons)
    assert m.seen[-1] == (list(range(2016, 2022)), None)  # final refit on everything
    assert m.calibrator_ is not None and set(m.calibration_scores_) == {"none", "temperature", "dirichlet"}


def test_fit_requires_enough_seasons():
    with pytest.raises(ValueError, match="training seasons"):
        SpyModel().fit(_toy_matches(seasons=range(2020, 2023)))


def test_poisson_model_end_to_end():
    rng = np.random.default_rng(5)
    n = 2400
    strength = rng.normal(size=n)
    hg, ag = rng.poisson(np.exp(0.3 + 0.4 * strength)), rng.poisson(np.exp(0.1 - 0.4 * strength))
    df = pd.DataFrame({
        "league": "EPL", "season": np.repeat(range(2016, 2022), n // 6), "x": strength,
        "home_goals": hg, "away_goals": ag,
        "result": np.where(hg > ag, "H", np.where(hg == ag, "D", "A")),
    })
    m = PoissonGoalsModel(["x"], dixon_coles=True).fit(df)
    pred = m.predict(df.tail(5))
    assert np.allclose(pred[metrics.PROB_COLS].sum(axis=1), 1)
    # stronger home side -> more expected home goals
    order = np.argsort(df.tail(5)["x"].to_numpy())
    assert np.all(np.diff(pred["exp_home_goals"].to_numpy()[order]) > 0)
