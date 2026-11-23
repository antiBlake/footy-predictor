import numpy as np
import pandas as pd
import pytest

from footy.config import get_league
from footy.data import teams
from footy.players import data as pdata
from footy.players import evaluate
from footy.players.model import (
    OWN_GOAL_SHARE, PlayerParams, per90_rates, player_positions, position_group,
    position_priors, predict_team_players,
)

POSITIONS = ["GK", "DC", "DC", "DL", "DR", "MC", "MC", "AML", "AMR", "FW", "FW"]


def synthetic(n_matches=30, seed=0):
    """Two teams playing weekly. Team A's striker A9 has true xG 0.6/90; A10 is a
    regular starter injured for the last 5 matches; A12 is a super-sub."""
    rng = np.random.default_rng(seed)
    rows, dates = [], [pd.Timestamp("2024-08-10") + pd.Timedelta(days=7 * i) for i in range(n_matches)]
    for i, d in enumerate(dates):
        for team in ("A", "B"):
            squad = [f"{team}{k}" for k in range(11)]
            if team == "A" and i >= n_matches - 5:
                squad[10] = "A11"  # A10 injured, A11 replaces him
            for k, pid in enumerate(squad):
                xg_rate = 0.6 if pid == "A9" else (0.3 if k >= 9 else 0.05)
                mins = 90 if k > 0 else 90
                xg = xg_rate * mins / 90
                rows.append(dict(date=d, team=team, player=pid, player_id=hash(pid) % 10**6,
                                 position=POSITIONS[k], minutes=mins, xg=xg,
                                 shots=rng.poisson(xg * 6), goals=rng.poisson(xg),
                                 xa=0.1 if k >= 5 else 0.02, assists=0, key_passes=1))
            if team == "A":
                rows.append(dict(date=d, team="A", player="A12", player_id=hash("A12") % 10**6,
                                 position="Sub", minutes=20, xg=0.1, shots=1,
                                 goals=rng.poisson(0.1), xa=0.0, assists=0, key_passes=0))
    return pd.DataFrame(rows), dates


def _pid(name):
    return hash(name) % 10**6


def test_position_groups():
    assert [position_group(p) for p in ["GK", "DC", "DMC", "AML", "FW", "Sub", None]] == \
        ["GK", "DEF", "MID", "MID", "FWD", "UNK", "UNK"]


def test_team_total_matches_match_model_lambda():
    pm, dates = synthetic()
    asof = dates[-1] + pd.Timedelta(days=7)
    out = predict_team_players(pm, "A", asof, dates, team_lambda=1.8)
    assert out["exp_goals"].sum() == pytest.approx(1.8 * (1 - OWN_GOAL_SHARE), rel=1e-6)
    assert out["p_score"].between(0, 1).all()


def test_availability_tracks_injury_and_subs():
    pm, dates = synthetic()
    asof = dates[-1] + pd.Timedelta(days=7)
    out = predict_team_players(pm, "A", asof, dates, 1.5).set_index("player")
    assert out.loc["A9", "p_start"] > 0.95
    assert out.loc["A11", "p_start"] > 0.65           # the replacement, 5 straight starts
    assert out.loc["A10", "p_start"] < 0.05           # missed 5 in a row: treated as out
    assert out.loc["A12", "p_start"] == 0 and out.loc["A12", "p_appear"] > 0.95
    assert out.loc["A12", "exp_minutes"] == pytest.approx(20, abs=1)
    assert out.index[0] == "A9"                       # striker is the likeliest scorer


def test_stronger_match_lifts_every_player():
    pm, dates = synthetic()
    asof = dates[-1] + pd.Timedelta(days=7)
    low = predict_team_players(pm, "A", asof, dates, 0.8).set_index("player_id")
    high = predict_team_players(pm, "A", asof, dates, 2.4).set_index("player_id")
    assert (high["p_score"] > low["p_score"]).all()
    assert (high["exp_shots"] > low["exp_shots"]).all()


def test_shrinkage_pulls_small_samples_to_position_prior():
    pm, _ = synthetic()
    fluke = pd.DataFrame([dict(date=pd.Timestamp("2025-01-01"), team="A", player="kid",
                               player_id=999, position="FW", minutes=30, xg=1.5, shots=3,
                               goals=1, xa=0, assists=0, key_passes=0)])
    pm2 = pd.concat([pm, fluke], ignore_index=True)
    positions = player_positions(pm2)
    priors = position_priors(pm2, positions)
    rates = per90_rates(pm2, positions, priors, PlayerParams())
    raw = 1.5 / 30 * 90
    assert rates.loc[999, "xg"] < raw / 5           # 4.5 xG/90 raw, heavily shrunk
    assert rates.loc[999, "xg"] > priors.loc["FWD", "xg"]


def test_no_leakage_from_matches_on_or_after_asof():
    pm, dates = synthetic()
    asof = dates[20]
    base = predict_team_players(pm, "A", asof, dates, 1.5)
    tampered = pm.copy()
    future = tampered["date"] >= asof
    tampered.loc[future, ["xg", "goals", "minutes"]] = [5.0, 5, 90]
    after = predict_team_players(tampered, "A", asof, dates, 1.5)
    pd.testing.assert_frame_equal(base, after)


def test_conform_maps_teams_and_dates():
    raw = pd.DataFrame({
        "game_id": [1, 1], "team": ["Manchester United", "Fulham"], "player_id": [7, 8],
        "player": ["P7", "P8"], "position": ["FW", "Sub"], "minutes": [90, 15],
        "goals": [1, 0], "own_goals": [0, 0], "shots": [3, 0], "xg": [0.8, 0.0],
        "xa": [0.1, 0.0], "assists": [0, 0], "key_passes": [2, 0], "season_id": [2024, 2024],
    }).set_index(["team", "player"])  # soccerdata returns these as the index
    schedule = pd.DataFrame({"game_id": [1], "date": [pd.Timestamp("2024-08-16 19:00")]})
    aliases = {"understat": {"EPL": {"Manchester United": "Man United", "Fulham": "Fulham"}}}
    out = pdata.conform(raw, schedule, get_league("EPL"), aliases)
    assert set(out["team"]) == {"Man United", "Fulham"}
    assert (out["date"] == pd.Timestamp("2024-08-16")).all()
    with pytest.raises(teams.UnmappedTeamError):
        pdata.conform(raw, schedule, get_league("EPL"), {"understat": {"EPL": {}}})


def test_backtest_runs_and_scores_all_players():
    pm, dates = synthetic(n_matches=40, seed=3)
    lambdas = pd.DataFrame({"date": dates * 2, "team": ["A"] * 40 + ["B"] * 40, "lam": 1.4})
    res, summary = evaluate.backtest(pm, lambdas, {"A": dates, "B": dates},
                                     dates[30], dates[-1])
    assert len(res) == len(pm[(pm["date"] >= dates[30]) & (pm["minutes"] > 0)])
    assert set(summary.index) == {"player_model", "baseline"}
    assert 0 < summary.loc["player_model", "listed_share"] <= 1
