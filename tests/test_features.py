import numpy as np
import pandas as pd
import pytest

from footy.config import get_league
from footy.features import elo, form, market
from footy.features.pipeline import build_features, feature_columns

EPL = get_league("EPL")


def _league(seasons=(2018, 2019), teams=("A", "B", "C", "D"), seed=0, relegate=None):
    """Synthetic double round-robin. relegate={season: (out, in)} swaps a team."""
    rng = np.random.default_rng(seed)
    rows, teams = [], list(teams)
    for season in seasons:
        if relegate and season in relegate:
            out, new = relegate[season]
            teams = [new if t == out else t for t in teams]
        day = pd.Timestamp(season, 8, 10)
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                hg, ag = rng.poisson(1.5), rng.poisson(1.1)
                rows.append(dict(
                    match_id=f"{season}{h}{a}", league="EPL", season=season, date=day,
                    kickoff="15:00", home_team=h, away_team=a, home_goals=hg, away_goals=ag,
                    result="H" if hg > ag else "D" if hg == ag else "A",
                    home_shots=hg * 4 + 5, away_shots=ag * 4 + 4,
                    odds_avg_home=2.0, odds_avg_draw=3.5, odds_avg_away=4.0,
                    odds_b365_home=2.0, odds_b365_draw=3.5, odds_b365_away=4.0,
                ))
                day += pd.Timedelta(days=7)
    df = pd.DataFrame(rows)
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    return df


def test_no_leakage_changing_a_result_only_affects_later_matches():
    df = _league()
    base = build_features(df, EPL)
    i = 7
    target_date = df.loc[i, "date"]
    changed = df.copy()
    changed.loc[i, ["home_goals", "away_goals"]] = [9, 0]
    changed.loc[i, "home_shots"] = 99
    after = build_features(changed, EPL)
    feats = [c for g in feature_columns(base).values() for c in g]
    up_to = base["date"] <= target_date
    pd.testing.assert_frame_equal(base.loc[up_to, feats], after.loc[up_to, feats])
    assert not base.loc[~up_to, feats].equals(after.loc[~up_to, feats])  # later rows do change


def test_future_fixtures_get_features_without_targets():
    df = _league(seasons=(2018,))
    fixture = df.iloc[[0]].assign(date=pd.Timestamp("2019-05-30"), home_goals=pd.NA,
                                  away_goals=pd.NA, result=pd.NA, match_id="future")
    out = build_features(df, EPL, fixtures=fixture)
    row = out[out["match_id"] == "future"].iloc[0]
    assert not row["is_usable"] and pd.isna(row["home_goals"])
    assert row["h_n_prev"] == 6 and not np.isnan(row["h_goals_for_ewms"])
    assert not np.isnan(row["elo_home"])


def test_excluded_match_does_not_feed_form_or_elo():
    df = _league(seasons=(2018,))
    a = build_features(df, EPL)
    df2 = df.copy()
    df2["exclude_reason"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df2.loc[0, "exclude_reason"] = "abandoned"
    b = build_features(df2, EPL)
    second_home = a.index[(a["home_team"] == df.loc[0, "home_team"])][1]
    assert b.loc[second_home, "h_n_prev"] == a.loc[second_home, "h_n_prev"] - 1
    assert not b.loc[0, "is_usable"]


def test_form_shift_first_match_has_no_history():
    out = form.team_form(_league(seasons=(2018,)).astype({"home_goals": float, "away_goals": float}))
    assert np.isnan(out.loc[0, "h_goals_for_ewms"]) and out.loc[0, "h_n_prev"] == 0


def test_promoted_team_resets_form_and_starts_at_relegated_rating():
    df = _league(seasons=(2018, 2019), relegate={2019: ("D", "E")})
    out = build_features(df, EPL)
    e_first = out[(out["season"] == 2019) & ((out["home_team"] == "E") | (out["away_team"] == "E"))].iloc[0]
    side = "h" if e_first["home_team"] == "E" else "a"
    assert e_first[f"{side}_is_promoted"] == 1.0 and e_first[f"{side}_n_prev"] == 0
    # E starts where relegated D ended (after regression to the mean)
    last18 = out[out["season"] == 2018]
    assert e_first[f"elo_{'home' if side == 'h' else 'away'}"] != elo.EloParams().initial
    assert out.loc[out["season"] == 2018, "h_is_promoted"].isna().all()  # unknown in first season
    assert len(last18) == 12


def test_elo_zero_sum_and_home_advantage():
    df = _league(seasons=(2018,)).astype({"home_goals": float, "away_goals": float})
    r = elo.compute_elo(df)
    assert r.loc[0, "elo_home"] == r.loc[0, "elo_away"] == 1500
    assert r.loc[0, "elo_p_home"] > 0.5  # home advantage
    assert elo.goal_multiplier(1) == 1 and elo.goal_multiplier(2) == 1.5
    assert elo.goal_multiplier(4) == pytest.approx(15 / 8)


def test_market_probabilities_sum_to_one_and_fallback():
    df = pd.DataFrame({
        "odds_avg_home": [2.0, np.nan], "odds_avg_draw": [3.5, np.nan], "odds_avg_away": [4.0, np.nan],
        "odds_b365_home": [2.1, 1.5], "odds_b365_draw": [3.4, 4.0], "odds_b365_away": [3.8, 7.0],
        "odds_avg_over25": [1.9, np.nan], "odds_avg_under25": [1.9, np.nan],
    })
    m = market.market_features(df)
    total = m[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].sum(axis=1)
    assert np.allclose(total, 1.0)
    assert m.loc[1, "mkt_p_home"] > 0.6  # from Bet365 fallback
    assert m.loc[0, "mkt_p_over25"] == pytest.approx(0.5)


def test_crowd_flag():
    from footy.features.context import crowd_restricted
    d = pd.DataFrame({"date": pd.to_datetime(["2020-06-16", "2020-06-17", "2021-05-16", "2021-05-18"])})
    assert crowd_restricted(d, "EPL").tolist() == [0, 1, 1, 0]
