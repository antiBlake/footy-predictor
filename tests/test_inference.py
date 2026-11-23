from datetime import date

import pandas as pd

from footy.config import get_league
from footy.data import football_data
from footy.inference import fixtures as fx_mod
from footy.inference.fixtures import upcoming_fixtures

EPL = get_league("EPL")


def _history():
    return pd.DataFrame({"season": [2026], "home_team": ["Arsenal"], "away_team": ["Chelsea"],
                         "date": [pd.Timestamp("2026-10-01")]})


def _fd():
    return pd.DataFrame({
        "league": "EPL", "season": 2026, "kickoff": "15:00",
        "date": pd.to_datetime(["2026-11-20", "2026-11-21", "2026-11-21"]),
        "home_team": ["Leeds", "Man City", "Arsenal"],
        "away_team": ["Hull", "Fulham", "Chelsea"],   # Arsenal v Chelsea already played
        "match_id": ["x", "y", "z"], "mkt_p_home": 0.5,
    })


def _of(monkeypatch, rows):
    of = pd.DataFrame(rows, columns=["season", "date", "home_team", "away_team", "kickoff"])
    of["date"] = pd.to_datetime(of["date"])
    monkeypatch.setattr(fx_mod, "_openfootball_fixtures", lambda league: of)


def test_sources_merge_and_dedupe(monkeypatch):
    _of(monkeypatch, [
        (2026, "2026-11-22", "Leeds", "Hull", None),          # FD has it on 20th: wrong date -> drop
        (2026, "2026-11-21", "Man City", "Fulham", "12:30"),  # duplicate of FD -> drop
        (2026, "2026-11-28", "Fulham", "Leeds", "15:00"),     # only in openfootball -> keep
        (2026, "2026-12-30", "Hull", "Man City", None),       # outside window -> drop
    ])
    out = upcoming_fixtures(EPL, _history(), _fd(), days=7, today=date(2026, 11, 21))
    pairs = list(zip(out["home_team"], out["away_team"], out["fixture_source"]))
    assert pairs == [("Man City", "Fulham", "football_data"),
                     ("Fulham", "Leeds", "openfootball")]
    assert out["match_id"].notna().all()


def test_stale_openfootball_fixture_warns(monkeypatch, caplog):
    _of(monkeypatch, [(2026, "2026-11-10", "Spurs", "Wolves", None)])
    upcoming_fixtures(EPL, _history(), _fd(), days=7, today=date(2026, 11, 21))
    assert "no result" in caplog.text


def test_fetch_fixtures_parses_canonical_schema(tmp_path, monkeypatch):
    csv = tmp_path / "fixtures.csv"
    csv.write_text(
        "﻿Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5\n"
        "E0,21/11/2026,12:30,Man City,Fulham,1.4,5,7,1.42,4.9,7.2,1.5,2.6\n"
        "B1,20/11/2026,19:45,Cercle Brugge,Westerlo,1.9,3.6,3.4,1.9,3.6,3.4,1.8,2\n",
        encoding="utf-8")
    monkeypatch.setattr(football_data, "fetch_cached", lambda *a, **k: csv)
    df = football_data.fetch_fixtures([EPL])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["home_team"] == "Man City" and row["season"] == 2026
    assert row["odds_avg_home"] == 1.42 and pd.isna(row["home_goals"])
    assert row["match_id"] == football_data.make_match_id(
        "EPL", pd.Series([pd.Timestamp("2026-11-21")]), pd.Series(["Man City"]),
        pd.Series(["Fulham"])).iloc[0]
