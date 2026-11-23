import pandas as pd
import pytest

from footy.config import get_league
from footy.data import build, schema, teams, understat

EPL = get_league("EPL")


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "INTERIM_DIR", tmp_path / "interim")
    monkeypatch.setattr(build, "PROCESSED_DIR", tmp_path / "processed")
    for sub in ("football_data", "understat"):
        (tmp_path / "interim" / sub).mkdir(parents=True)
    fd = schema.conform(pd.DataFrame({
        "match_id": ["a", "b"], "league": "EPL", "season": 2024,
        "date": pd.to_datetime(["2024-08-16", "2024-08-17"]),
        "home_team": ["Man United", "Arsenal"], "away_team": ["Fulham", "Wolves"],
        "home_goals": [1, 2], "away_goals": [0, 0], "result": ["H", "H"],
    }))
    fd.to_parquet(tmp_path / "interim" / "football_data" / "EPL.parquet")
    us = pd.DataFrame({c: [pd.NA] for c in understat.UNDERSTAT_COLUMNS}).assign(
        league="EPL", season=2024, understat_id=1, date=pd.Timestamp("2024-08-16 20:00"),
        home_team="Manchester United", away_team="Fulham", home_goals=1, away_goals=0,
        home_xg=2.43, away_xg=0.43)
    us.to_parquet(tmp_path / "interim" / "understat" / "EPL.parquet")
    monkeypatch.setattr(teams, "load_aliases", lambda: {
        "understat": {"EPL": {"Manchester United": "Man United", "Fulham": "Fulham"}}})
    return tmp_path


def test_build_joins_xg_and_keeps_unmatched(dirs):
    df = build.build_matches(EPL)
    assert len(df) == 2
    mu = df[df.home_team == "Man United"].iloc[0]
    assert mu["home_xg"] == pytest.approx(2.43)
    assert df[df.home_team == "Arsenal"]["home_xg"].isna().all()
    assert df["exclude_reason"].isna().all()


def test_build_without_understat_still_works(dirs):
    (dirs / "interim" / "understat" / "EPL.parquet").unlink()
    df = build.build_matches(EPL)
    assert "home_xg" in df.columns and df["home_xg"].isna().all()


def test_build_rejects_far_apart_dates(dirs):
    path = dirs / "interim" / "understat" / "EPL.parquet"
    us = pd.read_parquet(path).assign(date=pd.Timestamp("2024-12-01"))
    us.to_parquet(path)
    with pytest.raises(build.BuildError, match="differ by"):
        build.build_matches(EPL)


def test_exclusion_must_match_exactly_one_row(dirs, tmp_path):
    bad = tmp_path / "ex.yaml"
    bad.write_text("- {league: EPL, season: 2024, home_team: Nobody, away_team: Fulham, reason: x}")
    df = pd.read_parquet(dirs / "interim" / "football_data" / "EPL.parquet")
    with pytest.raises(build.BuildError, match="matched 0 rows"):
        build._apply_exclusions(df, EPL, bad)
