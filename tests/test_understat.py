import pandas as pd
import pytest

from footy.config import get_league
from footy.data import understat


def test_purge_removes_only_non_json(tmp_path):
    (tmp_path / "leagues.json").write_text("<!DOCTYPE html><html>We will come back soon")
    (tmp_path / "good.json").write_text('  {"stat": []}')
    (tmp_path / "empty.json").write_text("")
    assert understat.purge_poisoned_cache(tmp_path) == 2
    assert [p.name for p in tmp_path.iterdir()] == ["good.json"]


def _soccerdata_like_frame():
    row = {c: 1 for c in understat.COLUMN_MAP}
    row.update(game_id=26602, date="2024-08-16 19:00:00", home_team="Manchester United",
               away_team="Fulham", home_goals=1, away_goals=0, home_xg=2.43, away_xg=0.43,
               season_id=2024)
    unplayed = {**row, "game_id": 1, "home_goals": None, "away_goals": None}
    return pd.DataFrame([row, unplayed]).set_index(["home_team"])  # soccerdata returns an index


def test_conform_maps_columns_and_drops_unplayed():
    out = understat.conform(_soccerdata_like_frame(), get_league("EPL"))
    assert list(out.columns) == understat.UNDERSTAT_COLUMNS
    assert len(out) == 1
    assert out.loc[0, "season"] == 2024 and out.loc[0, "home_xg"] == pytest.approx(2.43)


def test_conform_fails_loudly_on_api_change():
    with pytest.raises(ValueError, match="missing columns"):
        understat.conform(pd.DataFrame({"game_id": [1]}), get_league("EPL"))
