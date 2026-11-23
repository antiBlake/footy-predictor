from pathlib import Path

import pandas as pd
import pytest

from footy.config import current_season, get_league, season_code, season_label
from footy.data import schema
from footy.data.football_data import parse_season, read_raw_csv

FIXTURE = Path(__file__).parent / "fixtures" / "E0_1819_messy.csv"


@pytest.fixture
def parsed() -> pd.DataFrame:
    raw = read_raw_csv(FIXTURE)
    return schema.validate(schema.conform(parse_season(raw, get_league("EPL"), 2018)))


def test_season_helpers():
    assert season_code(2024) == "2425"
    assert season_code(1999) == "9900"
    assert season_label(2024) == "2024-25"
    assert current_season(pd.Timestamp("2026-11-20").date()) == 2026
    assert current_season(pd.Timestamp("2027-03-01").date()) == 2026


def test_ragged_and_blank_rows_handled(parsed):
    # 5 data lines: 1 blank, 1 unplayed -> 3 matches
    assert len(parsed) == 3


def test_mixed_date_formats(parsed):
    assert parsed["date"].dt.year.unique().tolist() == [2018]
    assert parsed["date"].min() == pd.Timestamp("2018-08-11")


def test_result_inconsistency_is_fixed(parsed):
    chelsea = parsed[parsed["home_team"] == "Chelsea"].iloc[0]
    assert chelsea["result"] == "D"


def test_legacy_avg_odds_fallback_and_bad_odds_masked(parsed):
    mu = parsed[parsed["home_team"] == "Man United"].iloc[0]
    assert mu["odds_avg_home"] == pytest.approx(1.56)
    chelsea = parsed[parsed["home_team"] == "Chelsea"].iloc[0]
    assert pd.isna(chelsea["odds_avg_home"])  # 1.0 is implausible


def test_schema_columns_and_unique_ids(parsed):
    assert list(parsed.columns) == schema.ALL_COLUMNS
    assert parsed["match_id"].is_unique


def test_missing_required_column_raises():
    with pytest.raises(schema.SchemaError):
        schema.conform(pd.DataFrame({"league": ["EPL"]}))
