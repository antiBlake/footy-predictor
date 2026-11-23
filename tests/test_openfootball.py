from footy.config import get_league
from footy.data.openfootball import parse_season

PAYLOAD = {"matches": [
    {"date": "2019-08-09", "time": "20:00", "team1": "Liverpool FC", "team2": "Norwich City FC",
     "score": {"ht": [4, 0], "ft": [4, 1]}},
    {"date": "2019-08-10", "team1": "A FC", "team2": "B FC", "score": [2, 2]},
    {"date": "2019-07-01", "team1": "Arsenal FC", "team2": "Norwich City FC",
     "score": {"ft": [4, 0]}},                                     # wrong year upstream
    {"date": "2020-03-14", "team1": "C FC", "team2": "D FC", "status": "cancelled"},
    {"date": "2020-05-20", "team1": "E FC", "team2": "F FC"},      # not played yet
]}


def test_parse_handles_all_score_shapes_and_statuses():
    df = parse_season(PAYLOAD, get_league("EPL"), 2019)
    assert df["status"].tolist() == ["played", "played", "played", "cancelled", "scheduled"]
    assert df.loc[0, "ht_home_goals"] == 4
    assert df.loc[1, "home_goals"] == 2 and df.loc[1, "ht_home_goals"] is not None
    assert df["home_goals"].isna().sum() == 2


def test_wrong_year_is_corrected():
    df = parse_season(PAYLOAD, get_league("EPL"), 2019)
    assert str(df.loc[2, "date"].date()) == "2020-07-01"
