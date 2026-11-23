import datetime as dt
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from footy.api import schemas
from footy.api.app import create_app
from footy.api.service import ServiceState, UnknownTeamError
from footy.config import get_league
from footy.inference.predict import OUTPUT_COLUMNS


def _row(match_id, home, away, date, market=True):
    scores = [{"home": 1, "away": 1, "p": 0.12}, {"home": 1, "away": 0, "p": 0.1}]
    return {
        "league": "EPL", "date": pd.Timestamp(date), "kickoff": "15:00",
        "home_team": home, "away_team": away,
        "model": "ensemble[with_market]" if market else "ensemble[no_market]",
        "fixture_source": "football_data" if market else "openfootball",
        "p_home": 0.5, "p_draw": 0.25, "p_away": 0.25,
        "fair_odds_home": 2.0, "fair_odds_draw": 4.0, "fair_odds_away": 4.0,
        "mkt_p_home": 0.48 if market else float("nan"), "mkt_p_draw": 0.27 if market else float("nan"),
        "mkt_p_away": 0.25 if market else float("nan"),
        "exp_home_goals": 1.6, "exp_away_goals": 1.1, "p_over25": 0.52, "p_btts": 0.5,
        "top_scores": "1-1 12%", "top_scores_json": json.dumps(scores), "match_id": match_id,
    }


class StubState(ServiceState):
    """No files, no models: fixed predictions and a tiny team list."""

    def __init__(self, trained_last_match="2026-11-08"):
        super().__init__()
        self._last = trained_last_match
        self.requests = []

    def reload(self):
        self.bundle = {"stub": True}
        self.meta = {"trained_at": "2026-11-21T00:00:00+00:00", "seasons": [2016, 2026],
                     "last_match_date": self._last}
        self.predictions = pd.DataFrame([
            _row("a1", "Man City", "Fulham", "2026-11-21"),
            _row("b2", "Arsenal", "Chelsea", "2026-11-28", market=False),
        ], columns=OUTPUT_COLUMNS)
        self.generated_at = dt.datetime.now(dt.timezone.utc)

    def resolve_team(self, name, league):
        known = {"arsenal": "Arsenal", "chelsea": "Chelsea", "manchester united fc": "Man United"}
        if name.lower() in known:
            return known[name.lower()]
        raise UnknownTeamError(name, league.key, ["Arsenal"])

    def predict_fixture(self, req):
        self.requests.append(req)
        league = get_league(req.league)
        home = self.resolve_team(req.home_team, league)
        away = self.resolve_team(req.away_team, league)
        return pd.Series(_row("x9", home, away, req.date, market=req.odds_home is not None))

    def teams(self, league, season=None):
        return ["Arsenal", "Chelsea"]


@pytest.fixture
def client():
    with TestClient(create_app(StubState())) as c:
        yield c


def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["predictions_count"] == 2


def test_health_degraded_when_models_stale(monkeypatch):
    with TestClient(create_app(StubState(trained_last_match="2026-01-01"))) as c:
        body = c.get("/health").json()
    assert body["status"] == "degraded" and "footy daily" in body["issues"][0]


def test_upcoming_filters_and_shape(client):
    body = client.get("/predictions/upcoming", params={"date_to": "2026-11-21"}).json()
    assert body["count"] == 1
    p = body["predictions"][0]
    assert p["home_team"] == "Man City" and p["market"]["home"] == 0.48
    assert p["likely_scores"][0] == {"home": 1, "away": 1, "p": 0.12}
    no_mkt = client.get("/predictions/b2").json()
    assert no_mkt["market"] is None and no_mkt["model"] == "ensemble[no_market]"


def test_unknown_league_and_id_404(client):
    assert client.get("/predictions/upcoming", params={"league": "XYZ"}).status_code == 404
    assert client.get("/predictions/nope").status_code == 404
    assert client.get("/leagues/XYZ/teams").status_code == 404


def test_predict_resolves_aliases(client):
    r = client.post("/predict", json={"league": "EPL", "home_team": "Manchester United FC",
                                      "away_team": "chelsea", "date": "2026-11-29",
                                      "odds_home": 2.1, "odds_draw": 3.4, "odds_away": 3.6})
    assert r.status_code == 200
    assert r.json()["home_team"] == "Man United" and r.json()["model"] == "ensemble[with_market]"


def test_predict_validation(client):
    base = {"league": "EPL", "home_team": "Arsenal", "away_team": "Chelsea", "date": "2026-11-29"}
    assert client.post("/predict", json={**base, "odds_home": 2.0}).status_code == 422
    assert client.post("/predict", json={**base, "away_team": "Arsenal"}).status_code == 422
    assert client.post("/predict", json={**base, "odds_home": 0.9, "odds_draw": 3,
                                         "odds_away": 3}).status_code == 422
    r = client.post("/predict", json={**base, "home_team": "Arsenol"})
    assert r.status_code == 422 and r.json()["detail"]["suggestions"] == ["Arsenal"]


def test_admin_refresh_auth(client, monkeypatch):
    monkeypatch.delenv("FOOTY_API_KEY", raising=False)
    assert client.post("/admin/refresh").status_code == 403
    monkeypatch.setenv("FOOTY_API_KEY", "secret")
    assert client.post("/admin/refresh").status_code == 401
    assert client.post("/admin/refresh", headers={"X-API-Key": "nope"}).status_code == 401


def test_fixture_request_schema():
    with pytest.raises(ValueError):
        schemas.FixtureRequest(league="EPL", home_team="A", away_team="B",
                               date=dt.date(2026, 1, 1), odds_home=2.0)


def test_players_endpoint_unavailable_without_data(client):
    r = client.get("/predictions/a1/players")
    assert r.status_code == 404 and "footy players collect" in r.json()["detail"]


def test_players_endpoint_lists_by_scoring_chance():
    state = StubState()
    with TestClient(create_app(state)) as c:
        base = dict(team="Man City", side="home", position="FWD", p_start=0.9, p_appear=0.97,
                    exp_minutes=80.0, exp_assists=0.1, p_assist=0.1, exp_shots=2.0, match_id="a1")
        state.players = pd.DataFrame([
            {**base, "player": "Winger", "player_id": 2, "exp_goals": 0.2, "p_score": 0.18},
            {**base, "player": "Striker", "player_id": 1, "exp_goals": 0.6, "p_score": 0.45},
        ])
        body = c.get("/predictions/a1/players").json()
        assert [p["player"] for p in body["players"]] == ["Striker", "Winger"]
        assert c.get("/predictions/zzz/players").status_code == 404
