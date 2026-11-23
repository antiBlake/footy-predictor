"""Request/response models. These are the API contract: change with care."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Probabilities(BaseModel):
    home: float = Field(ge=0, le=1)
    draw: float = Field(ge=0, le=1)
    away: float = Field(ge=0, le=1)


class FairOdds(BaseModel):
    home: float
    draw: float
    away: float


class Goals(BaseModel):
    exp_home: float = Field(description="Expected home goals (Poisson mean)")
    exp_away: float
    p_over25: float = Field(ge=0, le=1)
    p_btts: float = Field(ge=0, le=1, description="Both teams to score")


class Scoreline(BaseModel):
    home: int
    away: int
    p: float


class MatchPrediction(BaseModel):
    match_id: str
    league: str
    date: dt.date
    kickoff: str | None = None
    home_team: str
    away_team: str
    model: str = Field(description="ensemble[with_market] when odds were available, else ensemble[no_market]")
    fixture_source: str | None = None
    probabilities: Probabilities
    fair_odds: FairOdds
    market: Probabilities | None = Field(None, description="Bookmaker-implied, margin removed")
    goals: Goals
    likely_scores: list[Scoreline]


class PlayerPrediction(BaseModel):
    player: str
    player_id: int
    team: str
    side: Literal["home", "away"]
    position: str
    p_start: float
    p_appear: float
    exp_minutes: float
    exp_goals: float
    p_score: float = Field(description="Anytime goalscorer probability")
    exp_assists: float
    p_assist: float
    exp_shots: float


class PlayerList(BaseModel):
    match_id: str
    count: int
    players: list[PlayerPrediction]


class PredictionList(BaseModel):
    generated_at: dt.datetime | None
    count: int
    predictions: list[MatchPrediction]


class Health(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    model_trained_at: str | None = None
    model_seasons: list[int] | None = None
    model_last_match_date: dt.date | None = None
    predictions_generated_at: dt.datetime | None = None
    predictions_count: int = 0
    issues: list[str] = []


class League(BaseModel):
    key: str
    name: str
    country: str


class FixtureRequest(BaseModel):
    league: str = Field(examples=["EPL"])
    home_team: str = Field(examples=["Arsenal"], description="Canonical name or any known alias")
    away_team: str = Field(examples=["Man City"])
    date: dt.date
    kickoff: str | None = Field(None, examples=["15:00"])
    odds_home: float | None = Field(None, gt=1, description="Decimal odds; give all three or none")
    odds_draw: float | None = Field(None, gt=1)
    odds_away: float | None = Field(None, gt=1)

    @model_validator(mode="after")
    def _odds_all_or_none(self) -> FixtureRequest:
        given = [o is not None for o in (self.odds_home, self.odds_draw, self.odds_away)]
        if any(given) and not all(given):
            raise ValueError("give odds_home, odds_draw and odds_away together, or none")
        if self.home_team == self.away_team:
            raise ValueError("home_team and away_team must differ")
        return self
