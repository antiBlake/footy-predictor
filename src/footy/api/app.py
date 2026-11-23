"""FastAPI app.

Run:  footy serve            (or: uvicorn footy.api.app:app)
Docs: http://localhost:8000/docs

Reads are served from the latest precomputed predictions (fast, deterministic).
POST /predict computes a single fixture on demand. POST /admin/refresh
recomputes the upcoming list; it is only enabled when FOOTY_API_KEY is set,
and needs that key in the X-API-Key header.
"""

from __future__ import annotations

import datetime as dt
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request

from footy import __version__
from footy.api import schemas
from footy.api.service import ServiceState, UnknownTeamError, to_schema
from footy.config import get_league, load_leagues


def _svc(request: Request) -> ServiceState:
    return request.app.state.svc


def _require_admin(x_api_key: str | None = Header(None)) -> None:
    expected = os.environ.get("FOOTY_API_KEY")
    if not expected:
        raise HTTPException(403, "admin endpoints disabled: set FOOTY_API_KEY to enable")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(401, "invalid or missing X-API-Key")


def _league_or_404(key: str):
    try:
        return get_league(key)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from None


def create_app(state: ServiceState | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.svc = state or ServiceState()
        app.state.svc.reload()
        yield

    app = FastAPI(title="footy", version=__version__, lifespan=lifespan,
                  description="Match outcome, goals and scoreline probabilities for the "
                              "Big 5 European leagues.")

    @app.get("/health", response_model=schemas.Health)
    def health(svc: ServiceState = Depends(_svc)):
        return svc.health()

    @app.get("/leagues", response_model=list[schemas.League])
    def leagues():
        return [schemas.League(key=lg.key, name=lg.name, country=lg.country)
                for lg in load_leagues().values()]

    @app.get("/leagues/{league}/teams", response_model=list[str])
    def league_teams(league: str, season: int | None = None, svc: ServiceState = Depends(_svc)):
        return svc.teams(_league_or_404(league), season)

    @app.get("/predictions/upcoming", response_model=schemas.PredictionList)
    def upcoming(
        league: str | None = Query(None, examples=["EPL"]),
        date_from: dt.date | None = None,
        date_to: dt.date | None = None,
        svc: ServiceState = Depends(_svc),
    ):
        if league:
            _league_or_404(league)
        df = svc.upcoming(league, date_from, date_to)
        preds = [to_schema(r) for _, r in df.iterrows()]
        return schemas.PredictionList(generated_at=svc.generated_at, count=len(preds),
                                      predictions=preds)

    @app.get("/predictions/{match_id}", response_model=schemas.MatchPrediction)
    def by_id(match_id: str, svc: ServiceState = Depends(_svc)):
        row = svc.by_id(match_id)
        if row is None:
            raise HTTPException(404, f"no upcoming prediction with id {match_id}")
        return to_schema(row)

    @app.get("/predictions/{match_id}/players", response_model=schemas.PlayerList)
    def players(match_id: str, svc: ServiceState = Depends(_svc)):
        df = svc.players_for(match_id)
        if df is None:
            raise HTTPException(404, "player predictions unavailable: run `footy players collect` "
                                     "(needs Understat) then `footy predict`")
        if df.empty:
            raise HTTPException(404, f"no player predictions for {match_id}")
        cols = list(schemas.PlayerPrediction.model_fields)
        rows = df.sort_values("p_score", ascending=False)[cols].to_dict("records")
        return schemas.PlayerList(match_id=match_id, count=len(rows),
                                  players=[schemas.PlayerPrediction(**r) for r in rows])

    @app.post("/predict", response_model=schemas.MatchPrediction)
    def predict(req: schemas.FixtureRequest, svc: ServiceState = Depends(_svc)):
        _league_or_404(req.league)
        try:
            return to_schema(svc.predict_fixture(req))
        except UnknownTeamError as exc:
            raise HTTPException(422, {"error": str(exc), "suggestions": exc.suggestions}) from None
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from None

    @app.post("/admin/refresh", dependencies=[Depends(_require_admin)])
    def refresh(days: int = Query(7, ge=1, le=60), svc: ServiceState = Depends(_svc)):
        try:
            n = svc.refresh(days)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None
        return {"predictions": n, "generated_at": svc.generated_at}

    return app


app = create_app()
