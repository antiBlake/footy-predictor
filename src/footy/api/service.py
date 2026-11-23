"""Service state behind the API: production models, latest predictions, team names.

Kept separate from the FastAPI app so it can be tested and reused (e.g. by an
agent) without HTTP.
"""

from __future__ import annotations

import datetime as dt
import difflib
import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from footy.api import schemas
from footy.config import League, current_season, get_league, load_leagues
from footy.data import teams
from footy.data.football_data import make_match_id
from footy.inference import predict as predict_mod
from footy.inference.production import PRODUCTION_DIR, load_production

log = logging.getLogger(__name__)

STALE_PREDICTIONS_HOURS = 36
STALE_RESULTS_DAYS = 16  # longer than an international break


class UnknownTeamError(ValueError):
    def __init__(self, name: str, league: str, suggestions: list[str]):
        self.name, self.league, self.suggestions = name, league, suggestions
        super().__init__(f"Unknown team {name!r} in {league}")


class ServiceState:
    def __init__(self, predictions_path: Path = predict_mod.LATEST_PATH,
                 models_dir: Path = PRODUCTION_DIR) -> None:
        self.predictions_path = predictions_path
        self.models_dir = models_dir
        self.bundle: dict | None = None
        self.meta: dict[str, Any] = {}
        self.predictions = pd.DataFrame(columns=predict_mod.OUTPUT_COLUMNS)
        self.generated_at: dt.datetime | None = None
        self.players = pd.DataFrame()
        self.load_error: str | None = None
        self._histories: dict[str, pd.DataFrame] = {}
        self._refresh_lock = threading.Lock()

    # -- loading ----------------------------------------------------------------------
    def reload(self) -> None:
        self._histories.clear()
        try:
            self.bundle, self.meta = load_production(self.models_dir)
            self.load_error = None
        except FileNotFoundError as exc:
            self.bundle, self.meta, self.load_error = None, {}, str(exc)
            log.error("%s", exc)
        if self.predictions_path.exists():
            self.predictions = pd.read_parquet(self.predictions_path)
            self.generated_at = dt.datetime.fromtimestamp(
                self.predictions_path.stat().st_mtime, tz=dt.timezone.utc)
        players_path = self.predictions_path.with_name(predict_mod.PLAYERS_LATEST_PATH.name)
        self.players = pd.read_parquet(players_path) if players_path.exists() else pd.DataFrame()
        log.info("Service loaded: %d predictions, models trained %s",
                 len(self.predictions), self.meta.get("trained_at"))

    def history(self, league: League) -> pd.DataFrame:
        if league.key not in self._histories:
            self._histories[league.key] = predict_mod.load_history(league)
        return self._histories[league.key]

    # -- health -----------------------------------------------------------------------
    def health(self) -> schemas.Health:
        issues = []
        if self.bundle is None:
            return schemas.Health(status="unavailable", issues=[self.load_error or "no models"])
        now = dt.datetime.now(dt.timezone.utc)
        if self.generated_at is None:
            issues.append("no predictions yet: run `footy predict`")
        elif now - self.generated_at > dt.timedelta(hours=STALE_PREDICTIONS_HOURS):
            issues.append(f"predictions older than {STALE_PREDICTIONS_HOURS}h")
        last = pd.Timestamp(self.meta["last_match_date"]).date()
        if (now.date() - last).days > STALE_RESULTS_DAYS:
            issues.append(f"models trained on results up to {last}: run `footy daily`")
        return schemas.Health(
            status="degraded" if issues else "ok",
            model_trained_at=self.meta.get("trained_at"),
            model_seasons=self.meta.get("seasons"),
            model_last_match_date=last,
            predictions_generated_at=self.generated_at,
            predictions_count=len(self.predictions),
            issues=issues,
        )

    # -- teams ------------------------------------------------------------------------
    def teams(self, league: League, season: int | None = None) -> list[str]:
        h = self.history(league)
        season = season if season is not None else int(h["season"].max())
        cur = h[h["season"] == season]
        return sorted(set(cur["home_team"]) | set(cur["away_team"]))

    def resolve_team(self, name: str, league: League) -> str:
        """Canonical name from a canonical name or any known alias (case-insensitive)."""
        h = self.history(league)
        canonical = set(h["home_team"]) | set(h["away_team"])
        lookup = {c.lower(): c for c in canonical}
        for source_tables in teams.load_aliases().values():
            for alias, canon in source_tables.get(league.key, {}).items():
                lookup.setdefault(alias.lower(), canon)
        key = name.strip().lower()
        if key in lookup:
            return lookup[key]
        suggestions = difflib.get_close_matches(key, list(lookup), n=3, cutoff=0.5)
        raise UnknownTeamError(name, league.key, sorted({lookup[s] for s in suggestions}))

    # -- predictions ------------------------------------------------------------------
    def upcoming(self, league: str | None, date_from: dt.date | None,
                 date_to: dt.date | None) -> pd.DataFrame:
        df = self.predictions
        if league:
            df = df[df["league"] == get_league(league).key]
        if date_from:
            df = df[df["date"] >= pd.Timestamp(date_from)]
        if date_to:
            df = df[df["date"] <= pd.Timestamp(date_to)]
        return df

    def by_id(self, match_id: str) -> pd.Series | None:
        hit = self.predictions[self.predictions["match_id"] == match_id]
        return None if hit.empty else hit.iloc[0]

    def players_for(self, match_id: str) -> pd.DataFrame | None:
        """None = no player data at all; empty frame = none for this match."""
        if self.players.empty:
            return None
        return self.players[self.players["match_id"] == match_id]

    def predict_fixture(self, req: schemas.FixtureRequest) -> pd.Series:
        if self.bundle is None:
            raise RuntimeError(self.load_error or "models not loaded")
        league = get_league(req.league)
        home = self.resolve_team(req.home_team, league)
        away = self.resolve_team(req.away_team, league)
        date = pd.Timestamp(req.date)
        fx = pd.DataFrame([{
            "league": league.key, "season": current_season(req.date), "date": date,
            "kickoff": req.kickoff, "home_team": home, "away_team": away,
            "odds_avg_home": req.odds_home, "odds_avg_draw": req.odds_draw,
            "odds_avg_away": req.odds_away, "fixture_source": "request",
        }])
        fx["match_id"] = make_match_id(league.key, fx["date"], fx["home_team"], fx["away_team"])
        # A fixture already in the history (same id) would be predicted with its own
        # result in the data; drop that row so the prediction is genuinely pre-match.
        history = self.history(league)
        history = history[history["match_id"] != fx["match_id"].iloc[0]]
        return predict_mod.predict_rows(league, history, fx, self.bundle).iloc[0]

    def refresh(self, days: int = 7) -> int:
        """Re-run predictions for upcoming fixtures and reload. Serialised by a lock."""
        if not self._refresh_lock.acquire(blocking=False):
            raise RuntimeError("a refresh is already running")
        try:
            out = predict_mod.predict_upcoming(list(load_leagues().values()), days=days,
                                               refresh=True)
            self.reload()
            return len(out)
        finally:
            self._refresh_lock.release()


def _f(x: Any) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x) else float(x)


def to_schema(row: pd.Series) -> schemas.MatchPrediction:
    market = None
    if all(_f(row.get(c)) is not None for c in ("mkt_p_home", "mkt_p_draw", "mkt_p_away")):
        market = schemas.Probabilities(home=row["mkt_p_home"], draw=row["mkt_p_draw"],
                                       away=row["mkt_p_away"])
    kickoff = row.get("kickoff")
    return schemas.MatchPrediction(
        match_id=row["match_id"], league=row["league"], date=pd.Timestamp(row["date"]).date(),
        kickoff=None if kickoff is None or pd.isna(kickoff) else str(kickoff),
        home_team=row["home_team"], away_team=row["away_team"], model=row["model"],
        fixture_source=None if pd.isna(row.get("fixture_source")) else row.get("fixture_source"),
        probabilities=schemas.Probabilities(home=row["p_home"], draw=row["p_draw"], away=row["p_away"]),
        fair_odds=schemas.FairOdds(home=row["fair_odds_home"], draw=row["fair_odds_draw"],
                                   away=row["fair_odds_away"]),
        market=market,
        goals=schemas.Goals(exp_home=row["exp_home_goals"], exp_away=row["exp_away_goals"],
                            p_over25=row["p_over25"], p_btts=row["p_btts"]),
        likely_scores=[schemas.Scoreline(**s) for s in json.loads(row["top_scores_json"])],
    )
