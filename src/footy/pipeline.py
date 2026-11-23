"""End-to-end jobs shared by the CLI (`footy update` / `footy daily`) and the bot."""

from __future__ import annotations

import logging

import requests

from footy.config import League, current_season, load_leagues
from footy.data import build, football_data, openfootball, teams, understat
from footy.data.http import SourceUnavailable
from footy.features import pipeline as features

log = logging.getLogger(__name__)

COLLECTORS = {
    "football_data": football_data.collect_league,
    "openfootball": openfootball.collect_league,
    "understat": understat.collect_league,
}


def run_update(leagues: list[League] | None = None, first_season: int = 2015) -> bool:
    """Re-collect every source, then rebuild matches + features. Returns success.

    Past seasons come from the raw cache; the current season is refetched when
    its cache is older than 6 hours. Understat being down is only a warning.
    """
    leagues = leagues or list(load_leagues().values())
    seasons = range(first_season, current_season() + 1)
    ok = True
    for source, collect in COLLECTORS.items():
        for league in leagues:
            try:
                collect(league, seasons)
            except (SourceUnavailable, requests.RequestException) as exc:
                # RequestException = network failure after retries (DNS, timeout, 5xx):
                # report it as a failed step instead of crashing the whole run
                optional = source == "understat"
                log.log(logging.WARNING if optional else logging.ERROR,
                        "%s %s: %s", source, league.key, exc)
                ok &= optional
                if optional:
                    break  # site down: no point trying each league
    for league in leagues:
        try:
            build.build_matches(league)
            features.build_league(league)
        except (build.BuildError, teams.UnmappedTeamError) as exc:
            log.error("%s", exc)
            ok = False
    return ok


def run_daily(days: int = 7) -> int:
    """update -> fit -> predict. Stops at the first hard failure so stale models are
    never re-served as fresh. Returns the number of upcoming predictions (-1 = failed)."""
    from footy.inference.predict import predict_upcoming
    from footy.inference.production import fit_production

    if not run_update():
        log.error("daily: update failed; models and predictions left unchanged")
        return -1
    fit_production()
    preds = predict_upcoming(list(load_leagues().values()), days=days, refresh=True)
    log.info("daily: done, %d upcoming predictions", len(preds))
    return len(preds)
