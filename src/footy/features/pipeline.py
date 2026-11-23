"""Assemble the feature table for one league.

Input:  data/processed/matches/<LEAGUE>.parquet (+ optional future fixtures)
Output: data/processed/features/<LEAGUE>.parquet

One row per match: identifiers, targets (NA for future fixtures) and
features. A match is "usable" (feeds Elo/form for later matches, and can be
a training row) only if it has a score and no `exclude_reason`.
"""

from __future__ import annotations

import logging

import pandas as pd

from footy.config import PROCESSED_DIR, League
from footy.features import context, elo, form, market

log = logging.getLogger(__name__)

ID_COLUMNS = ["match_id", "league", "season", "date", "kickoff", "home_team", "away_team"]
TARGET_COLUMNS = ["home_goals", "away_goals", "result"]
# stats that describe the match itself; blanked for unusable matches so they
# never leak into form features
MATCH_STAT_COLUMNS = sorted({c for pair in form.FORM_STATS.values() for c in pair})


def _mask_unusable(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    usable = df["home_goals"].notna() & df["away_goals"].notna()
    if "exclude_reason" in df:
        usable &= df["exclude_reason"].isna()
    masked = df.copy()
    present = [c for c in MATCH_STAT_COLUMNS if c in masked]
    masked[present] = masked[present].astype("float64").where(usable, axis=0)
    for c in MATCH_STAT_COLUMNS:
        if c not in masked:
            masked[c] = float("nan")
    return masked, usable


def build_features(
    matches: pd.DataFrame,
    league: League,
    fixtures: pd.DataFrame | None = None,
    elo_params: elo.EloParams = elo.EloParams(),
) -> pd.DataFrame:
    """Compute features for every row of matches (+ fixtures, if given)."""
    if fixtures is None:
        df = matches
    else:
        # align fixture columns/dtypes to the history so all-NA columns concat cleanly
        aligned = fixtures.reindex(columns=matches.columns).astype(matches.dtypes.to_dict())
        df = pd.concat([matches, aligned], ignore_index=True)
    df = df.sort_values(["date", "kickoff", "home_team"], na_position="first").reset_index(drop=True)
    masked, usable = _mask_unusable(df)

    mkt = market.market_features(df) if "odds_avg_home" in df else pd.DataFrame(index=df.index)
    ratings = elo.compute_elo(masked, elo_params)
    team = form.team_form(masked)
    ctx = pd.DataFrame({
        "crowd_restricted": context.crowd_restricted(df, league.key),
        "season_progress": context.season_progress(df, team["h_season_match_no"]),
    })

    out = pd.concat([
        df[ID_COLUMNS],
        df[TARGET_COLUMNS],
        usable.rename("is_usable"),
        df.get("exclude_reason", pd.Series(pd.NA, index=df.index, dtype="string")).rename("exclude_reason"),
        mkt, ratings, team, ctx,
    ], axis=1)
    out["elo_warmup"] = (out["season"] == out["season"].min()).astype(float)
    return out


def feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Feature names grouped for model configs and documentation."""
    excluded = set(ID_COLUMNS + TARGET_COLUMNS + ["is_usable", "exclude_reason"])
    groups: dict[str, list[str]] = {"market": [], "elo": [], "form": [], "context": []}
    for c in df.columns:
        if c in excluded:
            continue
        if c.startswith("mkt_"):
            groups["market"].append(c)
        elif c.startswith("elo_"):
            groups["elo"].append(c)
        elif c.startswith(("h_", "a_")):
            groups["form"].append(c)
        else:
            groups["context"].append(c)
    return groups


def build_league(league: League, *, write: bool = True) -> pd.DataFrame:
    path = PROCESSED_DIR / "matches" / f"{league.key}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `footy build --league {league.key}`")
    feats = build_features(pd.read_parquet(path), league)
    groups = feature_columns(feats)
    log.info("%s: %d rows, %s", league.key, len(feats),
             {g: len(cols) for g, cols in groups.items()})
    all_nan = [c for cols in groups.values() for c in cols if feats[c].isna().all()]
    if all_nan:
        log.warning("%s: %d features are entirely NA (e.g. no Understat yet): %s",
                    league.key, len(all_nan), all_nan[:6] + (["..."] if len(all_nan) > 6 else []))
    if write:
        out = PROCESSED_DIR / "features" / f"{league.key}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(out, index=False)
        log.info("Wrote %s", out)
    return feats
