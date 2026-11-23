"""Player predictions for one team in one match.

Three parts, each estimated only from matches before the prediction date:

1. Availability: over the team's last matches, weighted towards the most
   recent (half-life `halflife_team` team matches), how often did the player
   start, come on, or not play at all? A match the team played without him
   counts as "did not play", so injured or dropped players fade out quickly.
   That gives p_start, p_sub and expected minutes in each case.

2. Per-90 rates for xG, shots, xA, key passes: time-decayed over the
   player's own appearances (any team, any season) and shrunk towards his
   position's average (empirical Bayes):
       rate = (sum w*stat + prior_rate * prior_minutes) / (sum w*minutes + prior_minutes)
   so a striker with 200 minutes isn't credited with a fluke 1.5 xG/90.

3. Team context: player expectations are scaled so the team total equals
   the match model's expected goals for this fixture (minus own goals). The
   player table and the match prediction can't disagree, and a weak team
   away at a strong one lowers every player's numbers.

Goals and assists per player are Poisson given minutes, mixed over the
start / sub / no-play scenarios:
    P(scores) = p_start (1 - e^-lam_start) + p_sub (1 - e^-lam_sub)

Known limits: no lineups or injury news (a returning player with no
recent minutes is invisible until he plays), xG includes penalties, and
assists aren't modelled jointly with goals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RATE_STATS = ["xg", "shots", "xa", "key_passes"]
OWN_GOAL_SHARE = 0.03  # share of goals that are own goals (scored for, not by, a team's player)


@dataclass(frozen=True)
class PlayerParams:
    halflife_team: float = 4.0       # availability decay, in team matches
    lookback_team: int = 10          # team matches considered for availability
    missed_grace: int = 2            # missed matches before the absence penalty kicks in
    halflife_player: float = 15.0    # rate decay, in the player's appearances
    prior_minutes: float = 600.0     # shrinkage strength towards position mean
    default_start_minutes: float = 82.0
    default_sub_minutes: float = 20.0
    assists_per_goal: float = 0.72   # share of goals with an assist (top-flight average)


def position_group(position: str | None) -> str:
    """Understat positions look like 'FW', 'AML', 'DC', 'GK', 'Sub'."""
    if not position or position == "Sub":
        return "UNK"
    p = position.upper()
    if p.startswith("GK"):
        return "GK"
    if p.startswith("DM"):  # defensive midfielder (Understat 'DMC', 'DML', 'DMR')
        return "MID"
    if p.startswith("D"):
        return "DEF"
    if p.startswith(("FW", "F")):
        return "FWD"
    if p.startswith(("AM", "M", "DM")):
        return "MID"
    return "UNK"


def player_positions(pm: pd.DataFrame) -> pd.Series:
    """Most common non-Sub position group per player."""
    starts = pm[pm["position"] != "Sub"].assign(group=lambda d: d["position"].map(position_group))
    mode = starts.groupby("player_id")["group"].agg(lambda s: s.value_counts().index[0])
    return mode


def position_priors(pm: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
    """Per-90 means by position group (minutes-weighted), from the data given."""
    df = pm.assign(group=pm["player_id"].map(positions).fillna("UNK"))
    tot = df.groupby("group")[[*RATE_STATS, "minutes"]].sum()
    pri = tot[RATE_STATS].div(tot["minutes"].clip(lower=1), axis=0) * 90
    overall = df[RATE_STATS].sum() / max(df["minutes"].sum(), 1) * 90
    return pri.reindex(["GK", "DEF", "MID", "FWD", "UNK"]).fillna(overall)


def availability(pm_team: pd.DataFrame, team_dates: list[pd.Timestamp],
                 params: PlayerParams) -> pd.DataFrame:
    """p_start, p_sub, minutes when starting/subbing for players seen in the last
    `lookback_team` team matches. `team_dates` = team match dates before the cutoff."""
    recent = sorted(team_dates)[-params.lookback_team:]
    if not recent:
        return pd.DataFrame(columns=["p_start", "p_sub", "min_start", "min_sub"])
    age = {d: len(recent) - 1 - i for i, d in enumerate(recent)}
    w = {d: 0.5 ** (a / params.halflife_team) for d, a in age.items()}
    total_w = sum(w.values())
    seen = pm_team[pm_team["date"].isin(recent)].copy()
    seen["w"] = seen["date"].map(w)
    seen["started"] = seen["position"] != "Sub"
    rows = []
    for pid, g in seen.groupby("player_id"):
        st, sb = g[g["started"]], g[~g["started"] & (g["minutes"] > 0)]
        # consecutive most-recent team matches missed: without injury news this is
        # the best signal that a player is out, so it decays faster than the EWM
        played = set(g.loc[g["minutes"] > 0, "date"])
        streak = 0
        for d in reversed(recent):
            if d in played:
                break
            streak += 1
        absent = 0.5 ** max(streak - params.missed_grace, 0)
        rows.append({
            "player_id": pid,
            "p_start": st["w"].sum() / total_w * absent,
            "p_sub": sb["w"].sum() / total_w * absent,
            "min_start": np.average(st["minutes"], weights=st["w"]) if len(st) else params.default_start_minutes,
            "min_sub": np.average(sb["minutes"], weights=sb["w"]) if len(sb) else params.default_sub_minutes,
        })
    return pd.DataFrame(rows).set_index("player_id")


def per90_rates(pm_players: pd.DataFrame, positions: pd.Series, priors: pd.DataFrame,
                params: PlayerParams) -> pd.DataFrame:
    """Shrunk, time-decayed per-90 rates for each player in pm_players (all history)."""
    df = pm_players[pm_players["minutes"] > 0].sort_values("date")
    n = df.groupby("player_id").cumcount(ascending=False)  # 0 = most recent appearance
    df = df.assign(w=0.5 ** (n / params.halflife_player))
    wsum = df[RATE_STATS].mul(df["w"], axis=0).groupby(df["player_id"]).sum()
    wmin = (df["minutes"] * df["w"]).groupby(df["player_id"]).sum()
    group = wsum.index.to_series().map(positions).fillna("UNK")
    prior = priors.loc[group.to_numpy()].set_axis(wsum.index)
    k = params.prior_minutes
    rates = (wsum + prior * k / 90).div(wmin + k, axis=0) * 90
    rates["position"] = group
    rates["weighted_minutes"] = wmin
    return rates


def predict_team_players(
    pm: pd.DataFrame,
    team: str,
    asof: pd.Timestamp,
    team_dates: list[pd.Timestamp],
    team_lambda: float,
    params: PlayerParams = PlayerParams(),
    priors: pd.DataFrame | None = None,
    positions: pd.Series | None = None,
) -> pd.DataFrame:
    """Player expectations for `team` in a match on `asof` where the team is
    expected to score `team_lambda` goals. Uses only rows of pm dated before asof."""
    hist = pm[pm["date"] < asof]
    positions = positions if positions is not None else player_positions(hist)
    priors = priors if priors is not None else position_priors(hist, positions)
    avail = availability(hist[hist["team"] == team], [d for d in team_dates if d < asof], params)
    if avail.empty:
        return pd.DataFrame()
    rates = per90_rates(hist[hist["player_id"].isin(avail.index)], positions, priors, params)
    df = avail.join(rates, how="inner")
    names = hist.sort_values("date").groupby("player_id")["player"].last()
    df["player"] = names.reindex(df.index)

    # unscaled expected xG per scenario, then scale the team total to team_lambda
    lam_start = df["xg"] * df["min_start"] / 90
    lam_sub = df["xg"] * df["min_sub"] / 90
    raw_total = float((df["p_start"] * lam_start + df["p_sub"] * lam_sub).sum())
    scale = team_lambda * (1 - OWN_GOAL_SHARE) / raw_total if raw_total > 0 else 0.0
    df["attack_scale"] = scale

    def mix(rate: pd.Series, s: float) -> tuple[pd.Series, pd.Series]:
        ls = rate * df["min_start"] / 90 * s
        lb = rate * df["min_sub"] / 90 * s
        expected = df["p_start"] * ls + df["p_sub"] * lb
        p_any = df["p_start"] * (1 - np.exp(-ls)) + df["p_sub"] * (1 - np.exp(-lb))
        return expected, p_any

    df["exp_goals"], df["p_score"] = mix(df["xg"], scale)
    df["exp_shots"], _ = mix(df["shots"], scale)
    # assists: same idea, scaled so the team total = assisted share of expected goals
    raw_a = float((df["p_start"] * df["xa"] * df["min_start"] / 90
                   + df["p_sub"] * df["xa"] * df["min_sub"] / 90).sum())
    a_scale = team_lambda * params.assists_per_goal / raw_a if raw_a > 0 else 0.0
    df["exp_assists"], df["p_assist"] = mix(df["xa"], a_scale)
    df["p_appear"] = (df["p_start"] + df["p_sub"]).clip(upper=1)
    df["exp_minutes"] = df["p_start"] * df["min_start"] + df["p_sub"] * df["min_sub"]
    df["team"] = team
    cols = ["team", "player", "position", "p_start", "p_appear", "exp_minutes", "exp_goals",
            "p_score", "exp_assists", "p_assist", "exp_shots", "xg", "xa", "shots", "attack_scale"]
    out = df[cols].rename(columns={"xg": "xg_per90", "xa": "xa_per90", "shots": "shots_per90"})
    return out.sort_values("exp_goals", ascending=False).reset_index()
