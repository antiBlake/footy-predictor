"""Pure text formatting for Telegram (HTML parse mode). No Telegram imports, so
everything here is unit-testable."""

from __future__ import annotations

import html
import json
import re

import pandas as pd

from footy.config import load_leagues

TELEGRAM_LIMIT = 4096
SAFE_LIMIT = 3900  # headroom for HTML entities

LEAGUE_FLAGS = {"EPL": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "LALIGA": "🇪🇸", "SERIEA": "🇮🇹", "BUNDESLIGA": "🇩🇪", "LIGUE1": "🇫🇷"}


def esc(text: object) -> str:
    return html.escape(str(text), quote=False)


def pct(p: float) -> str:
    return "–" if p is None or pd.isna(p) else f"{p:.0%}"


def league_name(key: str) -> str:
    lg = load_leagues().get(key)
    return f"{LEAGUE_FLAGS.get(key, '')} {lg.name}".strip() if lg else key


def _favourite(row: pd.Series) -> str:
    probs = {"home": row["p_home"], "draw": row["p_draw"], "away": row["p_away"]}
    side = max(probs, key=probs.get)
    return {"home": row["home_team"], "away": row["away_team"], "draw": "Draw"}[side]


def fixture_line(row: pd.Series) -> str:
    """'15:00 Chelsea v Leeds  51 · 26 · 24' with the favourite in bold."""
    ko = row.get("kickoff")
    ko = "" if ko is None or pd.isna(ko) else f"{esc(ko)} "
    fav = _favourite(row)
    home = f"<b>{esc(row['home_team'])}</b>" if fav == row["home_team"] else esc(row["home_team"])
    away = f"<b>{esc(row['away_team'])}</b>" if fav == row["away_team"] else esc(row["away_team"])
    probs = f"{pct(row['p_home'])} · {pct(row['p_draw'])} · {pct(row['p_away'])}"
    return f"{ko}{home} v {away}  <code>{probs}</code>"


def fixtures_digest(df: pd.DataFrame, title: str) -> list[str]:
    """Fixtures grouped by day then league; split into Telegram-sized messages."""
    if df.empty:
        return [f"<b>{esc(title)}</b>\n\nNo fixtures."]
    df = df.sort_values(["date", "league", "kickoff", "home_team"], na_position="last")
    lines = [f"<b>{esc(title)}</b>", "<i>Home · Draw · Away win chance; favourite in bold</i>"]
    for day, by_day in df.groupby(df["date"].dt.date, sort=True):
        lines.append(f"\n<b>{pd.Timestamp(day):%a %d %b}</b>")
        for league, g in by_day.groupby("league", sort=False):
            lines.append(f"<u>{esc(league_name(league))}</u>")
            lines.extend(fixture_line(r) for _, r in g.iterrows())
    return split_message("\n".join(lines))


def match_card(row: pd.Series, players: pd.DataFrame | None = None, top_n: int = 4) -> str:
    ko = row.get("kickoff")
    when = pd.Timestamp(row["date"]).strftime("%a %d %b") + (
        "" if ko is None or pd.isna(ko) else f", {ko}")
    lines = [
        f"<b>{esc(row['home_team'])} v {esc(row['away_team'])}</b>",
        f"{esc(league_name(row['league']))} · {esc(when)}",
        "",
        f"🏠 Home  <b>{pct(row['p_home'])}</b>   (fair odds {row['fair_odds_home']:.2f})",
        f"🤝 Draw  <b>{pct(row['p_draw'])}</b>   (fair odds {row['fair_odds_draw']:.2f})",
        f"✈️ Away  <b>{pct(row['p_away'])}</b>   (fair odds {row['fair_odds_away']:.2f})",
    ]
    if not pd.isna(row.get("mkt_p_home")):
        lines.append(f"<i>Market: {pct(row['mkt_p_home'])} · {pct(row['mkt_p_draw'])} · "
                     f"{pct(row['mkt_p_away'])}</i>")
    lines += [
        "",
        f"⚽ Expected goals  {row['exp_home_goals']:.2f} – {row['exp_away_goals']:.2f}",
        f"Over 2.5: {pct(row['p_over25'])}   Both score: {pct(row['p_btts'])}",
    ]
    scores = _scores(row)
    if scores:
        lines.append("Likely scores: " + ", ".join(f"{s['home']}-{s['away']} ({pct(s['p'])})"
                                                   for s in scores[:3]))
    if players is not None and not players.empty:
        for side, team in (("home", row["home_team"]), ("away", row["away_team"])):
            top = players[players["side"] == side].nlargest(top_n, "p_score")
            if top.empty:
                continue
            lines.append(f"\n<b>{esc(team)}</b> · anytime scorer")
            lines.extend(f"{esc(p['player'])}  {pct(p['p_score'])}"
                         f"{'' if p['p_start'] >= 0.5 else '  <i>(bench?)</i>'}"
                         for _, p in top.iterrows())
    model = "with market odds" if "with_market" in str(row.get("model")) else "no odds available"
    lines.append(f"\n<i>Model: ensemble, {model}</i>")
    return "\n".join(lines)


def _scores(row: pd.Series) -> list[dict]:
    raw = row.get("top_scores_json")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return []


def split_message(text: str, limit: int = SAFE_LIMIT) -> list[str]:
    """Split on line boundaries so no message exceeds Telegram's limit."""
    chunks, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:  # pathological single line
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


# --- parsing ------------------------------------------------------------------------

_VS = re.compile(r"\s+(?:v|vs|vs\.|-|–)\s+", re.IGNORECASE)
_ODDS = re.compile(r"\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$")


def parse_predict_args(text: str) -> tuple[str, str, tuple[float, float, float] | None]:
    """'Arsenal v Man City 2.1 3.4 3.6' -> ('Arsenal', 'Man City', (2.1, 3.4, 3.6)).
    Raises ValueError with a user-facing message."""
    text = text.strip()
    odds = None
    m = _ODDS.search(text)
    if m:
        odds = tuple(float(x) for x in m.groups())
        if any(o <= 1 for o in odds):
            raise ValueError("Odds must be decimal odds above 1, e.g. 2.10 3.40 3.60")
        text = text[: m.start()]
    parts = _VS.split(text)
    if len(parts) != 2 or not all(p.strip() for p in parts):
        raise ValueError("Use: /predict Home v Away [home draw away odds]\n"
                         "e.g. /predict Arsenal v Man City 2.1 3.4 3.6")
    return parts[0].strip(), parts[1].strip(), odds  # type: ignore[return-value]
