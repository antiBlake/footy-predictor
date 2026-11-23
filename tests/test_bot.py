import datetime as dt
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from footy.bot import bot, formatting as fmt
from footy.bot.subscriptions import Subscribers


def _row(home="Chelsea", away="Leeds", ph=0.51, pd_=0.26, pa=0.23, date="2026-11-21",
         league="EPL", market=True, match_id="m1"):
    return pd.Series({
        "match_id": match_id, "league": league, "date": pd.Timestamp(date), "kickoff": "15:00",
        "home_team": home, "away_team": away, "model": "ensemble[with_market]",
        "p_home": ph, "p_draw": pd_, "p_away": pa,
        "fair_odds_home": 1 / ph, "fair_odds_draw": 1 / pd_, "fair_odds_away": 1 / pa,
        "mkt_p_home": 0.5 if market else float("nan"), "mkt_p_draw": 0.26 if market else float("nan"),
        "mkt_p_away": 0.24 if market else float("nan"),
        "exp_home_goals": 1.64, "exp_away_goals": 1.12, "p_over25": 0.52, "p_btts": 0.5,
        "top_scores_json": json.dumps([{"home": 1, "away": 1, "p": 0.12},
                                       {"home": 1, "away": 0, "p": 0.10}]),
    })


# --- formatting ---------------------------------------------------------------------

def test_fixture_line_bolds_favourite_and_escapes():
    line = fmt.fixture_line(_row(home="Brighton & Hove", away="<Leeds>", ph=0.2, pa=0.55))
    assert "<b>&lt;Leeds&gt;</b>" in line and "Brighton &amp; Hove" in line
    assert "20% · 26% · 55%" in line


def test_digest_groups_by_day_and_league():
    df = pd.DataFrame([_row(), _row("Lazio", "Lecce", league="SERIEA", match_id="m2"),
                       _row("Arsenal", "Man City", date="2026-11-28", market=False, match_id="m3")])
    (msg,) = fmt.fixtures_digest(df, "Next 7 days")
    assert msg.index("Sat 21 Nov") < msg.index("Premier League") < msg.index("Sat 28 Nov")
    assert "Serie A" in msg
    assert fmt.fixtures_digest(df.iloc[0:0], "Today")[0].endswith("No fixtures.")


def test_long_digest_is_split_under_limit():
    rows = [_row(f"Team{i}", f"Other{i}", match_id=str(i)) for i in range(400)]
    chunks = fmt.fixtures_digest(pd.DataFrame(rows), "Lots")
    assert len(chunks) > 1 and all(len(c) <= fmt.TELEGRAM_LIMIT for c in chunks)


def test_match_card_with_and_without_market_and_players():
    card = fmt.match_card(_row())
    assert "Market: 50%" in card and "1-1 (12%)" in card and "with market odds" in card
    no_mkt = fmt.match_card(_row(market=False))
    assert "Market" not in no_mkt
    players = pd.DataFrame([
        {"side": "home", "player": "Palmer", "p_score": 0.38, "p_start": 0.95},
        {"side": "home", "player": "Sub", "p_score": 0.1, "p_start": 0.1},
        {"side": "away", "player": "Calvert-Lewin", "p_score": 0.3, "p_start": 0.9},
    ])
    with_p = fmt.match_card(_row(), players)
    assert with_p.index("Palmer") < with_p.index("Sub") and "(bench?)" in with_p


@pytest.mark.parametrize("text,expected", [
    ("Arsenal v Man City", ("Arsenal", "Man City", None)),
    ("real madrid vs barcelona 2.1 3.5 3.4", ("real madrid", "barcelona", (2.1, 3.5, 3.4))),
    ("Inter - Milan", ("Inter", "Milan", None)),
])
def test_parse_predict_args(text, expected):
    assert fmt.parse_predict_args(text) == expected


@pytest.mark.parametrize("text", ["Arsenal", "Arsenal v", "A v B 0.5 3 3", ""])
def test_parse_predict_args_rejects(text):
    with pytest.raises(ValueError):
        fmt.parse_predict_args(text)


# --- subscriptions ------------------------------------------------------------------

def test_subscribers_roundtrip(tmp_path):
    s = Subscribers(tmp_path / "subs.json")
    assert s.add(1) and not s.add(1) and s.add(2)
    assert s.all() == {1, 2}
    assert s.remove(1) and not s.remove(1)
    assert Subscribers(tmp_path / "subs.json").all() == {2}


# --- handlers (no Telegram network) ---------------------------------------------------

class FakeSvc:
    def __init__(self):
        self.preds = pd.DataFrame([_row(date=str(dt.date.today()))])

    def upcoming(self, league, date_from, date_to):
        df = self.preds
        if league:
            df = df[df["league"] == league]
        return df

    def resolve_team(self, name, league):
        from footy.api.service import UnknownTeamError
        if league.key == "EPL" and name.lower() in {"chelsea", "leeds"}:
            return name.title()
        raise UnknownTeamError(name, league.key, ["Chelsea"] if league.key == "EPL" else [])

    def players_for(self, match_id):
        return None

    def predict_fixture(self, req):
        return _row(req.home_team, req.away_team)


def _update(chat_id=42):
    msg = SimpleNamespace(reply_text=AsyncMock(), reply_chat_action=AsyncMock())
    return SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id, username="u", title=None),
                           effective_message=msg)


def _context(args=(), allowed=(42,), tmp_path=None):
    return SimpleNamespace(args=list(args), bot_data={
        "svc": FakeSvc(), "allowed": set(allowed),
        "subs": Subscribers(tmp_path / "s.json") if tmp_path else None,
        "digest_time": dt.time(8, 0)})


def _sent(update):
    return "\n".join(c.args[0] for c in update.effective_message.reply_text.call_args_list)


async def test_unauthorised_chat_only_gets_its_id():
    upd = _update(chat_id=7)
    await bot.cmd_today(upd, _context(allowed=(42,)))
    text = _sent(upd)
    assert "Your chat id is 7" in text and "Chelsea" not in text


async def test_today_lists_fixtures():
    upd = _update()
    await bot.cmd_today(upd, _context())
    assert "Chelsea" in _sent(upd) and "Leeds" in _sent(upd)


async def test_match_finds_next_fixture_and_handles_unknown():
    upd = _update()
    await bot.cmd_match(upd, _context(["chelsea"]))
    assert "Expected goals" in _sent(upd)
    upd2 = _update()
    await bot.cmd_match(upd2, _context(["chelsae"]))
    assert "Did you mean: Chelsea" in _sent(upd2)


async def test_predict_command_runs_model_in_thread():
    upd = _update()
    await bot.cmd_predict(upd, _context(["chelsea", "v", "leeds", "2.0", "3.5", "4.0"]))
    assert "Chelsea v Leeds" in _sent(upd)
    upd2 = _update()
    await bot.cmd_predict(upd2, _context(["chelsea"]))
    assert "Use: /predict" in _sent(upd2)


async def test_subscribe_flow(tmp_path):
    ctx = _context(tmp_path=tmp_path)
    upd = _update()
    await bot.cmd_subscribe(upd, ctx)
    assert "Subscribed" in _sent(upd) and ctx.bot_data["subs"].all() == {42}
    await bot.cmd_unsubscribe(_update(), ctx)
    assert ctx.bot_data["subs"].all() == set()


def test_allowed_chat_ids_parsing(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", " 123, -100456 ,abc,")
    assert bot.allowed_chat_ids() == {123, -100456}
