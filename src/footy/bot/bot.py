"""Telegram bot.

Setup (see README "Telegram bot"):
  .env  TELEGRAM_BOT_TOKEN=...            from @BotFather
        TELEGRAM_ALLOWED_CHAT_IDS=123,456  who may use the bot (required)
        TELEGRAM_DIGEST_TIME=08:00         daily digest, local time (optional)
        TELEGRAM_DAILY_PIPELINE=1          run update+fit+predict before the digest
Run:   footy bot

Private by default: messages from chats not in TELEGRAM_ALLOWED_CHAT_IDS get
one reply telling them their chat id (so you can add yourself) and nothing else.

All model work is synchronous and CPU-bound, so handlers run it in a worker
thread (asyncio.to_thread) to keep the bot responsive.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
from collections.abc import Awaitable, Callable
from functools import wraps

from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from footy.api.schemas import FixtureRequest
from footy.api.service import ServiceState, UnknownTeamError
from footy.bot import formatting as fmt
from footy.bot.subscriptions import Subscribers
from footy.config import get_league, load_leagues

log = logging.getLogger(__name__)

Handler = Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]

COMMANDS = [
    ("today", "Today's matches with win chances"),
    ("week", "Next 7 days (optionally: /week EPL)"),
    ("match", "Details for a team's next match: /match Arsenal"),
    ("predict", "Any fixture: /predict Arsenal v Man City [odds]"),
    ("leagues", "League codes"),
    ("status", "Model and data freshness"),
    ("subscribe", "Daily digest of today's matches"),
    ("unsubscribe", "Stop the daily digest"),
    ("refresh", "Recompute upcoming predictions now"),
    ("help", "What I can do"),
]


def allowed_chat_ids() -> set[int]:
    raw = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "")
    return {int(x) for x in raw.replace(" ", "").split(",") if x.lstrip("-").isdigit()}


def restricted(handler: Handler) -> Handler:
    @wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if chat is None:
            return
        if chat.id not in context.bot_data["allowed"]:
            # no chat ids or usernames in logs: Actions logs of a public repo are public
            log.warning("Rejected a message from an unauthorised chat")
            if update.effective_message:
                await update.effective_message.reply_text(
                    f"This is a private bot. Your chat id is {chat.id}.\n"
                    "If this is your bot, add it to TELEGRAM_ALLOWED_CHAT_IDS in .env and restart.")
            return
        await handler(update, context)
    return wrapper


async def _send(update: Update, chunks: list[str] | str) -> None:
    for chunk in [chunks] if isinstance(chunks, str) else chunks:
        await update.effective_message.reply_text(chunk, parse_mode=ParseMode.HTML,
                                                  disable_web_page_preview=True)


def _svc(context: ContextTypes.DEFAULT_TYPE) -> ServiceState:
    return context.bot_data["svc"]


# --- commands -----------------------------------------------------------------------

@restricted
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = ["<b>⚽ footy: Big 5 match predictions</b>", ""]
    lines += [f"/{c} – {fmt.esc(d)}" for c, d in COMMANDS]
    if os.environ.get("FOOTY_BATCH_MODE"):
        lines += ["", "<i>I check for messages every few minutes, so replies aren't instant.</i>"]
    lines += ["", "<i>Probabilities from a model that matches the betting market in "
                  "backtests. For information, not betting advice.</i>"]
    await _send(update, "\n".join(lines))


@restricted
async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    today = today_()
    df = _svc(context).upcoming(None, today, today)
    await _send(update, fmt.fixtures_digest(df, f"Today, {today:%a %d %b}"))


@restricted
async def cmd_week(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    league = context.args[0].upper() if context.args else None
    if league:
        try:
            get_league(league)
        except ValueError:
            await _send(update, f"Unknown league {fmt.esc(league)}. Try /leagues")
            return
    today = today_()
    df = _svc(context).upcoming(league, today, today + dt.timedelta(days=7))
    title = f"Next 7 days{f' · {fmt.league_name(league)}' if league else ''}"
    await _send(update, fmt.fixtures_digest(df, title))


def find_team(svc: ServiceState, name: str) -> tuple[str, str]:
    """(league, canonical team) across all leagues; raises ValueError if unknown/ambiguous."""
    hits, suggestions = {}, set()
    for league in load_leagues().values():
        try:
            hits[league.key] = svc.resolve_team(name, league)
        except UnknownTeamError as exc:
            suggestions.update(exc.suggestions)
    if len(hits) == 1:
        return next(iter(hits.items()))
    if len(hits) > 1:
        raise ValueError(f"'{name}' matches teams in {', '.join(hits)}; be more specific.")
    hint = f" Did you mean: {', '.join(sorted(suggestions)[:4])}?" if suggestions else ""
    raise ValueError(f"I don't know a team called '{name}'.{hint}")


@restricted
async def cmd_match(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await _send(update, "Use: /match Arsenal")
        return
    svc = _svc(context)
    try:
        league, team = find_team(svc, " ".join(context.args))
    except ValueError as exc:
        await _send(update, fmt.esc(exc))
        return
    df = svc.upcoming(league, today_(), None)
    df = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("date")
    if df.empty:
        await _send(update, f"No upcoming {fmt.esc(team)} match in the current predictions. "
                            f"Try /predict {fmt.esc(team)} v Opponent")
        return
    row = df.iloc[0]
    await _send(update, fmt.match_card(row, svc.players_for(row["match_id"])))


@restricted
async def cmd_predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc = _svc(context)
    try:
        home_q, away_q, odds = fmt.parse_predict_args(" ".join(context.args))
        lg_h, home = find_team(svc, home_q)
        lg_a, away = find_team(svc, away_q)
    except ValueError as exc:
        await _send(update, fmt.esc(exc))
        return
    if lg_h != lg_a:
        await _send(update, f"{fmt.esc(home)} and {fmt.esc(away)} are in different leagues.")
        return
    req = FixtureRequest(league=lg_h, home_team=home, away_team=away, date=today_(),
                         odds_home=odds[0] if odds else None,
                         odds_draw=odds[1] if odds else None,
                         odds_away=odds[2] if odds else None)
    await update.effective_message.reply_chat_action("typing")
    try:
        row = await asyncio.to_thread(svc.predict_fixture, req)
    except RuntimeError as exc:
        await _send(update, f"Models not available: {fmt.esc(exc)}")
        return
    note = "" if odds else "\n<i>Tip: add home/draw/away odds for the stronger with-market model.</i>"
    await _send(update, fmt.match_card(row) + note)


@restricted
async def cmd_leagues(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = ["<b>Leagues</b>"] + [f"<code>{k}</code> {fmt.esc(fmt.league_name(k))}"
                                  for k in load_leagues()]
    await _send(update, "\n".join(lines))


@restricted
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    h = _svc(context).health()
    icon = {"ok": "✅", "degraded": "⚠️", "unavailable": "❌"}[h.status]
    lines = [f"{icon} <b>{h.status}</b>",
             f"Models trained: {fmt.esc(h.model_trained_at or '–')}",
             f"Latest result used: {fmt.esc(h.model_last_match_date or '–')}",
             f"Predictions: {h.predictions_count}, generated "
             f"{fmt.esc(h.predictions_generated_at.strftime('%d %b %H:%M UTC') if h.predictions_generated_at else '–')}"]
    lines += [f"• {fmt.esc(i)}" for i in h.issues]
    await _send(update, "\n".join(lines))


@restricted
async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    new = context.bot_data["subs"].add(update.effective_chat.id)
    when = context.bot_data["digest_time"].strftime("%H:%M")
    await _send(update, f"{'Subscribed' if new else 'Already subscribed'}: daily digest at {when}.")


@restricted
async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    gone = context.bot_data["subs"].remove(update.effective_chat.id)
    await _send(update, "Unsubscribed." if gone else "You weren't subscribed.")


@restricted
async def cmd_refresh(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _send(update, "Refreshing fixtures and predictions…")
    try:
        n = await asyncio.to_thread(_svc(context).refresh, 7)
    except Exception as exc:  # surface any failure to the user, keep the bot alive
        log.exception("refresh failed")
        await _send(update, f"Refresh failed: {fmt.esc(exc)}")
        return
    await _send(update, f"Done: {n} upcoming predictions.")


# --- scheduled jobs -----------------------------------------------------------------

async def job_daily(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Optional pipeline run, then the digest to every subscriber."""
    svc: ServiceState = context.bot_data["svc"]
    if context.bot_data.get("run_pipeline"):
        from footy.pipeline import run_daily
        n = await asyncio.to_thread(run_daily, 7)
        if n < 0:
            log.error("daily pipeline failed; sending digest from existing predictions")
        await asyncio.to_thread(svc.reload)
    await send_digest(context.bot, context.bot_data)


async def send_digest(tg_bot, bot_data: dict) -> int:
    """Send today's fixtures to every subscribed, allowed chat. Returns chats reached."""
    today = today_()
    df = bot_data["svc"].upcoming(None, today, today)
    if df.empty:
        log.info("digest: no matches today; nothing sent")
        return 0
    chunks = fmt.fixtures_digest(df, f"Today, {today:%a %d %b}")
    sent = 0
    for chat_id in bot_data["subs"].all() & bot_data["allowed"]:
        try:
            for chunk in chunks:
                await tg_bot.send_message(chat_id, chunk, parse_mode=ParseMode.HTML)
            sent += 1
        except Exception:  # one bad chat must not stop the others
            log.exception("digest to one chat failed")
    return sent


async def _post_init(app: Application) -> None:
    await app.bot.set_my_commands([BotCommand(c, d) for c, d in COMMANDS])


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("Unhandled error", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("Sorry, something went wrong. It's been logged.")


def bot_timezone() -> dt.tzinfo:
    """TELEGRAM_TZ (IANA name, e.g. 'Europe/London'); servers usually run on UTC, so
    set this to get the digest at your local time. Falls back to the machine's zone."""
    name = os.environ.get("TELEGRAM_TZ")
    if name:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    return dt.datetime.now().astimezone().tzinfo


def today_() -> dt.date:
    """'Today' in the bot's timezone, not the server's."""
    return dt.datetime.now(bot_timezone()).date()


async def job_bootstrap(context: ContextTypes.DEFAULT_TYPE) -> None:
    """First start on a fresh server: no models or predictions yet, so build everything."""
    from footy.pipeline import run_daily

    log.info("Bootstrap: no production models/predictions; running the full pipeline "
             "(first run downloads ~10 seasons, takes a few minutes)")
    n = await asyncio.to_thread(run_daily, 7)
    await asyncio.to_thread(context.bot_data["svc"].reload)
    log.info("Bootstrap finished: %s upcoming predictions", n)


def build_application(token: str, svc: ServiceState | None = None,
                      subs: Subscribers | None = None, schedule_jobs: bool = True) -> Application:
    allowed = allowed_chat_ids()
    log.info("Allowed chats configured: %d", len(allowed))  # count only: logs may be public
    if not allowed:
        log.warning("TELEGRAM_ALLOWED_CHAT_IDS is empty: the bot will only tell people their "
                    "chat id. Message it once, then add your id to .env.")
    digest = dt.time.fromisoformat(os.environ.get("TELEGRAM_DIGEST_TIME", "08:00"))
    digest = digest.replace(tzinfo=bot_timezone())

    app = Application.builder().token(token).post_init(_post_init).build()
    svc = svc or ServiceState()
    svc.reload()
    app.bot_data.update(svc=svc, subs=subs or Subscribers(), allowed=allowed,
                        digest_time=digest,
                        run_pipeline=os.environ.get("TELEGRAM_DAILY_PIPELINE", "1") == "1")
    handlers = {"start": cmd_help, "help": cmd_help, "today": cmd_today, "week": cmd_week,
                "match": cmd_match, "predict": cmd_predict, "leagues": cmd_leagues,
                "status": cmd_status, "subscribe": cmd_subscribe,
                "unsubscribe": cmd_unsubscribe, "refresh": cmd_refresh}
    for name, fn in handlers.items():
        app.add_handler(CommandHandler(name, fn))
    app.add_error_handler(_on_error)
    if schedule_jobs:
        app.job_queue.run_daily(job_daily, time=digest, name="daily")
        if svc.bundle is None or svc.predictions.empty:
            app.job_queue.run_once(job_bootstrap, when=1, name="bootstrap")
    return app


def _token() -> str:
    from footy.env import load_dotenv

    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN missing: create a bot with @BotFather and add the "
                         "token to .env (or GitHub Secrets)")
    return token


def _state_files(svc: ServiceState, subs: Subscribers) -> dict[str, float]:
    """mtimes of the files a tick can change (to decide whether to save state)."""
    out = {}
    for path in (svc.predictions_path, subs.path):
        out[str(path)] = path.stat().st_mtime if path.exists() else 0.0
    return out


async def tick() -> bool:
    """Batch mode: answer every pending message once, then exit.
    Returns True if persistent state changed (predictions refreshed, subscriptions)."""
    os.environ.setdefault("FOOTY_BATCH_MODE", "1")
    app = build_application(_token(), schedule_jobs=False)
    svc, subs = app.bot_data["svc"], app.bot_data["subs"]
    before = _state_files(svc, subs)
    await app.initialize()
    try:
        await _post_init(app)
        updates = await app.bot.get_updates(timeout=0, allowed_updates=Update.ALL_TYPES)
        for update in updates:
            await app.process_update(update)
        if updates:  # confirm, so Telegram doesn't send them again
            await app.bot.get_updates(offset=updates[-1].update_id + 1, timeout=0, limit=1)
        log.info("tick: handled %d updates", len(updates))
    finally:
        await app.shutdown()
    return _state_files(svc, subs) != before


async def digest() -> int:
    """Batch mode: send today's digest (after the daily pipeline has run)."""
    app = build_application(_token(), schedule_jobs=False)
    await app.initialize()
    try:
        return await send_digest(app.bot, app.bot_data)
    finally:
        await app.shutdown()


def _github_output(**values: object) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as fh:
            for k, v in values.items():
                fh.write(f"{k}={str(v).lower()}\n")


def run_tick() -> None:
    changed = asyncio.run(tick())
    _github_output(state_changed=changed)


def run_digest() -> None:
    n = asyncio.run(digest())
    log.info("digest sent to %d chat(s)", n)


def run() -> None:
    app = build_application(_token())
    log.info("Bot running; digest at %s", app.bot_data["digest_time"])
    app.run_polling(allowed_updates=Update.ALL_TYPES)



