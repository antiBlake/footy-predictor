"""footy command-line interface.

  footy collect --source football_data --league ALL --from 2015
  footy collect --source openfootball  --league EPL
  footy collect --source understat     --league EPL
  footy teams propose --source openfootball --league ALL   # writes config/team_aliases.yaml
  footy teams check   --source openfootball --league ALL   # verify mapping coverage
  footy build --league ALL                                  # processed match tables
  footy features --league ALL                               # pre-match feature tables
  footy train --test-from 2024                              # fit + compare models
  footy backtest --from 2020                                # walk-forward + report
  footy report                                              # re-render report only
  footy update                                              # refresh current season + rebuild
  footy fit                                                 # train production models
  footy predict --days 7                                    # predict upcoming fixtures
  footy daily                                               # update + fit + predict (for cron)
  footy serve --port 8000                                   # HTTP API (docs at /docs)
  footy bot                                                 # Telegram bot
  footy players collect --from 2021                         # Understat player stats
  footy players backtest --season 2025                      # score the player model
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from footy.config import INTERIM_DIR, League, current_season, get_league, load_leagues, season_label
from footy.data import build, football_data, openfootball, teams, understat
from footy.data.http import SourceUnavailable
from footy.features import pipeline as features
from footy.logging_utils import setup_logging

log = logging.getLogger("footy")

COLLECTORS = {
    "football_data": football_data.collect_league,
    "openfootball": openfootball.collect_league,
    "understat": understat.collect_league,
}
ALIASED_SOURCES = ["openfootball", "understat"]


def _leagues(key: str) -> list[League]:
    if key.upper() == "ALL":
        return list(load_leagues().values())
    return [get_league(key)]


def _read_interim(source: str, league: League) -> pd.DataFrame:
    path = INTERIM_DIR / source / f"{league.key}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `footy collect --source {source} --league {league.key}`"
        )
    return pd.read_parquet(path)


# --- collect ---------------------------------------------------------------------


def _print_football_data_summary(league: League, df: pd.DataFrame) -> None:
    summary = df.groupby("season").agg(
        matches=("match_id", "size"),
        teams=("home_team", "nunique"),
        goals_per_game=("home_goals", lambda s: (s + df.loc[s.index, "away_goals"]).mean()),
        home_win_pct=("result", lambda s: (s == "H").mean() * 100),
        avg_odds_coverage=("odds_avg_home", lambda s: s.notna().mean() * 100),
    )
    summary.index = [season_label(s) for s in summary.index]
    print(f"\n{league.name}: {len(df)} matches")
    print(summary.round(2).to_string())


def _cmd_collect(args: argparse.Namespace) -> int:
    last = args.to if args.to is not None else current_season()
    seasons = range(args.from_, last + 1)
    collector = COLLECTORS[args.source]
    failures = 0
    for league in _leagues(args.league):
        try:
            df = collector(league, seasons, refresh=args.refresh)
        except SourceUnavailable as exc:
            log.error("%s %s: %s", args.source, league.key, exc)
            failures += 1
            continue
        if args.source == "football_data":
            _print_football_data_summary(league, df)
    return 1 if failures else 0


# --- teams -----------------------------------------------------------------------


def _cmd_teams_propose(args: argparse.Namespace) -> int:
    aliases = teams.load_aliases()
    all_review: list[teams.AliasProposal] = []
    for league in _leagues(args.league):
        ref = _read_interim("football_data", league)
        other = _read_interim(args.source, league)
        proposals = teams.propose_aliases(ref, other)
        aliases, accepted, review = teams.merge_proposals(
            aliases, args.source, proposals,
            min_confidence=args.min_confidence, min_margin=args.min_margin,
        )
        log.info("%s/%s: %d accepted, %d need review", args.source, league.key,
                 len(accepted), len(review))
        all_review.extend(review)

    teams.save_aliases(aliases)
    log.info("Saved %s", teams.ALIASES_PATH)
    if all_review:
        print("\nNeeds manual review (add to config/team_aliases.yaml if correct):")
        for p in all_review:
            print(f"  {p.league:<11} {p.alias!r:<32} -> {p.canonical!r:<20} "
                  f"conf={p.confidence:.2f} votes={p.votes}/{p.appearances} "
                  f"runner-up={p.runner_up!r} ({p.runner_up_votes})")
        return 1
    return 0


def _cmd_teams_check(args: argparse.Namespace) -> int:
    failed = False
    for league in _leagues(args.league):
        ref = _read_interim("football_data", league)
        other = _read_interim(args.source, league)
        try:
            mapped = teams.to_canonical(other, args.source, league.key)
        except teams.UnmappedTeamError as exc:
            log.error("%s", exc)
            failed = True
            continue
        report = teams.coverage_report(ref, mapped)
        report.index = report.index.set_levels(
            [season_label(s) for s in report.index.levels[1]], level=1)
        print(f"\n{league.name}: Football-Data matches found in {args.source}")
        print(report.round(1).to_string())
    return 1 if failed else 0


# --- build -----------------------------------------------------------------------


def _cmd_build(args: argparse.Namespace) -> int:
    failed = False
    for league in _leagues(args.league):
        try:
            build.build_matches(league)
        except (build.BuildError, teams.UnmappedTeamError) as exc:
            log.error("%s", exc)
            failed = True
    return 1 if failed else 0


def _cmd_features(args: argparse.Namespace) -> int:
    failed = False
    for league in _leagues(args.league):
        try:
            features.build_league(league)
        except FileNotFoundError as exc:
            log.error("%s", exc)
            failed = True
    return 1 if failed else 0


def _cmd_train(args: argparse.Namespace) -> int:
    from footy.models.train import run_holdout  # heavy imports only when needed

    res = run_holdout(args.test_from, save=not args.no_save)
    pd.set_option("display.width", 160)
    print(f"\nHoldout: seasons {season_label(args.test_from)} onward, {int(res.scores['n'].iloc[0])} matches")
    print("\n1X2 scores (lower is better):")
    print(res.scores.drop(columns="n").round(4).to_string())
    print("\nLog loss by season:")
    print(res.by_season.rename(columns=season_label).round(4).to_string())
    print("\nChosen settings (calibrator picked by leave-one-season-out log loss):")
    print(res.chosen.round(4).to_string())
    print("\nGoals (Poisson models):")
    print(res.goals.round(3).to_string())
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    from footy.evaluation.backtest import walk_forward
    from footy.evaluation.render import build_report

    preds = walk_forward(args.from_, args.to)
    print(build_report(preds))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from footy.evaluation.backtest import PREDICTIONS_PATH
    from footy.evaluation.render import build_report

    if not PREDICTIONS_PATH.exists():
        log.error("%s missing; run `footy backtest` first", PREDICTIONS_PATH)
        return 1
    print(build_report(pd.read_parquet(PREDICTIONS_PATH)))
    return 0


def _cmd_update(args: argparse.Namespace) -> int:
    from footy.pipeline import run_update

    return 0 if run_update(_leagues(args.league), args.from_) else 1


def _cmd_fit(args: argparse.Namespace) -> int:
    from footy.inference.production import fit_production

    meta = fit_production()
    print(json.dumps({k: v for k, v in meta.items() if k != "models"}, indent=2))
    for name, info in meta["models"].items():
        print(f"  {name:<24} calibrator={info['calibrator']:<12} params={info['params']}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from footy.inference.predict import predict_upcoming

    preds = predict_upcoming(_leagues(args.league), days=args.days, refresh=args.refresh)
    if preds.empty:
        print("No fixtures in the window.")
        return 0
    view = pd.DataFrame({
        "date": preds["date"].dt.strftime("%a %d %b"),
        "ko": preds["kickoff"].fillna(""),
        "league": preds["league"],
        "match": preds["home_team"] + " v " + preds["away_team"],
        "H": preds["p_home"].map("{:.0%}".format),
        "D": preds["p_draw"].map("{:.0%}".format),
        "A": preds["p_away"].map("{:.0%}".format),
        "mkt H/D/A": [("-" if pd.isna(h) else f"{h:.0%}/{d:.0%}/{a:.0%}")
                      for h, d, a in preds[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].to_numpy()],
        "xG": preds["exp_home_goals"].map("{:.2f}".format) + "-"
              + preds["exp_away_goals"].map("{:.2f}".format),
        "o2.5": preds["p_over25"].map("{:.0%}".format),
        "likely scores": preds["top_scores"],
        "model": preds["model"].str.replace("ensemble", "", regex=False),
    })
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    print(view.to_string(index=False))
    return 0


def _cmd_daily(args: argparse.Namespace) -> int:
    from footy.pipeline import run_daily

    return 0 if run_daily(args.days) >= 0 else 1


def _cmd_bot(args: argparse.Namespace) -> int:
    from footy.bot.bot import run

    run()
    return 0


def _cmd_bot_tick(args: argparse.Namespace) -> int:
    from footy.bot.bot import run_tick

    run_tick()
    return 0


def _cmd_bot_digest(args: argparse.Namespace) -> int:
    from footy.bot.bot import run_digest

    run_digest()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("footy.api.app:app", host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_players_collect(args: argparse.Namespace) -> int:
    from footy.players.data import collect_league as collect_players

    failed = False
    for league in _leagues(args.league):
        try:
            collect_players(league, range(args.from_, current_season() + 1), refresh=args.refresh)
        except (SourceUnavailable, teams.UnmappedTeamError) as exc:
            log.error("players %s: %s", league.key, exc)
            failed = True
            if isinstance(exc, SourceUnavailable):
                break
    return 1 if failed else 0


def _cmd_players_backtest(args: argparse.Namespace) -> int:
    from footy.evaluation.backtest import PREDICTIONS_PATH
    from footy.inference.predict import load_history
    from footy.players import evaluate
    from footy.players.data import load_players
    from footy.players.predict import team_match_dates

    if not PREDICTIONS_PATH.exists():
        log.error("Run `footy backtest` first (player backtest uses its expected goals)")
        return 1
    wf = pd.read_parquet(PREDICTIONS_PATH)
    wf = wf[wf["model"] == args.model]
    summaries = []
    for league in _leagues(args.league):
        pm = load_players(league)
        if pm is None:
            log.error("%s: no player data; run `footy players collect`", league.key)
            return 1
        history = load_history(league)
        lambdas = evaluate.team_lambdas_from_predictions(wf, history)
        season = history[history["season"] == args.season]
        _, summary = evaluate.backtest(pm, lambdas, team_match_dates(history),
                                       season["date"].min(), season["date"].max())
        summaries.append(summary.assign(league=league.key))
    print(pd.concat(summaries).round(4).to_string())
    return 0


# --- parser ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="footy", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Download and parse a data source")
    collect.add_argument("--source", choices=sorted(COLLECTORS), default="football_data")
    collect.add_argument("--league", default="EPL", help="League key or ALL")
    collect.add_argument("--from", dest="from_", type=int, default=2015,
                         help="First season start year (2015 = 2015-16)")
    collect.add_argument("--to", type=int, default=None,
                         help="Last season start year (default: current season)")
    collect.add_argument("--refresh", action="store_true", help="Ignore the raw cache")
    collect.set_defaults(func=_cmd_collect)

    teams_p = sub.add_parser("teams", help="Team-name mapping tools")
    teams_sub = teams_p.add_subparsers(dest="teams_command", required=True)
    propose = teams_sub.add_parser("propose", help="Infer aliases from fixtures")
    check = teams_sub.add_parser("check", help="Report mapping coverage vs Football-Data")
    for p in (propose, check):
        p.add_argument("--source", choices=ALIASED_SOURCES, required=True)
        p.add_argument("--league", default="ALL")
    propose.add_argument("--min-confidence", type=float, default=0.8)
    propose.add_argument("--min-margin", type=float, default=3.0)
    propose.set_defaults(func=_cmd_teams_propose)
    check.set_defaults(func=_cmd_teams_check)

    build_p = sub.add_parser("build", help="Join sources into processed match tables")
    build_p.add_argument("--league", default="ALL")
    build_p.set_defaults(func=_cmd_build)

    feat_p = sub.add_parser("features", help="Compute pre-match features")
    feat_p.add_argument("--league", default="ALL")
    feat_p.set_defaults(func=_cmd_features)

    train_p = sub.add_parser("train", help="Fit all models and score them on a holdout")
    train_p.add_argument("--test-from", type=int, default=2024,
                         help="First holdout season start year")
    train_p.add_argument("--no-save", action="store_true")
    train_p.set_defaults(func=_cmd_train)

    bt = sub.add_parser("backtest", help="Walk-forward backtest (refits each season) + report")
    bt.add_argument("--from", dest="from_", type=int, default=2020, help="First test season")
    bt.add_argument("--to", type=int, default=None, help="Last test season (default: current)")
    bt.set_defaults(func=_cmd_backtest)

    rp = sub.add_parser("report", help="Rebuild reports/backtest.md from saved predictions")
    rp.set_defaults(func=_cmd_report)

    up = sub.add_parser("update", help="Refresh current-season data and rebuild features")
    up.add_argument("--league", default="ALL")
    up.add_argument("--from", dest="from_", type=int, default=2015)
    up.set_defaults(func=_cmd_update)

    fit_p = sub.add_parser("fit", help="Train production models on all seasons")
    fit_p.set_defaults(func=_cmd_fit)

    pr = sub.add_parser("predict", help="Predict upcoming fixtures")
    pr.add_argument("--league", default="ALL")
    pr.add_argument("--days", type=int, default=7)
    pr.add_argument("--refresh", action="store_true", help="Refetch fixtures.csv now")
    pr.set_defaults(func=_cmd_predict)

    daily = sub.add_parser("daily", help="update + fit + predict (schedule this)")
    daily.add_argument("--days", type=int, default=7)
    daily.set_defaults(func=_cmd_daily)

    serve = sub.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)

    bot = sub.add_parser("bot", help="Run the Telegram bot (token in .env)")
    bot.set_defaults(func=_cmd_bot)
    sub.add_parser("bot-tick", help="Answer pending Telegram messages once and exit "
                                    "(GitHub Actions)").set_defaults(func=_cmd_bot_tick)
    sub.add_parser("bot-digest", help="Send today's digest once and exit "
                                      "(GitHub Actions)").set_defaults(func=_cmd_bot_digest)

    pl = sub.add_parser("players", help="Player data and models")
    pl_sub = pl.add_subparsers(dest="players_command", required=True)
    plc = pl_sub.add_parser("collect", help="Collect Understat per-player match stats")
    plc.add_argument("--league", default="ALL")
    plc.add_argument("--from", dest="from_", type=int, default=2021)
    plc.add_argument("--refresh", action="store_true")
    plc.set_defaults(func=_cmd_players_collect)
    plb = pl_sub.add_parser("backtest", help="Score anytime-scorer predictions for one season")
    plb.add_argument("--league", default="ALL")
    plb.add_argument("--season", type=int, default=current_season() - 1)
    plb.add_argument("--model", default="poisson_dc[with_market]",
                     help="walk-forward model whose expected goals drive the players")
    plb.set_defaults(func=_cmd_players_backtest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
