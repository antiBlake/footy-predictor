"""Write reports/backtest.md (+ reliability.png) from walk-forward predictions."""

from __future__ import annotations

import logging

import pandas as pd

from footy.config import PROCESSED_DIR, PROJECT_ROOT, season_label
from footy.evaluation import betting, report

log = logging.getLogger(__name__)
REPORTS_DIR = PROJECT_ROOT / "reports"


def _md(df: pd.DataFrame, digits: int = 4) -> str:
    return df.round(digits).to_markdown()


def _load_matches() -> pd.DataFrame:
    paths = sorted((PROCESSED_DIR / "matches").glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def build_report(preds: pd.DataFrame) -> str:
    REPORTS_DIR.mkdir(exist_ok=True)
    preds = report.common_rows(preds)
    n = preds["match_id"].nunique()
    seasons = sorted(preds["season"].unique())

    overall = report.score_table(preds).set_index("model").drop(columns="n").sort_values("log_loss")
    by_season = (report.score_table(preds, ["season"])
                 .pivot(index="model", columns="season", values="log_loss")
                 .loc[overall.index].rename(columns=season_label))
    by_league = (report.score_table(preds, ["league"])
                 .pivot(index="model", columns="league", values="log_loss").loc[overall.index])

    models = [m for m in overall.index if m not in ("market", "base_rate")]
    diffs = pd.DataFrame([vars(report.bootstrap_diff(preds, m)) for m in models]) \
        .set_index("model").drop(columns="reference")

    best_with = next(m for m in overall.index if "with_market" in m)
    best_without = next(m for m in overall.index if "no_market" in m)
    chart_models = ["market", best_with, best_without]
    report.plot_reliability(preds, chart_models, REPORTS_DIR / "reliability.png")
    eces = pd.DataFrame({m: report.ece(report.reliability(preds, m)) for m in chart_models}).T
    rel_table = report.reliability(preds, best_with).pivot(
        index="bin", columns="outcome", values=["mean_pred", "observed"])

    matches = _load_matches()
    bet_tables = {}
    for m in sorted({best_without, *[x for x in models if "no_market" in x]}):
        bets = betting.value_bets(preds, matches, m)
        bet_tables[m] = betting.summarise(bets)

    calib = (preds.dropna(subset=["calibrator"]).drop_duplicates(["model", "season"])
             .pivot(index="model", columns="season", values="calibrator")
             .rename(columns=season_label)) if "calibrator" in preds else pd.DataFrame()

    lines = [
        "# Walk-forward backtest",
        "",
        f"{n:,} out-of-sample matches, seasons {season_label(seasons[0])} to "
        f"{season_label(seasons[-1])}, all five leagues. Each season is predicted by models "
        "trained only on earlier seasons. Lower is better for every score.",
        "",
        "## Overall",
        _md(overall),
        "",
        "## Model minus market (log loss), week-block bootstrap 95% interval",
        "Negative = better than the market. `share_better` = share of bootstrap samples "
        "in which the model beats the market.",
        _md(diffs),
        "",
        "## Log loss by season",
        _md(by_season),
        "",
        "## Log loss by league",
        _md(by_league),
        "",
        "## Calibration",
        "![reliability](reliability.png)",
        "",
        "Expected calibration error (count-weighted |observed - predicted|):",
        _md(eces),
        "",
        f"Reliability table for `{best_with}` (10 equal-count bins):",
        _md(rel_table, 3),
        "",
        "Calibrator chosen in each season's fit:",
        _md(calib) if not calib.empty else "(not recorded)",
        "",
        "## Value-betting backtest (research only)",
        "Flat 1-unit stakes at Bet365 pre-match odds on the best-edge outcome when "
        "edge > threshold. CLV = expected return if the margin-free *closing* market is "
        "the truth. Every threshold is shown, so none is cherry-picked.",
    ]
    for m, t in bet_tables.items():
        lines += ["", f"### {m}", _md(t, 3)]
    text = "\n".join(lines) + "\n"
    (REPORTS_DIR / "backtest.md").write_text(text)
    log.info("Wrote %s", REPORTS_DIR / "backtest.md")
    return text
