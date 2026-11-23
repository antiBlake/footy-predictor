"""Scores, uncertainty and calibration for walk-forward predictions.

All comparisons use the same matches for every model: matches where the
market benchmark exists (all but one in the data).

Uncertainty: per-match log-loss differences are strongly correlated within a
weekend (same news, same weather, same refereeing trends), so the bootstrap
resamples whole *weeks*, not single matches. That gives honest (wider)
intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from footy.evaluation.metrics import EPS, OUTCOMES, PROB_COLS, encode, one_hot, score


def common_rows(preds: pd.DataFrame) -> pd.DataFrame:
    """Keep matches that every model (incl. market) has a prediction for."""
    ok = preds.dropna(subset=PROB_COLS)
    n_models = ok["model"].nunique()
    counts = ok.groupby("match_id")["model"].nunique()
    return ok[ok["match_id"].isin(counts[counts == n_models].index)]


def score_table(preds: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    rows = []
    keys = ["model"] + (by or [])
    for key, g in preds.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(keys, key)),
                     **score(g[PROB_COLS].to_numpy(), encode(g["result"]))})
    return pd.DataFrame(rows)


def per_match_log_loss(g: pd.DataFrame) -> pd.Series:
    p = g[PROB_COLS].to_numpy()
    y = encode(g["result"])
    return pd.Series(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1)), index=g["match_id"].to_numpy())


@dataclass
class Diff:
    model: str
    reference: str
    mean: float      # model minus reference; negative = model better
    lo: float
    hi: float
    share_better: float  # share of bootstrap samples where model < reference


def bootstrap_diff(preds: pd.DataFrame, model: str, reference: str = "market",
                   n_boot: int = 2000, seed: int = 0) -> Diff:
    a = per_match_log_loss(preds[preds["model"] == model])
    b = per_match_log_loss(preds[preds["model"] == reference])
    d = (a - b.reindex(a.index)).dropna()
    weeks = (preds.drop_duplicates("match_id").set_index("match_id")["date"]
             .reindex(d.index).dt.to_period("W").astype(str))
    sums = d.groupby(weeks.to_numpy()).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums["sum"].to_numpy()[idx].sum(1) / sums["count"].to_numpy()[idx].sum(1)
    return Diff(model, reference, float(d.mean()), float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)), float((boot < 0).mean()))


def reliability(preds: pd.DataFrame, model: str, n_bins: int = 10) -> pd.DataFrame:
    """Per outcome: equal-count bins of predicted probability vs observed frequency."""
    g = preds[preds["model"] == model]
    p = g[PROB_COLS].to_numpy()
    y = one_hot(encode(g["result"]))
    rows = []
    for k, outcome in enumerate(OUTCOMES):
        bins = pd.qcut(p[:, k], n_bins, labels=False, duplicates="drop")
        for b in np.unique(bins):
            m = bins == b
            rows.append({"outcome": outcome, "bin": int(b), "n": int(m.sum()),
                         "mean_pred": float(p[m, k].mean()), "observed": float(y[m, k].mean())})
    return pd.DataFrame(rows)


def ece(rel: pd.DataFrame) -> pd.Series:
    """Expected calibration error per outcome (count-weighted |observed - predicted|)."""
    return rel.groupby("outcome").apply(
        lambda r: float((r["n"] * (r["observed"] - r["mean_pred"]).abs()).sum() / r["n"].sum()),
        include_groups=False,
    ).reindex(OUTCOMES)


# -- chart -------------------------------------------------------------------------
# Reference palette (dataviz skill): fixed categorical order, recessive ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def plot_reliability(preds: pd.DataFrame, models: list[str], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(models) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} models per chart")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=False, facecolor=SURFACE)
    names = {"H": "Home win", "D": "Draw", "A": "Away win"}
    for ax, outcome in zip(axes, OUTCOMES):
        ax.set_facecolor(SURFACE)
        rels = {}
        for m in models:
            rel = reliability(preds, m)
            rels[m] = rel[rel["outcome"] == outcome]
        lo = min(r[["mean_pred", "observed"]].min().min() for r in rels.values()) - 0.02
        hi = max(r[["mean_pred", "observed"]].max().max() for r in rels.values()) + 0.02
        ax.plot([lo, hi], [lo, hi], color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1,
                label="Perfect calibration")
        for color, (m, r) in zip(SERIES, rels.items()):
            ax.plot(r["mean_pred"], r["observed"], color=color, lw=2, marker="o", ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, label=m)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(names[outcome], color=INK, fontsize=12, loc="left")
        ax.set_xlabel("Predicted probability", color=INK_2, fontsize=10)
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("Observed frequency", color=INK_2, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=10, labelcolor=INK)
    fig.suptitle("Calibration, walk-forward out-of-sample predictions (10 equal-count bins)",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
