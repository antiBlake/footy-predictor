"""Production models: trained on every usable season up to today.

The served model is the backtest winner: an equal-weight average of
Dixon-Coles Poisson and LightGBM (log loss 0.9723 with market features,
0.9832 without, in the 2020-21+ walk-forward). Both feature sets are kept:
'with_market' for fixtures that have odds, 'no_market' for those that don't.

Refit after each matchday or at least monthly (`footy fit`); the models
themselves change slowly, but the most recent season matters most.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from footy.config import PROJECT_ROOT
from footy.models.base import FEATURE_SETS, FIRST_TRAIN_SEASON, load_features, select_features
from footy.models.fitting import MatchModel
from footy.models.outcome import GBMOutcomeModel
from footy.models.poisson import PoissonGoalsModel

log = logging.getLogger(__name__)

PRODUCTION_DIR = PROJECT_ROOT / "models" / "production"
METADATA = "metadata.json"


def _members(feature_set: str, feats: list[str]) -> dict[str, MatchModel]:
    return {"poisson_dc": PoissonGoalsModel(feats, dixon_coles=True),
            "lightgbm": GBMOutcomeModel(feats)}


def fit_production(directory: Path = PRODUCTION_DIR) -> dict[str, Any]:
    df = load_features()
    train = df[df["is_usable"] & (df["season"] >= FIRST_TRAIN_SEASON)]
    directory.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seasons": [int(train["season"].min()), int(train["season"].max())],
        "n_matches": int(len(train)),
        "last_match_date": str(train["date"].max().date()),
        "models": {},
    }
    for fs in FEATURE_SETS:
        feats = select_features(df, fs)
        for name, model in _members(fs, feats).items():
            log.info("Fitting %s[%s] on %d matches", name, fs, len(train))
            model.fit(train)
            file = f"{name}[{fs}].joblib"
            joblib.dump(model, directory / file)
            meta["models"][f"{name}[{fs}]"] = {
                "file": file,
                "params": {k: (float(v) if isinstance(v, float) else v)
                           for k, v in (model.params_ or {}).items()},
                "calibrator": model.calibrator_.name if model.calibrator_ else "off",
                "n_features": len(feats),
            }
    (directory / METADATA).write_text(json.dumps(meta, indent=2))
    log.info("Saved production models to %s", directory)
    return meta


def load_production(directory: Path = PRODUCTION_DIR) -> tuple[dict[str, dict[str, MatchModel]], dict]:
    """Return ({feature_set: {member: model}}, metadata)."""
    meta_path = directory / METADATA
    if not meta_path.exists():
        raise FileNotFoundError(f"No production models in {directory}; run `footy fit` first")
    meta = json.loads(meta_path.read_text())
    bundle: dict[str, dict[str, MatchModel]] = {fs: {} for fs in FEATURE_SETS}
    for key, info in meta["models"].items():
        name, fs = key.rstrip("]").split("[")
        bundle[fs][name] = joblib.load(directory / info["file"])
    return bundle, meta
