"""Project paths, league registry and season helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml

def _find_project_root() -> Path:
    """FOOTY_ROOT if set; else the source checkout this file lives in (editable install /
    running from src); else the current directory. A regular `pip install .` puts this
    file in site-packages, where parents[2] is not the project."""
    if os.environ.get("FOOTY_ROOT"):
        return Path(os.environ["FOOTY_ROOT"])
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "config" / "leagues.yaml").exists():
        return source_root
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = Path(os.environ.get("FOOTY_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"


@dataclass(frozen=True)
class League:
    key: str
    name: str
    country: str
    football_data_code: str
    openfootball_code: str
    soccerdata_id: str
    understat_name: str


@lru_cache(maxsize=1)
def load_leagues(path: Path | None = None) -> dict[str, League]:
    """Load the league registry from config/leagues.yaml."""
    path = path or CONFIG_DIR / "leagues.yaml"
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    return {key: League(key=key, **fields) for key, fields in raw["leagues"].items()}


def get_league(key: str) -> League:
    leagues = load_leagues()
    try:
        return leagues[key.upper()]
    except KeyError:
        raise ValueError(f"Unknown league {key!r}. Known: {sorted(leagues)}") from None


# --- Seasons -----------------------------------------------------------------
# A season is identified by its starting year: 2024 means 2024-25.
# Football-Data encodes it as "2425".

SEASON_START_MONTH = 7  # seasons roll over in July


def season_code(start_year: int) -> str:
    """2024 -> '2425'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2024 -> '2024-25'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def current_season(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= SEASON_START_MONTH else today.year - 1
