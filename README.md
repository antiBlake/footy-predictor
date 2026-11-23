# footy — Big 5 football prediction system

Predicts, for the Premier League, La Liga, Serie A, Bundesliga and Ligue 1:

1. Match outcome probabilities (Home / Draw / Away)
2. Expected goals for each side
3. Player stats (goals, assists, shots) for likely starters

**Where it stands (23 Nov 2026):** stages 1–6 and 8 run end to end on real data:
about 20,000 matches, daily predictions, and an HTTP API. The match model is
statistically level with the betting market and well calibrated. Stage 7 (players)
is built and tested but has no data yet, because Understat, its only free
source, has been down since 20 Nov 2026. See **Handover** at the bottom.

## Status

| Stage | What | State |
|---|---|---|
| 1 | Football-Data.co.uk collector, all Big 5 | ✅ done |
| 2 | openfootball fixtures, team-name mapping, Understat xG, joined match tables | ✅ done (Understat run pending: site down 20 Nov 2026) |
| 3 | Feature engineering (market, Elo, rolling form, context) | ✅ done (xG features fill in once Understat is collected) |
| 4a | Goals model: independent Poisson + Dixon-Coles | ✅ done |
| 4b | Outcome model: multinomial logistic + LightGBM, calibrated | ✅ done |
| 5 | Walk-forward evaluation: log loss, Brier, RPS, calibration, betting backtest | ✅ done |
| 6 | Inference pipeline for upcoming fixtures | ✅ done |
| 7 | Player models (minutes × per-90 rates × team xG share) | ✅ built and tested; waiting for Understat data |
| 8 | CLI / FastAPI service, daily job, Docker | ✅ done |

## Project layout

```
football-predictor/
├── config/
│   ├── leagues.yaml          # league registry: add a league here
│   ├── team_aliases.yaml     # source team name -> canonical (Football-Data) name
│   ├── match_exclusions.yaml # matches kept but not used for training
│   └── crowd_restrictions.yaml # COVID closed-doors periods per league
├── data/                     # git-ignored
│   ├── raw/                  # untouched downloads, one file per source/league/season
│   ├── interim/              # parsed into the canonical schema, per source
│   └── processed/            # joined, team-mapped, feature tables
├── src/footy/
│   ├── config.py             # paths, league registry, season helpers
│   ├── logging_utils.py
│   ├── cli.py                # every `footy ...` command
│   ├── data/
│   │   ├── schema.py         # canonical match schema + validation
│   │   ├── football_data.py  # Football-Data.co.uk collector
│   │   ├── http.py           # retries, raw cache, atomic writes
│   │   ├── openfootball.py   # fixtures (incl. future) for mapping + inference
│   │   ├── understat.py      # xG / PPDA / deep / xPts via soccerdata
│   │   ├── teams.py          # name normalisation + fixture-based alias inference
│   │   └── build.py          # join sources -> data/processed/matches/<LEAGUE>.parquet
│   ├── features/             # market.py, elo.py, form.py, context.py, pipeline.py
│   ├── models/               # base, fitting, calibration, poisson, outcome, train
│   ├── evaluation/           # metrics, backtest (walk-forward), report, betting, render
│   ├── inference/            # fixtures.py, production.py, predict.py
│   ├── bot/                  # Telegram bot: bot.py, formatting.py, subscriptions.py
│   ├── players/              # data.py (Understat), model.py, predict.py, evaluate.py
│   └── api/                  # schemas.py, service.py, app.py (FastAPI)
├── tests/
└── notebooks/                # exploration only; no logic lives here
```

## Setup

Requires Python 3.11+. Recommended with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
uv pip install -e ".[dev,model,understat,api]"
brew install libomp        # macOS only: OpenMP runtime LightGBM needs
```

Optional: put `API_FOOTBALL_KEY=...` in `.env` (git-ignored). It's not used yet: the
free API-Football plan only covers 2022–2024, not the current season.

## Usage

```bash
# 1. collect (2015-16 to current season by default; --league EPL|...|ALL; --refresh)
footy collect --source football_data --league ALL
footy collect --source openfootball  --league ALL
footy collect --source understat     --league ALL

# 2. team names: infer aliases from fixtures, then verify coverage
footy teams propose --source understat            # updates config/team_aliases.yaml
footy teams check   --source understat

# 3. join into data/processed/matches/<LEAGUE>.parquet
footy build --league ALL

# 4. pre-match features -> data/processed/features/<LEAGUE>.parquet
footy features --league ALL

# 5. fit all models, score on 2024-25 onward (saves models/ and predictions)
footy train --test-from 2024

# 6. walk-forward backtest (refits each season) -> reports/backtest.md + reliability.png
footy backtest --from 2020
footy report            # rebuild the report from saved predictions

# 7. weekly routine
footy update            # refresh current season, rebuild matches + features
footy fit               # retrain production models (after each matchday, or at least monthly)
footy predict --days 7  # -> data/predictions/upcoming_<date>.csv/.parquet

pytest
```

## Key conventions

- **Seasons** are identified by their start year: `2024` = 2024-25.
- **Caching:** finished seasons are downloaded once. The current season is
  fetched again when the cached copy is older than 6 hours.
- **No silent failures:** bad data either raises `SchemaError` or is fixed
  with a logged warning. A season that doesn't exist upstream is skipped
  with a warning.
- **Never a random split:** all evaluation is walk-forward by date (stage 5).

## Features (Stage 3)

Each feature uses only matches that kicked off before the match it describes. The
test `test_no_leakage_...` changes one result and checks that nothing on or
before that date moves. Future fixtures go through the same code (pass
`fixtures=`), so training and inference share one feature path.

| Group | Features | Notes |
|---|---|---|
| market (6) | `mkt_p_home/draw/away`, `mkt_overround`, `mkt_p_over25`, `mkt_log_ratio` | Pre-match market average with the margin removed proportionally, Bet365 as fallback. Closing odds are not used (they're only known at kick-off). |
| elo (5) | `elo_home/away/diff`, `elo_p_home`, `elo_warmup` | Elo weighted by goal difference, K=20, home advantage 60. Ratings shrink 20% toward the league mean each summer. Promoted teams start at the relegated teams' average. |
| form (76) | `h_/a_{goals,shots,sot,corners,xg,npxg,ppda,deep}_{for,against}_ewm{s,l}`, `points`, `n_prev`, `rest_days`, `is_promoted` | Exponentially weighted averages with half-lives of 3 and 10 matches. Form resets when a team returns to the top flight. |
| context (2) | `crowd_restricted`, `season_progress` | `config/crowd_restrictions.yaml` |

Each row also has `is_usable` (has a score and isn't excluded), which marks the rows models train on.

**Reference points (2016-17 to now, all five leagues, 18,400 matches):** log loss
for always predicting the average outcome rates is 1.0705; the market scores
0.9647. Matches with restricted crowds had a 40.0% home-win rate against 44.6%
otherwise.

## Models (Stage 4)

All five leagues are trained together, with a league indicator. Each model is
trained twice, once with the market features and once without (`no_market`).

| Model | What it is |
|---|---|
| `poisson_raw` | Two L2-regularised Poisson GLMs (home goals, away goals) over the features; 1X2 from the scoreline grid. No calibration. |
| `poisson_dc` | Same plus the Dixon-Coles low-score correction (rho ≈ -0.055, adding 0-0 and 1-1 probability). Also gives expected goals, over 2.5 and both-teams-to-score. |
| `logistic` | Multinomial logistic regression on standardised features (median imputation plus missing-value flags). |
| `lightgbm` | Shallow, slow-learning gradient boosting that handles missing values itself. Falls back to sklearn's HistGradientBoosting if LightGBM won't load (macOS: `brew install libomp`). |

**Fit procedure (`models/fitting.py`, the same for every model):** settings are
tuned on each of the last 3 training seasons in turn, each time training only
on the seasons before it. The calibrator (none / temperature / Dirichlet) is
chosen by training it on two of those seasons and scoring it on the third,
then the model is refit on all training seasons. An early version calibrated
on a single season (2023-24, which had 26.4% draws) and made every model
worse. A test now checks that tuning never sees its own season or a later one.

**Holdout result (train 2016-17 to 2023-24, test 2024-25 onward, 4,004 matches; log loss):**

| market | lightgbm[with_market] | poisson_dc[with_market] | logistic[with_market] | poisson_dc[no_market] | base rate |
|---|---|---|---|---|---|
| **0.9682** | 0.9708 | 0.9721 | 0.9738 | 0.9820 | 1.0731 |

## Evaluation (Stage 5)

`footy backtest` predicts each season from 2020-21 onward using models trained only
on earlier seasons: 11,233 out-of-sample matches. Intervals come from a bootstrap
that resamples whole **weeks**, because matches played on the same weekend share
news and conditions and aren't independent.

| | log loss | vs market (95% interval) |
|---|---|---|
| market | **0.9715** | n/a |
| ensemble[with_market] (Dixon-Coles + LightGBM average) | 0.9723 | +0.0008 (-0.0004, +0.0020): no significant difference |
| poisson_dc[with_market] | 0.9736 | +0.0022 (+0.0007, +0.0037) |
| ensemble[no_market] | 0.9832 | +0.0118 (+0.0094, +0.0142) |

- The ensemble beats the market in the Bundesliga and Serie A and in 2 of 7 seasons. Overall the market is still ahead, and the gap is within noise.
- Calibration is good: the ensemble's expected calibration error is at most 0.014 for each outcome, lower than the market's own.
- **Betting backtest (research only):** every no-market model loses at every edge
  threshold (ROI -5% to -13%), and the average closing-line value is about -4%. The
  model's "edges" are mostly its own errors. Without xG (and lineups), there is
  no evidence of a betting edge.

## Predictions (Stage 6)

`footy predict` combines two fixture sources:

- **Football-Data `fixtures.csv`:** about the next week, with odds and already-canonical names. These fixtures use `ensemble[with_market]`.
- **openfootball:** later fixtures, without odds. These use `ensemble[no_market]`. Any pairing Football-Data already has, as a result or a fixture, is dropped, because openfootball's dates can be wrong.

The ensemble is the backtest winner: an average of the calibrated Dixon-Coles Poisson and LightGBM probabilities.

Output per fixture: home/draw/away probabilities and fair odds, the market's probabilities, expected goals, over 2.5, both teams to score, and the three most likely scorelines. The goals outputs come from the Poisson member of the ensemble.

A fixture that openfootball dates in the past but that has no result produces a warning. It's either postponed or the data needs `footy update`.

## API (Stage 8)

```bash
footy serve --port 8000                  # interactive docs at http://localhost:8000/docs
FOOTY_API_KEY=... footy serve            # also enables POST /admin/refresh
```

| Endpoint | What it does |
|---|---|
| `GET /health` | `ok` / `degraded` / `unavailable`, plus model training date, last result used, prediction age and a list of issues |
| `GET /leagues`, `GET /leagues/{key}/teams` | League list and canonical team names |
| `GET /predictions/upcoming?league=&date_from=&date_to=` | Latest precomputed predictions |
| `GET /predictions/{match_id}` | A single upcoming match |
| `GET /predictions/{match_id}/players` | Likely players for that match: chance to start or appear, expected minutes, anytime-scorer and assist probabilities, expected shots. Returns 404 until player data is collected. |
| `POST /predict` | Any fixture on demand. Team names can be canonical names or any known alias, in any case, and an unknown name returns 422 with suggestions. Pass all three odds to use the with-market model. |
| `POST /admin/refresh` | Recompute upcoming predictions. Needs the `X-API-Key` header, and is disabled unless `FOOTY_API_KEY` is set. |

Each prediction includes the home/draw/away probabilities, fair odds, the market's
probabilities (when odds exist), expected goals, over 2.5, both teams to score, the
five most likely scorelines, and which model was used.

**Scheduling.** `footy daily` runs update, then fit, then predict, and stops at the
first hard failure so it never serves stale models as fresh. Example crontab,
07:00 every day:
```
0 7 * * * cd ~/football-predictor && .venv/bin/footy daily >> logs/daily.log 2>&1
```
A running server picks up the new predictions on its next refresh or restart.

**Docker:** `docker build -t footy .`, then run with `data/` and `models/` mounted
(see the header of the `Dockerfile`).

## Telegram bot

```bash
footy bot      # long-polling, so no public URL or webhook is needed
```

**Setup (5 minutes)**
1. In Telegram, message **@BotFather**, send `/newbot`, and pick a name. Copy the token.
2. Paste it into `.env` as `TELEGRAM_BOT_TOKEN=...` (the placeholder lines are already there).
3. Run `footy bot` and message your bot. It replies with your chat id.
4. Put that id in `TELEGRAM_ALLOWED_CHAT_IDS`, then restart the bot.

The bot is **private**: any chat not on that list only gets told its own chat id.

| Command | What it does |
|---|---|
| `/today`, `/week [EPL]` | Fixtures with home · draw · away chances, favourite in bold |
| `/match Arsenal` | The team's next match: probabilities, market odds, expected goals, likely scores, likely scorers (once player data exists) |
| `/predict Arsenal v Man City [2.4 3.4 2.9]` | Any fixture on demand. Adding odds uses the stronger with-market model. |
| `/status` | Model and data freshness |
| `/subscribe`, `/unsubscribe` | Daily digest of the day's matches at `TELEGRAM_DIGEST_TIME` |
| `/refresh` | Re-fetch fixtures and recompute predictions now |

Each morning, before the digest, the bot runs the full `update → fit → predict`
pipeline. Set `TELEGRAM_DAILY_PIPELINE=0` to skip it, for example if cron already runs
`footy daily`. Model work runs in a background thread, so the bot keeps responding.

**Running on GitHub Actions (current setup):** `.github/workflows/tick.yml` answers
messages and `daily.yml` refreshes, retrains, predicts and sends the digest. Data and
models live in the Actions cache, and the token, chat IDs and timezone are in repo
Secrets. **GitHub's scheduled runs are best effort:** in practice they arrive hours
apart, so replies and the digest can be hours late. Telegram discards messages not
fetched within 24 hours. For a reply now, run the `bot-tick` workflow manually
(Actions tab, or `gh workflow run tick.yml`). Two ways to get fast replies:
host `footy bot` on an always-on server, or have a Cloudflare Worker start the
workflow when each message arrives.

**Keeping it running elsewhere:** it only works while `footy bot` is running. Use a
terminal or `tmux`, or a launchd agent / systemd service for an always-on machine.

## Players (Stage 7)

`players/model.py` estimates, for each team in a fixture:

1. **Who plays.** Recent team matches, weighted toward the latest, give each
   player's chance of starting or coming on and his expected minutes in each
   case. A player who misses three or more matches in a row is treated as
   probably out, since there's no injury feed.
2. **How productive.** xG, shots, xA and key passes per 90 minutes, decayed over
   his own appearances and shrunk toward his position's average. A player with
   only 30 minutes of football doesn't get credited with a fluke 4.5 xG per 90.
3. **Team context.** Player numbers are scaled so the team's total expected goals
   equals the match model's expected goals for this fixture, less about 3% for own
   goals. Player tables can never contradict the match prediction.

Anytime-scorer probability mixes a Poisson model over the start and substitute
cases. `footy predict` adds `upcoming_players_latest.parquet` automatically once
player data exists. `footy players backtest` scores the model against a
"goals per appearance over the last 38" baseline. Every player who actually
played is scored, so missing a scorer counts against the model.

**Limits:** there are no lineups or injury news, so a player returning from
injury is invisible until he plays. xG includes penalties, so penalty takers
aren't modelled separately. Goals and assists aren't modelled jointly.

## Team names

Canonical names are Football-Data.co.uk's (e.g. `Man United`, `Ath Madrid`).
Other sources map onto them through `config/team_aliases.yaml`, scoped by league.
`footy teams propose` builds that file automatically: matches from two sources
with the same date (±2 days) and the same score "vote" for a name pairing, and only
clear winners are accepted. Anything unclear is printed for you to review, and
entries you edit by hand are never overwritten. Unmapped names raise an error
listing every missing name; they are never dropped silently.

## Data notes

- **openfootball is not a source of results.** Use it only for name mapping and
  future fixtures. Known problems: Premier League 2019-20 post-restart games are
  dated a year early (fixed on load), and La Liga and Serie A are missing the
  last matchday of 2024-25.
- **Awarded results:** Ligue 1 2016-17 Bastia v Lyon (abandoned, awarded 0-3) is
  flagged in `exclude_reason`. Serie A 2020-21 Verona v Roma keeps its on-pitch 0-0.
- **Understat outages** return an HTML page with HTTP 200, and soccerdata caches it
  as JSON. The collector checks the site first and deletes any broken cache files.

- Pinnacle odds (`PSH/PSD/PSA`) stop partway through 2025-26 (55% coverage,
  0% in 2026-27). Models should use the **market average** (`odds_avg_*`),
  which has full coverage, and treat Pinnacle as optional.
- Betbrain averages (`BbAv*`) were replaced by `Avg*` in 2019-20; the
  collector merges them into one `odds_avg_*` column.

## Handover

**Routine** (or schedule `footy daily`):
```bash
footy update && footy fit && footy predict --days 7
footy serve
```

**When Understat is back** (`curl -s https://understat.com | grep -c "come back soon"` prints 0):
```bash
footy collect --source understat --league ALL         # team xG, PPDA, deep, xPts
footy teams propose --source understat && footy teams check --source understat
footy build && footy features                         # xG form features fill in
footy backtest --from 2020                            # does xG close the gap to the market?
footy players collect --from 2021                     # slow: one request per match
footy players backtest --season 2025                  # check the player model first
footy fit && footy predict                            # player tables now included
```

**Results so far** (walk-forward, 2020-21 to now, 11,233 matches): the ensemble
scores 0.9723 log loss against the market's 0.9715, a difference within noise,
and is well calibrated. There's no betting edge: closing-line value is -4%.

**Most likely improvements, in order:** xG features (Understat), then confirmed
lineups and injuries (paid API-Football plan), then shot-level xG models.

**Decided against:** FBref through soccerdata. It hits a Cloudflare CAPTCHA and
tries to click through it with GUI automation, which is a way around the site's
bot protection.

**Tests:** `pytest` runs 86 tests in about 4 seconds, with no network or trained
models needed. They cover leakage checks for features, model fitting, the
walk-forward backtest and the player model; schema and team-name mapping; the
API; and the Telegram bot.
