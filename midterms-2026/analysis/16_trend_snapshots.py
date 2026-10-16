"""
16_trend_snapshots.py
=====================
Build a forecast time series by replaying the model at past as-of dates.

    python3 analysis/16_trend_snapshots.py                 # fill any missing dates
    python3 analysis/16_trend_snapshots.py --from 2026-08-01
    python3 analysis/16_trend_snapshots.py --every 3       # every 3 days
    python3 analysis/16_trend_snapshots.py --dates 2026-10-06,2026-10-09

Why this works
--------------
Nothing archives a run: save_stage overwrites, so every past forecast this
project has produced is gone. But the forecast is a pure function of the polls
it can see, and `aggregate_polls` filters on `end_date <= FORECAST_ASOF`. Set
that variable to a past date and the pipeline reconstructs what it would have
said from the poll set as it stood. History can therefore be rebuilt
retroactively instead of having to be captured as it happens.

Only four stages depend on the as-of date: 08 poll aggregation, 11 the
hierarchical fit, 12 the blend and 13 the simulation. Economic data,
fundamentals, ratings and the rating calibration do not move with it, so they
are left alone and the replay costs one PyMC fit per date.

What this is NOT
----------------
Two things make a replay different from the forecast that was actually
published on a past date, and both run in the same direction:

1. It uses TODAY'S CODE. The poll-age decay, the rating band projection, the
   House district scraping and the independent-candidate handling were all
   added during October. An August replay shows what the current model thinks
   about August's polls, not what would have been published in August.

2. It uses TODAY'S RATINGS. race_ratings_2026 carries no as-of date, so the
   Cook ratings published on 2026-10-15 are applied to every replayed date.
   Where a rating moved, the replay credits the model with having known it
   early.

So the series is "one consistent method applied to a growing poll set", which
is the right object for a "how the race moved" chart and the wrong one for
"was the forecast wrong in August". Say which one it is when publishing it.

Outputs (data_store/processed/)
-------------------------------
trend_chambers.parquet   asof_date, office, p_dem_control, median, p10, p90, n_polls
trend_races.parquet      asof_date, race_id, office, p_dem, p_dem_caucus, blend_margin,
                         rating, n_polls

The date column is called asof_date, not asof: "asof" shadows
DataFrame.asof() and attribute access would silently return a method.

Both are APPENDED to and de-duplicated on their keys, so re-running only pays
for dates that are missing. The live outputs are restored at the end by
replaying today's date last.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
import config  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("trend_snapshots")

STAGES = ["model/08_poll_aggregation.py", "model/11_hierarchical_model.py",
          "model/12_blend_stacking.py", "simulation/13_monte_carlo.py"]
CHAMBERS = _ROOT / "data_store" / "processed" / "trend_chambers.parquet"
RACES = _ROOT / "data_store" / "processed" / "trend_races.parquet"
DEFAULT_START = date(2026, 6, 1)


def run_asof(d: date) -> bool:
    """Run the as-of-dependent stages with FORECAST_ASOF pinned to `d`."""
    env = {**os.environ, "FORECAST_ASOF": d.isoformat()}
    for stage in STAGES:
        r = subprocess.run([sys.executable, "-u", str(_ROOT / stage)], env=env,
                           capture_output=True, text=True, cwd=str(_ROOT))
        if r.returncode != 0:
            log.error("  %s failed for %s:\n%s", Path(stage).name, d, r.stderr[-1500:])
            return False
    return True


def harvest(d: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    cc = json.loads((config.OUTPUTS / "chamber_control.json").read_text())
    rp = pd.read_csv(config.OUTPUTS / "race_probabilities.csv")
    polled = pd.read_parquet(config.DATA_PROCESSED / "poll_estimates_2026.parquet")
    n_by_race = dict(zip(polled.race_id, polled.n_polls))

    ch = pd.DataFrame([{
        "asof_date": d, "office": office,
        "p_dem_control": cc[office]["p_dem_control"],
        "median": cc[office]["dem_seats_median"],
        "p10": cc[office]["dem_seats_p10"], "p90": cc[office]["dem_seats_p90"],
        "n_polls": int(sum(v for k, v in n_by_race.items()
                           if k.startswith({"House": "H-", "Senate": "S-", "Governor": "G-"}[office]))),
    } for office in ("House", "Senate", "Governor")])

    cols = [c for c in ("race_id", "office", "p_dem", "p_dem_caucus", "blend_margin", "rating")
            if c in rp.columns]
    rc = rp[cols].copy()
    rc["asof_date"] = d
    rc["n_polls"] = rc.race_id.map(n_by_race).fillna(0).astype(int)
    return ch, rc


def append(path: Path, new: pd.DataFrame, keys: list[str]) -> None:
    old = pd.read_parquet(path) if path.exists() else None
    df = pd.concat([old, new], ignore_index=True) if old is not None else new
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.date
    df = df.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)
    df.to_parquet(path, index=False)


def wanted_dates(argv: list[str]) -> list[date]:
    today = config.FORECAST_ASOF
    if "--dates" in argv:
        raw = argv[argv.index("--dates") + 1]
        return sorted({date.fromisoformat(x.strip()) for x in raw.split(",") if x.strip()})
    start = (date.fromisoformat(argv[argv.index("--from") + 1])
             if "--from" in argv else DEFAULT_START)
    step = int(argv[argv.index("--every") + 1]) if "--every" in argv else 7
    out, d = [], start
    while d < today:
        out.append(d)
        d += timedelta(days=step)
    out.append(today)          # today's run goes last, which restores live state
    return out


def main(argv: list[str]) -> None:
    dates = wanted_dates(argv)
    done: set[date] = set()
    if CHAMBERS.exists() and "--force" not in argv:
        done = set(pd.to_datetime(pd.read_parquet(CHAMBERS)["asof_date"]).dt.date)
    todo = [d for d in dates if d not in done or d == config.FORECAST_ASOF]
    log.info("%d date(s) requested, %d already stored, %d to run",
             len(dates), len(dates) - len(todo), len(todo))
    if not todo:
        log.info("nothing to do; pass --force to rebuild")
        return

    budget = float(argv[argv.index("--budget") + 1]) if "--budget" in argv else None
    import time
    t0 = time.time()
    for i, d in enumerate(todo, 1):
        if budget and time.time() - t0 > budget and d != config.FORECAST_ASOF:
            log.info("time budget reached; %d date(s) left for the next run", len(todo) - i + 1)
            break
        log.info("[%d/%d] replaying %s ...", i, len(todo), d)
        if not run_asof(d):
            log.warning("  skipped %s", d)
            continue
        ch, rc = harvest(d)
        append(CHAMBERS, ch, ["asof_date", "office"])
        append(RACES, rc, ["asof_date", "race_id"])
        row = ch.set_index("office")
        log.info("  House %.1f%% / %d   Senate %.1f%% / %d   Governor %.1f%% / %d   (%d polls)",
                 100 * row.loc["House", "p_dem_control"], row.loc["House", "median"],
                 100 * row.loc["Senate", "p_dem_control"], row.loc["Senate", "median"],
                 100 * row.loc["Governor", "p_dem_control"], row.loc["Governor", "median"],
                 int(row["n_polls"].sum()))

    # Always leave the live outputs on the real as-of date.
    if todo and todo[-1] != config.FORECAST_ASOF:
        log.info("restoring live run at %s", config.FORECAST_ASOF)
        run_asof(config.FORECAST_ASOF)
    log.info("wrote %s and %s", CHAMBERS.name, RACES.name)


if __name__ == "__main__":
    main(sys.argv[1:])
