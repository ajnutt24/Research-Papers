"""
check_data.py
=============
Reports where every input in the current run actually came from.

    python3 check_data.py        # terminal
    %run check_data.py           # notebook

"live"/"cache"  = fetched from the real source
"mirror"        = public GitHub copy of an official dataset (real data)
"manual"        = a CSV you filled in yourself (real data)
"fixture"       = SIMULATED PLACEHOLDER - not real, do not publish
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    _ROOT = Path(__file__).resolve().parent
except NameError:
    _ROOT = next((d for c in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
                  for d in (c, c / "midterms-2026") if (d / "config.py").is_file()), Path.cwd())
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from utils import load_meta  # noqa: E402

REAL = {"live", "cache", "mirror", "manual"}

# stage output -> (what it is, which script produces it, how to fix a fixture)
INPUTS = [
    ("economic_cycle_features", "CPI / unemployment / gas (FRED)", "01",
     "needs fred.stlouisfed.org reachable; set FRED_API_KEY for the official API"),
    ("gas_prices", "AAA daily gas price", "02",
     "needs gasprices.aaa.com reachable; EIA (in stage 01) is the backup series"),
    ("polls_2026", "2026 polls", "03",
     "point config.POLL_WORKBOOK at MidtermPolls2026.xlsx, or fill "
     "data_store/manual/polls_2026.csv (template in manual/templates/)"),
    ("approval_2026", "presidential approval", "04",
     "the workbook's US-APPROVAL rows, or data_store/manual/approval_manual.csv"),
    ("race_ratings_2026", "Cook / Sabato / Inside ratings", "05",
     "the workbook's Ratings sheet, or data_store/manual/race_ratings_2026.csv"),
    ("fundamentals_2026", "partisan lean, incumbency, fundraising", "06",
     "add pvi_manual.csv and incumbency overrides; FEC needs api.open.fec.gov"),
    ("historical_results", "1976-2024 election results", "07", "-"),
    ("rating_calibration", "rating tier calibration", "10", "-"),
    ("stacking_weights", "LOO stacking weights", "12", "-"),
]

print(f"Forecast date: {config.FORECAST_ASOF}  ({config.DAYS_TO_ELECTION} days to the election)\n")

# The polling workbook is the primary input, and a workbook that was not saved
# this week is the likeliest way to get a confident forecast built on old polls.
try:
    import datetime as _dt

    import workbook as _wb
    _p = _wb.find_workbook(quiet=True)
    if _p is None:
        print("Polling workbook: NOT FOUND (falling back to the Wikipedia scraper)")
        print("   looked in:")
        for _c in _wb.candidate_paths():
            print(f"      {_c}")
    else:
        _m = _dt.datetime.fromtimestamp(_p.stat().st_mtime)
        _age = (_dt.datetime.now() - _m).days
        print(f"Polling workbook: {_p}")
        print(f"   last saved {_m:%Y-%m-%d %H:%M} ({_age} day(s) ago)"
              + ("   <-- STALE, re-save it before trusting this run" if _age > 10 else ""))
    print()
except Exception as _e:  # noqa: BLE001
    print(f"Polling workbook: could not be checked ({type(_e).__name__}: {_e})\n")
print(f"{'INPUT':38s} {'SOURCE':9s} {'STAGE':5s}")
print("-" * 70)

fixtures = []
missing = []
for name, label, stage, fix in INPUTS:
    m = load_meta(name)
    prov = m.get("provenance")
    if prov is None:
        missing.append((label, stage))
        mark, prov = " ", "not run"
    elif prov in REAL:
        mark = " "
    else:
        mark = "!"
        fixtures.append((label, stage, fix))
    print(f"{mark} {label:36s} {prov:9s} {stage:5s}")

print()
if fixtures:
    print(f"{len(fixtures)} input(s) are PLACEHOLDERS, so the forecast is not yet real:\n")
    for label, stage, fix in fixtures:
        print(f"  - {label} (stage {stage})")
        print(f"      fix: {fix}")
    print("\nAfter fixing any of these, re-run:   %run run_pipeline.py --update")
else:
    print("All inputs are real. The forecast in outputs/ is usable.")

if missing:
    print("\nNot yet generated (run the pipeline):", ", ".join(f"{l} ({s})" for l, s in missing))

# headline numbers, if present
cc = config.OUTPUTS / "chamber_control.json"
if cc.exists():
    import json
    d = json.loads(cc.read_text())
    print(f"\nLast simulation: {d['n_sims']:,} runs, overall provenance = {d['provenance']}")
    for ch in ("House", "Senate", "Governor"):
        print(f"   {ch:9s} P(D control) = {d[ch]['p_dem_control']:.0%}   median D seats = {d[ch]['dem_seats_median']:.0f}")
