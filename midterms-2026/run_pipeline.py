"""
run_pipeline.py
===============
Convenience runner. Three modes:

  python3 run_pipeline.py --update      # new polls dropped: 03 -> 08 -> 11 -> 12 -> 13   (~1 min)
  python3 run_pipeline.py --full        # everything, 01 -> 14                              (~6 min)
  python3 run_pipeline.py 03 08 13      # any explicit list of stage numbers, in that order

Each stage reads the cached outputs of the stages it depends on, so the
update mode only re-runs what new polls can change: poll aggregation, the
hierarchical model, the blend (cached stacking weights), and the simulation.
Fundamentals (09) and ratings (10) are untouched unless you ask for them.
Nothing runs on a timer: put the --update line in cron or a scheduled task if
you want it hands-off (e.g. `0 7 * * * cd /path/midterms-2026 && python3 run_pipeline.py --update`).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
STAGES = {
    "01": "data/01_fetch_economic_data.py", "02": "data/02_fetch_gas_prices.py",
    "03": "data/03_fetch_polls.py", "04": "data/04_fetch_approval.py",
    "05": "data/05_fetch_race_ratings.py", "07": "data/07_fetch_historical_results.py",
    "06": "data/06_fetch_fundamentals.py", "08": "model/08_poll_aggregation.py",
    "09": "model/09_fundamentals_model.py", "10": "model/10_rating_to_margin_calibration.py",
    "11": "model/11_hierarchical_model.py", "12": "model/12_blend_stacking.py",
    "13": "simulation/13_monte_carlo.py", "14": "validation/14_backtest.py",
}
FULL = ["01", "02", "03", "04", "05", "07", "06", "08", "09", "10", "11", "12", "13", "14"]
UPDATE = ["03", "08", "11", "12", "13"]


def main(argv: list[str]):
    if "--full" in argv:
        order = FULL
    elif "--update" in argv or not argv:
        order = UPDATE
    else:
        order = [a.zfill(2) for a in argv if a.zfill(2) in STAGES]
    extra = [a for a in argv if a.startswith("--") and a not in ("--full", "--update")]
    for st in order:
        t0 = time.perf_counter()
        print(f"\n===== stage {st}: {STAGES[st]} =====", flush=True)
        rc = subprocess.call([sys.executable, str(ROOT / STAGES[st]), *extra], cwd=ROOT)
        print(f"===== stage {st} finished in {time.perf_counter() - t0:.0f}s (exit {rc}) =====", flush=True)
        if rc != 0:
            sys.exit(f"stage {st} failed; later stages not run")
    import json
    cc = ROOT / "outputs" / "chamber_control.json"
    if cc.exists() and "13" in order:
        d = json.loads(cc.read_text())
        print(f"\nprovenance={d['provenance']}  asof={d['asof']}  ({d['days_to_election']} days out)")
        for ch in ("House", "Senate", "Governor"):
            print(f"  {ch:9s} P(D control)={d[ch]['p_dem_control']:.0%}  median D seats={d[ch]['dem_seats_median']:.0f}")


if __name__ == "__main__":
    main(sys.argv[1:])
