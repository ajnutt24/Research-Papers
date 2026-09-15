"""
show_results.py
===============
Print the forecast. Finds the project folder itself, so it works no matter
which directory the notebook is in, and explains what is missing if the
pipeline has not produced results yet.

    %run show_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FOLDER = "midterms-2026"

try:
    PROJECT = Path(__file__).resolve().parent
except NameError:
    here = Path.cwd().resolve()
    PROJECT = None
    for base in [here, *here.parents]:
        for cand in (base, base / FOLDER, base / "research-papers" / FOLDER):
            if (cand / "config.py").is_file() and (cand / "run_pipeline.py").is_file():
                PROJECT = cand
                break
        if PROJECT:
            break
    if PROJECT is None:
        raise SystemExit("Could not find the midterms-2026 folder. Run START_HERE.py first.")

OUT = PROJECT / "outputs"
if Path.cwd().resolve() != PROJECT:
    print(f"(notebook is in {Path.cwd()}; reading results from {PROJECT})\n")

cc = OUT / "chamber_control.json"
rp = OUT / "race_probabilities.csv"

if not cc.exists():
    print("No results yet: outputs/chamber_control.json does not exist.\n")
    print(f"Looked in: {OUT}")
    produced = sorted(p.name for p in OUT.glob("*")) if OUT.exists() else []
    print(f"Files currently in outputs/: {produced if produced else '(none)'}\n")
    print("chamber_control.json is written by stage 13, which needs 11 and 12 first.")
    print("Run:\n    %run run_pipeline.py --from 09\n")
    log = OUT / "pipeline_log.txt"
    if log.exists():
        lines = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        print("Last 25 lines of the previous run (outputs/pipeline_log.txt):")
        print("-" * 70)
        for l in lines[-25:]:
            print(l)
    raise SystemExit(0)

d = json.loads(cc.read_text())
print(f"Forecast as of {d['asof']}  ({d['days_to_election']} days to the election)")
print(f"{d['n_sims']:,} simulations   data quality: {d['provenance']}\n")
print(f"{'CHAMBER':10s} {'P(D control)':>13s} {'median D seats':>15s} {'80% range':>14s}")
print("-" * 56)
for ch in ("House", "Senate", "Governor"):
    v = d[ch]
    print(f"{ch:10s} {v['p_dem_control']:>12.0%} {v['dem_seats_median']:>15.0f} "
          f"{str(int(v['dem_seats_p10'])) + '-' + str(int(v['dem_seats_p90'])):>14s}")

if d["provenance"] == "fixture":
    print("\n" + "!" * 56)
    print("Some inputs are PLACEHOLDERS, so these numbers are a pipeline test,")
    print("not a real forecast. Run  %run check_data.py  to see which inputs.")
    print("!" * 56)

if rp.exists():
    import pandas as pd
    r = pd.read_csv(rp)
    comp = r[(r.p_dem > .25) & (r.p_dem < .75)].sort_values("p_dem")
    print(f"\n{len(comp)} competitive races (win probability between 25% and 75%):")
    cols = ["race_id", "office", "polled", "rating", "blend_margin", "p_dem"]
    with pd.option_context("display.max_rows", 100, "display.width", 120):
        print(comp[cols].to_string(index=False,
              formatters={"blend_margin": "{:+.1f}".format, "p_dem": "{:.0%}".format}))
    print(f"\nFull table: {rp}")
