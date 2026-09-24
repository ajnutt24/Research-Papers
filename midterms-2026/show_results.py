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
    cols = ["race_id", "office", "polled", "rating", "blend_margin", "p_dem"]
    fmt = {"blend_margin": "{:+.1f}".format, "p_dem": "{:.0%}".format}

    watch_file = PROJECT / "data_store" / "manual" / "watchlist.csv"
    use_watch = "--watchlist" in sys.argv or "--watch" in sys.argv
    if use_watch and not watch_file.exists():
        print(f"\nNo watchlist at {watch_file}")
        use_watch = False

    if use_watch:
        wl = pd.read_csv(watch_file, comment="#")
        sub = r[r.race_id.isin(set(wl.race_id))].copy()
        missing = set(wl.race_id) - set(r.race_id)
        print(f"\nWatchlist: {len(sub)} of {len(wl)} races"
              + (f" ({len(missing)} not in the forecast: {sorted(missing)})" if missing else ""))
        exp = sub.groupby("office").p_dem.sum().round(1)
        print("\nExpected Democratic seats from the watchlist alone:")
        for office, v in exp.items():
            n = int((sub.office == office).sum())
            print(f"   {office:9s} {v:5.1f} of {n}")
        for office in ["Senate", "House"]:
            part = sub[sub.office == office].sort_values("p_dem")
            if not len(part):
                continue
            print(f"\n{office} watchlist ({len(part)} races), least to most Democratic:")
            with pd.option_context("display.max_rows", 200, "display.width", 130):
                print(part[cols].to_string(index=False, formatters=fmt))
    else:
        comp = r[(r.p_dem > .25) & (r.p_dem < .75)].sort_values("p_dem")
        print(f"\n{len(comp)} competitive races (win probability between 25% and 75%):")
        with pd.option_context("display.max_rows", 100, "display.width", 120):
            print(comp[cols].to_string(index=False, formatters=fmt))
        if watch_file.exists():
            print("\n(a watchlist exists: run  %run show_results.py --watchlist  to see just those races)")
    print(f"\nFull table: {rp}")
