"""
test_wikipedia_live.py
======================
Checks that Wikipedia poll scraping works from THIS machine, and shows what
it found. Run it before a full pipeline run if you want fast feedback.

    %run test_wikipedia_live.py            # a few representative races
    %run test_wikipedia_live.py --all      # every 2026 race article
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    _ROOT = Path(__file__).resolve().parent
except NameError:
    _ROOT = Path.cwd()
sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import config  # noqa: E402
import wikipolls  # noqa: E402
from utils import fetch_text  # noqa: E402

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/html/"
SAMPLE = [
    ("GENERIC", "2026_United_States_House_of_Representatives_elections"),
    ("S-GA", "2026_United_States_Senate_election_in_Georgia"),
    ("S-NC", "2026_United_States_Senate_election_in_North_Carolina"),
    ("S-MI", "2026_United_States_Senate_election_in_Michigan"),
    ("S-ME", "2026_United_States_Senate_election_in_Maine"),
    ("G-AZ", "2026_Arizona_gubernatorial_election"),
]

if "--all" in sys.argv:
    sys.path.insert(0, str(_ROOT / "data"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("s3", _ROOT / "data" / "03_fetch_polls.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    targets = m.wiki_titles()
else:
    targets = SAMPLE

print(f"Checking {len(targets)} Wikipedia article(s)\n")
print(f"{'RACE':16s} {'POLLS':>6s}  {'LATEST':>12s}  NOTE")
print("-" * 72)
total, reachable, rows_all = 0, 0, []
for rid, title in targets:
    total += 1
    try:
        html, prov = fetch_text(WIKI_API + title, f"wiki_{rid}.html", max_age_hours=12)
    except Exception as e:
        print(f"{rid:16s} {'-':>6s}  {'-':>12s}  unreachable ({type(e).__name__})")
        continue
    reachable += 1
    df = wikipolls.parse_html(html, rid, default_year=config.CYCLE)
    if len(df):
        rows_all.append(df)
        latest = max(df.end_date)
        print(f"{rid:16s} {len(df):>6d}  {str(latest):>12s}  ok ({prov})")
    else:
        print(f"{rid:16s} {0:>6d}  {'-':>12s}  article reached, no poll table found")

print("-" * 72)
print(f"{reachable}/{total} articles reachable")
if rows_all:
    allp = pd.concat(rows_all, ignore_index=True)
    print(f"{len(allp)} polls parsed across {allp.race_id.nunique()} races\n")
    print("Most recent 10:")
    cols = ["race_id", "pollster", "end_date", "sample_size", "population", "dem_pct", "rep_pct"]
    print(allp.sort_values("end_date", ascending=False).head(10)[cols].to_string(index=False))
    print("\nWikipedia scraping works here. Run:  %run run_pipeline.py --update")
else:
    print("\nNo polls parsed. Either the articles have no polling tables yet, or")
    print("this machine cannot reach Wikipedia. Send this output and I can adjust.")
