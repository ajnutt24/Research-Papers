"""
diagnose_polls.py
=================
Shows exactly what the scraper sees on one race's Wikipedia article: every
table, its heading, its columns, and whether the parser accepted or rejected
it, with the reason. Use it when a race returns fewer polls than the article
actually contains.

    %run diagnose_polls.py S-NE
    %run diagnose_polls.py S-NE --dump      # also print the raw rows

Send the output and the parser can be corrected for that table shape.
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

args = [a for a in sys.argv[1:] if not a.startswith("--")]
race = args[0].upper() if args else "S-NE"
dump = "--dump" in sys.argv

sys.path.insert(0, str(_ROOT / "data"))
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("s3", _ROOT / "data" / "03_fetch_polls.py")
s3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s3)

title = dict(s3.wiki_titles()).get(race)
if title is None:
    print(f"{race} is not in the guessed-title list; trying link discovery ...")
    title = s3.discover_race_articles().get(race)
if title is None:
    raise SystemExit(f"No Wikipedia article known for {race}.")

print(f"Race   : {race}")
print(f"Article: https://en.wikipedia.org/wiki/{title}\n")

try:
    html, prov = fetch_text(WIKI_API + title, f"wiki_{race}.html", max_age_hours=0)  # always fresh
except Exception as e:
    raise SystemExit(
        f"Could not reach Wikipedia: {type(e).__name__}\n"
        "This machine has no route to en.wikipedia.org. Run this on Colab, "
        "where it is reachable."
    )
print(f"Fetched {len(html):,} characters ({prov})\n")

accepted = rejected = 0
rows_total = 0
for i, (heading, df) in enumerate(wikipolls.iter_tables_with_sections(html)):
    cols = [wikipolls._flatten(c) for c in df.columns]
    df2 = df.copy()
    df2.columns = cols
    roles = wikipolls.identify_columns(cols)
    skip = ""
    if wikipolls.PRIMARY_RE.search(heading or ""):
        skip = "SKIPPED: heading looks like a primary/runoff"
    elif wikipolls.HYPOTHETICAL_RE.search(heading or ""):
        skip = "SKIPPED: heading looks hypothetical"
    elif roles is None:
        need = []
        if not any(wikipolls.POLLSTER_COL.search(c.lower()) for c in cols):
            need.append("no pollster column")
        if not any(wikipolls.DATE_COL.search(c.lower()) for c in cols):
            need.append("no date column")
        if not any(wikipolls.DEM_COL.search(c) for c in cols):
            need.append("no Democratic candidate column")
        if not any(wikipolls.REP_COL.search(c) for c in cols):
            need.append("no Republican candidate column")
        skip = "REJECTED: " + "; ".join(need or ["columns did not map"])

    print(f"[table {i}] heading: {heading!r}")
    print(f"           shape  : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"           columns: {cols}")
    if skip:
        print(f"           {skip}")
        rejected += 1
    else:
        part = wikipolls.parse_tables([df], race, config.CYCLE)
        accepted += 1
        rows_total += len(part)
        print(f"           ACCEPTED -> {len(part)} poll rows parsed "
              f"(from {df.shape[0]} table rows)")
        if len(part) < df.shape[0]:
            print(f"           NOTE: {df.shape[0] - len(part)} row(s) dropped "
                  f"(unreadable date or percentages)")
        if dump and len(part):
            print(part[["pollster", "end_date", "sample_size", "population",
                        "dem_pct", "rep_pct"]].to_string(index=False))
    print()

print("=" * 60)
print(f"{accepted} table(s) accepted, {rejected} skipped or rejected, {rows_total} poll rows total")
if rows_total:
    df = wikipolls.parse_html_sections(html, lambda h: race, config.CYCLE)
    if len(df):
        print(f"date range: {df.end_date.min()} to {df.end_date.max()}")
print("\nIf a table you expected is REJECTED, send this output: the column list")
print("shows what the parser needs to learn about that table's shape.")
