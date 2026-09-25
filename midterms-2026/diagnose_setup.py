"""
diagnose_setup.py
=================
Answers one question: why are the polls still placeholders on this machine?

There are only a few possible causes, and this separates them:
  * the code is older than the parser fixes
  * this machine cannot reach Wikipedia (the poll source)
  * the pipeline has not been re-run since the code was updated

    %run diagnose_setup.py

Prints a verdict and the exact next command. Changes nothing.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def _root() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd().resolve()
        for cand in [here, *here.parents]:
            for d in (cand, cand / "midterms-2026"):
                if (d / "config.py").is_file():
                    return d
        return Path.cwd()


ROOT = _root()
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

print(f"Project : {ROOT}")
print(f"Code ver: {getattr(config, 'CODE_VERSION', '(pre-versioning)')}\n")

problems = []

# 1. Is the code new enough to contain the parser fixes?
print("1. CODE VERSION")
try:
    import wikipolls
    needed = ["parse_approval", "IND_COL", "AGGREGATOR_COL", "parse_html_sections"]
    missing = [n for n in needed if not hasattr(wikipolls, n)]
    if missing:
        print(f"   OUT OF DATE - missing: {', '.join(missing)}")
        problems.append("stale_code")
    else:
        print("   up to date (independent-candidate and aggregator fixes present)")
except Exception as e:
    print(f"   could not import wikipolls: {type(e).__name__}: {e}")
    problems.append("stale_code")

# 2. Can this machine reach the sources?
print("\n2. NETWORK")
from utils import SESSION  # noqa: E402

CHECKS = [
    ("Wikipedia (polls)", "https://en.wikipedia.org/api/rest_v1/page/html/2026_United_States_Senate_elections"),
    ("FRED (economics)", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"),
    ("FEC (candidates)", f"https://api.open.fec.gov/v1/candidates/?api_key={config.FEC_API_KEY}&cycle=2026&office=S&per_page=1"),
]
for label, url in CHECKS:
    try:
        r = SESSION.get(url)
        print(f"   OK    {label}  ({len(r.text):,} bytes)")
    except Exception as e:
        msg = str(e)
        # A 429 is throttling, not a block: the host is reachable and will
        # answer again shortly. Treating it as unreachable would send you to
        # Colab for a problem that fixes itself.
        throttled = "429" in msg
        short = "rate limited (429) - temporary, reachable" if throttled else type(e).__name__
        print(f"   {'WARN' if throttled else 'FAIL'}  {label}  -> {short}")
        if "Wikipedia" in label:
            problems.append("wikipedia_throttled" if throttled else "no_wikipedia")

# 3. What did the last run actually do?
print("\n3. LAST PIPELINE RUN")
meta_p = ROOT / "data_store" / "processed" / "polls_2026.meta.json"
if not meta_p.exists():
    print("   stage 03 has never run here")
    problems.append("never_run")
else:
    m = json.loads(meta_p.read_text())
    written = m.get("written", "?")
    age = ""
    try:
        age = f" ({(datetime.now() - datetime.fromisoformat(written)).days} days ago)"
    except Exception:
        pass
    print(f"   written   : {written}{age}")
    print(f"   provenance: {m.get('provenance')}")
    print(f"   polls     : {m.get('rows')} across {m.get('n_races_polled')} races")
    print(f"   sources   : {m.get('sources')}")
    if m.get("provenance") == "fixture":
        problems.append("ran_but_fixture")

# ---- verdict ----
print("\n" + "=" * 62)
if "stale_code" in problems:
    print("VERDICT: the code here predates the poll-parser fixes.")
    print("Re-run the setup cell, which refreshes the code AND reloads stale")
    print("modules, then run the pipeline:")
    print("\n    %run run_pipeline.py --from 03")
elif "wikipedia_throttled" in problems:
    print("VERDICT: Wikipedia is reachable but currently rate-limiting this")
    print("machine (HTTP 429). That is temporary and clears on its own.")
    print("Wait a few minutes, then run:")
    print("\n    %run run_pipeline.py --from 03")
    print("\nIf the last run above already says provenance=live, your polls are")
    print("real and nothing needs fixing.")
elif "no_wikipedia" in problems:
    print("VERDICT: this machine cannot reach Wikipedia, which is where the")
    print("polls come from. Everything else can still run, but the polls will")
    print("stay placeholders here.")
    print("\nOptions:")
    print("  - run the pipeline on Colab instead, where Wikipedia is reachable")
    print("  - or enter polls by hand:  %run add_polls.py --example")
    print("  - if this is a work network, a proxy or firewall is the likely cause")
elif "ran_but_fixture" in problems or "never_run" in problems:
    print("VERDICT: the code and network are fine, but stage 03 has not produced")
    print("real polls yet on this machine. Run it now:")
    print("\n    %run run_pipeline.py --from 03")
    print("\nThe first poll fetch takes about 9 minutes (115 Wikipedia articles).")
else:
    print("VERDICT: code, network and the last run all look correct.")
    print("If check_data.py still says fixture, re-run it: it reads the files")
    print("on disk, so it needs the pipeline to have finished first.")
print("=" * 62)
