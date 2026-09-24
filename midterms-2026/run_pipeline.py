"""
run_pipeline.py
===============
Convenience runner. Three modes:

  python3 run_pipeline.py --update      # new polls dropped: 03 -> 08 -> 11 -> 12 -> 13   (~1 min)
  python3 run_pipeline.py --full        # everything, 01 -> 14                             (~5 min)
  python3 run_pipeline.py --fast        # 01 -> 13, skips the backtest (stage 14)
  python3 run_pipeline.py 03 08 13      # any explicit list of stage numbers, in that order
  python3 run_pipeline.py --from 09     # resume the full order starting at stage 09

Each stage reads the cached outputs of the stages it depends on, so the
update mode only re-runs what new polls can change: poll aggregation, the
hierarchical model, the blend (cached stacking weights), and the simulation.

Output from every stage is streamed live AND saved to outputs/pipeline_log.txt.
If a stage fails, the error is reprinted at the end so it is not buried in
hundreds of lines of sampler output.
"""
from __future__ import annotations

import os
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

# Which stage produces which output, and what each stage needs before it can
# run. Colab wipes its filesystem between sessions, so "--update" on a fresh
# runtime would otherwise fail deep inside stage 08 with a missing-file error
# for something stage 06 produces. The runner uses this to pull in whatever
# prerequisites are genuinely absent.
PRODUCES = {
    "01": ["economic_monthly", "economic_cycle_features"],
    "02": ["gas_prices"],
    "03": ["polls_2026"],
    "04": ["approval_2026", "approval_history"],
    "05": ["race_ratings_2026", "race_ratings_historical"],
    "07": ["historical_results", "historical_national"],
    "06": ["fundamentals_2026"],
    "08": ["poll_estimates_2026", "generic_ballot_trend", "national_environment",
           "polling_error_history", "house_effects_2026"],
    "09": ["fundamentals_national", "fundamentals_estimates_2026"],
    "10": ["rating_calibration", "ratings_estimates_2026"],
    "11": ["hierarchical_estimates_2026"],
    "12": ["stacking_weights", "blend_2026"],
    "13": [],
    "14": [],
}
REQUIRES = {
    "01": [], "02": [], "03": [], "04": [], "05": [], "07": [],
    "06": ["historical_results"],
    "08": ["polls_2026", "fundamentals_2026"],
    "09": ["economic_cycle_features", "approval_history", "fundamentals_2026",
           "historical_results", "historical_national"],
    "10": ["race_ratings_2026", "race_ratings_historical", "historical_results"],
    "11": ["fundamentals_2026", "fundamentals_estimates_2026", "fundamentals_national",
           "national_environment", "poll_estimates_2026"],
    "12": ["hierarchical_estimates_2026", "fundamentals_estimates_2026",
           "fundamentals_national", "ratings_estimates_2026", "national_environment"],
    "13": ["blend_2026"],
    "14": ["national_environment", "historical_results", "rating_calibration"],
}
PRODUCER = {out: st for st, outs in PRODUCES.items() for out in outs}


def output_exists(name: str) -> bool:
    return (ROOT / "data_store" / "processed" / f"{name}.parquet").exists()


def resolve_prerequisites(order: list[str]) -> tuple[list[str], list[str]]:
    """Expand `order` with any missing upstream stages. Returns (order, added)."""
    planned = list(order)
    added: list[str] = []
    changed = True
    while changed:
        changed = False
        for st in list(planned):
            for need in REQUIRES.get(st, []):
                if output_exists(need):
                    continue
                producer = PRODUCER.get(need)
                if producer is None or producer in planned:
                    continue
                planned.append(producer)
                added.append(producer)
                changed = True
    # keep the canonical order so dependencies run before dependents
    planned = [st for st in FULL if st in planned]
    return planned, added
# --fast: everything needed for a forecast, skipping the backtest (stage 14),
# which re-scores four historical cycles and is the second-slowest stage.
FAST = [s for s in FULL if s != "14"]
UPDATE = ["03", "08", "11", "12", "13"]
LOG = ROOT / "outputs" / "pipeline_log.txt"


def run_stage(stage: str, extra: list[str], log) -> tuple[int, list[str]]:
    """Run one stage, streaming its output live and capturing it for the log."""
    cmd = [sys.executable, "-u", str(ROOT / STAGES[stage]), *extra]
    lines: list[str] = []
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            errors="replace")
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()          # keep the log current so a stall is diagnosable
        lines.append(line)
    proc.wait()
    return proc.returncode, lines


def main(argv: list[str]):
    if "--from" in argv:
        start = argv[argv.index("--from") + 1].zfill(2)
        base = FAST if "--fast" in argv else FULL
        order = base[base.index(start):] if start in base else base
    elif "--fast" in argv:
        order = FAST
    elif "--full" in argv:
        order = FULL
    elif "--update" in argv or not any(a for a in argv if not a.startswith("--")):
        order = UPDATE
    else:
        order = [a.zfill(2) for a in argv if a.zfill(2) in STAGES]
    skip = {"--full", "--fast", "--update", "--from"}
    drop = set()
    if "--from" in argv:
        drop.add(argv[argv.index("--from") + 1])
    extra = [a for a in argv if a.startswith("--") and a not in skip and a not in drop]

    order, added = resolve_prerequisites(order)
    if added:
        print(f"Adding missing prerequisite stage(s): {', '.join(sorted(set(added)))}")
        print("(their outputs are not on disk yet, e.g. after a fresh Colab session)\n")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    failed = None
    with open(LOG, "w", encoding="utf-8") as log:
        log.write(f"pipeline start {time.strftime('%Y-%m-%d %H:%M:%S')}  stages={order}\n")
        for st in order:
            banner = f"\n===== stage {st}: {STAGES[st]} =====  started {time.strftime('%H:%M:%S')}"
            print(banner, flush=True)
            log.write(banner + "\n")
            log.flush()
            t0 = time.perf_counter()
            rc, lines = run_stage(st, extra, log)
            done = f"===== stage {st} finished in {time.perf_counter() - t0:.0f}s (exit {rc}) ====="
            print(done, flush=True)
            log.write(done + "\n")
            log.flush()
            if rc != 0:
                failed = (st, lines)
                break

    if failed:
        st, lines = failed
        tail = [ln for ln in lines if ln.strip()][-30:]
        print("\n" + "=" * 72)
        print(f"STAGE {st} FAILED ({STAGES[st]}). The error was:")
        print("=" * 72)
        for ln in tail:
            print(ln)
        print("=" * 72)
        print(f"Full log saved to: {LOG}")
        print("Later stages were not run. Send the lines above if you need help.")
        return 1

    cc = ROOT / "outputs" / "chamber_control.json"
    if cc.exists() and "13" in order:
        import json
        d = json.loads(cc.read_text())
        print(f"\nprovenance={d['provenance']}  asof={d['asof']}  ({d['days_to_election']} days out)")
        for ch in ("House", "Senate", "Governor"):
            print(f"  {ch:9s} P(D control)={d[ch]['p_dem_control']:.0%}  median D seats={d[ch]['dem_seats_median']:.0f}")
        if d["provenance"] == "fixture":
            print("\nNOTE: some inputs are placeholders. Run  %run check_data.py  to see which.")
    return 0


if __name__ == "__main__":
    code = main(sys.argv[1:])
    # In a notebook, sys.exit raises a confusing SystemExit traceback; just report.
    if code and not any("ipykernel" in m for m in sys.modules):
        sys.exit(code)
