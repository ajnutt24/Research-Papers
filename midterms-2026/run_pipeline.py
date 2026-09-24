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
