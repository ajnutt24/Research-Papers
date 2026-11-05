#!/usr/bin/env python3
"""Refresh ONLY the Winner column of the scorecard workbook, in place.

Rebuilding the workbook from scratch would be wrong once anyone has touched it.
Everything else in the file is frozen: the model's probabilities are the
2026-11-01 forecast and must not move, the candidates are fixed, and Outcome,
Model agree, the row colours and the scoreboard are formulas that recompute
themselves the moment a Winner cell changes. The results are the only live
part, so they are the only part this writes.

The rule that makes it safe to re-run: a cell you have already filled is never
overwritten. Wikipedia's live reporting is unofficial and gets corrected, and a
certified result typed in by hand outranks it. Where the two disagree the cell
is left alone and the disagreement is printed, so it can be judged rather than
silently resolved.

    python3 analysis/20_update_results.py              # fill blanks, report conflicts
    python3 analysis/20_update_results.py --dry-run    # report only, write nothing
"""
import re
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

WB = ROOT / "outputs" / "midterms_2026_scorecard.xlsx"
DRY = "--dry-run" in sys.argv


def winner_party(marker: str) -> str:
    m = str(marker or "")
    if "(Democratic" in m or "(DFL" in m:
        return "D"
    if "(Republican" in m:
        return "R"
    return ""          # independents and uncalled races stay blank on purpose


def scrape_all() -> dict[str, str]:
    """race_id -> 'D' / 'R' from the three Wikipedia summary pages."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fp", ROOT / "data" / "03_fetch_polls.py")
    fp = importlib.util.module_from_spec(spec)
    sys.modules["fp"] = fp
    spec.loader.exec_module(fp)
    sys.path.insert(0, str(ROOT / "analysis"))
    spec2 = importlib.util.spec_from_file_location("src", ROOT / "analysis" / "18_scorecard_sources.py")

    out = {}
    # House comes from the rundown script's own CSV, which 17 refreshes.
    rundown = ROOT / "outputs" / "house_rundown_2026.csv"
    if rundown.exists():
        h = pd.read_csv(rundown)
        for r in h.itertuples():
            w = winner_party(getattr(r, "wiki_winner_marker", ""))
            if w:
                out[r.race_id] = w
    for name in ("sen_cands.csv", "gov_cands.csv"):
        p = ROOT / "outputs" / name
        if p.exists():
            d = pd.read_csv(p)
            for r in d.itertuples():
                w = winner_party(getattr(r, "wiki_winner_marker", ""))
                if w:
                    out[r.race_id] = w
    return out


def main() -> None:
    if not WB.exists():
        sys.exit(f"workbook not found: {WB}\nrun analysis/19_build_scorecard.py first")
    latest = scrape_all()
    print(f"results available for {len(latest)} races")

    wb = load_workbook(WB)
    filled = conflicts = already = 0
    conflict_rows = []
    for sheet in ("House", "Senate", "Governor"):
        ws = wb[sheet]
        hdr = {c.value: c.column for c in ws[1]}
        c_id, c_win = hdr["Race ID"], hdr["Winner"]
        for row in range(2, ws.max_row + 1):
            rid = ws.cell(row=row, column=c_id).value
            new = latest.get(rid)
            if not new:
                continue
            cur = (ws.cell(row=row, column=c_win).value or "").strip()
            if cur == "":
                if not DRY:
                    ws.cell(row=row, column=c_win, value=new)
                filled += 1
            elif cur != new:
                conflicts += 1
                conflict_rows.append((sheet, rid, cur, new))
            else:
                already += 1

    print(f"  filled {filled} blank cell(s)")
    print(f"  {already} already matched")
    print(f"  {conflicts} conflict(s): left as they are, listed below")
    for sheet, rid, cur, new in conflict_rows:
        print(f"      {sheet:9s} {rid:10s} sheet says {cur!r}, Wikipedia says {new!r}")

    if DRY:
        print("\n--dry-run: nothing written")
        return
    if filled:
        wb.save(WB)
        print(f"\nsaved {WB}")
    else:
        print("\nnothing to write; workbook untouched")


if __name__ == "__main__":
    main()
