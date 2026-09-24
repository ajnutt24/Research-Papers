"""
add_polls.py
============
Add real polls to the model without wrestling with CSV columns.

    python3 add_polls.py --example        show the accepted formats
    python3 add_polls.py --file mine.txt  read polls from a text file
    python3 add_polls.py --check          show what is currently loaded

Accepted line format (one poll per line, commas between fields):

    race_id, pollster, end_date, sample, population, dem_pct, rep_pct

for example:

    GENERIC, Quinnipiac, 2026-09-18, 1500, rv, 49, 43
    S-GA, Emerson College, 2026-09-15, 800, lv, 51, 45
    S-NC, Marist, 2026-09-16, 900, lv, 48, 46

race_id values: GENERIC for the national generic ballot, S-XX for Senate
(S-OH-special, S-FL-special for the two specials), G-XX for governor,
H-XX-NN for a House district (two-digit district, e.g. H-NE-02).
population: lv (likely voters), rv (registered voters), a (adults).

Rows are appended to data_store/manual/polls_2026.csv and de-duplicated, so
running this twice with the same poll is harmless. Once that file exists the
model stops generating placeholder polls.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

try:
    _ROOT = Path(__file__).resolve().parent
except NameError:
    _ROOT = Path.cwd()
sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import config  # noqa: E402

TARGET = config.DATA_MANUAL / "polls_2026.csv"
COLUMNS = ["race_id", "pollster", "sponsor", "partisan", "start_date", "end_date",
           "sample_size", "population", "methodology", "dem_pct", "rep_pct",
           "hypothetical", "dem_candidate", "rep_candidate"]

EXAMPLE = """\
One poll per line, seven comma-separated fields:

    race_id, pollster, end_date, sample, population, dem_pct, rep_pct

Examples:

    GENERIC, Quinnipiac, 2026-09-18, 1500, rv, 49, 43
    S-GA, Emerson College, 2026-09-15, 800, lv, 51, 45
    S-NC, Marist, 2026-09-16, 900, lv, 48, 46
    G-AZ, Noble Predictive, 2026-09-12, 600, lv, 47, 46
    H-NE-02, Split Ticket, 2026-09-10, 500, lv, 52, 44

race_id:     GENERIC | S-XX | S-OH-special | S-FL-special | G-XX | H-XX-NN
population:  lv (likely voters) | rv (registered) | a (adults)
dem_pct/rep_pct: the two-way numbers as published, e.g. 49 and 43.

For an independent who would caucus with a party (Osborn in Nebraska), put
them in that party's column and note it in the pollster field.
"""


def parse_line(line: str, lineno: int) -> dict | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7:
        raise ValueError(f"line {lineno}: expected 7 fields, got {len(parts)}: {line!r}")
    rid, pollster, end, n, pop, dem, rep = parts[:7]
    rid = rid.upper().replace("SPECIAL", "special")
    try:
        end_d = pd.to_datetime(end).date()
    except Exception:
        raise ValueError(f"line {lineno}: could not read the date {end!r} (use YYYY-MM-DD)")
    pop = pop.lower()
    if pop not in ("lv", "rv", "a"):
        raise ValueError(f"line {lineno}: population must be lv, rv or a (got {pop!r})")
    try:
        dem_f, rep_f, n_f = float(dem), float(rep), float(n)
    except ValueError:
        raise ValueError(f"line {lineno}: sample/dem/rep must be numbers")
    return {"race_id": rid, "pollster": pollster, "sponsor": "", "partisan": "",
            "start_date": end_d, "end_date": end_d, "sample_size": n_f,
            "population": pop, "methodology": "", "dem_pct": dem_f, "rep_pct": rep_f,
            "hypothetical": False, "dem_candidate": "", "rep_candidate": ""}


def validate(rows: list[dict]) -> list[str]:
    valid = set(config.race_universe().race_id) | {"GENERIC"}
    problems = []
    for r in rows:
        if r["race_id"] not in valid:
            problems.append(f"unknown race_id {r['race_id']!r} (e.g. S-GA, H-NE-02, G-AZ, GENERIC)")
        if not (0 < r["dem_pct"] < 100 and 0 < r["rep_pct"] < 100):
            problems.append(f"{r['race_id']}: dem/rep percentages look wrong ({r['dem_pct']}, {r['rep_pct']})")
        if r["end_date"] > config.ELECTION_DATE:
            problems.append(f"{r['race_id']}: end_date {r['end_date']} is after election day")
    return problems


def add(rows: list[dict]) -> None:
    new = pd.DataFrame(rows)[COLUMNS]
    if TARGET.exists():
        old = pd.read_csv(TARGET, comment="#")
        for c in COLUMNS:
            if c not in old:
                old[c] = ""
        combined = pd.concat([old[COLUMNS], new], ignore_index=True)
    else:
        combined = new
    before = len(combined)
    combined = combined.drop_duplicates(subset=["race_id", "pollster", "end_date", "sample_size"], keep="last")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(TARGET, index=False)
    print(f"Wrote {len(combined)} polls to {TARGET}"
          f"{f' ({before - len(combined)} duplicate(s) collapsed)' if before != len(combined) else ''}")
    print(f"\nRaces covered: {combined.race_id.nunique()}")
    print(combined.race_id.value_counts().head(15).to_string())
    print("\nNow run:  %run run_pipeline.py --update")


def check() -> None:
    if not TARGET.exists():
        print(f"No manual poll file yet at {TARGET}")
        print("The model is using placeholder polls. Run  python3 add_polls.py --example")
        return
    df = pd.read_csv(TARGET, comment="#")
    print(f"{len(df)} polls in {TARGET}")
    print(f"Races covered: {df.race_id.nunique()}")
    print(f"Date range: {df.end_date.min()} to {df.end_date.max()}")
    print(df.race_id.value_counts().to_string())


def main(argv: list[str]) -> None:
    if "--example" in argv or not argv:
        print(EXAMPLE)
        return
    if "--check" in argv:
        check()
        return
    if "--file" in argv:
        path = Path(argv[argv.index("--file") + 1])
        text = path.read_text()
    else:
        print("Paste polls, one per line. Finish with an empty line.\n")
        lines = []
        while True:
            try:
                ln = input()
            except EOFError:
                break
            if not ln.strip():
                break
            lines.append(ln)
        text = "\n".join(lines)

    rows, errors = [], []
    for i, ln in enumerate(text.splitlines(), 1):
        try:
            parsed = parse_line(ln, i)
        except ValueError as e:
            errors.append(str(e))
            continue
        if parsed:
            rows.append(parsed)
    if errors:
        print("Could not read these lines, nothing written:\n")
        for e in errors:
            print("  -", e)
        return
    if not rows:
        print("No polls read.")
        return
    problems = validate(rows)
    if problems:
        print("Problems found, nothing written:\n")
        for p in problems:
            print("  -", p)
        return
    add(rows)


if __name__ == "__main__":
    main(sys.argv[1:])
