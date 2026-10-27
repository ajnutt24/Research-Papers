#!/usr/bin/env python3
"""Convert the New York Times poll export into the project's wide poll schema.

The NYT publishes its polling database in the long FiveThirtyEight layout: one
row per CANDIDATE per QUESTION per poll. Our schema is one row per question with
a two-party margin, so the conversion is a pivot plus an identifier mapping.

Why this is a separate stage rather than a tweak to `fetch_nyt_csv`:
`fetch_nyt_csv` hands its download straight to `standardise()`, which expects a
frame that already has one row per question with named Democratic and
Republican columns. Feeding it the long layout silently produces nothing usable,
because the candidate percentages live in a single `pct` column that
`standardise` has no way to split by party.

What the export gives us that the Wikipedia scrape does not:

  * `sponsors` and `partisan`, the sponsor's name and partisan lean. The
    Wikipedia tables carry a party marker in the pollster name at best, and
    often not even that. This is the field that tells us National Public
    Affairs' Nebraska and Ohio polls were paid for by the God, Family, &
    Country PAC.
  * `question_id`, which separates the several ballot questions one survey can
    ask. The scrape has to infer that from repeated rows.
  * ranked-choice rounds, flagged explicitly, so Alaska's final-round result can
    be read off instead of reconstructed.

Deliberate choices, each of which is a place this could go wrong:

1. ONE QUESTION PER SURVEY IS NOT PICKED HERE. Every general-election question
   is emitted and `dedupe_polls` in stage 03 collapses them. That function
   already prefers the larger sample and then the higher two-party total, which
   is exactly the rule for telling a head-to-head from a question that also
   lists minor candidates. Re-implementing the choice here would mean two rules
   that have to agree forever.

2. POLLSTER NAMES REUSE THE SPELLING ALREADY IN THE DATASET.
   `workbook.normalise_pollster` preserves the "(D)"/"(R)"/"(I)" marker on
   purpose, so "GBAO (D)" and "GBAO" are different pollsters by design, and a
   name that disagrees with the one already on file would neither dedupe
   against it nor share its house effect. So when a firm is already in the
   dataset under some spelling, that spelling wins; only a firm we have never
   seen gets a name synthesised from `partisan`.

3. THE MAIN CANDIDATE IS NOT ALWAYS THE DEMOCRAT. Idaho and South Dakota have
   no Democrat on the ballot and the export codes those challengers `IND`,
   while Nebraska's Osborn is coded `DEM` despite running as an independent.
   For the races config marks as independent-led, an `IND` row wins over a
   `DEM` row; otherwise the Democrat is the main candidate. Montana is why the
   preference is needed rather than a fallback: it has both a Democrat (21.2)
   and the independent the model actually tracks (29.4).

4. RANKED CHOICE COLLAPSES TO THE FINAL ROUND. A ranked-choice question appears
   once per round, and summing or averaging rounds would be meaningless. Where
   any round is flagged final, only that round is kept; where rounds exist but
   none is flagged final, the question is dropped rather than guessed at.

Writes `data_store/manual/polls_nyt.csv`, which stage 03 picks up through
`load_nyt_export()`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "config.py").exists():
            return p
    raise RuntimeError("config.py not found above this file")


ROOT = _find_project_root()
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import workbook as _wb  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("import_nyt")

SRC_DIR = config.DATA_MANUAL / "nyt"
OUT = config.DATA_MANUAL / "polls_nyt.csv"

FILES = {"Senate": "senate.csv", "House": "house.csv", "Governor": "governor.csv"}

# The export dates both Ohio's and Florida's 2026 Senate contests to the general
# election day with no seat marker, because each state has exactly one Senate
# race this cycle and it is the special. config.SENATE_2026 agrees: both are
# flagged special, for Husted and Moody respectively.
SENATE_SPECIAL_STATES = {
    st for st, _cls, special, *_rest in config.SENATE_2026 if special
}

OUT_COLS = ["poll_id", "race_id", "office", "state", "district", "pollster", "sponsor",
            "partisan", "start_date", "end_date", "sample_size", "population",
            "methodology", "dem_pct", "rep_pct", "hypothetical",
            "dem_candidate", "rep_candidate", "main_is_independent"]


def existing_pollster_spellings() -> dict[str, str]:
    """Map a bare pollster name to the spelling already used in the dataset.

    Keyed on the name with any partisan marker and punctuation removed, so that
    the export's "National Public Affairs" finds our "National Public Affairs
    (R)". Where the dataset holds a firm under more than one spelling, the one
    carrying more polls wins, that being the one whose house effect rests on
    more evidence.
    """
    path = config.DATA_PROCESSED / "polls_2026.parquet"
    if not path.exists():
        return {}
    have = pd.read_parquet(path, columns=["pollster"])
    counts = have.pollster.value_counts()
    out: dict[str, str] = {}
    for name, n in counts.items():
        bare = _wb.normalise_pollster(name)
        bare = bare.rsplit(" (", 1)[0].strip().lower() if bare.endswith(")") else bare.lower()
        if bare not in out:          # value_counts is already sorted descending
            out[bare] = name
    return out


def race_id_for(office: str, row: pd.Series) -> str | None:
    state = _s(row.get("state")).upper()
    if office == "House":
        if state in ("US", "", "NAN"):
            return "GENERIC"
        num = pd.to_numeric(row.get("seat_number"), errors="coerce")
        if pd.isna(num):
            return None              # district not identified; a guess would be worse
        return f"H-{state}-{int(num):02d}"
    if office == "Senate":
        if len(state) != 2:
            return None
        return f"S-{state}-special" if state in SENATE_SPECIAL_STATES else f"S-{state}"
    if office == "Governor":
        return f"G-{state}" if len(state) == 2 else None
    return None


def resolve_ranked_choice(q: pd.DataFrame) -> pd.DataFrame | None:
    """Reduce a ranked-choice question to its final round, or drop it."""
    rounds = pd.to_numeric(q.get("ranked_choice_round"), errors="coerce")
    if rounds.isna().all():
        return q
    final = q[q.get("ranked_choice_final").astype(str).str.upper() == "TRUE"]
    if len(final):
        return final
    return None


def pivot_question(q: pd.DataFrame, independent_led: bool) -> dict | None:
    """Turn one question's candidate rows into a two-party row."""
    q = q.copy()
    q["pct_f"] = pd.to_numeric(q["pct"], errors="coerce")
    q = q.dropna(subset=["pct_f"])
    if not len(q):
        return None
    party = q["party"].astype(str).str.upper()

    rep = q[party == "REP"]
    dem = q[party == "DEM"]
    ind = q[party == "IND"]
    if not len(rep):
        return None

    # Which non-Republican is the candidate the model tracks. See choice 3.
    if independent_led and len(ind):
        main = ind
    elif len(dem):
        main = dem
    elif len(ind):
        main = ind
    else:
        return None

    # A question can list two candidates of a party (a jungle ballot). The
    # leading one is the one the two-party margin is about.
    main = main.loc[[main.pct_f.idxmax()]]
    rep = rep.loc[[rep.pct_f.idxmax()]]
    first = q.iloc[0]
    return {
        # Matches workbook.py's definition: the main candidate is not a Democrat.
        # Independent of the party CODE, because the export files Nebraska's
        # Osborn under DEM while Idaho's and South Dakota's independents get IND.
        "main_is_independent": bool(independent_led) or not len(dem),
        "dem_pct": float(main.pct_f.iloc[0]),
        "rep_pct": float(rep.pct_f.iloc[0]),
        "dem_candidate": _s(main.candidate_name.iloc[0]),
        "rep_candidate": _s(rep.candidate_name.iloc[0]),
        "sample_size": pd.to_numeric(first.get("sample_size"), errors="coerce"),
        "population": _s(first.get("population")).lower() or "unknown",
        "methodology": _s(first.get("methodology")),
        "start_date": pd.to_datetime(first.get("start_date"), format="%m/%d/%y", errors="coerce"),
        "end_date": pd.to_datetime(first.get("end_date"), format="%m/%d/%y", errors="coerce"),
        "sponsor": _s(first.get("sponsors")),
        "partisan_raw": _s(first.get("partisan")).upper(),
        "pollster_raw": _s(first.get("pollster")),
        "poll_id_raw": _s(first.get("poll_id")),
        "question_id": _s(first.get("question_id")),
    }


SUFFIX = {"REP": " (R)", "DEM": " (D)", "IND": " (I)"}


def _s(v) -> str:
    """Blank for a missing cell. `str(nan)` is "nan", and `nan or ""` keeps the
    nan because nan is truthy, so both of the obvious spellings put the literal
    text "nan" into sponsor and candidate columns."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def convert(office: str, path: Path, spellings: dict[str, str],
            independent_races: set[str]) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, low_memory=False)
    n_all = len(raw)
    g = raw[(raw.cycle.astype(str) == str(config.CYCLE))
            & (raw.stage.astype(str).str.lower() == "general")].copy()
    log.info("%s: %d rows -> %d rows in the %s general election",
             office, n_all, len(g), config.CYCLE)

    rows, dropped_rc, dropped_map, dropped_pivot = [], 0, 0, 0
    for qid, q in g.groupby("question_id", sort=False):
        rid = race_id_for(office, q.iloc[0])
        if rid is None:
            dropped_map += 1
            continue
        q2 = resolve_ranked_choice(q)
        if q2 is None:
            dropped_rc += 1
            continue
        rec = pivot_question(q2, rid in independent_races)
        if rec is None:
            dropped_pivot += 1
            continue
        rec["race_id"] = rid
        rec["office"] = "Generic" if rid == "GENERIC" else office
        rows.append(rec)

    log.info("%s: %d questions kept; dropped %d unmapped, %d ranked-choice without a "
             "final round, %d without a usable two-party pair",
             office, len(rows), dropped_map, dropped_rc, dropped_pivot)
    if not rows:
        return pd.DataFrame(columns=OUT_COLS)

    d = pd.DataFrame(rows)

    # Pollster spelling: reuse the dataset's, else synthesise from `partisan`.
    def name_for(r) -> str:
        bare_key = _wb.normalise_pollster(r.pollster_raw).lower()
        if bare_key in spellings:
            return spellings[bare_key]
        return r.pollster_raw + SUFFIX.get(r.partisan_raw, "")

    d["pollster"] = [name_for(r) for r in d.itertuples()]
    d["partisan"] = d.partisan_raw.map(lambda p: p if p in ("REP", "DEM", "IND") else "")
    d["state"] = d.race_id.map(
        lambda r: "US" if r == "GENERIC" else (r.split("-")[1] if "-" in r else ""))
    d["district"] = d.race_id.map(
        lambda r: int(r.split("-")[2]) if r.startswith("H-") and r != "GENERIC" else 0)
    d["hypothetical"] = False
    # Stable, source-tagged identifier. The export's poll_id repeats across the
    # questions of one survey, so the question_id has to be part of the key or
    # the final drop_duplicates("poll_id") in stage 03 would keep just one.
    d["poll_id"] = ("NYT-" + d.poll_id_raw.str.slice(0, 8) + "-" + d.question_id.str.slice(0, 8))
    return d[OUT_COLS]


def main() -> None:
    missing = [f for f in FILES.values() if not (SRC_DIR / f).exists()]
    if missing:
        log.error("no NYT export found. Expected these under %s: %s",
                  SRC_DIR, ", ".join(missing))
        sys.exit(1)

    spellings = existing_pollster_spellings()
    log.info("reusing %d pollster spellings already in the dataset", len(spellings))
    independent_races = set(getattr(config, "INDEPENDENT_RACES_2026", {}))
    log.info("independent-led races: %s", ", ".join(sorted(independent_races)) or "none")

    frames = [convert(office, SRC_DIR / fn, spellings, independent_races)
              for office, fn in FILES.items()]
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["end_date", "dem_pct", "rep_pct"])
    out = out.sort_values(["race_id", "end_date"])
    out.to_csv(OUT, index=False)

    log.info("wrote %d rows to %s", len(out), OUT)
    log.info("races covered: %d | newest field end: %s",
             out.race_id.nunique(), pd.to_datetime(out.end_date).max().date())
    log.info("rows carrying a sponsor: %d | a partisan lean: %d",
             int((out.sponsor.fillna("") != "").sum()), int((out.partisan.fillna("") != "").sum()))
    print(out.groupby("office").agg(rows=("poll_id", "size"), races=("race_id", "nunique")))


if __name__ == "__main__":
    main()
