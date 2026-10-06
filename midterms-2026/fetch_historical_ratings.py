"""
fetch_historical_ratings.py
===========================
Build `data_store/manual/historical_ratings.csv`: archived expert race ratings
for past midterms, so the rating-to-margin calibration in stage 10 rests on more
than one cycle.

    python3 fetch_historical_ratings.py              # write the CSV
    python3 fetch_historical_ratings.py --dry-run    # parse and report, write nothing
    python3 fetch_historical_ratings.py --cycles 2014,2022

Why this exists
---------------
Stage 10 turns a qualitative rating ("Lean R") into a win probability and a
margin band using the raters' empirical track record. Out of the box that record
is FiveThirtyEight's 2018 forecast-review file: one cycle, and because 435 of
its 506 races are House districts it is overwhelmingly Safe seats. The tiers
that actually decide a forecast come out thin:

    Safe 376 races | Likely 82 | Lean 33 | Toss-up 15

With 15 Toss-ups the posterior is pinned at the prior, so the model is not
learning the raters' accuracy where it matters most. Senate and Governor races
are a much richer source of competitive-tier observations than House districts,
because a far larger share of them are rated Lean or Toss-up in the first place.

Source and why this one
-----------------------
Wikipedia's per-cycle election articles carry a "Final pre-election predictions"
table (titled variously "Election ratings", "Predictions", "Election
predictions") listing each race's final rating from every major rater, with the
as-of date in the column header. Cook's and Inside Elections' own archives are
paywalled and Sabato's are spread across individual newsletter posts, so this is
the only public source that covers four cycles in one consistent shape. The
tables cite each rater's own final release, which is the quantity wanted: the
rating as it stood on election eve, not an earlier one.

The House is NOT available this way. Those articles have an "Election ratings"
heading but it carries prose about the number of competitive seats, not a
district-by-district table. District-level ratings for past cycles would have to
come from Daily Kos Elections' spreadsheets or a Cook subscription. The
consequence is recorded honestly in the output metadata: the added cycles are
Senate and Governor only, which shifts the pooled sample toward statewide races.

Which raters, and why only three
--------------------------------
Cook Political Report, Sabato's Crystal Ball, and Inside Elections. These are
the three the 2026 pipeline actually reads (see `SOURCES` in script 05), and a
calibration is only meaningful for the raters it will be applied to. The tables
also carry RealClearPolitics, FiveThirtyEight, the New York Times, Daily Kos,
CQ, Politico, Fox and others; those are poll-driven models or media aggregates
rather than expert ratings, they are not in the 2026 feed, and including them
would calibrate a mapping on one population and apply it to another.

Inside Elections is treated as the continuation of the Rothenberg Political
Report, which it is: the same publication under its founder Stuart Rothenberg
was renamed Inside Elections in 2017 after Nathan Gonzales took over as editor.
The 2006 and 2010 tables use the Rothenberg name, the 2022 table uses Inside
Elections, and the series is continuous.

What the output is, and is not
------------------------------
One row per race per rater, plus a `consensus` row per race holding the median
tier (rounded toward Toss-up, matching how script 05 forms the 2026 consensus).
Stage 10 reads only the consensus rows, because three raters looking at the same
race are three correlated judgements, not three independent observations, and
counting them separately would inflate the sample threefold while understating
the uncertainty.

Margins and winners are NOT taken from the tables' "Result" column. They come
from `historical_results`, the project's own canonical results stage, so the
ratings and the outcomes cannot disagree about who won.

Ratings for independent candidates ("Lean I" for Lieberman in 2006, "Lean I" for
Walker in Alaska 2014) are kept in the file with an empty tier_code and a note,
and excluded from the calibration. A tier code is signed D-positive and an
independent has no side on that axis; inventing one would be worse than
dropping four races.
"""
from __future__ import annotations

import datetime as dt
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    PROJECT = Path(__file__).resolve().parent
except NameError:
    PROJECT = Path.cwd()
sys.path.insert(0, str(PROJECT))

import config  # noqa: E402
from utils import fetch_text, get_logger, load_stage  # noqa: E402

log = get_logger("hist_ratings")

CYCLES = [2006, 2010, 2014, 2022]
ARTICLES = {
    "Senate": "{c}_United_States_Senate_elections",
    "Governor": "{c}_United_States_gubernatorial_elections",
}
OUT = config.DATA_MANUAL / "historical_ratings.csv"

# A rater column is matched on its header text. Order matters only for
# reporting. Rothenberg and Inside Elections are one series; see the docstring.
RATERS = {
    "cook": re.compile(r"\bcook\b", re.I),
    "sabato": re.compile(r"\bsabato\b|crystal\s*ball", re.I),
    "inside": re.compile(r"\binside\s*elections\b|\bIE\b|\brothenberg\b", re.I),
}
# Columns that look like raters but are poll-driven models or aggregates. Listed
# explicitly so a new one shows up as "unmatched" in the log instead of being
# silently folded into a rater by a loose regex.
NOT_RATERS = re.compile(
    r"\bRCP\b|realclear|\b538\b|fivethirtyeight|\bNYT\b|new york times|daily kos|"
    r"\bCQ\b|politico|\bfox\b|\bDDHQ\b|econ|\bED\b|rasmussen|\bCBS\b|desart|\bSSP\b|"
    r"result|winner|incumbent|last (election|race)|\bPVI\b|constituency|state|overall",
    re.I)

RATING_RE = re.compile(
    r"\b(safe|solid|likely|lean[s]?|tilt|toss\s*-?\s*up|tossup)\b\s*([DRI])?\b", re.I)
LEVEL_OF = {"safe": "Safe", "solid": "Safe", "likely": "Likely",
            "lean": "Lean", "leans": "Lean", "tilt": "Lean",
            "toss-up": "Toss-up", "tossup": "Toss-up", "toss up": "Toss-up"}
# Signed tier code, D-positive, matching script 05's TIER_CODE. Tilt collapses
# into Lean rather than getting its own 0.5 code: only three of the four cycles
# use "Tilt" at all, and script 10 buckets to four levels regardless.
CODE = {("Safe", "D"): 3, ("Likely", "D"): 2, ("Lean", "D"): 1, ("Toss-up", None): 0,
        ("Lean", "R"): -1, ("Likely", "R"): -2, ("Safe", "R"): -3}

STATE_CODE = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI",
    "Wyoming": "WY",
}
RAT_WORDS = re.compile(r"(safe|solid|likely|lean|tilt|toss)", re.I)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_rating(cell: str) -> tuple[str, str | None] | None:
    """('Lean', 'R') from a cell, or None if it holds no rating.

    Takes the FIRST rating in the cell. Some cells carry a parenthetical such as
    "Lean R (flip)", which is a note about whether the seat changes hands, not a
    second rating.
    """
    m = RATING_RE.search(str(cell))
    if not m:
        return None
    level = LEVEL_OF.get(m.group(1).lower().replace("  ", " "))
    if level is None:
        return None
    party = (m.group(2) or "").upper() or None
    if level == "Toss-up":
        party = None            # a toss-up has no favoured side by definition
    elif party not in ("D", "R", "I"):
        return None             # "Lean" with no party is unusable
    return level, party


def best_ratings_table(slug: str, force: bool = False) -> pd.DataFrame | None:
    """The table in the article with the most rating words in it."""
    from bs4 import BeautifulSoup
    try:
        txt, _ = fetch_text(f"https://en.wikipedia.org/wiki/{slug}", f"wiki_{slug}.html",
                            force=force, max_age_hours=24 * 30)
    except Exception as e:                                        # noqa: BLE001
        log.warning("%s: fetch failed (%s: %s)", slug, type(e).__name__, e)
        return None
    soup = BeautifulSoup(txt, "lxml")
    best = (0, None)
    for tbl in soup.find_all("table"):
        n = len(RAT_WORDS.findall(tbl.get_text(" ", strip=True)))
        if n > best[0]:
            best = (n, tbl)
    if best[1] is None or best[0] < 15:
        log.warning("%s: no table with enough ratings in it (best had %d)", slug, best[0])
        return None
    try:
        # flavor pinned to lxml: html5lib is not a declared dependency and
        # pandas falls back to it silently, which fails the whole article.
        df = pd.read_html(io.StringIO(str(best[1])), header=0, flavor="lxml")[0]
    except Exception as e:                                        # noqa: BLE001
        log.warning("%s: table found but would not parse (%s: %s)", slug, type(e).__name__, e)
        return None
    # Some years nest a second header row ("2010 election ratings.1" in the
    # column names, the real rater names in row 0). Promote it.
    if any(re.search(r"election ratings", str(c), re.I) for c in df.columns):
        df.columns = [str(x) for x in df.iloc[0]]
        df = df.iloc[1:].reset_index(drop=True)
    return df


def clean_state(raw: str) -> tuple[str | None, bool]:
    """('DE', True) from 'Delaware (special)'. Footnote markers are stripped.

    Deliberately tolerant about the parentheses. Stripping a footnote marker can
    leave an unbalanced bracket behind ("South Carolina [x](special)" becomes
    "South Carolina special)"), so "special" is detected as a word anywhere in
    the label and then every non-letter is discarded, rather than requiring a
    well-formed "(special)".
    """
    s = re.sub(r"\[[^\]]*\]", "", str(raw))
    special = bool(re.search(r"\bspecial\b", s, re.I))
    s = re.sub(r"\b(special|regular|class\s*[0-9IVX]+)\b", " ", s, flags=re.I)
    s = re.sub(r"[^A-Za-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return STATE_CODE.get(s), special


def race_id_for(office: str, state: str, special: bool) -> str:
    prefix = "S" if office == "Senate" else "G"
    return f"{prefix}-{state}-special" if special else f"{prefix}-{state}"


def parse_cycle_office(cycle: int, office: str, force: bool = False) -> pd.DataFrame:
    df = best_ratings_table(ARTICLES[office].format(c=cycle), force=force)
    if df is None:
        return pd.DataFrame()

    # Columns are addressed BY POSITION, not by label. These tables carry
    # duplicate headers in some years and a literal NaN header in others, and
    # df[label] then returns a DataFrame rather than a column.
    cols = [str(c) for c in df.columns]
    def column(j: int) -> list[str]:
        return [str(v) for v in df.iloc[:, j].tolist()]

    rater_cols: dict[str, int] = {}
    unmatched = []
    for j, c in enumerate(cols):
        if NOT_RATERS.search(c):
            continue
        hit = next((name for name, rx in RATERS.items() if rx.search(c)), None)
        if hit and hit not in rater_cols:
            rater_cols[hit] = j
        elif hit is None and RAT_WORDS.search(" ".join(column(j)[:8])):
            unmatched.append(c)
    if unmatched:
        log.info("%d %s: columns that hold ratings but match no known rater: %s",
                 cycle, office, unmatched)
    if not rater_cols:
        log.warning("%d %s: none of Cook / Sabato / Inside Elections found in %s",
                    cycle, office, cols)
        return pd.DataFrame()
    log.info("%d %s: raters %s", cycle, office,
             {k: re.sub(r"\[[^\]]*\]", "", cols[j]).strip() for k, j in rater_cols.items()})

    rows, skipped = [], []
    for i in range(len(df)):
        state, special = clean_state(df.iloc[i, 0])
        if state is None:
            raw = re.sub(r"\s+", " ", str(df.iloc[i, 0])).strip()
            if raw and raw.lower() not in ("nan", "state", "overall", "constituency"):
                skipped.append(raw)
            continue
        rid = race_id_for(office, state, special)
        for rater, j in rater_cols.items():
            got = parse_rating(df.iloc[i, j])
            if got is None:
                continue
            level, party = got
            rows.append({"cycle": cycle, "race_id": rid, "office": office,
                         "source": rater, "rating": _label(level, party),
                         "tier_code": CODE.get((level, party), np.nan),
                         "source_column": re.sub(r"\[[^\]]*\]", "", cols[j]).strip()})
    if skipped:
        log.warning("%d %s: %d row label(s) not recognised as a state: %s",
                    cycle, office, len(skipped), skipped[:6])
    out = pd.DataFrame(rows)
    log.info("%d %s: %d rating(s) across %d race(s)", cycle, office,
             len(out), out.race_id.nunique() if len(out) else 0)
    return out


def _label(level: str, party: str | None) -> str:
    return "Toss-up" if level == "Toss-up" else f"{level} {party}"


def reconcile_keys(df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Match the constructed race_ids to the keys the results stage actually uses.

    The two do not always agree about specials. The results mirror suffixes a
    special election only when it would otherwise collide with a regular race in
    the same cycle: Delaware 2010 is `S-DE-special` because Delaware also held a
    regular Senate race, but Utah's 2010 gubernatorial special is plain `G-UT`,
    since there was no regular Utah race to distinguish it from. Rather than
    encode that rule twice and risk the two copies drifting, a constructed key
    with no result falls back to its unsuffixed form when the results stage has
    that one and nothing else has claimed it.

    Anything still unmatched is left alone and reported. A rating with no result
    contributes nothing to a calibration, and silently renaming it to whatever
    is nearby would be worse than losing it.
    """
    have = set(zip(results.cycle, results.race_key))
    claimed = set(zip(df.cycle, df.race_id)) & have
    fixes = {}
    for cycle, rid in sorted(set(zip(df.cycle, df.race_id))):
        if (cycle, rid) in have or not rid.endswith("-special"):
            continue
        bare = rid[: -len("-special")]
        if (cycle, bare) in have and (cycle, bare) not in claimed:
            fixes[(cycle, rid)] = bare
            claimed.add((cycle, bare))
    if fixes:
        log.info("%d race id(s) re-keyed to match historical_results: %s", len(fixes),
                 {f"{c} {r}": v for (c, r), v in fixes.items()})
        df = df.copy()
        key = list(zip(df.cycle, df.race_id))
        df["race_id"] = [fixes.get(k, k[1]) for k in key]
    return df


# --------------------------------------------------------------------------
# Consensus
# --------------------------------------------------------------------------
def add_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """One `consensus` row per race: the median tier across raters.

    Rounded TOWARD Toss-up, matching script 05's convention for 2026. Rounding
    away from zero would make the consensus more confident than any individual
    rater on a split call, which is the wrong direction for a calibration whose
    whole purpose is to stop over-confidence.

    A race gets NO consensus row if ANY of its raters rated it for an
    independent. Dropping just that rater's row and taking the median of the
    rest looks tidier and is wrong: in Rhode Island 2010 two of the three raters
    said "Lean I" and Lincoln Chafee duly won, but their ratings carry no tier
    code, so a median over the survivors is Cook's lone "Toss-up" and the race
    enters the calibration as a toss-up the Democrat lost. The raters were
    right and the record would say they were wrong. A race where the serious
    possibility is an independent is not described by a D-versus-R tier axis at
    all, so it belongs outside the calibration rather than inside it with a
    distorted label. The same applies in Alaska 2014, where Bill Walker won.
    """
    bad = set(map(tuple, df.loc[df.tier_code.isna(), ["cycle", "race_id"]].values))
    usable = df.dropna(subset=["tier_code"])
    usable = usable[[(c, r) not in bad for c, r in zip(usable.cycle, usable.race_id)]]
    rows = []
    for (cycle, rid), g in usable.groupby(["cycle", "race_id"]):
        med = float(np.median(g.tier_code.values))
        code = int(np.sign(med) * np.floor(abs(med)))   # toward zero
        level = {3: "Safe", 2: "Likely", 1: "Lean", 0: "Toss-up"}[abs(code)]
        party = None if code == 0 else ("D" if code > 0 else "R")
        rows.append({"cycle": cycle, "race_id": rid, "office": g.office.iloc[0],
                     "source": "consensus", "rating": _label(level, party),
                     "tier_code": code,
                     "source_column": f"median of {len(g)}: " +
                                      ", ".join(f"{r.source}={r.rating}" for r in g.itertuples())})
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


# --------------------------------------------------------------------------
def main() -> None:
    argv = sys.argv[1:]
    cycles = CYCLES
    if "--cycles" in argv:
        cycles = [int(x) for x in argv[argv.index("--cycles") + 1].split(",")]
    force = "--force" in argv

    frames = [parse_cycle_office(c, o, force=force) for c in cycles for o in ARTICLES]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise SystemExit("Nothing parsed. The article structure may have changed; "
                         "run with --dry-run and read the warnings above.")
    df = pd.concat(frames, ignore_index=True)

    ind = df[df.tier_code.isna()]
    if len(ind):
        log.info("%d rating(s) for independent candidates. The whole race is kept in "
                 "the file with no consensus row, so it never reaches the "
                 "calibration: %s", len(ind), sorted(set(zip(ind.cycle, ind.race_id))))
    res = load_stage("historical_results")
    df = reconcile_keys(df, res)
    df = add_consensus(df)

    # Cross-check against the project's own results: a race_id with no result is
    # either a parsing error or a race the mirror does not cover, and either way
    # it contributes nothing to a calibration. Report both, do not drop.
    have = set(zip(res.cycle, res.race_key))
    cons = df[df.source == "consensus"].copy()
    cons["has_result"] = [(c, r) in have for c, r in zip(cons.cycle, cons.race_id)]
    missing = cons[~cons.has_result]
    if len(missing):
        log.warning("%d consensus race(s) have no row in historical_results and will "
                    "not reach the calibration: %s", len(missing),
                    sorted(zip(missing.cycle, missing.race_id))[:12])

    print(f"\n{'='*72}\nArchived ratings parsed\n{'='*72}")
    per = df[df.source == "consensus"].groupby(["cycle", "office"]).size().unstack(fill_value=0)
    print(per.to_string())
    print(f"\nconsensus races: {len(cons)}, of which {int(cons.has_result.sum())} have a "
          f"result on file")
    lvl = cons[cons.has_result].rating.str.replace(r" [DR]$", "", regex=True).value_counts()
    print(f"\nby tier (these are what stage 10 will add):")
    for k in ("Safe", "Likely", "Lean", "Toss-up"):
        print(f"   {k:9s} {int(lvl.get(k, 0)):4d}")
    print(f"\nper-rater rows: {df[df.source != 'consensus'].groupby('source').size().to_dict()}")

    if "--dry-run" in argv:
        print("\n--dry-run: nothing written.")
        return

    header = (
        "# Archived expert race ratings for past midterms, for the stage 10\n"
        "# rating-to-margin calibration. Generated by fetch_historical_ratings.py;\n"
        f"# regenerate with: python3 fetch_historical_ratings.py\n"
        f"# Written {dt.date.today()}. Cycles: {', '.join(map(str, cycles))}. Senate and\n"
        "# Governor only: Wikipedia's House articles carry no per-district ratings table.\n"
        "#\n"
        "# Raters are Cook Political Report, Sabato's Crystal Ball and Inside Elections\n"
        "# (continuing the Rothenberg Political Report), which are the three the 2026\n"
        "# pipeline reads. Each race also carries a `consensus` row, the median tier\n"
        "# rounded toward Toss-up, and that is the only row stage 10 uses: three raters\n"
        "# on one race are correlated judgements, not three observations.\n"
        "#\n"
        "# source_column records the exact Wikipedia column, including the rater's\n"
        "# as-of date, so any row can be traced back to the release it came from.\n"
        "# Ratings for independent candidates have an empty tier_code and are excluded\n"
        "# from the calibration.\n"
        "#\n"
        "# Hand edits are preserved only until this script is re-run. To correct a\n"
        "# rating permanently, fix it here and do not regenerate, or fix the parser.\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        fh.write(header)
        df.sort_values(["cycle", "office", "race_id", "source"]).to_csv(fh, index=False)
    print(f"\nWrote {len(df)} row(s) to {OUT}")
    print("Re-run stages 05 and 10 to fold them into the calibration:\n"
          "    python3 run_pipeline.py 05 10 12 13 14")


if __name__ == "__main__":
    main()
