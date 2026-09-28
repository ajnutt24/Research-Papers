"""
workbook.py
===========
Read the hand-maintained Excel polling workbook (MidtermPolls2026.xlsx).

Why this module exists
----------------------
Scraping Wikipedia got the model to 1,058 polls, but with two limits it could
not scrape its way out of. First, Wikipedia carries statewide polls in
abundance and district polls almost not at all, so all 435 House forecasts ran
with no district survey anywhere in them. Second, a scraper cannot tell a
live general-election matchup from a pre-primary ballot test or a withdrawn
candidate without a human deciding. The workbook solves both: it carries
polls for 88 House districts, and every row has an `include_in_model` flag
that records a human's judgement about whether the row belongs in a forecast.

The workbook is therefore treated as the authoritative poll source, not as one
more scraper to average in. Its own README says polls are entered in one place
and that is what the model should read, and its `include_in_model = N` rows are
deliberate exclusions (pre-primary ballots, hypotheticals, superseded
duplicates). Re-scraping Wikipedia alongside it would put those excluded rows
back and double-count everything the workbook already took from Wikipedia,
because the same poll carries different identifiers in the two sources and
would survive de-duplication twice.

What it reads
-------------
Only the four input sheets. The per-state tabs are generated views that say
"Do not edit here", so reading them would mean reading the same polls twice
through a slower path.

  Polls         master table, one row per poll version
  Races         registry of all 508 races: type, state, district, special flag
  Ratings       Cook / Inside Elections / Sabato, as a time series
  Rating_Scale  rating label -> numeric score (read for validation only)

Identifier translation
----------------------
The workbook and the model number races differently, and the `Races` sheet is
used to translate rather than parsing identifier strings, because the sheet is
authoritative and string parsing would silently mis-handle the six at-large
states.

  US-GB        -> GENERIC          AL-SEN   -> S-AL
  US-APPROVAL  -> (stage 04)       OH-SEN-S -> S-OH-special
  AL-02        -> H-AL-02          AL-GOV   -> G-AL
  AK-AL        -> H-AK-01          (at-large; the model numbers these 01)

Conventions handled
-------------------
* Percentages are stored as fractions (0.51). They are scaled to percent, but
  the scale is detected rather than assumed, so the workbook switching to whole
  numbers later does not silently divide the whole forecast by 100.
* Option A is the Democrat, or the independent where there is no Democrat
  (Nebraska, Idaho, South Dakota). Option B is the Republican. This matches the
  model's two-party margin convention, which is main-candidate minus
  Republican, so an Osborn-style race is read correctly rather than discarded.
* `DFL` is Minnesota's Democratic-Farmer-Labor party and counts as Democratic.
* Sample size is recovered from the margin of error where it is missing, since
  the workbook notes that its Pollsmax rows carry no MoE and some carry no n.
* Pollster names are normalised, because the workbook warns that its two
  sources spell the same pollster differently ("Economist / YouGov" against
  "The Economist/YouGov"). Left unnormalised, one pollster becomes two and its
  house effect is estimated twice from half the data each time.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
from utils import get_logger

log = get_logger("workbook")

SHEET_POLLS = "Polls"
SHEET_RACES = "Races"
SHEET_RATINGS = "Ratings"

REQUIRED_POLL_COLS = ["race_id", "pollster", "field_end", "opt_a_pct", "opt_b_pct", "include_in_model"]
REQUIRED_RACE_COLS = ["race_id", "race_type", "state_abbr", "district"]
REQUIRED_RATING_COLS = ["race_id", "rater", "rating"]

APPROVAL_ID = "US-APPROVAL"
GENERIC_ID = "US-GB"

# Workbook rater label -> the source key the model's rating stage expects.
RATER_TO_SOURCE = {"cook": "cook", "inside elections": "inside", "inside": "inside",
                   "sabato": "sabato", "sabato's crystal ball": "sabato"}

# The workbook records the survey population as LV / RV / A / V. The model uses
# lv / rv / a / unknown. "V" is only documented as "voters", which does not say
# registered or likely, so it maps to unknown rather than guessing: the model's
# fitted population effects differ by more than 4 points between rv and
# unknown, so a wrong guess here moves real numbers.
POPULATION_MAP = {"LV": "lv", "RV": "rv", "A": "a", "V": "unknown"}

DEM_PARTIES = {"D", "DFL", "DNL"}
IND_PARTIES = {"I", "I?", "IND", "L", "G", "C", "AF"}


# --------------------------------------------------------------------------
# Locating the file
# --------------------------------------------------------------------------
def candidate_paths() -> list[Path]:
    """Every place the workbook might live, in priority order.

    The environment variable comes first so a run can be pointed at a copy
    without editing anything. The configured path is the author's own machine.
    The repository-local copy is what makes the same code work in Colab, in a
    container, and on a second machine, none of which have the Windows path.
    """
    out: list[Path] = []
    env = os.environ.get("MIDTERM_WORKBOOK", "").strip()
    if env:
        out.append(Path(env))
    if getattr(config, "POLL_WORKBOOK", ""):
        out.append(Path(config.POLL_WORKBOOK))
    out.append(config.DATA_MANUAL / "MidtermPolls2026.xlsx")
    for extra in getattr(config, "WORKBOOK_EXTRA_PATHS", []):
        out.append(Path(extra))
    return out


def find_workbook(quiet: bool = False) -> Path | None:
    """First candidate path that exists, or None."""
    tried = []
    for p in candidate_paths():
        try:
            if p.is_file():
                return p
        except OSError:
            pass                      # a Windows path on Linux, or similar
        tried.append(str(p))
    if not quiet:
        log.info("no polling workbook found; looked in:")
        for t in tried:
            log.info("    %s", t)
    return None


def _read(path: Path, sheet: str, required: list[str]) -> pd.DataFrame:
    """Read one sheet and fail with a message that names the fix.

    Nothing here is cached. The workbook is edited by hand every week, and a
    cache that served last week's copy would be indistinguishable from the
    refresh not having landed.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet)
    except ImportError as e:
        raise ImportError(f"reading .xlsx needs openpyxl ({e}). Run: pip install openpyxl") from e
    except ValueError as e:
        raise ValueError(f"workbook {path.name} has no sheet named {sheet!r}: {e}") from e
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"sheet {sheet!r} in {path.name} is missing column(s) {missing}. "
                         f"Found: {list(df.columns)}")
    return df


# --------------------------------------------------------------------------
# Identifier translation
# --------------------------------------------------------------------------
def race_id_map(races: pd.DataFrame) -> dict[str, str]:
    """Workbook race_id -> model race_id, built from the Races sheet.

    Built from the sheet's own type/state/district/special columns rather than
    by parsing the identifier text, because the six at-large states write their
    district as "AL" and the model numbers them 01.
    """
    out: dict[str, str] = {}
    for r in races.itertuples():
        wid = str(r.race_id).strip()
        kind = str(r.race_type).strip().lower()
        st = str(r.state_abbr).strip().upper()
        if kind == "generic ballot":
            out[wid] = "GENERIC"
        elif kind == "presidential approval":
            out[wid] = APPROVAL_ID          # handled by stage 04, not a race
        elif kind == "senate":
            special = str(getattr(r, "special", "") or "").strip().upper().startswith("Y")
            out[wid] = f"S-{st}-special" if special else f"S-{st}"
        elif kind == "governor":
            out[wid] = f"G-{st}"
        elif kind == "house":
            d = str(r.district).strip().upper()
            n = 1 if d in ("AL", "AT-LARGE", "", "NAN") else int(re.sub(r"\D", "", d) or 0)
            out[wid] = f"H-{st}-{n:02d}"
        else:
            log.warning("unrecognised race_type %r for %s; skipping", r.race_type, wid)
    return out


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------
def _to_percent(s: pd.Series) -> pd.Series:
    """Scale shares to percentage points, detecting the scale in use.

    The workbook stores fractions today. Detection rather than assumption means
    that if it is ever switched to whole numbers, the model does not silently
    read a 51-point lead as half a point.
    """
    v = pd.to_numeric(s, errors="coerce")
    finite = v.dropna()
    if len(finite) and float(finite.max()) <= 1.5:
        return v * 100.0
    return v


def _n_from_moe(moe: pd.Series) -> pd.Series:
    """Recover an effective sample size from a 95% margin of error.

    For a proportion at its widest, moe = 1.96 * sqrt(0.25 / n), so
    n = (0.98 / moe)^2. This is only used where sample size is missing: the
    workbook notes that some rows carry no reported n. Getting an approximate n
    matters because the aggregator weights each poll by its variance, and a
    missing n means a poll is either dropped or given a default weight.
    """
    m = pd.to_numeric(moe, errors="coerce")
    m = m.where(m > 0)
    m = m.where(m <= 1.0, m / 100.0)          # accept 3.1 as well as 0.031
    return (0.98 / m) ** 2


def normalise_pollster(name: str) -> str:
    """Collapse the spelling differences between the workbook's two sources.

    The workbook's own quality notes warn that Pollsmax and Wikipedia spell the
    same pollster differently, and that the names must be unified before house
    effects are modelled. A pollster split in two gets its house effect
    estimated twice from half the evidence each time, and the shrinkage toward
    zero that protects against noisy estimates is applied to both halves.

    The partisan marker is deliberately preserved: "Public Policy Polling (D)"
    must not collapse into "Public Policy Polling", because the sponsor is
    information about the poll, not noise in its name.
    """
    s = str(name or "").strip()
    if not s:
        return ""
    party = ""
    m = re.search(r"\(\s*([DRI])\s*\)\s*$", s)
    if m:
        party = f" ({m.group(1).upper()})"
        s = s[:m.start()].strip()
    s = re.sub(r"^the\s+", "", s, flags=re.I)        # "The Economist/YouGov"
    s = re.sub(r"\s*/\s*", "/", s)                   # "Economist / YouGov"
    s = re.sub(r"\s*&\s*", " and ", s)
    s = re.sub(r"[.,]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s + party


# --------------------------------------------------------------------------
# Polls
# --------------------------------------------------------------------------
def load_polls(path: Path | None = None) -> pd.DataFrame | None:
    """Poll rows in the shape script 03's `standardise` expects, or None.

    Only rows the workbook marks `include_in_model = Y` are returned. That flag
    is the human judgement the scrapers could not make, and honouring it is the
    whole reason to prefer this source. `Review` rows are additional versions
    of a release that is already represented by its primary row, so including
    them would count the same poll twice; set WORKBOOK_INCLUDE_REVIEW to
    override that.
    """
    path = path or find_workbook()
    if path is None:
        return None
    polls = _read(path, SHEET_POLLS, REQUIRED_POLL_COLS)
    races = _read(path, SHEET_RACES, REQUIRED_RACE_COLS)
    mapping = race_id_map(races)

    n_all = len(polls)
    flag = polls["include_in_model"].astype(str).str.strip().str.upper()
    keep = flag.eq("Y")
    if getattr(config, "WORKBOOK_INCLUDE_REVIEW", False):
        keep |= flag.eq("REVIEW")
    polls = polls[keep].copy()

    polls["model_race_id"] = polls["race_id"].astype(str).str.strip().map(mapping)
    unmapped = polls["model_race_id"].isna()
    if unmapped.any():
        bad = sorted(polls.loc[unmapped, "race_id"].astype(str).unique())[:8]
        log.warning("    %d poll row(s) have a race_id absent from the Races sheet (%s); skipped",
                    int(unmapped.sum()), ", ".join(bad))
        polls = polls[~unmapped]
    polls = polls[polls["model_race_id"] != APPROVAL_ID]        # approval is stage 04

    a_party = polls.get("opt_a_party", pd.Series("", index=polls.index)).astype(str).str.strip().str.upper()
    out = pd.DataFrame({
        "race_id": polls["model_race_id"].values,
        "pollster": [normalise_pollster(x) for x in polls["pollster"]],
        "sponsor": polls.get("partisan_sponsor", pd.Series("", index=polls.index)).fillna("").astype(str).values,
        "start_date": polls.get("field_start", polls["field_end"]).values,
        "end_date": polls["field_end"].values,
        "sample_size": pd.to_numeric(polls.get("sample_n"), errors="coerce").values,
        "population": polls.get("population", pd.Series(np.nan, index=polls.index))
                           .map(lambda v: POPULATION_MAP.get(str(v).strip().upper(), "unknown")).values,
        "methodology": polls.get("mode", pd.Series("", index=polls.index)).fillna("").astype(str).values,
        "partisan": polls.get("partisan_sponsor", pd.Series("", index=polls.index)).fillna("").astype(str).values,
        "dem_pct": _to_percent(polls["opt_a_pct"]).values,
        "rep_pct": _to_percent(polls["opt_b_pct"]).values,
        "hypothetical": False,
        "dem_candidate": polls.get("opt_a_label", pd.Series("", index=polls.index)).fillna("").astype(str).values,
        "rep_candidate": polls.get("opt_b_label", pd.Series("", index=polls.index)).fillna("").astype(str).values,
        "main_is_independent": a_party.isin(IND_PARTIES).values,
        "workbook_poll_id": polls.get("poll_id", pd.Series("", index=polls.index)).astype(str).values,
    })

    # Recover a sample size where it is missing but a margin of error is given.
    if "moe" in polls.columns:
        need = out["sample_size"].isna()
        if need.any():
            filled = _n_from_moe(polls["moe"]).values
            out.loc[need, "sample_size"] = filled[need.values]
            got = int((need & out["sample_size"].notna()).sum())
            if got:
                log.info("    recovered sample size from margin of error for %d poll(s)", got)

    out = out[out["dem_pct"].notna() & out["rep_pct"].notna()]
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    out = out[out["end_date"].notna()]

    n_ind = int(out["main_is_independent"].sum())
    log.info("workbook %s: %d of %d rows marked include_in_model=Y, %d usable polls across %d races",
             path.name, int(keep.sum()), n_all, len(out), out.race_id.nunique())
    log.info("    field dates %s to %s%s", out.end_date.min().date(), out.end_date.max().date(),
             f", {n_ind} with an independent as the main candidate" if n_ind else "")
    return out


# --------------------------------------------------------------------------
# Presidential approval
# --------------------------------------------------------------------------
def load_approval(path: Path | None = None) -> pd.DataFrame | None:
    """Approval rows as (date, approve, disapprove, source), or None.

    The workbook files approval as ordinary poll rows under race_id
    US-APPROVAL, with option A = Approve and option B = Disapprove.
    """
    path = path or find_workbook()
    if path is None:
        return None
    polls = _read(path, SHEET_POLLS, REQUIRED_POLL_COLS)
    rid = polls["race_id"].astype(str).str.strip()
    ap = polls[rid.eq(APPROVAL_ID)].copy()
    if not len(ap):
        return None
    flag = ap["include_in_model"].astype(str).str.strip().str.upper()
    keep = flag.eq("Y")
    if getattr(config, "WORKBOOK_INCLUDE_REVIEW", False):
        keep |= flag.eq("REVIEW")
    ap = ap[keep]
    out = pd.DataFrame({
        "date": pd.to_datetime(ap["field_end"], errors="coerce"),
        "approve": _to_percent(ap["opt_a_pct"]).values,
        "disapprove": _to_percent(ap["opt_b_pct"]).values,
        "source": "workbook",
    }).dropna(subset=["date", "approve", "disapprove"]).sort_values("date")
    if not len(out):
        return None
    log.info("workbook approval: %d polls, %s to %s, last approve %.1f",
             len(out), out.date.min().date(), out.date.max().date(), out.approve.iloc[-1])
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Expert ratings
# --------------------------------------------------------------------------
def load_ratings(path: Path | None = None) -> pd.DataFrame | None:
    """Latest rating per rater and race as (race_id, source, rating), or None.

    The Ratings sheet is a time series: a new row is added when a rater moves a
    race, and nothing is overwritten. Only the most recent row per rater and
    race is a current rating, so the sheet is reduced by `as_of` date. Its own
    `is_latest` column is an Excel formula, which arrives as a cached value and
    goes stale the moment a row is added without recalculating, so the date is
    used instead of trusting it.
    """
    path = path or find_workbook()
    if path is None:
        return None
    try:
        rat = _read(path, SHEET_RATINGS, REQUIRED_RATING_COLS)
    except ValueError as e:
        log.warning("workbook ratings unavailable: %s", e)
        return None
    races = _read(path, SHEET_RACES, REQUIRED_RACE_COLS)
    mapping = race_id_map(races)

    rat = rat.copy()
    rat["model_race_id"] = rat["race_id"].astype(str).str.strip().map(mapping)
    rat = rat[rat["model_race_id"].notna()]
    rat = rat[~rat["model_race_id"].isin([APPROVAL_ID, "GENERIC"])]
    rat["source"] = rat["rater"].astype(str).str.strip().str.lower().map(RATER_TO_SOURCE)
    unknown = rat["source"].isna()
    if unknown.any():
        log.warning("    unrecognised rater(s): %s; skipped",
                    sorted(rat.loc[unknown, "rater"].astype(str).unique()))
        rat = rat[~unknown]
    rat["as_of"] = pd.to_datetime(rat.get("as_of"), errors="coerce")
    rat = rat.sort_values("as_of").groupby(["model_race_id", "source"], as_index=False).last()

    out = pd.DataFrame({"race_id": rat["model_race_id"].values,
                        "source": rat["source"].values,
                        "rating": rat["rating"].astype(str).str.strip().values})
    out = out[out["rating"].str.lower().ne("nan") & out["rating"].ne("")]
    log.info("workbook ratings: %d current ratings across %d races from %s",
             len(out), out.race_id.nunique(), sorted(out.source.unique()))
    return out.reset_index(drop=True)


if __name__ == "__main__":       # quick check: python3 workbook.py
    p = find_workbook()
    print(f"workbook: {p}")
    if p:
        import datetime as _dt
        print(f"last modified: {_dt.datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M}")
        for fn in (load_polls, load_approval, load_ratings):
            d = fn(p)
            print(f"{fn.__name__}: {0 if d is None else len(d)} rows")
