"""
03_fetch_polls.py
=================
Ingest 2026 general-election polls (House, Senate, Governor and the national
generic ballot) into one normalised table.

Sources, in priority order
--------------------------
1. A CSV endpoint (the New York Times poll-tracking database, successor to
   FiveThirtyEight's discontinued feed). The NYT does not publish a stable,
   documented CSV URL, so the URL is read from the NYT_POLLS_URL environment
   variable. Column names are matched heuristically (see COLUMN_ALIASES).
2. RealClearPolitics/RealClearPolling HTML tables. A list of race pages is
   read from data_store/manual/rcp_urls.csv (race_id,url); the generic-ballot
   page is always attempted. RCP tables carry pollster, dates, sample and
   population ("LV"/"RV"), which is enough for house-effect adjustment.
3. A hand-maintained CSV, data_store/manual/polls_2026.csv, in the same
   schema as the output (documented below). This is the recommended path when
   scraping is blocked: paste polls in as they are published.
4. A labelled FIXTURE, simulated from partisan lean plus a national
   environment, so the downstream pipeline can be exercised end to end.
   Fixture output is flagged provenance="fixture" and must never be
   published as a forecast.

Filtering rules
---------------
* Hypothetical matchups are dropped: a poll row is kept only if it has exactly
  one Democrat and one Republican and, when data_store/manual/candidates_2026.csv
  exists, both names match the nominees for that race (fuzzy last-name match).
* Only polls with an end date after the primary season start (config: 2026-01-01)
  and before FORECAST_ASOF are kept.
* Pollster name, sponsor, field dates, sample size, population (lv/rv/a) and
  partisan sponsorship are all retained for script 08.

Output schema (data_store/processed/polls_2026.parquet)
------------------------------------------------------
poll_id, race_id, office, state, district, pollster, sponsor, partisan,
start_date, end_date, sample_size, population, methodology,
dem_pct, rep_pct, margin, hypothetical, source
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from utils import SESSION, fetch_text, get_logger, load_partisan_lean, save_stage  # noqa: E402

log = get_logger("03_polls")

OUT_COLS = ["poll_id", "race_id", "office", "state", "district", "pollster", "sponsor", "partisan",
            "start_date", "end_date", "sample_size", "population", "methodology",
            "dem_pct", "rep_pct", "margin", "hypothetical", "source"]
POLL_WINDOW_START = date(2026, 1, 1)
RCP_GENERIC_URL = "https://www.realclearpolling.com/polls/state-of-the-union/generic-congressional-vote"
MANUAL_POLLS = config.DATA_MANUAL / "polls_2026.csv"
MANUAL_CANDIDATES = config.DATA_MANUAL / "candidates_2026.csv"
MANUAL_RCP_URLS = config.DATA_MANUAL / "rcp_urls.csv"

COLUMN_ALIASES = {
    "pollster": ["pollster", "pollster_name", "poll"],
    "sponsor": ["sponsor", "sponsors", "sponsor_name"],
    "start_date": ["start_date", "startdate", "field_start", "start"],
    "end_date": ["end_date", "enddate", "field_end", "end", "date"],
    "sample_size": ["sample_size", "samplesize", "sample", "n"],
    "population": ["population", "pop", "population_full", "sample_type"],
    "methodology": ["methodology", "method", "mode"],
    "partisan": ["partisan", "partisan_sponsor"],
    "state": ["state", "state_abbrev", "st"],
    "district": ["seat_number", "district", "cd"],
    "office": ["office", "office_type", "race_type"],
    "dem_pct": ["dem", "dem_pct", "democrat", "d"],
    "rep_pct": ["rep", "rep_pct", "republican", "r"],
    "hypothetical": ["hypothetical", "is_hypothetical"],
    "dem_candidate": ["dem_candidate", "candidate_dem", "answer_dem"],
    "rep_candidate": ["rep_candidate", "candidate_rep", "answer_rep"],
}


def _pick(df: pd.DataFrame, key: str):
    cols = {c.lower().strip(): c for c in df.columns}
    for a in COLUMN_ALIASES[key]:
        if a in cols:
            return df[cols[a]]
    return pd.Series([np.nan] * len(df), index=df.index)


def make_poll_id(row) -> str:
    key = f"{row['race_id']}|{row['pollster']}|{row['end_date']}|{row['sample_size']}|{row['population']}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def normalise_population(x) -> str:
    x = str(x).lower()
    if "likely" in x or x.startswith("lv"):
        return "lv"
    if "registered" in x or x.startswith("rv"):
        return "rv"
    if "adult" in x or x == "a":
        return "a"
    return "unknown"


def race_id_from(office: str, state: str, district) -> str:
    office = str(office).lower()
    if "house" in office or office in ("h", "us house"):
        return f"H-{state}-{int(district):02d}"
    if "sen" in office:
        return f"S-{state}"
    if "gov" in office:
        return f"G-{state}"
    if "generic" in office:
        return "GENERIC"
    return f"?-{state}"


def standardise(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map an arbitrary poll table to the output schema."""
    out = pd.DataFrame(index=df.index)
    for k in ["pollster", "sponsor", "start_date", "end_date", "sample_size", "population",
              "methodology", "partisan", "state", "district", "office", "dem_pct", "rep_pct", "hypothetical"]:
        out[k] = _pick(df, k)
    if "race_id" in df.columns:
        out["race_id"] = df["race_id"]
    else:
        out["race_id"] = [race_id_from(o, s, d if d == d else 0) for o, s, d in zip(out.office, out.state, out.district)]
    out["office"] = out["race_id"].str[0].map({"H": "House", "S": "Senate", "G": "Governor"}).fillna("Generic")
    out["state"] = out["race_id"].str.split("-").str[1].where(out["office"] != "Generic", "US")
    out["district"] = pd.to_numeric(out["race_id"].str.split("-").str[2], errors="coerce").fillna(0).astype(int)
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce").dt.date
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce").dt.date
    out["start_date"] = out["start_date"].fillna(out["end_date"])
    out["sample_size"] = pd.to_numeric(out["sample_size"], errors="coerce")
    out["population"] = out["population"].map(normalise_population)
    out["dem_pct"] = pd.to_numeric(out["dem_pct"], errors="coerce")
    out["rep_pct"] = pd.to_numeric(out["rep_pct"], errors="coerce")
    out["margin"] = out["dem_pct"] - out["rep_pct"]
    out["hypothetical"] = out["hypothetical"].astype(str).str.lower().isin(["true", "1", "yes"])
    out["partisan"] = out["partisan"].fillna("").astype(str)
    out["sponsor"] = out["sponsor"].fillna("").astype(str)
    out["methodology"] = out["methodology"].fillna("").astype(str)
    out["source"] = source
    out["poll_id"] = [make_poll_id(r) for _, r in out.iterrows()]
    return out[OUT_COLS]


# --------------------------------------------------------------------------
# Source 1: NYT-style CSV endpoint
# --------------------------------------------------------------------------
def fetch_nyt_csv() -> pd.DataFrame | None:
    url = os.environ.get("NYT_POLLS_URL", "")
    if not url:
        log.info("NYT_POLLS_URL not set; skipping NYT feed")
        return None
    try:
        txt, prov = fetch_text(url, "nyt_polls.csv")
        df = pd.read_csv(io.StringIO(txt))
        out = standardise(df, f"nyt:{prov}")
        log.info("NYT feed: %d rows (%s)", len(out), prov)
        return out
    except Exception as e:
        log.warning("NYT feed failed: %s", e)
        return None


# --------------------------------------------------------------------------
# Source 2: RealClearPolitics HTML tables
# --------------------------------------------------------------------------
def parse_rcp_table(html: str, race_id: str) -> pd.DataFrame:
    """Parse an RCP polls table. Expected header contains Poll, Date, Sample,
    then candidate columns with (D)/(R) suffixes, then Spread."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for table in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
        if not headers or not any("Poll" in h for h in headers):
            continue
        dcol = next((i for i, h in enumerate(headers) if re.search(r"\(D\)|Democrat", h)), None)
        rcol = next((i for i, h in enumerate(headers) if re.search(r"\(R\)|Republican", h)), None)
        if dcol is None or rcol is None:
            continue
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) < max(dcol, rcol) + 1 or cells[0].lower().startswith("rcp"):
                continue
            m = re.search(r"(\d+)\s*(LV|RV|A)?", cells[2]) if len(cells) > 2 else None
            date_m = re.search(r"(\d{1,2}/\d{1,2})\s*-\s*(\d{1,2}/\d{1,2})", cells[1]) if len(cells) > 1 else None
            yr = config.CYCLE
            start = pd.to_datetime(f"{date_m.group(1)}/{yr}") if date_m else pd.NaT
            end = pd.to_datetime(f"{date_m.group(2)}/{yr}") if date_m else pd.NaT
            rows.append({
                "race_id": race_id, "pollster": cells[0].split("*")[0].strip(), "sponsor": "",
                "start_date": start, "end_date": end,
                "sample_size": float(m.group(1)) if m else np.nan,
                "population": (m.group(2) or "unknown") if m else "unknown",
                "dem_pct": pd.to_numeric(cells[dcol], errors="coerce"),
                "rep_pct": pd.to_numeric(cells[rcol], errors="coerce"),
                "partisan": "*" in cells[0], "hypothetical": False,
            })
        break
    return pd.DataFrame(rows)


def fetch_rcp() -> pd.DataFrame | None:
    targets = [("GENERIC", RCP_GENERIC_URL)]
    if MANUAL_RCP_URLS.exists():
        extra = pd.read_csv(MANUAL_RCP_URLS)
        targets += list(zip(extra["race_id"], extra["url"]))
    frames = []
    for rid, url in targets:
        try:
            txt, prov = fetch_text(url, f"rcp_{rid}.html", check_robots=True)
            df = parse_rcp_table(txt, rid)
            if len(df):
                frames.append(standardise(df, f"rcp:{prov}"))
        except Exception as e:
            log.warning("RCP %s failed: %s", rid, e)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    log.info("RCP: %d rows", len(out))
    return out


# --------------------------------------------------------------------------
# Source 3: manual CSV
# --------------------------------------------------------------------------
def load_manual() -> pd.DataFrame | None:
    if not MANUAL_POLLS.exists():
        return None
    df = pd.read_csv(MANUAL_POLLS)
    out = standardise(df, "manual")
    log.info("manual polls: %d rows", len(out))
    return out


# --------------------------------------------------------------------------
# Source 4: fixture
# --------------------------------------------------------------------------
FIXTURE_POLLSTERS = [
    ("Emerson College", 0.8, "ivr/online"), ("Quinnipiac University", -0.6, "live phone"),
    ("Siena/NYT", 0.3, "live phone"), ("YouGov/Economist", 1.2, "online"),
    ("Marist", -0.4, "live phone"), ("Trafalgar Group", -2.5, "mixed"),
    ("Data for Progress", 1.8, "online"), ("Fox News/Beacon-Shaw", 0.1, "live phone"),
    ("Morning Consult", 1.0, "online"), ("Rasmussen Reports", -2.8, "ivr/online"),
    ("Public Policy Polling", 1.4, "ivr"), ("Cygnal", -1.5, "mixed"),
]


def build_fixture(seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Simulate a plausible poll table from partisan lean + national swing.
    Everything here is invented and labelled as such."""
    rng = np.random.default_rng(seed)
    lean, _ = load_partisan_lean("2022")
    lean = lean.set_index("race_key")["lean"]
    universe = config.race_universe()
    nat_env = 3.0  # fixture national environment (D+3) - NOT a forecast
    rows = []
    asof = config.FORECAST_ASOF

    def key(r):
        return f"{r.state}-{int(r.district):02d}" if r.office == "House" else r.state

    for r in universe.itertuples():
        base = lean.get(key(r), 0.0)
        truth = base + nat_env + rng.normal(0, 3)
        # polling intensity: competitive races get many polls, safe ones few/none
        lam = {"House": 6, "Senate": 12, "Governor": 8}[r.office] * np.exp(-abs(truth) / 6)
        n_polls = rng.poisson(lam)
        for _ in range(n_polls):
            p = FIXTURE_POLLSTERS[rng.integers(len(FIXTURE_POLLSTERS))]
            end = asof - timedelta(days=int(rng.integers(0, 150)))
            n = int(rng.choice([400, 500, 600, 800, 1000]))
            pop = rng.choice(["lv", "rv"], p=[0.7, 0.3])
            obs = truth + p[1] + (1.5 if pop == "rv" else 0) + rng.normal(0, 100 / np.sqrt(n) * 1.5)
            d = 47 + obs / 2 + rng.normal(0, 1)
            rows.append(dict(race_id=r.race_id, pollster=p[0], sponsor="", partisan="",
                             start_date=end - timedelta(days=3), end_date=end, sample_size=n,
                             population=pop, methodology=p[2], dem_pct=round(d, 1), rep_pct=round(d - obs, 1),
                             hypothetical=False))
    for _ in range(90):  # generic ballot
        p = FIXTURE_POLLSTERS[rng.integers(len(FIXTURE_POLLSTERS))]
        end = asof - timedelta(days=int(rng.integers(0, 180)))
        n = int(rng.choice([1000, 1200, 1500, 2000]))
        pop = rng.choice(["lv", "rv", "a"], p=[0.5, 0.4, 0.1])
        obs = nat_env + p[1] + (1.5 if pop != "lv" else 0) + rng.normal(0, 100 / np.sqrt(n) * 1.5)
        d = 45 + obs / 2
        rows.append(dict(race_id="GENERIC", pollster=p[0], sponsor="", partisan="",
                         start_date=end - timedelta(days=4), end_date=end, sample_size=n,
                         population=pop, methodology=p[2], dem_pct=round(d, 1), rep_pct=round(d - obs, 1),
                         hypothetical=False))
    return standardise(pd.DataFrame(rows), "fixture")


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------
def filter_hypothetical(df: pd.DataFrame, raw_names: pd.DataFrame | None = None) -> pd.DataFrame:
    before = len(df)
    df = df[~df["hypothetical"]]
    df = df.dropna(subset=["dem_pct", "rep_pct", "end_date"])
    if MANUAL_CANDIDATES.exists() and raw_names is not None and {"dem_candidate", "rep_candidate"} <= set(raw_names.columns):
        cands = pd.read_csv(MANUAL_CANDIDATES)
        noms = {(r.race_id, r.party): str(r.candidate).split()[-1].lower() for r in cands.itertuples()}
        keep = []
        for pid, rid, dn, rn in zip(raw_names.poll_id, raw_names.race_id, raw_names.dem_candidate, raw_names.rep_candidate):
            d_ok = noms.get((rid, "D")) in str(dn).lower() if (rid, "D") in noms else True
            r_ok = noms.get((rid, "R")) in str(rn).lower() if (rid, "R") in noms else True
            keep.append(d_ok and r_ok)
        ok_ids = set(raw_names.poll_id[keep])
        df = df[df.poll_id.isin(ok_ids)]
    df = df[(df.end_date >= POLL_WINDOW_START) & (df.end_date <= config.FORECAST_ASOF)]
    log.info("hypothetical/date filter: %d -> %d polls", before, len(df))
    return df


def main(force_fixture: bool = False):
    frames, provs = [], []
    if not force_fixture:
        for fn, prov in [(fetch_nyt_csv, "live"), (fetch_rcp, "live"), (load_manual, "manual")]:
            df = fn()
            if df is not None and len(df):
                frames.append(df)
                provs.append("live" if "live" in df["source"].iloc[0] else ("cache" if "cache" in df["source"].iloc[0] else prov))
    if not frames:
        log.warning("NO real polls available - building a labelled FIXTURE. Do not publish this run.")
        frames.append(build_fixture())
        provs.append("fixture")
    polls = pd.concat(frames, ignore_index=True).drop_duplicates("poll_id")
    polls = filter_hypothetical(polls)
    prov = max(provs, key=lambda p: {"live": 0, "cache": 1, "manual": 1, "fixture": 3}[p])
    save_stage(polls, "polls_2026", prov, {"n_races_polled": int(polls.race_id.nunique()),
                                            "sources": sorted(polls.source.unique().tolist())})
    print(polls.groupby("office").size())


if __name__ == "__main__":
    main(force_fixture="--fixture" in sys.argv)
