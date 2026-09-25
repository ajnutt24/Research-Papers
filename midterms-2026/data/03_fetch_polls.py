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
1b. Wikipedia polling tables, one article per race (see wikipolls.py). This is
   the default working source: free, structured, comprehensive and legally
   reusable, and each row cites its original pollster. Commercial aggregators
   render their tables in JavaScript, so a plain fetch gets nothing.
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
import time
import io
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _find_project_root() -> Path:
    """Locate the midterms-2026 folder (the one holding config.py and utils.py).

    Works when run as a script, and when the code is pasted into a Jupyter cell
    (where __file__ does not exist): then it searches the working directory, its
    parents, and any 'midterms-2026' subfolder.
    """
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        pass
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        for d in (cand, cand / "midterms-2026"):
            if (d / "config.py").is_file() and (d / "utils.py").is_file():
                return d
    raise RuntimeError(
        "Cannot find the project folder. config.py and utils.py must exist as FILES "
        "(pasting their contents into a notebook cell does not create them).\n"
        f"Current working directory: {here}\n"
        "Fix: download the midterms-2026 folder from the repository, then in a cell run\n"
        "    %cd /path/to/midterms-2026"
    )


_ROOT = _find_project_root()
sys.path.insert(0, str(_ROOT))
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


def office_of(race_id: pd.Series) -> pd.Series:
    """Map race_id to office.

    "GENERIC" must be tested before the first-letter rule, because it also
    starts with "G" and would otherwise be filed as a governor's race. That
    mistake is quiet and expensive: several hundred national generic-ballot
    polls get attributed to state governor contests, and the generic ballot,
    which is the model's single most important national input, goes missing.
    """
    rid = race_id.astype("string")
    return rid.str[0].map({"H": "House", "S": "Senate", "G": "Governor"}) \
              .where(rid != "GENERIC", "Generic").fillna("Generic")


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
    out["office"] = office_of(out["race_id"])
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
    for c in ("dem_candidate", "rep_candidate", "main_is_independent"):
        out[c] = df[c].values if c in df.columns else ("" if c != "main_is_independent" else False)
    return out[OUT_COLS + ["dem_candidate", "rep_candidate", "main_is_independent"]]


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
        log.warning("RealClearPolitics returned no polls (it blocks automated "
                    "requests). Wikipedia is the primary source; this is expected.")
        return None
    out = pd.concat(frames, ignore_index=True)
    log.info("RCP: %d rows", len(out))
    return out


# --------------------------------------------------------------------------
# Source 2b: Wikipedia polling tables
# --------------------------------------------------------------------------
# Wikipedia is the only comprehensive, free, structured and legally reusable
# source of 2026 general-election polls now that FiveThirtyEight's database is
# gone. Each race's article carries a "Polling" wikitable that pandas can read
# directly, and every row cites its original pollster. See wikipolls.py.
WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/html/"


def wiki_titles() -> list[tuple[str, str]]:
    """(race_id, article title) for every race Wikipedia is likely to cover."""
    # The generic congressional ballot lives on the cycle-wide "2026 United
    # States elections" article, not on the House article. The House article
    # carries only an aggregate-of-aggregators table, which this parser
    # deliberately rejects: aggregator rows are not independent observations,
    # and feeding six of them in as six polls would badly understate
    # uncertainty. The cycle article carries the individual polls themselves
    # (600+ rows across 2024-2026), which is what the Kalman filter needs.
    out = [("GENERIC", "2026_United_States_elections")]
    for st, cls, special, inc, open_seat, note in config.SENATE_2026:
        name = config.STATE_NAMES[st].replace(" ", "_")
        if special:
            out.append((f"S-{st}-special", f"2026_United_States_Senate_special_election_in_{name}"))
        else:
            out.append((f"S-{st}", f"2026_United_States_Senate_election_in_{name}"))
    for st, inc, open_seat, note in config.GOVERNOR_2026:
        name = config.STATE_NAMES[st].replace(" ", "_")
        out.append((f"G-{st}", f"2026_{name}_gubernatorial_election"))
    return out


HUB_PAGES = [
    "2026_United_States_Senate_elections",
    "2026_United_States_gubernatorial_elections",
    "2026_United_States_House_of_Representatives_elections",
]
RACE_LINK_RE = re.compile(
    r"2026.*(Senate (special )?election in|gubernatorial election|"
    r"House of Representatives elections in)", re.I)


def title_to_race_id(title: str) -> str | None:
    """Map a Wikipedia article title to a race_id, or None if it is not one.

    Statewide House articles return "H-<ST>-*", a marker meaning the article
    covers many districts and must be parsed section by section.
    """
    t = title.replace("_", " ")
    name_to_abbr = {v.lower(): k for k, v in config.STATE_NAMES.items()}

    m = re.search(r"Senate (special )?election in (.+)$", t, re.I)
    if m:
        st = name_to_abbr.get(m.group(2).strip().lower())
        if st:
            return f"S-{st}-special" if m.group(1) else f"S-{st}"
        return None
    m = re.search(r"^2026 (.+?) gubernatorial election", t, re.I)
    if m:
        st = name_to_abbr.get(m.group(1).strip().lower())
        return f"G-{st}" if st else None
    m = re.search(r"House of Representatives elections in (.+)$", t, re.I)
    if m:
        st = name_to_abbr.get(m.group(1).strip().lower())
        return f"H-{st}-*" if st else None
    return None


def discover_race_articles() -> dict[str, str]:
    """Crawl the hub pages for links to individual race articles.

    Guessing titles covers most races, but Wikipedia's naming varies and
    articles get renamed or redirected, so the real links are harvested too.
    Returns {race_id: article title}.
    """
    sys.path.insert(0, str(_ROOT))
    import wikipolls
    from utils import require
    require(wikipolls, "parse_html_sections", "discover_links", "district_from_heading")

    found: dict[str, str] = {}
    for hub in HUB_PAGES:
        try:
            html, _ = fetch_text(WIKI_API + hub, f"wiki_hub_{hub[:40]}.html", max_age_hours=12)
        except Exception as e:
            log.debug("hub %s unreachable: %s", hub, e)
            continue
        for title in wikipolls.discover_links(html, RACE_LINK_RE):
            rid = title_to_race_id(title)
            if rid:
                found.setdefault(rid, title)
    if found:
        log.info("Wikipedia link discovery: %d race articles found on the hub pages", len(found))
    return found


def fetch_wikipedia(limit: int | None = None) -> pd.DataFrame | None:
    """Fetch and parse polling tables for every 2026 race.

    Two passes: the hub articles are crawled for real links to race articles,
    and those are merged with directly-guessed titles so a naming change on
    either side does not lose a race. Each article is then parsed section by
    section, which keeps per-district House tables attached to the right seat
    and drops primary and hypothetical-matchup tables.
    """
    sys.path.insert(0, str(_ROOT))
    import wikipolls
    from utils import require
    require(wikipolls, "parse_html_sections", "discover_links", "district_from_heading")

    targets: dict[str, str] = dict(wiki_titles())          # guessed
    discovered = discover_race_articles()                  # crawled
    for rid, title in discovered.items():
        targets.setdefault(rid, title)                     # adds House state pages
    items = list(targets.items())
    if limit:
        items = items[:limit]

    frames, found = [], 0
    reasons = {"missing_article": 0, "rate_limited": 0, "network": 0, "parse_error": 0}
    consecutive_network = 0
    for rid, title in items:
        html = None
        for attempt in range(3):
            try:
                html, prov = fetch_text(WIKI_API + title, f"wiki_{rid.replace('*', 'all')}.html",
                                        max_age_hours=12)
                consecutive_network = 0
                break
            except Exception as e:
                msg = str(e)
                if "404" in msg:
                    # No article for this race. Entirely normal, and never a
                    # reason to stop: many races have no page yet.
                    reasons["missing_article"] += 1
                    break
                if "429" in msg:
                    # Throttling, not refusal. Wikipedia rate-limits a long run
                    # of requests. Pause briefly and move on rather than
                    # spending minutes per article: whatever is missed this run
                    # is picked up on the next, because successful fetches are
                    # cached for 12 hours and the cache accumulates.
                    reasons["rate_limited"] += 1
                    time.sleep(8)
                    continue
                reasons["network"] += 1
                consecutive_network += 1
                break
        if html is None:
            # Only a genuine network fault, repeated, means the host is gone.
            if consecutive_network >= 8 and not frames:
                log.warning("Wikipedia unreachable after %d consecutive network errors; "
                            "giving up on this source", consecutive_network)
                break
            continue

        if rid.endswith("-*"):        # statewide House article: one race per section
            state = rid.split("-")[1]
            n_seats = config.HOUSE_SEATS_BY_STATE.get(state, 0)

            def rid_for(heading, _st=state, _n=n_seats):
                d = wikipolls.district_from_heading(heading)
                if d is None and _n == 1:
                    return f"H-{_st}-01"      # at-large states have no district heading
                return f"H-{_st}-{d:02d}" if d and 1 <= d <= _n else None
        else:
            def rid_for(heading, _rid=rid):
                return _rid

        # A parse failure on one article must never discard the articles that
        # already succeeded. An earlier run lost 50 articles' worth of real
        # polls to a single ImportError raised deep inside pandas.read_html.
        try:
            df = wikipolls.parse_html_sections(html, rid_for, default_year=config.CYCLE)
            if len(df):
                frames.append(standardise(df, f"wikipedia:{prov}"))
                found += 1
        except Exception as e:  # noqa: BLE001
            reasons["parse_error"] += 1
            log.warning("    %s: parse failed (%s: %s)", rid, type(e).__name__, e)
        if (found + sum(reasons.values())) % 25 == 0 and (found + sum(reasons.values())) > 0:
            log.info("    ... %d articles processed, %d with polls so far",
                     found + sum(reasons.values()), found)

    n_rows = sum(len(f) for f in frames)
    log.info("Wikipedia: %d of %d articles had polls, %d poll rows", found, len(items), n_rows)
    log.info("    skipped: %d no article, %d rate-limited, %d network error, %d parse error",
             reasons["missing_article"], reasons["rate_limited"], reasons["network"],
             reasons["parse_error"])
    if reasons["rate_limited"] > len(items) // 4:
        log.warning("    Wikipedia throttled a large share of requests. Re-running will "
                    "reuse what was cached and fetch the rest.")
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    by_office = office_of(out.race_id)
    log.info("    polls by office: %s", by_office.value_counts().to_dict())
    return out


# --------------------------------------------------------------------------
# Source 3: manual CSV
# --------------------------------------------------------------------------
def load_manual() -> pd.DataFrame | None:
    if not MANUAL_POLLS.exists():
        return None
    df = pd.read_csv(MANUAL_POLLS, comment="#")
    df = df.dropna(subset=["race_id"])
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
        cands = pd.read_csv(MANUAL_CANDIDATES, comment="#")
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



def filter_to_nominees(df: pd.DataFrame) -> pd.DataFrame:
    """Drop polls of match-ups that are no longer on the ballot.

    Wikipedia keeps historical tables for candidates who lost a primary or
    withdrew. Those are real polls of an unreal contest, and averaging them in
    drags a race toward a match-up no voter will see.

    Three rules keep this from doing more harm than good, each learned from a
    way an earlier version of it destroyed real data:

    1. Only the hand-verified candidate file is consulted. The auto-derived
       file is built from Federal Election Commission (FEC) filings, and an
       FEC filing is a Statement of Candidacy, not a nomination: anyone may
       file. Treating the alphabetically-first filer as the nominee dropped
       Peggy Flanagan's Minnesota polls in favour of a filer named Angie
       Craig, James Talarico's Texas polls in favour of one named Colin
       Allred, and every poll of sitting senators Jack Reed, Mark Warner and
       Ed Markey. The auto file remains the right input for the candidate
       experience index, which is what it was built for, and the wrong input
       for deciding who is on the ballot.

    2. A race is only filtered when the named nominee actually appears in at
       least one of its polls. If the name never appears, the more likely
       explanation is that the candidate record is wrong or spelled
       differently, not that every poll of the race is stale. Filtering on a
       name the polls have never heard of would silently delete the race.

    3. Nothing is dropped without saying which race lost what, so a bad
       candidate record shows up in the log instead of hiding in a seat count.
    """
    manual = config.DATA_MANUAL / "candidates_2026.csv"
    if not manual.exists() or "dem_candidate" not in df.columns:
        return df
    c = pd.read_csv(manual, comment="#")
    names: dict[str, str] = {}
    for r in c.itertuples():
        nm = str(getattr(r, "main_name", "") or "").strip()
        if nm and nm.lower() != "nan":
            names[r.race_id] = nm.split()[-1].lower()
    if not names:
        return df

    got = df["dem_candidate"].fillna("").astype(str).str.strip().str.lower()
    keep = pd.Series(True, index=df.index)
    for rid, want in names.items():
        in_race = df["race_id"] == rid
        if not in_race.any():
            continue
        matches = in_race & got.str.contains(want, regex=False)
        named = in_race & (got != "")
        if not matches.any():
            # Rule 2: the nominee is absent from every poll of this race, so
            # the record is more suspect than the polls.
            if named.any():
                log.warning("    %s: nominee '%s' appears in none of its %d polls; "
                            "keeping them all rather than emptying the race",
                            rid, want, int(named.sum()))
            continue
        dropped = named & ~matches
        if dropped.any():
            log.info("    %s: dropped %d of %d polls (not the nominee: %s)", rid,
                     int(dropped.sum()), int(named.sum()),
                     ", ".join(sorted(set(df.loc[dropped, "dem_candidate"].astype(str)))[:4]))
        keep &= ~dropped

    out = df[keep]
    n = len(df) - len(out)
    if n:
        log.info("dropped %d poll(s) of candidates who are not the nominee", n)
    return out


def main(force_fixture: bool = False):
    frames, provs = [], []
    report = []
    if not force_fixture:
        for name, fn, prov in [("NYT feed (NYT_POLLS_URL)", fetch_nyt_csv, "live"),
                               ("Wikipedia polling tables", fetch_wikipedia, "live"),
                               ("RealClearPolitics scrape", fetch_rcp, "live"),
                               ("manual CSV (data_store/manual/polls_2026.csv)", load_manual, "manual")]:
            try:
                df = fn()
            except Exception as e:  # noqa: BLE001
                report.append((name, f"error: {type(e).__name__}: {e}"))
                continue
            if df is None:
                report.append((name, "not available"))
            elif not len(df):
                report.append((name, "reachable but returned 0 usable polls"))
            else:
                report.append((name, f"{len(df)} polls"))
                frames.append(df)
                provs.append("live" if "live" in df["source"].iloc[0] else ("cache" if "cache" in df["source"].iloc[0] else prov))

    log.info("poll sources tried:")
    for name, outcome in report:
        log.info("    %-48s %s", name, outcome)

    if not frames:
        log.warning("=" * 70)
        log.warning("NO REAL POLLS FOUND - building a labelled FIXTURE.")
        log.warning("The forecast from this run is a pipeline test, not a forecast.")
        log.warning("")
        log.warning("To fix, put real polls in:")
        log.warning("    %s", MANUAL_POLLS)
        log.warning("A ready-to-fill template with the right columns is at:")
        log.warning("    %s", config.DATA_MANUAL / "templates" / "polls_2026.template.csv")
        log.warning("Or run:  python3 add_polls.py --example   to see the format.")
        log.warning("=" * 70)
        frames.append(build_fixture())
        provs.append("fixture")
    polls = pd.concat(frames, ignore_index=True).drop_duplicates("poll_id")
    polls = filter_hypothetical(polls)
    polls = filter_to_nominees(polls)
    prov = max(provs, key=lambda p: {"live": 0, "cache": 1, "manual": 1, "fixture": 3}[p])
    save_stage(polls, "polls_2026", prov, {"n_races_polled": int(polls.race_id.nunique()),
                                            "sources": sorted(polls.source.unique().tolist())})
    print(polls.groupby("office").size())


if __name__ == "__main__":
    main(force_fixture="--fixture" in sys.argv)
