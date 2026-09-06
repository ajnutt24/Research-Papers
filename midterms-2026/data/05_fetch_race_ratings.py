"""
05_fetch_race_ratings.py
========================
Current expert race ratings (Cook Political Report, Sabato's Crystal Ball,
Inside Elections) for every 2026 race, plus a historical ratings table for
calibration (script 10) and backtesting (script 14).

Methodology
-----------
* Ratings are qualitative; we map them to an ordinal tier code
  (-3 Safe R ... 0 Toss-up ... +3 Safe D). "Tilt" ratings map to +/-0.5 and
  are rounded toward Toss-up. The consensus rating is the median tier across
  available sources, which is robust to one outlier rater.
* Scrapers: each site is fetched once, robots.txt permitting, and parsed with
  a tolerant pattern that looks for a race label ("TX-23", "Texas 23rd",
  "AZ Senate", "Arizona Governor") near a rating keyword. Sites change their
  markup often, so when a scraper returns nothing the script falls back to
  data_store/manual/race_ratings_2026.csv (race_id, source, rating) which you
  can populate by hand in ten minutes from the three sites' summary tables.
* Fixture: if neither scrape nor manual file is available, tiers are derived
  from partisan lean thresholds (|lean| < 3 Toss-up, < 8 Lean, < 15 Likely,
  else Safe) and labelled source="fixture_pvi_derived".
* Historical ratings: data_store/manual/historical_ratings.csv
  (cycle, race_id, source, rating) if supplied; otherwise FiveThirtyEight's
  2018 forecast-review file, which records a tier category for all 506 races
  with the actual winner. That gives a one-cycle empirical calibration; add
  Cook's archived ratings for 2010/2014/2022 to strengthen it.

Outputs
-------
race_ratings_2026.parquet       race_id, source, rating, tier_code (+ consensus rows source="consensus")
race_ratings_historical.parquet cycle, race_id, source, rating, tier_code, dem_won
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from utils import GH, fetch_text, get_logger, load_partisan_lean, save_stage, tier_from_lean  # noqa: E402

log = get_logger("05_ratings")

SOURCES = {
    "cook": {"House": "https://www.cookpolitical.com/ratings/house-race-ratings",
             "Senate": "https://www.cookpolitical.com/ratings/senate-race-ratings",
             "Governor": "https://www.cookpolitical.com/ratings/governor-race-ratings"},
    "sabato": {"House": "https://centerforpolitics.org/crystalball/2026-house/",
               "Senate": "https://centerforpolitics.org/crystalball/2026-senate/",
               "Governor": "https://centerforpolitics.org/crystalball/2026-governor/"},
    "inside": {"House": "https://insideelections.com/ratings/house",
               "Senate": "https://insideelections.com/ratings/senate",
               "Governor": "https://insideelections.com/ratings/governor"},
}
MANUAL = config.DATA_MANUAL / "race_ratings_2026.csv"
MANUAL_HIST = config.DATA_MANUAL / "historical_ratings.csv"
FR2018 = f"{GH}/fivethirtyeight/data/master/forecast-review/forecast_results_2018.csv"

TIER_CODE = {"Safe D": 3, "Solid D": 3, "Likely D": 2, "Lean D": 1, "Tilt D": 0.5, "Toss-up": 0,
             "Tossup": 0, "Toss Up": 0, "Tilt R": -0.5, "Lean R": -1, "Likely R": -2,
             "Safe R": -3, "Solid R": -3}
CODE_TIER = {3: "Safe D", 2: "Likely D", 1: "Lean D", 0: "Toss-up", -1: "Lean R", -2: "Likely R", -3: "Safe R"}
RATING_RE = re.compile(r"\b(Safe|Solid|Likely|Lean|Tilt|Toss[- ]?up)\s*(D|R|Dem|Rep|Democrat|Republican)?\b", re.I)


def normalise_rating(txt: str) -> str | None:
    m = RATING_RE.search(str(txt))
    if not m:
        return None
    kind = m.group(1).lower().replace(" ", "").replace("-", "")
    side = (m.group(2) or "").upper()[:1]
    if kind == "tossup":
        return "Toss-up"
    if not side:
        return None
    kind = {"safe": "Safe", "solid": "Safe", "likely": "Likely", "lean": "Lean", "tilt": "Tilt"}[kind]
    return f"{kind} {side}"


def tier_code(rating: str) -> float:
    return TIER_CODE.get(rating, np.nan)


def code_to_tier(code: float) -> str:
    return CODE_TIER[int(np.sign(code) * np.floor(abs(code)))]   # tilt -> toss-up


def parse_rating_page(html: str, office: str) -> pd.DataFrame:
    """Tolerant parser: scan table rows / list items for a race label and a rating."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    name_to_abbr = {v.upper(): k for k, v in config.STATE_NAMES.items()}
    rows = []
    for el in soup.find_all(["tr", "li", "div"]):
        txt = el.get_text(" ", strip=True)
        if len(txt) > 200:
            continue
        rating = normalise_rating(txt)
        if not rating:
            continue
        rid = None
        m = re.search(r"\b([A-Z]{2})-(\d{1,2})\b", txt)
        if office == "House" and m:
            rid = f"H-{m.group(1)}-{int(m.group(2)):02d}"
        elif office == "House":
            m2 = re.search(r"([A-Z][a-z]+(?: [A-Z][a-z]+)?)\s+(\d{1,2})", txt)
            if m2 and m2.group(1).upper() in name_to_abbr:
                rid = f"H-{name_to_abbr[m2.group(1).upper()]}-{int(m2.group(2)):02d}"
        else:
            m3 = re.search(r"\b([A-Z]{2})\b", txt)
            m4 = re.search(r"([A-Z][a-z]+(?: [A-Z][a-z]+)?)", txt)
            st = m3.group(1) if m3 and m3.group(1) in config.STATES else (
                name_to_abbr.get(m4.group(1).upper()) if m4 else None)
            if st:
                prefix = "S" if office == "Senate" else "G"
                rid = f"{prefix}-{st}-special" if ("special" in txt.lower() and office == "Senate") else f"{prefix}-{st}"
        if rid:
            rows.append({"race_id": rid, "rating": rating})
    return pd.DataFrame(rows).drop_duplicates("race_id") if rows else pd.DataFrame(columns=["race_id", "rating"])


def scrape_all() -> pd.DataFrame | None:
    frames = []
    for src, pages in SOURCES.items():
        for office, url in pages.items():
            try:
                html, prov = fetch_text(url, f"ratings_{src}_{office}.html", check_robots=True)
                df = parse_rating_page(html, office)
                if len(df):
                    df["source"] = src
                    frames.append(df)
                    log.info("%s %s: %d ratings (%s)", src, office, len(df), prov)
            except Exception as e:
                log.warning("%s %s failed: %s", src, office, e)
    return pd.concat(frames, ignore_index=True) if frames else None


def fixture_ratings() -> pd.DataFrame:
    lean, _ = load_partisan_lean("2022")
    lean = lean.set_index("race_key")["lean"]
    uni = config.race_universe()
    key = np.where(uni.office == "House", uni.state + "-" + uni.district.map("{:02d}".format), uni.state)
    uni["lean"] = [lean.get(k, 0.0) for k in key]
    # incumbency bonus for the fixture only
    inc = uni.get("incumbent_party", pd.Series([np.nan] * len(uni))).map({"D": 2.0, "R": -2.0}).fillna(0)
    uni["rating"] = [tier_from_lean(v) for v in uni["lean"] + inc]
    out = uni[["race_id", "rating"]].copy()
    out["source"] = "fixture_pvi_derived"
    return out


def consensus(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tier_code"] = df["rating"].map(tier_code)
    df = df.dropna(subset=["tier_code"])
    cons = df.groupby("race_id")["tier_code"].median().reset_index()
    cons["rating"] = cons["tier_code"].map(code_to_tier)
    cons["source"] = "consensus"
    cons["n_sources"] = df.groupby("race_id").size().values
    return pd.concat([df, cons], ignore_index=True)


def historical() -> tuple[pd.DataFrame, str]:
    if MANUAL_HIST.exists():
        h = pd.read_csv(MANUAL_HIST)
        h["rating"] = h["rating"].map(normalise_rating)
        prov = "manual"
    else:
        txt, prov = fetch_text(FR2018, "fivethirtyeight_forecast_results_2018.csv", max_age_hours=24 * 30)
        fr = pd.read_csv(io.StringIO(txt))
        fr = fr[fr["version"] == "deluxe"]   # deluxe = the version that folds in expert ratings
        def rid(row):
            st, seat = row["race"].split("-")
            if row["branch"] == "House":
                return f"H-{st}-{int(seat):02d}"
            if row["branch"] == "Senate":
                return f"S-{st}-special" if seat.endswith("2") else f"S-{st}"
            return f"G-{st}"
        h = pd.DataFrame({"cycle": 2018, "race_id": fr.apply(rid, axis=1),
                          "source": "fivethirtyeight_category",
                          "rating": fr["category"].map(normalise_rating),
                          "dem_won": fr["Democrat_Won"].astype(float)})
        prov = "mirror"
    h["tier_code"] = h["rating"].map(tier_code)
    h = h.dropna(subset=["tier_code"])
    h["rating"] = h["tier_code"].map(code_to_tier)
    return h, prov


def main():
    df = scrape_all()
    prov = "live"
    if (df is None or len(df) < 50) and MANUAL.exists():
        df = pd.read_csv(MANUAL)
        df["rating"] = df["rating"].map(normalise_rating)
        prov = "manual"
        log.info("ratings from manual file: %d rows", len(df))
    if df is None or len(df) < 50:
        log.warning("no ratings from scrape or manual file; using FIXTURE derived from partisan lean")
        df = fixture_ratings()
        prov = "fixture"
    out = consensus(df)
    uni = config.race_universe()
    missing = set(uni.race_id) - set(out.race_id)
    if missing:
        log.warning("%d races have no rating; they will be treated as 'unrated' downstream", len(missing))
    save_stage(out, "race_ratings_2026", prov, {"n_rated": int(out[out.source == "consensus"].race_id.nunique()),
                                                 "n_missing": len(missing)})
    hist, hprov = historical()
    save_stage(hist, "race_ratings_historical", hprov, {"cycles": sorted(hist.cycle.unique().tolist())})
    print(out[out.source == "consensus"].rating.value_counts())


if __name__ == "__main__":
    main()
