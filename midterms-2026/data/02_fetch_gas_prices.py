"""
02_fetch_gas_prices.py
======================
Scrape AAA's daily national-average regular gasoline price and cross-check it
against EIA's weekly official series (FRED: GASREGW).

Methodology
-----------
* AAA publishes no API; the number lives in an HTML table on
  https://gasprices.aaa.com/. We fetch that page at most once per
  CACHE_MAX_AGE_HOURS, only after confirming robots.txt allows it, with a
  descriptive User-Agent and a 2-second inter-request floor. Each observation
  is appended to a local daily log (data_store/raw/aaa_gas_log.csv) so a time
  series accumulates without re-scraping history.
* AAA is a private commercial source; EIA's series is public domain, so the
  EIA number is the "series of record". The stage output stores both and the
  gap between them, and script 09 uses EIA when the two disagree by more than
  15 cents (a sign the scrape parsed the wrong cell).
* Gas prices are a *salience* variable (they are posted on every corner), and
  are stored for use as an optional add-on to CPI in the fundamentals model;
  they are not used by default because CPI already contains motor fuel.

Output: data_store/processed/gas_prices.parquet
"""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from utils import SESSION, cache_is_fresh, cache_path, get_logger, load_stage, save_stage  # noqa: E402

log = get_logger("02_gas")
AAA_URL = "https://gasprices.aaa.com/"
LOG_FILE = cache_path("aaa_gas_log.csv")


def parse_aaa(html: str) -> dict[str, float]:
    """Extract Current/Yesterday/Week Ago/Month Ago/Year Ago regular prices
    from AAA's national-average table. Tolerant to markup changes: it looks
    for a row label followed by a $x.xxx cell."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    out = {}
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) >= 2 and re.search(r"(Current|Yesterday|Week Ago|Month Ago|Year Ago)", cells[0], re.I):
            m = re.search(r"\$?(\d\.\d{2,3})", cells[1])
            if m:
                key = re.sub(r"[^a-z]", "_", cells[0].lower()).strip("_")
                out[key] = float(m.group(1))
    if not out:  # last resort: regex on raw text
        m = re.search(r"Current Avg\.?\s*\$?(\d\.\d{3})", html)
        if m:
            out["current_avg"] = float(m.group(1))
    return out


def scrape_aaa(force: bool = False) -> tuple[dict, str]:
    p = cache_path("aaa_gasprices.html")
    if not force and cache_is_fresh(p):
        return parse_aaa(p.read_text()), "cache"
    r = SESSION.get(AAA_URL, check_robots=True)
    p.write_text(r.text)
    return parse_aaa(r.text), "live"


def append_log(today: date, price: float, prov: str):
    row = pd.DataFrame([{"date": today.isoformat(), "aaa_regular": price, "provenance": prov}])
    if LOG_FILE.exists():
        old = pd.read_csv(LOG_FILE)
        old = old[old["date"] != today.isoformat()]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(LOG_FILE, index=False)


def main(force: bool = False):
    today = config.FORECAST_ASOF
    prov = "fixture"
    aaa_now = float("nan")
    try:
        parsed, prov = scrape_aaa(force=force)
        key = next((k for k in parsed if k.startswith("current")), None)
        if key is None:
            raise ValueError(f"could not find current average in AAA page; parsed={parsed}")
        aaa_now = parsed[key]
        append_log(today, aaa_now, prov)
        log.info("AAA national regular = $%.3f (%s)", aaa_now, prov)
    except Exception as e:
        log.warning("AAA scrape unavailable (%s).", e)
        if LOG_FILE.exists():
            hist = pd.read_csv(LOG_FILE)
            aaa_now = float(hist.iloc[-1]["aaa_regular"])
            prov = "cache"
            log.info("using last logged AAA value $%.3f", aaa_now)

    # EIA cross-check from stage 01
    econ = load_stage("economic_monthly", required=False)
    eia_now = float("nan")
    eia_prov = "missing"
    if econ is not None:
        s = econ.dropna(subset=["gasregw"]).sort_values("date")
        if len(s):
            eia_now = float(s.iloc[-1]["gasregw"])
            eia_prov = str(s.iloc[-1]["provenance"])
    if pd.isna(aaa_now) and not pd.isna(eia_now):
        aaa_now, prov = eia_now, "fixture"   # placeholder equal to EIA, labelled
        log.warning("no AAA observation; using EIA value as labelled placeholder")
    elif pd.isna(aaa_now):
        aaa_now, prov = 3.20, "fixture"
        log.warning("no gas data at all; using FIXTURE $3.20")

    gap = aaa_now - eia_now if not pd.isna(eia_now) else float("nan")
    flag = bool(abs(gap) > 0.15) if gap == gap else False
    out = pd.DataFrame([{
        "date": today, "aaa_regular": aaa_now, "eia_gasregw": eia_now,
        "gap_aaa_minus_eia": gap, "cross_check_failed": flag, "eia_provenance": eia_prov,
    }])
    if LOG_FILE.exists():
        hist = pd.read_csv(LOG_FILE)
        hist["date"] = pd.to_datetime(hist["date"]).dt.date
        out = pd.concat([hist.rename(columns={"provenance": "aaa_provenance"}), out], ignore_index=True)
        out = out.drop_duplicates("date", keep="last")
    save_stage(out, "gas_prices", prov, {"cross_check_failed": flag})
    print(out.tail(3).to_string(index=False))


if __name__ == "__main__":
    main(force="--force" in sys.argv)
