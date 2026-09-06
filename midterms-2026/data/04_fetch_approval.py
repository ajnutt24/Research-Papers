"""
04_fetch_approval.py
====================
Presidential approval, used as a fundamentals predictor in place of a bare
"party in power" dummy. A dummy says only *which* party is exposed to the
midterm penalty; approval says *how much* of a penalty to expect (Tufte 1975,
Abramowitz's "referendum" framing).

Sources
-------
1. FiftyPlusOne's public approval tracker or the NYT tracker. Neither has a
   documented stable CSV, so candidate URLs are listed in APPROVAL_URLS and
   any that returns a parsable table wins. Set APPROVAL_URL to override.
2. data_store/manual/approval_manual.csv  (date, approve, disapprove, source)
   - paste in the tracker's topline by hand when scraping is blocked.
3. Labelled fixture.

Historical table
----------------
`approval_history` holds the final pre-midterm Gallup approval for every
midterm 1946-2022, the President's party, and a war-salience covariate on a
0-1 scale (an approximate news-attention measure for an ongoing U.S. conflict:
Korea 1950, Vietnam 1966/70, Gulf 1990, Afghanistan/Iraq 2002-2010, ISIS 2014,
Ukraine 2022). These are seeded from published figures, flagged verify=True,
and written to data_store/manual/approval_history.csv on first run so they can
be edited. The 2026 war-salience value (Iran conflict) is read from
data_store/manual/war_salience_2026.csv (weeks_since_escalation, attention_index)
and is a placeholder until you supply a measured series (e.g. GDELT volume).

Outputs
-------
approval_2026.parquet      date, approve, disapprove, net
approval_history.parquet   cycle, pres_party, approval, war_salience,
                           weeks_since_escalation, verify
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from utils import fetch_text, get_logger, save_stage  # noqa: E402

log = get_logger("04_approval")

APPROVAL_URLS = [
    os.environ.get("APPROVAL_URL", ""),
    "https://fiftyplusone.news/approval/data.csv",
    "https://static01.nyt.com/newsgraphics/2025-01-20-trump-approval/data.csv",
]
MANUAL = config.DATA_MANUAL / "approval_manual.csv"
HIST = config.DATA_MANUAL / "approval_history.csv"
WAR_2026 = config.DATA_MANUAL / "war_salience_2026.csv"

# cycle, president's party, final pre-midterm Gallup approval (approx.), war salience 0-1
HISTORY_SEED = [
    (1946, "D", 33, 0.10), (1950, "D", 39, 0.90), (1954, "R", 61, 0.10), (1958, "R", 57, 0.10),
    (1962, "D", 61, 0.30), (1966, "D", 44, 0.80), (1970, "R", 58, 0.80), (1974, "R", 54, 0.10),
    (1978, "D", 49, 0.05), (1982, "R", 42, 0.05), (1986, "R", 63, 0.10), (1990, "R", 58, 0.60),
    (1994, "D", 46, 0.05), (1998, "D", 66, 0.10), (2002, "R", 63, 0.70), (2006, "R", 38, 0.80),
    (2010, "D", 45, 0.40), (2014, "D", 42, 0.30), (2018, "R", 40, 0.10), (2022, "D", 40, 0.30),
]


def parse_any_table(txt: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(io.StringIO(txt))
    except Exception:
        return None
    cols = {c.lower(): c for c in df.columns}
    dcol = next((cols[c] for c in cols if "date" in c or c in ("enddate", "end")), None)
    acol = next((cols[c] for c in cols if c.startswith("approv") or c == "yes"), None)
    dcol2 = next((cols[c] for c in cols if c.startswith("disapprov") or c == "no"), None)
    if not (dcol and acol and dcol2):
        return None
    out = pd.DataFrame({"date": pd.to_datetime(df[dcol], errors="coerce").dt.date,
                        "approve": pd.to_numeric(df[acol], errors="coerce"),
                        "disapprove": pd.to_numeric(df[dcol2], errors="coerce")}).dropna()
    return out if len(out) else None


def fetch_live() -> tuple[pd.DataFrame | None, str]:
    for url in [u for u in APPROVAL_URLS if u]:
        try:
            txt, prov = fetch_text(url, "approval_" + str(abs(hash(url)))[:8] + ".csv")
            df = parse_any_table(txt)
            if df is not None:
                log.info("approval from %s (%s): %d rows", url, prov, len(df))
                return df, prov
        except Exception as e:
            log.warning("approval source %s failed: %s", url, e)
    return None, "missing"


def main():
    df, prov = fetch_live()
    if df is None and MANUAL.exists():
        df = pd.read_csv(MANUAL)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        prov = "manual"
        log.info("approval from manual file: %d rows", len(df))
    if df is None:
        log.warning("no approval data; writing FIXTURE (approve 41 / disapprove 55)")
        idx = pd.date_range(config.FORECAST_ASOF - pd.Timedelta(days=120), config.FORECAST_ASOF)
        rng = np.random.default_rng(4)
        df = pd.DataFrame({"date": idx.date,
                           "approve": 41 + np.cumsum(rng.normal(0, 0.1, len(idx))),
                           "disapprove": 55 - np.cumsum(rng.normal(0, 0.1, len(idx)))})
        prov = "fixture"
    df = df.sort_values("date")
    df["net"] = df["approve"] - df["disapprove"]
    save_stage(df, "approval_2026", prov, {"latest_approve": float(df.approve.iloc[-1]),
                                            "latest_net": float(df.net.iloc[-1])})

    # historical table (editable seed)
    if not HIST.exists():
        pd.DataFrame(HISTORY_SEED, columns=["cycle", "pres_party", "approval", "war_salience"]) \
            .assign(weeks_since_escalation=np.nan, verify=True).to_csv(HIST, index=False)
        log.info("seeded %s - please verify the figures", HIST.name)
    hist = pd.read_csv(HIST)
    # current cycle
    if WAR_2026.exists():
        w = pd.read_csv(WAR_2026).iloc[-1]
        war_2026, weeks_2026, wprov = float(w["attention_index"]), float(w["weeks_since_escalation"]), "manual"
    else:
        war_2026, weeks_2026, wprov = 0.5, np.nan, "fixture"
        log.warning("war_salience_2026.csv missing; using placeholder attention_index=0.5")
    cur = pd.DataFrame([{"cycle": config.CYCLE, "pres_party": "R", "approval": float(df.approve.iloc[-1]),
                         "war_salience": war_2026, "weeks_since_escalation": weeks_2026, "verify": True}])
    hist = pd.concat([hist[hist.cycle != config.CYCLE], cur], ignore_index=True).sort_values("cycle")
    save_stage(hist, "approval_history", "manual" if prov != "fixture" else "fixture",
               {"war_salience_2026_provenance": wprov})
    print(hist.tail(4).to_string(index=False))


if __name__ == "__main__":
    main()
