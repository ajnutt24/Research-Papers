"""
01_fetch_economic_data.py
=========================
Pull the macro-economic regressors: CPI (CPIAUCSL), unemployment (UNRATE) and
the EIA weekly regular-gasoline price (GASREGW, the public-domain cross-check
for the AAA scrape in script 02) from FRED.

Methodology
-----------
* CPI is the *primary* economic regressor, used as year-over-year inflation at
  the last full month before the election (October). The "referendum" strand
  of the fundamentals literature (Hibbs's bread-and-peace model, Abramowitz's
  time-for-change model) uses real income growth; CPI inflation is the
  component of that story voters feel most directly and the one with the
  cleanest monthly history back to 1913.
* Unemployment is fetched but NOT automatically used as a second predictor.
  Both series proxy "how the economy feels"; this script prints their
  correlation over the historical midterm sample and stores it in the stage
  metadata so script 09 can decide (default: CPI only, unemployment ignored
  unless |corr| < 0.5).
* Fallback chain: FRED API (needs FRED_API_KEY) -> FRED's keyless fredgraph
  CSV -> the `datasets/cpi-us` GitHub mirror (CPI only, monthly since 1913)
  -> a labelled fixture. Each output row carries its provenance.

Outputs (data_store/processed/)
-------------------------------
economic_monthly.parquet          date, cpi, cpi_yoy, unrate, gasregw
economic_cycle_features.parquet   cycle, cpi_yoy_oct, unrate_oct  (1946..2026)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from utils import fetch_text, get_logger, save_stage, worst_provenance  # noqa: E402

log = get_logger("01_econ")

FRED_API = "https://api.stlouisfed.org/fred/series/observations"
FRED_GRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
CPI_MIRROR = "https://raw.githubusercontent.com/datasets/cpi-us/main/data/cpiai.csv"
MIDTERM_CYCLES = list(range(1946, 2027, 4))


def fred_series(series_id: str, force: bool = False) -> tuple[pd.Series, str]:
    """Return (monthly series indexed by date, provenance)."""
    # 1. official API
    if config.FRED_API_KEY:
        url = (f"{FRED_API}?series_id={series_id}&api_key={config.FRED_API_KEY}"
               f"&file_type=json&observation_start=1913-01-01")
        try:
            txt, prov = fetch_text(url, f"fred_{series_id}.json", force=force)
            import json
            obs = json.loads(txt)["observations"]
            s = pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in obs if o["value"] != "."})
            return s.sort_index(), prov
        except Exception as e:
            log.warning("FRED API failed for %s: %s", series_id, e)
    # 2. keyless CSV endpoint
    try:
        txt, prov = fetch_text(FRED_GRAPH.format(sid=series_id), f"fredgraph_{series_id}.csv", force=force)
        df = pd.read_csv(io.StringIO(txt))
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = pd.Series(df["value"].values, index=pd.to_datetime(df["date"])).dropna()
        return s.sort_index(), prov
    except Exception as e:
        log.warning("fredgraph failed for %s: %s", series_id, e)
    # 3. GitHub mirror (CPI only)
    if series_id == "CPIAUCSL":
        try:
            txt, prov = fetch_text(CPI_MIRROR, "cpi_mirror.csv", force=force, max_age_hours=24 * 7)
            df = pd.read_csv(io.StringIO(txt))
            s = pd.Series(df["Index"].values, index=pd.to_datetime(df["Date"]))
            return s.sort_index(), "mirror"
        except Exception as e:
            log.warning("CPI mirror failed: %s", e)
    return pd.Series(dtype=float), "missing"


def fixture_series(series_id: str) -> pd.Series:
    """Labelled placeholder so downstream stages can run offline."""
    idx = pd.date_range("1947-01-01", config.FORECAST_ASOF, freq="MS")
    rng = np.random.default_rng(1)
    if series_id == "CPIAUCSL":
        vals = 20 * np.exp(np.cumsum(rng.normal(0.003, 0.003, len(idx))))
    elif series_id == "UNRATE":
        vals = np.clip(5.5 + np.cumsum(rng.normal(0, 0.15, len(idx))) * 0.3, 2.5, 11)
    else:  # GASREGW weekly-ish, keep monthly for simplicity
        vals = np.clip(3.2 + np.cumsum(rng.normal(0, 0.05, len(idx))) * 0.2, 1.5, 5.5)
    return pd.Series(vals, index=idx)


def main(force: bool = False):
    provs = []
    series = {}
    for sid in ["CPIAUCSL", "UNRATE", "GASREGW"]:
        s, prov = fred_series(sid, force=force)
        if s.empty:
            log.warning("%s unavailable from every source; using FIXTURE", sid)
            s, prov = fixture_series(sid), "fixture"
        series[sid] = s
        provs.append(prov)
        log.info("%s: %d obs, last=%s (%s)", sid, len(s), s.index.max().date(), prov)

    monthly = pd.DataFrame({
        "cpi": series["CPIAUCSL"].resample("MS").mean(),
        "unrate": series["UNRATE"].resample("MS").mean(),
        "gasregw": series["GASREGW"].resample("MS").mean(),
    })
    monthly["cpi_yoy"] = 100 * (monthly["cpi"] / monthly["cpi"].shift(12) - 1)
    monthly = monthly.reset_index().rename(columns={"index": "date"})
    if "date" not in monthly.columns:
        monthly = monthly.rename(columns={monthly.columns[0]: "date"})

    # election-cycle features: value at October of the election year (or the
    # latest available month for the current cycle)
    feats = []
    m = monthly.set_index("date")
    for cy in MIDTERM_CYCLES:
        target = pd.Timestamp(year=cy, month=10, day=1)
        avail = m.loc[:target]
        if avail.empty:
            continue
        # take the latest non-missing value per column (series end at different months)
        def last_valid(col):
            v = avail[col].dropna()
            return (float(v.iloc[-1]), v.index[-1].date()) if len(v) else (float("nan"), None)
        cpi_v, cpi_m = last_valid("cpi_yoy")
        un_v, _ = last_valid("unrate")
        gas_v, _ = last_valid("gasregw")
        feats.append({"cycle": cy, "feature_month": cpi_m,
                      "cpi_yoy_oct": cpi_v, "unrate_oct": un_v, "gasregw_oct": gas_v})
    feats = pd.DataFrame(feats)

    hist = feats[feats.cycle < config.CYCLE].dropna(subset=["cpi_yoy_oct", "unrate_oct"])
    corr = float(hist["cpi_yoy_oct"].corr(hist["unrate_oct"])) if len(hist) > 3 else float("nan")
    log.info("corr(CPI YoY, unemployment) across %d midterms = %.2f", len(hist), corr)
    # stage provenance follows the PRIMARY regressor (CPI); others are recorded in meta
    prov = provs[0]
    save_stage(monthly, "economic_monthly", prov, {"sources": dict(zip(["CPIAUCSL", "UNRATE", "GASREGW"], provs))})
    save_stage(feats, "economic_cycle_features", prov,
               {"cpi_unrate_corr": corr,
                "use_unemployment": bool(abs(corr) < 0.5) if corr == corr else False})
    print(feats.tail(6).to_string(index=False))


if __name__ == "__main__":
    main(force="--force" in sys.argv)
