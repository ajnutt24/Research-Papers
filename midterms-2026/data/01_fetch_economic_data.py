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



# --------------------------------------------------------------------------
# State personal income (Hummel & Rothschild: the most predictive economic
# variable for Senate races, and only as a TREND, not a level)
# --------------------------------------------------------------------------
# FRED publishes quarterly total personal income per state. The series id is
# the two-letter state code followed by OTOT (for example MIOTOT). Naming has
# changed before, so several patterns are tried and whichever returns data
# wins. This is optional: if FRED is unreachable the rest of the pipeline is
# unaffected and the income term is simply absent.
STATE_INCOME_PATTERNS = ["{st}OTOT", "{st}PCPI", "{st}NQGSP"]


def state_income_growth(force: bool = False) -> tuple[pd.DataFrame, str]:
    """Year-over-year growth in personal income, per state, per quarter."""
    frames, pattern_used, failures = [], None, 0
    for st in config.STATES:
        got = None
        for pat in ([pattern_used] if pattern_used else STATE_INCOME_PATTERNS):
            sid = pat.format(st=st)
            ser, prov = fred_series(sid, force=force)
            if not ser.empty:
                got, pattern_used = ser, pat
                break
        if got is None:
            failures += 1
            if failures >= 5 and not frames:
                log.warning("state personal income unavailable from FRED; skipping this term")
                return pd.DataFrame(), "missing"
            continue
        g = 100 * (got / got.shift(4) - 1) if pattern_used.endswith("OTOT") else 100 * (got / got.shift(1) - 1)
        frames.append(pd.DataFrame({"state": st, "date": g.index, "income_yoy": g.values}))
    if not frames:
        return pd.DataFrame(), "missing"
    out = pd.concat(frames, ignore_index=True).dropna(subset=["income_yoy"])
    log.info("state personal income: %d states via pattern %s", out.state.nunique(), pattern_used)
    return out, "live"


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
    inc, inc_prov = state_income_growth(force=force)
    if len(inc):
        save_stage(inc, "state_income_growth", inc_prov,
                   {"n_states": int(inc.state.nunique())})
    else:
        log.info("no state income data; the state-economy term will be absent")

    save_stage(feats, "economic_cycle_features", prov,
               {"cpi_unrate_corr": corr,
                "use_unemployment": bool(abs(corr) < 0.5) if corr == corr else False})
    print(feats.tail(6).to_string(index=False))


if __name__ == "__main__":
    main(force="--force" in sys.argv)
