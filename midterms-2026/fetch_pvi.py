"""
fetch_pvi.py
============
Build per-district partisan leans for the 2026 maps from The Downballot's
presidential-results-by-district data, and write them to
data_store/manual/pvi_manual.csv, which stage 06 already reads.

    python3 fetch_pvi.py            # fetch, validate, write
    python3 fetch_pvi.py --dry-run  # fetch and validate, write nothing
    python3 fetch_pvi.py --all      # also overwrite unchanged districts

Why this script exists
----------------------
Ten states redrew their congressional maps for 2026: Texas, Missouri, North
Carolina, Alabama, Florida, Louisiana, Tennessee, Ohio, California and Utah.
That is 173 of 435 districts. For those, FiveThirtyEight's partisan-lean file
describes lines that no longer exist, so stage 06 falls back to the district's
STATE lean, which cannot tell a safe seat from a marginal one: every California
seat, safe and competitive alike, came out at D+25.7, and California's 22nd,
which every rater calls a Toss-up, was forecast at D+26.9 and 99% Democratic.

Source
------
The Downballot (formerly Daily Kos Elections) recomputed the 2024 presidential
result for every district on the new lines, published 9 July 2026, sponsored by
Grassroots Analytics, with a public methodology statement. It is the standard
open source for this and is free to use. Cook's own 2026 PVI is subscriber-only.

The sheet gives 2024 for every district and 2020 only for districts whose lines
did not change, because 2020 was never recomputed on the new boundaries. So the
lean here is built from 2024 alone, which is also the more informative election:
the 2024 coalition is the one 2026 runs on.

Two methodology choices worth stating
-------------------------------------
1. **Normalised to the national result, not raw margin.** A lean has to be
   relative to the country or it confounds "this district is Republican" with
   "it was a Republican year". The national margin is computed by summing the
   sheet's own district vote totals rather than taken from memory, which keeps
   it self-consistent with the numerator: it comes out at Trump +1.68.

2. **Bridged onto the FiveThirtyEight scale before use.** This is the part that
   is easy to get wrong. Script 09 fits its `b_lean` coefficients on historical
   cycles whose leans come from FiveThirtyEight, so feeding a differently-scaled
   lean into the same coefficient mis-predicts. Regressing the FiveThirtyEight
   lean on this one across the 262 districts that have both gives

       lean_538 = -2.09 + 1.024 * lean_2024      r = 0.968, residual sd 7.9

   The slope is essentially 1 and the intercept says the 2024 basis is about
   two points more Republican, which is corrected here. The 7.9-point residual
   is not correctable and is not noise: it is genuine movement between the
   2016-2020 coalition and the 2024 one. It means a new-map district's lean
   carries more uncertainty than an unchanged one's, which is already handled,
   because config.NEW_MAP_SHRINK_MULTIPLIER (1.75) widens precisely these
   districts' offsets by enough to cover it.

By default only the redistricted districts are written, since the other 262
already have a lean on the scale the coefficients were fitted on and changing
them is a separate decision. `--all` writes every district, which makes the
House internally consistent on a single 2024 basis at the cost of moving 262
districts off the fitted scale.
"""
from __future__ import annotations

import io
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
from utils import get_logger  # noqa: E402

log = get_logger("fetch_pvi")

# The Downballot's new-map workbook (2026 lines), "actual vote tallies" tab.
SHEET_ID = "1eZfaFI-c-PFOoKx1-zZA2MP0_dxRq_LVK0re3BOQqy0"
GID = "1491069057"
URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
SOURCE_NOTE = ("The Downballot, 2024 presidential results by 2026 congressional district "
               "(published 2026-07-09); lean = district margin minus national margin, "
               "bridged onto the FiveThirtyEight scale")

OUT = config.DATA_MANUAL / "pvi_manual.csv"
COLS = ["district", "incumbent", "party", "_pad", "h24", "t24", "tot24",
        "h24p", "t24p", "m24", "_gap", "b20", "t20", "tot20", "b20p", "t20p", "m20"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False)
                         .str.replace("%", "", regex=False).str.strip(), errors="coerce")


def race_id_of(code: str) -> str | None:
    """'TX-35' -> 'H-TX-35'; 'AK-AL' -> 'H-AK-01' (the model numbers at-large 01)."""
    parts = str(code).strip().upper().split("-")
    if len(parts) != 2 or len(parts[0]) != 2:
        return None
    st, d = parts
    if d in ("AL", "ATLARGE", "AT-LARGE"):
        return f"H-{st}-01"
    digits = "".join(c for c in d if c.isdigit())
    return f"H-{st}-{int(digits):02d}" if digits else None


def fetch() -> pd.DataFrame:
    """Download the sheet and return district, 2024 margin, and vote totals.

    The header spans four rows (a sponsor banner, a credit line, a year group
    header and the column names), so the data starts at row 5. The row count is
    checked against 435 rather than assumed, because a layout change upstream
    would otherwise shift every column silently.
    """
    import requests
    log.info("fetching %s", URL)
    r = requests.get(URL, timeout=90)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), skiprows=3)
    if len(df.columns) != len(COLS):
        raise ValueError(f"expected {len(COLS)} columns from the sheet, got {len(df.columns)}: "
                         f"{list(df.columns)}. The layout upstream has changed; "
                         f"check the header rows before trusting any parse.")
    df.columns = COLS
    df = df[df.district.notna() & df.district.astype(str).str.match(r"^[A-Z]{2}-")].copy()
    if len(df) != 435:
        raise ValueError(f"expected 435 districts, parsed {len(df)}. Refusing to write a "
                         f"partial lean file.")
    for c in ("h24", "t24", "m24"):
        df[c] = _num(df[c])
    df["race_id"] = [race_id_of(x) for x in df.district]
    bad = df.race_id.isna()
    if bad.any():
        raise ValueError(f"could not map district code(s): {df.loc[bad, 'district'].tolist()}")
    return df


def main(dry_run: bool = False, do_all: bool = False) -> None:
    df = fetch()

    # National margin summed from the sheet's own totals, so numerator and
    # denominator come from the same place.
    h, t = float(df.h24.sum()), float(df.t24.sum())
    nat = 100.0 * (h - t) / (h + t)
    log.info("national 2024 margin from the sheet itself: %+.2f (Harris %s / Trump %s)",
             nat, f"{h:,.0f}", f"{t:,.0f}")
    df["lean_2024"] = df.m24 - nat

    # --- bridge onto the FiveThirtyEight scale -------------------------------
    fund = PROJECT / "data_store" / "processed" / "fundamentals_2026.parquet"
    if not fund.exists():
        raise SystemExit(f"{fund} not found. Run the pipeline through stage 06 once first, so "
                         f"this script can calibrate against the existing leans.")
    f = pd.read_parquet(fund)
    f = f[f.race_id.str.startswith("H-")][["race_id", "lean", "lean_source", "new_map"]]
    m = df.merge(f, on="race_id", how="inner")

    ref = m[(m.lean_source == "538_2022")].dropna(subset=["lean", "lean_2024"])
    if len(ref) < 100:
        raise SystemExit(f"only {len(ref)} districts carry a FiveThirtyEight lean to calibrate "
                         f"against; too few to trust the bridge. Not writing.")
    slope, intercept = np.polyfit(ref.lean_2024, ref.lean, 1)
    resid = ref.lean - (intercept + slope * ref.lean_2024)
    r = float(np.corrcoef(ref.lean_2024, ref.lean)[0, 1])
    log.info("bridge fitted on %d unchanged districts: lean_538 = %+.2f + %.3f * lean_2024",
             len(ref), intercept, slope)
    log.info("    r = %.4f, residual sd = %.2f pts (real 2016-2020 vs 2024 coalition change, "
             "not noise)", r, float(resid.std()))
    if r < 0.85:
        raise SystemExit(f"the two lean series correlate at only {r:.3f}. That is too weak to "
                         f"bridge; something is wrong with the parse. Not writing.")
    m["lean_bridged"] = intercept + slope * m.lean_2024

    target = m if do_all else m[m.new_map.astype(bool)]
    target = target.dropna(subset=["lean_bridged"])
    log.info("writing leans for %d district(s)%s", len(target),
             "" if do_all else " on new maps (the ones with no usable lean today)")

    # Show the races this changes most, which is where to sanity-check it.
    chk = target.assign(shift=target.lean_bridged - target.lean).reindex(
        target.assign(shift=(target.lean_bridged - target.lean).abs()).sort_values(
            "shift", ascending=False).index)
    log.info("largest corrections (old lean -> new lean):")
    for _, x in chk.head(12).iterrows():
        log.info("    %-9s %+7.1f -> %+7.1f   (%s)", x.race_id, x.lean, x.lean_bridged,
                 str(x.incumbent)[:24])

    if dry_run:
        log.info("--dry-run: nothing written")
        return

    out = pd.DataFrame({"race_id": target.race_id.values,
                        "pvi": target.lean_bridged.round(2).values})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# {SOURCE_NOTE}\n"
              f"# bridge: lean_538 = {intercept:+.2f} + {slope:.3f} * lean_2024 "
              f"(r={r:.4f}, residual sd={resid.std():.2f})\n"
              f"# national 2024 margin used: {nat:+.2f}\n"
              f"# regenerate with: python3 fetch_pvi.py\n")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(header)
        out.to_csv(fh, index=False)
    log.info("wrote %s (%d rows)", OUT, len(out))
    log.info("now re-run:  python3 run_pipeline.py --from 06")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv, do_all="--all" in sys.argv)
