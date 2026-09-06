"""
config.py
=========
Shared parameters for the 2026 midterm forecasting pipeline.

Everything that a downstream script might need to *agree* on lives here:
file paths, the election date, the race universe (which seats are on the
ballot), the redistricting-status table, API keys (read from environment
variables, never hard-coded), and the hyper-parameters that control the
uncertainty structure and the Monte Carlo simulation.

Methodology notes
-----------------
* Margins are expressed everywhere as **Democratic minus Republican, in
  percentage points** (D+5 = +5.0, R+3 = -3.0). The national environment is
  expressed on the same scale (generic-ballot margin).
* The race universe is *declared* here rather than scraped so that the model
  runs even when every external source is down, and so that the seat count is
  auditable. Update `SENATE_2026`, `GOVERNOR_2026` and `HOUSE_SEATS_BY_STATE`
  by hand when something changes (a resignation, a court-ordered map).
* Redistricting status is a maintained table, not a fact the model can
  discover. It drives (a) which partisan-lean vintage is used for a district
  and (b) how much extra shrinkage the hierarchical model applies to districts
  whose lean is a mapmaker's assumption rather than an electoral track record.

Environment variables
---------------------
FRED_API_KEY   Federal Reserve Economic Data key (https://fred.stlouisfed.org/docs/api/api_key.html)
FEC_API_KEY    FEC OpenFEC key (https://api.open.fec.gov/developers/); DEMO_KEY works with low limits
NYT_POLLS_URL  optional override for the poll CSV endpoint (see 03_fetch_polls.py)
MODEL_USER_AGENT  contact string sent with scraping requests
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data_store" / "raw"          # cached downloads
DATA_MANUAL = PROJECT_ROOT / "data_store" / "manual"    # hand-maintained inputs
DATA_PROCESSED = PROJECT_ROOT / "data_store" / "processed"  # stage outputs
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
for _p in (DATA_RAW, DATA_MANUAL, DATA_PROCESSED, OUTPUTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------
ELECTION_DATE = date(2026, 11, 3)
CYCLE = 2026
# "As of" date for the forecast. Override with FORECAST_ASOF=YYYY-MM-DD to
# freeze a run (useful for reproducibility and for backtesting snapshots).
FORECAST_ASOF = date.fromisoformat(os.environ.get("FORECAST_ASOF", date.today().isoformat()))
DAYS_TO_ELECTION = max((ELECTION_DATE - FORECAST_ASOF).days, 0)

# Historical midterm cycles used for backtesting (script 14) and calibration
BACKTEST_CYCLES = [2010, 2014, 2018, 2022]
# Cycles whose polling *miss* informs the correlated-polling-error prior.
# 2016 and 2020 are included because the spec asks that the polling-error
# prior reflect those misses, not just this cycle's poll-to-poll spread.
POLL_MISS_CYCLES = [2014, 2016, 2018, 2020, 2022]

# --------------------------------------------------------------------------
# Credentials & HTTP behaviour
# --------------------------------------------------------------------------
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FEC_API_KEY = os.environ.get("FEC_API_KEY", "DEMO_KEY")
USER_AGENT = os.environ.get(
    "MODEL_USER_AGENT",
    "midterms-2026-forecast/0.1 (research; contact via repository issues)",
)
HTTP_TIMEOUT = 30            # seconds
HTTP_MIN_INTERVAL = 2.0      # seconds between requests to the same host
HTTP_RETRIES = 3
CACHE_MAX_AGE_HOURS = 12     # re-use a cached download younger than this

# --------------------------------------------------------------------------
# Race universe
# --------------------------------------------------------------------------
STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL",
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
    "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]
STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

# 2020-census apportionment. Sums to 435. Mid-decade redistricting changes
# district *lines*, never the number of seats per state.
HOUSE_SEATS_BY_STATE = {
    "AL": 7, "AK": 1, "AZ": 9, "AR": 4, "CA": 52, "CO": 8, "CT": 5, "DE": 1,
    "FL": 28, "GA": 14, "HI": 2, "ID": 2, "IL": 17, "IN": 9, "IA": 4, "KS": 4,
    "KY": 6, "LA": 6, "ME": 2, "MD": 8, "MA": 9, "MI": 13, "MN": 8, "MS": 4,
    "MO": 8, "MT": 2, "NE": 3, "NV": 4, "NH": 2, "NJ": 12, "NM": 3, "NY": 26,
    "NC": 14, "ND": 1, "OH": 15, "OK": 5, "OR": 6, "PA": 17, "RI": 2, "SC": 7,
    "SD": 1, "TN": 9, "TX": 38, "UT": 4, "VT": 1, "VA": 11, "WA": 10, "WV": 2,
    "WI": 8, "WY": 1,
}
assert sum(HOUSE_SEATS_BY_STATE.values()) == 435, "House apportionment must sum to 435"

HOUSE_MAJORITY = 218
SENATE_MAJORITY = 51          # 50 + VP tiebreak goes to the President's party (R)
SENATE_SEATS_NOT_UP = {"D": 34, "R": 31}   # 47 D-caucus, 53 R, minus seats on ballot
GOVERNORS_NOT_UP = {"D": 6, "R": 8}        # 50 governors minus the 36 on the ballot

# Senate seats on the 2026 ballot: 33 Class II seats + 2 specials (OH, FL).
# incumbent_party = party holding the seat now; open = incumbent not running.
# `verify` flags entries the maintainer should re-confirm against current news.
SENATE_2026 = [
    # state, class, special, incumbent_party, open_seat, note
    ("AL", 2, False, "R", True,  "Tuberville running for governor"),
    ("AK", 2, False, "R", False, "Sullivan"),
    ("AR", 2, False, "R", False, "Cotton"),
    ("CO", 2, False, "D", False, "Hickenlooper"),
    ("DE", 2, False, "D", False, "Coons"),
    ("GA", 2, False, "D", False, "Ossoff"),
    ("ID", 2, False, "R", False, "Risch"),
    ("IL", 2, False, "D", True,  "Durbin retiring"),
    ("IA", 2, False, "R", True,  "Ernst retiring (verify)"),
    ("KS", 2, False, "R", False, "Marshall"),
    ("KY", 2, False, "R", True,  "McConnell retiring"),
    ("LA", 2, False, "R", False, "Cassidy (verify primary outcome)"),
    ("ME", 2, False, "R", False, "Collins"),
    ("MA", 2, False, "D", False, "Markey (verify primary outcome)"),
    ("MI", 2, False, "D", True,  "Peters retiring"),
    ("MN", 2, False, "D", True,  "Smith retiring"),
    ("MS", 2, False, "R", False, "Hyde-Smith"),
    ("MT", 2, False, "R", False, "Daines"),
    ("NE", 2, False, "R", False, "Ricketts"),
    ("NH", 2, False, "D", True,  "Shaheen retiring"),
    ("NJ", 2, False, "D", False, "Booker"),
    ("NM", 2, False, "D", False, "Lujan"),
    ("NC", 2, False, "R", True,  "Tillis retiring"),
    ("OK", 2, False, "R", False, "Mullin"),
    ("OR", 2, False, "D", False, "Merkley"),
    ("RI", 2, False, "D", False, "Reed"),
    ("SC", 2, False, "R", False, "Graham"),
    ("SD", 2, False, "R", False, "Rounds"),
    ("TN", 2, False, "R", False, "Hagerty"),
    ("TX", 2, False, "R", False, "Cornyn (verify primary/runoff outcome)"),
    ("VA", 2, False, "D", False, "Warner"),
    ("WV", 2, False, "R", False, "Capito"),
    ("WY", 2, False, "R", False, "Lummis"),
    ("OH", 3, True,  "R", False, "special: Husted (appointed)"),
    ("FL", 3, True,  "R", False, "special: Moody (appointed)"),
]
assert len(SENATE_2026) == 35

# Governorships on the 2026 ballot (36). Same conventions as above.
GOVERNOR_2026 = [
    ("AL", "R", True,  "Ivey term-limited"),
    ("AK", "R", True,  "Dunleavy term-limited"),
    ("AZ", "D", False, "Hobbs"),
    ("AR", "R", False, "Sanders"),
    ("CA", "D", True,  "Newsom term-limited"),
    ("CO", "D", True,  "Polis term-limited"),
    ("CT", "D", False, "Lamont (verify)"),
    ("FL", "R", True,  "DeSantis term-limited"),
    ("GA", "R", True,  "Kemp term-limited"),
    ("HI", "D", False, "Green"),
    ("ID", "R", False, "Little (verify)"),
    ("IL", "D", False, "Pritzker"),
    ("IA", "R", True,  "Reynolds retiring"),
    ("KS", "D", True,  "Kelly term-limited"),
    ("ME", "D", True,  "Mills term-limited"),
    ("MD", "D", False, "Moore"),
    ("MA", "D", False, "Healey"),
    ("MI", "D", True,  "Whitmer term-limited"),
    ("MN", "D", True,  "Walz not seeking re-election (verify)"),
    ("NE", "R", False, "Pillen"),
    ("NV", "R", False, "Lombardo"),
    ("NH", "R", False, "Ayotte"),
    ("NM", "D", True,  "Lujan Grisham term-limited"),
    ("NY", "D", False, "Hochul"),
    ("OH", "R", True,  "DeWine term-limited"),
    ("OK", "R", True,  "Stitt term-limited"),
    ("OR", "D", False, "Kotek"),
    ("PA", "D", False, "Shapiro"),
    ("RI", "D", False, "McKee (verify primary)"),
    ("SC", "R", True,  "McMaster term-limited"),
    ("SD", "R", False, "Rhoden (appointed; verify primary)"),
    ("TN", "R", True,  "Lee term-limited"),
    ("TX", "R", False, "Abbott"),
    ("VT", "R", False, "Scott (verify)"),
    ("WI", "D", True,  "Evers retiring"),
    ("WY", "R", True,  "Gordon term-limited"),
]
assert len(GOVERNOR_2026) == 36

# --------------------------------------------------------------------------
# Redistricting status tracker
# --------------------------------------------------------------------------
# status values:
#   "census_map"   -> the post-2020-census map (used in 2022/2024) is in effect
#   "new_map"      -> a mid-decade map is in effect for the 2026 general
#   "litigation"   -> a map change is pending / under challenge; treat lean as
#                     less reliable and widen shrinkage
# lean_vintage: which partisan-lean file applies ("2022" = 538's post-2020
# census leans; "manual" = a district-level file the maintainer drops into
# data_store/manual/pvi_manual.csv for redrawn districts).
# This table is NOT final. Missouri's ruling is under appeal to the U.S. Supreme
# Court; Virginia and Florida have unresolved litigation. Re-verify before every
# published run.
REDISTRICTING_STATUS = {
    st: {"status": "census_map", "lean_vintage": "2022", "new_map_for_2026": False,
         "note": "", "last_verified": "2026-09-06"}
    for st in STATES
}
REDISTRICTING_STATUS.update({
    "TX": {"status": "new_map", "lean_vintage": "manual", "new_map_for_2026": True,
           "note": "Mid-decade map enacted Aug 2025; district-court block stayed by SCOTUS Dec 2025. Verify.",
           "last_verified": "2026-09-06"},
    "CA": {"status": "new_map", "lean_vintage": "manual", "new_map_for_2026": True,
           "note": "Prop 50 (Nov 2025) map in effect through 2030. Verify.",
           "last_verified": "2026-09-06"},
    "NC": {"status": "new_map", "lean_vintage": "manual", "new_map_for_2026": True,
           "note": "Oct 2025 legislative map; challenge not granted before 2026. Verify.",
           "last_verified": "2026-09-06"},
    "OH": {"status": "new_map", "lean_vintage": "manual", "new_map_for_2026": True,
           "note": "Redistricting commission map adopted Oct 2025. Verify.",
           "last_verified": "2026-09-06"},
    "UT": {"status": "new_map", "lean_vintage": "manual", "new_map_for_2026": True,
           "note": "Court-ordered map (Nov 2025). Appeals pending. Verify.",
           "last_verified": "2026-09-06"},
    "MO": {"status": "census_map", "lean_vintage": "2022", "new_map_for_2026": False,
           "note": ("SPECIAL CASE: Aug 2026 primary ran on the 2025 map; state Supreme Court "
                    "blocked that map for the general, so the 2020-census map applies in "
                    "November. Ruling under appeal to SCOTUS. Candidates must be re-mapped "
                    "from primary district to general district via "
                    "data_store/manual/missouri_candidate_map.csv."),
           "last_verified": "2026-09-06"},
    "FL": {"status": "litigation", "lean_vintage": "2022", "new_map_for_2026": False,
           "note": "Unresolved litigation / possible mid-decade redraw. Verify.",
           "last_verified": "2026-09-06"},
    "VA": {"status": "litigation", "lean_vintage": "2022", "new_map_for_2026": False,
           "note": "Redistricting amendment / litigation unresolved. Verify.",
           "last_verified": "2026-09-06"},
    "LA": {"status": "litigation", "lean_vintage": "2022", "new_map_for_2026": False,
           "note": "Louisiana v. Callais (VRA Section 2) outcome may affect map. Verify.",
           "last_verified": "2026-09-06"},
})

# Missouri: candidates must be re-mapped by hand. The general election uses the
# 2020-census districts (MO-01 .. MO-08). See data_store/manual/missouri_candidate_map.csv
MISSOURI_GENERAL_MAP_VINTAGE = "2022"

# --------------------------------------------------------------------------
# Rating tiers (Cook / Sabato / Inside Elections share this vocabulary)
# --------------------------------------------------------------------------
RATING_TIERS = ["Safe D", "Likely D", "Lean D", "Toss-up", "Lean R", "Likely R", "Safe R"]
# Literature / published-track-record priors for P(Dem win) by tier, used as a
# Beta prior that the empirical calibration (script 10) updates. These are
# deliberately soft (prior weight ~ 20 races per tier).
RATING_TIER_PRIOR_PDEM = {
    "Safe D": 0.995, "Likely D": 0.93, "Lean D": 0.78, "Toss-up": 0.50,
    "Lean R": 0.22, "Likely R": 0.07, "Safe R": 0.005,
}
RATING_TIER_PRIOR_WEIGHT = 20.0

# --------------------------------------------------------------------------
# Uncertainty structure
# --------------------------------------------------------------------------
# Correlated polling-error prior: if the historical data (script 08) cannot be
# loaded, fall back to this standard deviation (pct points of margin) for the
# shared, all-races polling shock. ~3.5 reflects the 2016/2020 House/Senate
# misses (538: ~ 2-6 pts toward Democrats) balanced by better cycles.
POLL_SHOCK_SD_FALLBACK = 3.5
# Correlated national/fundamentals shock fallback (pct points)
NATIONAL_SHOCK_SD_FALLBACK = 3.0
# State-level shared shock (pct points) - regional/state polling & turnout error
STATE_SHOCK_SD = 2.0
# Idiosyncratic race noise floor (pct points) added to every race
RACE_NOISE_FLOOR_SD = 4.0
# Extra shrinkage for districts on new maps (multiplier on district-offset sd)
NEW_MAP_SHRINK_MULTIPLIER = 1.75

# Stacking-weight horizon curve. FiveThirtyEight's poll archive holds only the
# final three weeks before each election, so weights are *estimated* for
# horizons <= 21 days and *extrapolated* beyond that with an exponential decay
# toward a fundamentals-heavy floor:  w(h) = floor + (w21 - floor) * exp(-(h-21)/tau).
# Replace with estimated values once a long-horizon poll archive is available.
STACK_HORIZONS_ESTIMATED = [1, 3, 7, 14, 21]
STACK_EXTRAPOLATION = {"tau_days": 60.0, "floor_hier": 0.35}

# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
N_SIMS_DEFAULT = 69_420
N_SIMS_FALLBACK = 42_069
N_SIMS_BENCHMARK = 1_000
SIM_RUNTIME_LIMIT_SECONDS = 120
RANDOM_SEED = 20261103

# --------------------------------------------------------------------------
# PyMC sampling defaults (small models; increase for publication runs)
# --------------------------------------------------------------------------
MCMC_DRAWS = int(os.environ.get("MCMC_DRAWS", 1000))
MCMC_TUNE = int(os.environ.get("MCMC_TUNE", 1000))
MCMC_CHAINS = int(os.environ.get("MCMC_CHAINS", 4))
MCMC_TARGET_ACCEPT = 0.9


def race_universe():
    """Return a DataFrame of every 2026 race with a stable `race_id`.

    race_id conventions:  H-TX-23  (House),  S-GA  /  S-OH-special (Senate),
    G-AZ (Governor). Built here so every stage keys on the same identifiers.
    """
    import pandas as pd

    rows = []
    for st, n in HOUSE_SEATS_BY_STATE.items():
        rs = REDISTRICTING_STATUS[st]
        for d in range(1, n + 1):
            rows.append({
                "race_id": f"H-{st}-{d:02d}", "office": "House", "state": st,
                "district": d, "special": False,
                "new_map": rs["new_map_for_2026"],
                "redistricting_status": rs["status"],
                "lean_vintage": rs["lean_vintage"],
            })
    for st, cls, special, inc, open_seat, note in SENATE_2026:
        rid = f"S-{st}-special" if special else f"S-{st}"
        rows.append({
            "race_id": rid, "office": "Senate", "state": st, "district": 0,
            "special": special, "incumbent_party": inc, "open_seat": open_seat,
            "note": note, "new_map": False, "redistricting_status": "n/a",
            "lean_vintage": "2022",
        })
    for st, inc, open_seat, note in GOVERNOR_2026:
        rows.append({
            "race_id": f"G-{st}", "office": "Governor", "state": st, "district": 0,
            "special": False, "incumbent_party": inc, "open_seat": open_seat,
            "note": note, "new_map": False, "redistricting_status": "n/a",
            "lean_vintage": "2022",
        })
    df = pd.DataFrame(rows)
    df["district"] = df["district"].astype(int)
    return df


if __name__ == "__main__":
    u = race_universe()
    print(u.groupby("office").size())
    print("Days to election:", DAYS_TO_ELECTION, "as of", FORECAST_ASOF)
    print("States with new maps:", [s for s, v in REDISTRICTING_STATUS.items() if v["new_map_for_2026"]])
