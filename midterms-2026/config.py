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
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:  # imported from a notebook cell
    PROJECT_ROOT = Path.cwd()
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
# Bumped whenever a change adds a function other modules call, so a stale
# copy reports a version mismatch instead of an AttributeError.
CODE_VERSION = "2026.09.24.1"

ELECTION_DATE = date(2026, 11, 3)
CYCLE = 2026
# "As of" date for the forecast. Override with FORECAST_ASOF=YYYY-MM-DD to
# freeze a run (useful for reproducibility and for backtesting snapshots).
FORECAST_ASOF = date.fromisoformat(os.environ.get("FORECAST_ASOF", date.today().isoformat()))
DAYS_TO_ELECTION = max((ELECTION_DATE - FORECAST_ASOF).days, 0)

# Historical midterm cycles used for backtesting (script 14) and calibration
BACKTEST_CYCLES = [2010, 2014, 2018, 2022]
# Cycles whose polling *miss* informs how much correlated polling RISK the
# forecast carries. 2016 and 2020 are included deliberately: a presidential-
# scale miss could happen in a midterm, and the prudent thing is to let that
# possibility widen the intervals rather than assume it away.
POLL_MISS_CYCLES = [2014, 2016, 2018, 2020, 2022]

# Cycles that inform how far the shared polling error is allowed to DRIFT FROM
# ZERO when the model fits it (the prior sd on poll_bias in script 11).
#
# This is a different question from the one above, and answering both with one
# number was a mistake. "How badly could polls miss?" is about risk, and there
# the presidential years belong. "How badly do polls usually miss in a midterm?"
# is about the expected central value, and there they do not, because the
# historical record separates cleanly:
#
#     midterms      2014 +3.92   2018 -0.16   2022 +0.10   mean +1.29  RMS 2.27
#     presidential  2016 +4.29   2020 +6.90                mean +5.59  RMS 5.74
#
# (Positive means polls overstated Democrats. Poll margin minus actual result,
# final three weeks, averaged per race then per cycle.)
#
# The two most recent midterms were essentially unbiased. The mechanism usually
# offered for the 2016 and 2020 misses, low-education and low-trust voters
# under-represented in samples when Trump himself was on the ballot, is specific
# to a presidential ballot. 2026 is a midterm with Trump not on the ballot,
# which is the 2018 and 2022 situation.
#
# Pooling all five cycles gave a drift prior of sd 4.03, wide enough that the
# gap between polls and fundamentals could push poll_bias to +3.08, which
# haircut every polled race by three points. Calibrating the drift prior on the
# right comparison class does not assume the bias away; it just stops the model
# expecting a presidential-year miss in a midterm.
#
# Caveats, because three cycles is thin: 2014 was a real +3.92 midterm miss, so
# midterm bias is not zero, and this narrows rather than eliminates the term.
# Set POLL_BIAS_PRIOR=pooled in the environment to restore the pooled behaviour.
# An environment switch rather than an edit, so a sensitivity run is reproducible
# and nobody has to remember to put the file back afterwards. Note that stage 08
# reads this too, so changing it means re-running 08, not just 11.
_PRIOR_SET = os.environ.get("POLL_BIAS_PRIOR", "midterm")
if _PRIOR_SET not in ("midterm", "pooled"):
    raise ValueError(f"POLL_BIAS_PRIOR must be 'midterm' or 'pooled', got {_PRIOR_SET!r}")
POLL_BIAS_PRIOR_CYCLES = [2014, 2018, 2022] if _PRIOR_SET == "midterm" else POLL_MISS_CYCLES

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
# Races with a significant independent / third-party candidate
# --------------------------------------------------------------------------
# The model is two-party: every race is scored as a margin between the leading
# non-Republican and the Republican. That breaks in three ways this cycle, and
# Hummel & Rothschild (2014) drop races where a third candidate clears 10% for
# exactly this reason, so these need to be handled explicitly rather than
# forced into a D-vs-R frame.
#
#   has_democrat=False : no Democrat on the ballot, so the "Democratic" column
#                        holds the independent. Treating the seat as Safe R
#                        because the Democratic share is zero would be wrong.
#   three_way=True     : Democrat AND independent both running; the anti-R vote
#                        may split, which a two-party margin cannot express.
#   caucus_prob_dem    : if the independent wins, the probability they caucus
#                        with the Democrats. This decides chamber control and is
#                        a JUDGEMENT, not an estimate. 0.5 is a deliberate
#                        "genuinely unknown". Set it per race from what the
#                        candidate has actually said.
INDEPENDENT_RACES_2026 = {
    "S-NE": {"has_democrat": False, "three_way": False, "caucus_prob_dem": 0.90,
             "independent": "Dan Osborn",
             "note": "Osborn (I) vs Ricketts (R); no Democrat. 0.90 rather than 1.0 because "
                     "Osborn has publicly said he would caucus with neither party, so a small "
                     "amount of doubt is warranted even on a confident expectation."},
    "S-ID": {"has_democrat": False, "three_way": False, "caucus_prob_dem": 0.5,
             "independent": "Achilles",
             "note": "Achilles (I) vs Risch (R); no Democrat. Caucus intent unknown, so 0.5 "
                     "is a deliberate coin flip rather than a guess dressed as an estimate."},
    "S-SD": {"has_democrat": False, "three_way": False, "caucus_prob_dem": 0.5,
             "independent": "Beng",
             "note": "Beng (I) vs Rounds (R); no Democrat. Caucus intent unknown; 0.5."},
    "S-MT": {"has_democrat": True, "three_way": True, "caucus_prob_dem": 0.5,
             "independent": "", "note": "Three-way: Daines (R) vs a Democrat vs an independent."},
}
# How much a three-way race widens that seat's uncertainty (pct points, added
# in quadrature). A split anti-incumbent vote is genuinely less predictable.
THREE_WAY_EXTRA_SD = 6.0

# How far an independent runs ahead of where a generic Democrat would, in
# points. The partisan-lean fundamentals describe a generic Democrat, so
# without this an independent in a deep-red state is modelled as a Democrat
# losing badly, which is the wrong candidate.
#
# Nebraska is measurable rather than assumed. In 2024 the state ran two Senate
# races on the same day: the independent lost by 6.7 while the Democrat in the
# concurrent special lost by 25.2, an 18.5-point gap in a same-state,
# same-electorate comparison. That is a clean natural experiment but a single
# observation, and it flatters the independent (a well-funded challenger
# against a complacent incumbent). The default shrinks it by half toward zero.
# Idaho and South Dakota get 0: their independents have no comparable record,
# and inventing a bonus for them would be fabrication.
INDEPENDENT_BONUS_PTS = {
    "S-NE": 9.0,     # half of the measured 18.5; raise toward 18.5 if warranted
    "S-ID": 0.0,
    "S-SD": 0.0,
}

# --------------------------------------------------------------------------
# Candidate experience (Hummel & Rothschild 2014, Table 2)
# --------------------------------------------------------------------------
# Points a candidate's prior office is worth, relative to a candidate with no
# political experience. These are the paper's published estimates, applied as a
# prior-driven offset rather than fitted here: the 2018-2022 training set has
# no experience coding, so there is nothing in it to estimate them from. The
# model uses the DIFFERENCE between the two candidates, and incumbency stays a
# separate fitted term so the two do not double-count.
EXPERIENCE_POINTS_SENATE = {
    "governor": 10.8,       # most valuable experience for a Senate candidate
    "senator": 10.8,        # former senator seeking to return
    "us_house": 5.5,        # paper: statewide office is worth about the same
    "statewide": 5.5,       # AG, secretary of state, treasurer
    "lt_governor": 5.0,
    "local": 5.0,           # "nearly as beneficial as having served in the House"
    "state_legislator": 4.0,  # least valuable elected experience
    "none": 0.0,            # no elected office (party chair, first-time candidate)
    "unknown": None,        # candidate not identified yet: contributes NO edge.
                            # Distinct from "none": an unfilled row must not
                            # quietly penalise a candidate we simply have not
                            # looked up.
}
EXPERIENCE_POINTS_GOVERNOR = dict(EXPERIENCE_POINTS_SENATE, senator=9.4, governor=9.4)
# Scale applied to the experience difference. 1.0 takes the paper at face
# value; lower it to shrink toward the fitted incumbency term.
EXPERIENCE_WEIGHT = 1.0

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

# For a district on a brand-new map with no per-district partisan lean, the
# "fundamentals" estimate is just its STATE's lean, which carries no
# district-level information at all: every California seat, safe and marginal
# alike, comes out at D+25.7. The blend nevertheless gave the fundamentals 0.91
# of the weight against 0.02 for the race rating, so California's 22nd, which
# every rater calls a Toss-up, was forecast at D+26.9 and 99% Democratic.
#
# That ordering is backwards for these races specifically. The rating is the
# only input that knows where the new lines fall, so this moves most of the
# fundamentals weight to the rating wherever the lean is a state fallback AND a
# rating exists. It is a mitigation, not a fix: the real repair is a
# per-district lean for the new maps, from the workbook's Races.pvi column or
# pvi_manual.csv, after which no race takes this path at all.
WEAK_LEAN_FUND_TO_RATING = float(os.environ.get("WEAK_LEAN_FUND_TO_RATING", 0.75))

# Stacking-weight horizon curve. FiveThirtyEight's poll archive holds only the
# final three weeks before each election, so weights are *estimated* for
# horizons <= 21 days and *extrapolated* beyond that with an exponential decay
# toward a fundamentals-heavy floor:  w(h) = floor + (w21 - floor) * exp(-(h-21)/tau).
# Replace with estimated values once a long-horizon poll archive is available.
STACK_HORIZONS_ESTIMATED = [1, 3, 7, 14, 21]
# --------------------------------------------------------------------------
# Poll de-duplication
# --------------------------------------------------------------------------
# One survey should contribute one row. The aggregator weights by inverse
# variance and treats every row as fresh evidence, so a survey present twice
# gets double weight and halves the uncertainty it ought to leave alone.
#
# Three distinct things look like duplicates and only two of them are:
#
# 1. The SAME SURVEY, TWO QUESTIONS. Glengariff's Michigan governor poll of
#    2026-01-06 appears as 32/34 and as 47/45, same pollster, same date, same
#    600 respondents. Those are a multi-candidate or undecided-heavy question
#    and the clean head-to-head from one survey, not two surveys. Keeping both
#    counts 600 people twice, so one row is kept: the one whose two-party total
#    is highest, which is the base head-to-head because it has the fewest
#    respondents parked in "other" or "undecided".
#
# 2. The SAME SURVEY, TWO SOURCES, DATES ONE DAY APART. Cygnal's generic-ballot
#    poll is in the workbook ending 2026-05-06 and in the Wikipedia scrape
#    ending 2026-05-07, both n=1500, both D+7. One poll, two spellings of its
#    field period. A date window catches it.
#
# 3. GENUINELY SEPARATE POLLS CLOSE TOGETHER. Trafalgar polled New Hampshire's
#    Senate race ending 09-25 (n=1090, D+4) and 09-27 (n=1082, D+6). Two polls.
#    Collapsing them would discard real evidence, which is why the date window
#    alone is not enough: the margins and sample sizes have to agree too.
#
# So the fuzzy rule fires only when the dates are within the window AND the
# margins agree within the tolerance AND the sample sizes are equal or one is
# missing. That keeps case 2 and spares case 3.
POLL_DEDUP_DATE_WINDOW_DAYS = 2
POLL_DEDUP_MARGIN_TOL = 1.0
# Every dropped row is written here with the row it lost to and the reason, so
# a destructive step is auditable rather than taken on trust.
POLL_DEDUP_AUDIT = True

# --------------------------------------------------------------------------
# Poll recency
# --------------------------------------------------------------------------
# A poll's weight in the Kalman filter is its inverse observation variance, so
# "an old poll should count less" is implemented by inflating that variance with
# the poll's age. The weight below is multiplicative on the weight, i.e. the
# variance is divided by it.
#
# Why this is needed on top of the filter. The filter already discounts old
# polls through the latent random walk: the state has drifted since an old
# observation, so it pins today's value less tightly. But the drift is scaled at
# RANDOM_WALK_SD_PER_DAY["race"] = 0.18 points per day, and a random walk
# accumulates as the square root of time, so that implies a race margin moves
# only 0.18 * sqrt(243) = 2.8 points over eight months. Real race margins move
# far more than that, so the filter was letting stale polls hold their ground.
# H-WA-04 was the symptom: one poll from 5 February 2026, showing D+18 in a
# district at R+24.6, still carried 48% of the weight in October.
#
# Exponential, not a step. A cliff at 90 days would give a poll at 89 days three
# times the influence of one at 91, make a forecast jump as the calendar
# advances rather than as evidence arrives, and treat a 10-day-old poll and an
# 89-day-old one as equals. The decay is calibrated to pass through the chosen
# anchor exactly: tau = REF_DAYS / ln(1 / WEIGHT_AT_REF).
#
# At the default anchor (1/3 of full weight at 90 days, tau = 81.98 days):
#     14 days  0.84      90 days  0.33
#     30 days  0.69     180 days  0.11
#     60 days  0.48     243 days  0.05
#
# The floor stops the variance multiplier running away on a very old poll and
# keeps one from being discarded outright, which matters for a race whose only
# poll is old: better a 2%-weight poll than none.
# Environment-overridable so a sensitivity run needs no edit, the same way
# POLL_BIAS_MODE works. POLL_AGE_WEIGHT_AT_REF=0.999 effectively turns the decay
# off, which is the way to produce a before/after comparison:
#     POLL_AGE_WEIGHT_AT_REF=0.999 python3 run_pipeline.py 08 09 10 11 12 13 14
POLL_AGE_REF_DAYS = float(os.environ.get("POLL_AGE_REF_DAYS", 90.0))
POLL_AGE_WEIGHT_AT_REF = float(os.environ.get("POLL_AGE_WEIGHT_AT_REF", 1.0 / 3.0))
POLL_AGE_MIN_WEIGHT = float(os.environ.get("POLL_AGE_MIN_WEIGHT", 0.02))
if not (0.0 < POLL_AGE_WEIGHT_AT_REF < 1.0):
    raise ValueError(f"POLL_AGE_WEIGHT_AT_REF must be in (0, 1), got {POLL_AGE_WEIGHT_AT_REF}")
# House effects are estimated from the FULL weight of every poll, not the aged
# weight. The decay answers "how much does this poll tell me about the race
# today", which decays; a pollster's lean is a property of the pollster and an
# old poll is just as informative about that, so ageing it would throw away the
# evidence that makes the house-effect estimate usable at all.
POLL_AGE_APPLIES_TO_HOUSE_EFFECTS = False

STACK_EXTRAPOLATION = {"tau_days": 60.0, "floor_hier": 0.35}

# Minimum historical races per FITTED WEIGHT before a horizon's stack is
# trusted. A three-component stack therefore needs 3x this many races.
#
# Why this exists. The public poll archive thins out fast as the horizon grows:
# at 21 days only 65 rated polled races remain, against 238 at 14 days. A
# three-weight simplex fitted on 65 races lands on the boundary, and because the
# forecast horizon (28 days) is beyond the archive, the extrapolation anchors on
# exactly that noisiest point. The symptom was a weight of 0.171 at 21 days
# against 0.061 at 14, and a 2-component w_hier of exactly 1.000 on 82 races
# where 364 races gave 0.982. Neither is a finding about the world.
#
# With this rule a horizon that cannot support its stack is dropped from the
# curve and the extrapolation anchors on the last horizon that can. 50 is not
# a tuned number: it is the order of magnitude at which a weight on the unit
# simplex stops being dominated by its boundary, and the file previously used a
# flat 40 for a 3-way stack, which is the same idea applied too loosely.
STACK_MIN_ROWS_PER_COMPONENT = 50

# Where the fund:rating split on polled races comes from.
#
# "stack" uses the three-way polled_rated fit directly, which is the internally
# consistent choice. The alternative, and what the code used to do, was to take
# the MAGNITUDE of the non-poll slice from the two-way polled_all_cycles stack
# and only the RATIO from the three-way one. That ratio is computed as
# r/(f+r), and the three-way fit puts the fundamentals at exactly 0.000 on
# polled races at every horizon, so the ratio evaluates to exactly 1.0 with no
# information about how well determined it is. Before archived ratings existed
# the rating weight was the one pinned at 0.000 and the same expression returned
# exactly 0.0. A 7% slice of weight therefore flipped wholesale from one
# component to the other, moving the Senate probability by 1.7 points, on the
# strength of which of two boundary-clipped weights happened to be the zero.
#
# "ratio_shrunk" keeps the old two-stack construction but shrinks the ratio
# toward an even split, as a Beta-style prior on a proportion, with the weight
# below. Use it to reproduce the old behaviour's shape without its brittleness.
POLLED_RATING_SHARE_MODE = "stack"      # "stack" | "ratio_shrunk"
POLLED_RATING_SHARE_PRIOR_ROWS = 100    # only read when mode is "ratio_shrunk"

# Project a rating's margin onto its tier's empirical band instead of using the
# tier mean.
#
# A "Safe" rating historically spans +14 to +64 points at the 10th and 90th
# percentiles, with a mean of +35.6. Feeding +35.6 into the blend as a point
# estimate for a specific safe race is the error: the blend is a mixture, so its
# variance carries a (component - mean)^2 term, and mixing a 35-point constant
# with a tight race estimate manufactures uncertainty that is not there. It put
# the Oklahoma governor's race at a 26% Democratic chance with a 18.3-point band,
# and Colorado at 69%.
#
# What a rating actually says is "this race is somewhere in this band", not
# "this race is at the band's mean". So the component's mean is projected onto
# the band: if the other components already place the race inside it, the rating
# is redundant and contributes nothing; if they place it outside, the rating
# shifts the estimate to the nearest edge of its band, which preserves genuine
# disagreement (polls at R+2 against a Safe R rating) without overstating it.
RATING_BAND_PROJECTION = True
RATING_BAND_QUANTILES = (10.0, 90.0)

# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
N_SIMS_DEFAULT = 69_420
N_SIMS_FALLBACK = 42_069
N_SIMS_BENCHMARK = 1_000
SIM_RUNTIME_LIMIT_SECONDS = 120
RANDOM_SEED = 20261103

# --------------------------------------------------------------------------
# Hand-maintained polling workbook (MidtermPolls2026.xlsx)
# --------------------------------------------------------------------------
# When this workbook is found it becomes THE poll source and the scrapers are
# skipped, rather than being averaged in alongside them. Two reasons, both
# about not undoing work a person has already done:
#
#   Its include_in_model column records a human decision that a scraper cannot
#   make, separating live general-election matchups from pre-primary ballot
#   tests, withdrawn candidates and superseded duplicates. Re-scraping would
#   put the excluded rows back.
#
#   The workbook was built partly from the same Wikipedia articles the scraper
#   reads. The same poll carries a different identifier in each source, so it
#   would survive de-duplication twice and be counted twice.
#
# Set MIDTERM_WORKBOOK to point a run at a different copy without editing this
# file. Paths are tried in the order workbook.candidate_paths() lists them:
# the environment variable, then POLL_WORKBOOK, then the repository-local copy,
# then WORKBOOK_EXTRA_PATHS.
POLL_WORKBOOK = os.environ.get(
    "MIDTERM_WORKBOOK",
    r"C:\Users\slima\OneDrive\Election Model Project\Election Model Project"
    r"\Midterm Polls Toolkit\MidtermPolls2026.xlsx")

# Further places to look. The repository-local copy at
# data_store/manual/MidtermPolls2026.xlsx is always tried and is what makes the
# same code run in Colab, in a container and on a second machine, none of which
# have the OneDrive path above.
WORKBOOK_EXTRA_PATHS = [
    Path.home() / "OneDrive" / "Election Model Project" / "Election Model Project"
    / "Midterm Polls Toolkit" / "MidtermPolls2026.xlsx",
    Path.home() / "Downloads" / "MidtermPolls2026.xlsx",
]

# The workbook marks additional versions of an already-included release as
# "Review" (a different population, leaners pushed, or reworded question).
# Including them counts the same release more than once, and the aggregator
# treats each row as independent evidence, so the default is to take only the
# primary row. Turn this on to inspect the effect of the alternates.
WORKBOOK_INCLUDE_REVIEW = os.environ.get("WORKBOOK_INCLUDE_REVIEW", "0") not in ("0", "", "false", "False")

# Whether the workbook may also supply presidential approval (stage 04) and
# expert ratings (stage 05). Both live in the same file, and both are otherwise
# weak spots: approval is scraped, and ratings fall back to partisan lean for
# every race a hand-written CSV does not cover.
# Let the Wikipedia scraper add polls that field-closed AFTER the newest poll
# the workbook holds for that race, leaving everything else to the workbook.
# The workbook is updated by hand once a week while polls drop daily, so without
# this the model is always a few days stale in exactly the window a "how are the
# races trending" question is about. The per-race cutoff is what makes it safe:
# every row the workbook deliberately excluded is older than its own newest
# included row, so none can be restored, and no poll is counted from both
# sources. Set to False to use the workbook alone.
WORKBOOK_TOPUP_SCRAPE = os.environ.get("WORKBOOK_TOPUP_SCRAPE", "1") not in ("0", "", "false", "False")

# Which races the top-up scrape covers: "watchlist" (the watchlist plus the
# generic ballot, about a dozen articles) or "all" (116 articles). Wikipedia
# serves these 1-4MB pages slowly enough that a full crawl can exceed an hour,
# so the default covers the races the forecast turns on and leaves the rest to
# the weekly workbook refresh. Articles are cached either way.
WORKBOOK_TOPUP_RACES = os.environ.get("WORKBOOK_TOPUP_RACES", "watchlist")

WORKBOOK_FEEDS_APPROVAL = True
WORKBOOK_FEEDS_RATINGS = True

# --------------------------------------------------------------------------
# PyMC sampling defaults (small models; increase for publication runs)
# --------------------------------------------------------------------------
MCMC_DRAWS = int(os.environ.get("MCMC_DRAWS", 1000))
MCMC_TUNE = int(os.environ.get("MCMC_TUNE", 1000))
# 4 chains is the standard minimum for trustworthy convergence diagnostics:
# R-hat and effective sample size are both between-chain statistics, so with
# 2 chains they have almost no power to detect a chain stuck in the wrong
# part of the posterior. 4 chains is the Vehtari et al. (2021) recommendation
# and what Stan and PyMC default to. Set MCMC_CHAINS=2 to trade diagnostic
# power for roughly half the sampling time.
MCMC_CHAINS = int(os.environ.get("MCMC_CHAINS", 4))
# Number of worker PROCESSES PyMC may use for parallel chains.
# Default 1 (sequential) on purpose. Each stage already runs inside a
# subprocess whose stdout is a pipe; letting PyMC spawn its own workers on top
# of that deadlocks on Windows, where multiprocessing uses spawn and the
# workers inherit the pipe. Sequential sampling costs roughly 2x wall time for
# each chain and removes that failure mode entirely. Set MCMC_CORES=4 to opt
# in on Linux or macOS, where it is safe and roughly 4x faster.
MCMC_CORES = int(os.environ.get("MCMC_CORES", 1))
MCMC_TARGET_ACCEPT = 0.9

# --------------------------------------------------------------------------
# How the shared polling error is treated in the hierarchical fit
# --------------------------------------------------------------------------
# This is the most consequential modelling choice in the project, worth roughly
# 35 points of Senate win probability, so it is an explicit switch.
#
# `poll_bias` is the shared polling error: one number added to every poll
# observation, representing polls being collectively wrong in one direction.
# Its job is not only to shift the polls. It is what makes the polls
# CORRELATED with each other, and that second role turns out to matter more.
#
#   "estimated"  poll_bias ~ N(0, poll_bias_prior_sd), free, prior centred on
#                zero. Not the default; see the note under "symmetric".
#                Because the polls share a term, 58 statewide polls count as
#                strong but not independent evidence about the national
#                environment. On 2026 data the posterior lands near +3.0,
#                meaning the model reads polled races as about 3 points less
#                Democratic than their polls. The historical record agrees
#                independently: final-three-week polls overstated Democrats in
#                4 of the last 5 cycles (+3.9, +4.3, -0.2, +6.9, +0.1),
#                averaging +2.97.
#                The honest caveat is that this estimate is confounded. The
#                fundamentals are badly wrong in exactly the races that get
#                polled, for reasons unrelated to polling: they cannot see the
#                independent candidates in South Dakota (+17.5) or Idaho
#                (+12.3), Alaska's politics (+12.8) or Kansas (+14.6), and they
#                miss in both directions (Vermont -20.2, Massachusetts -17.2).
#                Some of the +3.0 is therefore fundamentals error wearing a
#                polling-error label. Treat the Senate number as conditional on
#                this parameter, and read the sensitivity below before quoting
#                it.
#
#   "symmetric"  poll_bias pinned to 0 in the fit, with the correlated polling
#                error entering only as a zero-mean shared shock in the
#                simulation. THIS IS THE DEFAULT, chosen deliberately by the
#                project owner: it makes no claim about the direction of a
#                polling miss, which is the position modellib's own
#                historical_polling_error documents ("the sign of the miss is
#                unknowable in advance, so we do not centre it"), and it is
#                closest to how the published forecasters treat the question.
#
#                Its cost is real and must be quoted with its output. Removing
#                the parameter also removes the CORRELATION between polls, so
#                the fit treats 58 correlated statewide polls as 58 independent
#                readings of the national environment and over-determines it.
#                Measured: nat_dev posterior sd falls to 1.96 here against 2.20
#                under "estimated", i.e. pinning a parameter made the model MORE
#                confident, and pushed national uncertainty below the 4.03-point
#                polling miss the model itself estimates. The visible symptom is
#                a House probability that prints as 100%.
#
#                That 100% is a statement about the model, not about the world.
#                No forecast 36 days out should be read as certainty. Treat it
#                as ">99% under this specification" and note that the same
#                over-determination makes the Senate and Governor figures more
#                confident than the evidence strictly supports.
#
# Sensitivity, 28 days out, same polls in all three (reproduce with
# POLL_BIAS_MODE=estimated POLL_BIAS_PRIOR=pooled python3 run_pipeline.py 08 11 12 13):
#                                     House   Senate   Governor   poll_bias     nat_dev sd
#   symmetric (default)                98%      67%       62%      0 (fixed)        1.10
#   estimated, drift prior 2.27        97%      62%       52%      +1.05 +/- 1.60    2.14
#   estimated, drift prior 4.03        90%      53%       45%      +1.58 +/- 1.97    2.50
#   median D seats, in the same order: 239/52/27, 234/51/26, 231/51/25
#
# The last two differ only in POLL_BIAS_PRIOR_CYCLES.
#
# THE SPECIFICATIONS HAVE CONVERGED as polls accumulated, and that is the main
# thing to know about this parameter now. At 36 days out the Senate ran 62% /
# 38% / 24% across these three rows, a 38-point spread that made the assumption
# the dominant fact about the forecast. It is now 67% / 62% / 53%, a 14-point
# spread. The cause is visible in the fitted poll_bias: it has fallen from +1.96
# to +1.05 under the midterm prior and from +3.08 to +1.58 under the pooled one.
# That term is inferred from the gap between polls and fundamentals, and 1,000-odd
# polls have closed most of that gap. The Senate is a modest Democratic favourite
# under every specification, which is a defensible thing to publish; it was not
# in September.
#
# Where the assumption now bites hardest is the GOVERNORS, not the Senate: 62%
# under symmetric against 45% under the pooled prior, which crosses from
# favourite to underdog. Quote that one with the assumption attached.
#
# The over-determination critique still stands and is visible in the last
# column: symmetric puts national uncertainty at sd 1.10, against 2.14 and 2.50
# for the estimated runs and a 4.03-point polling miss the model itself measures.
# Read the House number as ">95% under this specification" rather than as a
# precise figure. It is a property of the specification, not a symptom of thin
# House data, so more district polling will not fix it.
#
# The backtest does not settle the choice between these: its calibration table is
# unchanged across all three, because it scores the blend components at historical
# cycles and never re-fits this cycle's poll_bias. The case for the midterm drift
# prior is the reference-class argument in POLL_BIAS_PRIOR_CYCLES, not backtest
# evidence, and it should be quoted that way.
#
# Neither is close to a published forecaster's Senate number, and this
# parameter is why. Anyone quoting the Senate figure should quote the
# assumption with it.
POLL_BIAS_MODE = os.environ.get("POLL_BIAS_MODE", "symmetric")
if POLL_BIAS_MODE not in ("estimated", "symmetric"):
    raise ValueError(f"POLL_BIAS_MODE must be 'estimated' or 'symmetric', got {POLL_BIAS_MODE!r}")


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
