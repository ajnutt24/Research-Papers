# 2026 Midterm Election Forecasting Model

A Bayesian hierarchical forecast and poll aggregator for the November 3, 2026
midterms: all 435 House races, the 35 Senate races on the ballot (33 Class II
seats plus the Ohio and Florida specials) and the 36 governorships. Polls,
economic and political fundamentals and expert race ratings are blended with
weights learned by leave-one-out stacking on the 2010, 2014, 2018 and 2022
midterms, then a vectorised Monte Carlo simulation produces seat-count
distributions and chamber-control probabilities.

Every margin in the project is **Democratic minus Republican, in percentage
points**.

## Data provenance: read this first

Every stage output carries a `provenance` column and a `.meta.json` sidecar:

| value | meaning |
|---|---|
| `live` | fetched from the primary source during this run |
| `cache` | re-used from `data_store/raw/` (younger than `CACHE_MAX_AGE_HOURS`) |
| `manual` | read from a hand-maintained file in `data_store/manual/` |
| `mirror` | public GitHub mirror of an official dataset (FiveThirtyEight's election-results and poll archives, `datasets/cpi-us`, `unitedstates/congress-legislators`) |
| `fixture` | **simulated placeholder** so the pipeline can run; never publish |

Downstream stages inherit the worst provenance they consumed, so
`outputs/chamber_control.json` says honestly whether the numbers rest on real
polls. The environment this was built in blocked every non-GitHub host
(FRED, AAA, the FEC, NYT, RealClearPolitics, Cook, Sabato, Inside Elections,
FiftyPlusOne), so the committed `data_store/processed/` outputs for the
2026 polls, approval, ratings, fundraising, unemployment and gas prices are
fixtures. The historical inputs (results, polls, partisan leans, CPI) are
real. **Re-run scripts 01-06 from a machine with network access, or fill in
the manual CSVs, before reading anything into the 2026 numbers.**

## Pipeline order and data flow

Run from the project directory (`midterms-2026/`). Each script reads the
cached parquet outputs of the stages it depends on, so you can re-run any one
stage without re-running the others.

```
config.py  -----------------------------------------------------------------.
                                                                            |
data/01_fetch_economic_data.py   -> economic_monthly, economic_cycle_features
data/02_fetch_gas_prices.py      -> gas_prices                     (reads 01)
data/03_fetch_polls.py           -> polls_2026
data/04_fetch_approval.py        -> approval_2026, approval_history
data/05_fetch_race_ratings.py    -> race_ratings_2026, race_ratings_historical
data/07_fetch_historical_results.py -> historical_results, historical_national
data/06_fetch_fundamentals.py    -> fundamentals_2026              (reads 07)

model/08_poll_aggregation.py     -> poll_estimates_2026, generic_ballot_trend,
                                    national_environment, polling_error_history,
                                    house_effects_2026                (reads 03, 06)
model/09_fundamentals_model.py   -> fundamentals_national, fundamentals_estimates_2026,
                                    national_model_idata.nc, seat_model_idata.nc
                                                                      (reads 01, 04, 06, 07)
model/10_rating_to_margin_calibration.py -> rating_calibration, ratings_estimates_2026
                                                                      (reads 05, 07)
model/11_hierarchical_model.py   -> hierarchical_estimates_2026, hier_draws.npz
                                                                      (reads 06, 08, 09)
model/12_blend_stacking.py       -> stacking_weights, blend_2026      (reads 08-11 + history)
simulation/13_monte_carlo.py     -> outputs/race_probabilities.csv,
                                    seat_distribution_*.csv, chamber_control.json
                                                                      (reads 11, 12)
validation/14_backtest.py        -> outputs/backtest_*.csv            (reads history, 10, 08 meta)
```

Note that 07 runs before 06: the lagged-result fundamental needs the
historical results table. `model/modellib.py` holds the numerical code that
08, 09, 12 and 14 share (Kalman filter, house effects, fundamentals fits,
stacking) so the backtest and the live forecast use identical code paths.

One-shot run:

```bash
cd midterms-2026
pip install pandas numpy scipy pyarrow requests beautifulsoup4 lxml pyyaml matplotlib pymc arviz
export FRED_API_KEY=...   FEC_API_KEY=...        # optional but recommended
for s in data/01 data/02 data/03 data/04 data/05 data/07 data/06 \
         model/08 model/09 model/10 model/11 model/12 simulation/13 validation/14; do
  python3 ${s}_*.py || break
done
```

Useful environment variables: `FORECAST_ASOF=YYYY-MM-DD` freezes the as-of
date; `MCMC_DRAWS`, `MCMC_TUNE`, `MCMC_CHAINS` control PyMC; `NYT_POLLS_URL`
and `APPROVAL_URL` point the poll and approval fetchers at a CSV endpoint.
`python3 model/12_blend_stacking.py --refit` recomputes the stacking weights
(a few minutes); otherwise the cached table is reused.

## What each stage does, briefly

* **01 Economic data.** CPI is the primary economic regressor (year-over-year
  inflation in October of the election year). Unemployment is fetched and its
  correlation with CPI across the 19 historical midterms is logged
  (`-0.22` on the current data) but it is not used as a second regressor by
  default. Falls back from the FRED API to the keyless FRED CSV to a GitHub
  CPI mirror.
* **02 Gas prices.** AAA daily national average scraped politely (robots.txt
  check, 2-second floor, one fetch per 12 hours, appended to a local log);
  EIA's GASREGW is the public-domain series of record and a 15-cent
  disagreement flags a bad parse. Gas is stored as an optional salience
  covariate; CPI already contains motor fuel.
* **03 Polls.** NYT CSV endpoint (URL via env var), RealClearPolitics tables,
  or a manual CSV; hypothetical matchups are dropped (one D, one R, names
  matching the nominee file when present). Pollster, sponsor, field dates,
  sample size, population type and partisan flag are kept.
* **04 Approval.** Tracker CSV or manual file. Also seeds
  `approval_history.csv` (Gallup final pre-midterm approval 1946-2022, the
  President's party, and a 0-1 war-salience index) with `verify=True`.
* **05 Race ratings.** Cook, Sabato and Inside Elections scraped with a
  tolerant parser, else a manual file, else a lean-derived fixture. Consensus
  is the median tier. Historical ratings default to FiveThirtyEight's 2018
  category file (506 races with outcomes); add Cook archives for more cycles.
* **06 Fundamentals inputs.** Partisan lean (Daily Kos PVI for maps in effect
  in November via `pvi_manual.csv`, else FiveThirtyEight's 2022 lean; new-map
  districts without a manual PVI get their state lean and a weak-prior flag),
  lagged same-seat result, incumbency (current members from
  `congress-legislators` plus `incumbency_overrides_2026.csv` for
  retirements and primary losses), FEC individual-contribution share.
* **07 Historical results.** MIT Election Lab files if you drop them into
  `data_store/manual/`, otherwise FiveThirtyEight's results mirror
  (House/Senate/Governor 1998-2024, fusion ballot lines merged per candidate).
* **08 Poll aggregation.** House effects by penalised backfitting shrunk to
  each pollster's historical bias; a local-level Kalman filter per race
  (recency and sample size weighting fall out of the filter); the estimate is
  projected to Election Day. National environment from the generic-ballot
  filter and from the seat-implied swing. The correlated polling-error prior
  is the RMS of cycle-level poll bias in 2014-2022 (4.0 points on the archive,
  driven by 2020's 6.9-point and 2014/2016's 4-point misses).
* **09 Fundamentals model.** PyMC "referendum" model of the national House
  vote on 20 midterms with Abramowitz/Hibbs-style priors, a post-1994
  realignment term, and a war-salience coefficient whose prior is centred on
  zero so the data set its sign. A seat-level model with office-specific
  coefficients (national swing, lean, incumbency, fundraising) trained on
  2018-2022.
* **10 Rating calibration.** Beta-binomial win rates and shrunk margins per
  tier from historical ratings versus results, with sd floors (8-12 points) so
  a qualitative rating never becomes a hairline estimate.
* **11 Hierarchical model.** PyMC: national deviation and polling bias as two
  separate shared terms, state random effects, race offsets scaled by the
  fundamentals residual sd and widened by 1.75x on newly drawn maps. Polled
  races update the posterior; unpolled races inherit their state's swing.
* **12 Blend.** ArviZ LOO stacking over backtest cycles at horizons 1-21
  days; extrapolated beyond 21 days (see limitations). The blend is a mixture,
  so between-component disagreement widens the band.
* **13 Monte Carlo.** One national error and one polling error per draw
  (through the hierarchical posterior draw), state shocks, mixture component
  choice, idiosyncratic noise; numpy-vectorised in chunks; benchmarked on
  1,000 draws and falls back from 69,420 to 42,069 runs if the projection
  exceeds two minutes, logging which count was used and why.
* **14 Backtest.** Election-eve rebuild of every component for 2010, 2014,
  2018 and 2022 with leave-one-cycle-out stacking weights; Brier score, log
  loss, accuracy, expected-vs-actual seats, a 10-bin calibration table and the
  explicit "do 60-80% races win 60-80% of the time" check.

## Redistricting status (NOT final)

`config.REDISTRICTING_STATUS` is a maintained table with one row per state
(`census_map`, `new_map`, `litigation`). It drives which lean vintage a
district uses and how much extra shrinkage the hierarchical model applies.
As committed it marks Texas, California, North Carolina, Ohio and Utah as on
new maps for 2026 and Florida, Virginia and Louisiana as under litigation.

**Missouri is a special case.** The August 2026 primary was run under the 2025
map, which the Missouri Supreme Court has since blocked for the general
election, so the 2020-census map applies in November and the model uses the
2022 partisan leans for MO-01 to MO-08. Nominees must be re-mapped by hand
from their primary district to their general-election district in
`data_store/manual/missouri_candidate_map.csv`. Missouri's ruling is under
appeal to the U.S. Supreme Court, and Virginia and Florida have unresolved
litigation, so this table should be re-verified before every published run
and must not be treated as final.

## Known limitations and what to fix first

1. **Fixture inputs.** See the provenance section. The 2026 poll, approval,
   rating and FEC pulls could not be exercised against live endpoints from
   the build environment; the parsers are written defensively but untested
   against current markup. The manual CSV path is the reliable fallback.
2. **Long-horizon stacking weights are extrapolated.** The public poll
   archive covers the final 21 days of each cycle only. Two months out, the
   polls-vs-fundamentals split comes from `config.STACK_EXTRAPOLATION`, not
   from data. Supplying a long-horizon archive (or accumulating this cycle's
   polls) lets `12` estimate the full curve.
3. **Ratings are calibrated on one cycle (2018)** and in-sample in the
   backtest. Add archived Cook/Sabato/Inside ratings for 2010, 2014 and 2022.
4. **2010/2014 leans are lagged-result proxies**; FiveThirtyEight leans start
   in 2018. MIT Election Lab data would extend the seat model's training set.
5. **Incumbency overrides must be maintained.** Without
   `incumbency_overrides_2026.csv` every current member is assumed to be
   running; retirements and primary losses are missing from the committed
   fixture-era run.
6. **Historical national-conditions table** (`approval_history.csv`,
   `historical_house_vote.csv`) is seeded from published figures with
   `verify=True`; check them.
7. **War salience** for 2026 is a placeholder (0.5) until
   `war_salience_2026.csv` carries a measured series (e.g. news-attention
   volume, weeks since escalation).

## Backtest results (election-eve, leave-one-cycle-out weights)

From `outputs/backtest_scores.csv` on the committed run (real historical
inputs; the backtest does not use any fixture):

| cycle | races | Brier | log loss | accuracy | expected D seats (all offices) | actual |
|---|---|---|---|---|---|---|
| 2010 | 421 | 0.056 | 0.178 | 93.1% | 188 | 178 |
| 2014 | 392 | 0.033 | 0.109 | 95.7% | 163 | 157 |
| 2018 | 457 | 0.046 | 0.154 | 93.4% | 225 | 234 |
| 2022 | 464 | 0.060 | 0.189 | 90.3% | 194 | 232 |

A coin flip scores Brier 0.25. The blend beats fundamentals alone in every
cycle and office (e.g. 2022 Senate: 0.043 vs 0.110) and matches the pooled
polls-plus-fundamentals component, which stacking gives about 98% of the
weight in the final three weeks. Calibration: races given 60-80% for the
Democrat (mean 71%) went Democratic 82% of the time (94 races); on the
favourite's side the 60-80% bucket (mean 70%, 192 races) saw the favourite
win 74%. The model is mildly under-confident in the middle bins and its
biggest miss is 2022, where approval of 40 and 7.7% inflation made the
fundamentals predict a Republican wave that did not arrive; that is the
historical-fundamentals error the national shock term is there to represent.

Two stacking findings worth knowing: (1) the rating component receives
roughly zero weight against the pooled estimate, but the only historical
ratings available here are FiveThirtyEight's 2018 categories, which are
themselves model output; archived Cook/Sabato/Inside ratings may change that;
(2) for unpolled races the plain fundamentals beat the state-pooled estimate,
so the state random effect mostly matters through the PyMC model's shared
terms rather than as a point-estimate shift.

## Outputs

`outputs/chamber_control.json`, `outputs/race_probabilities.csv`,
`outputs/seat_distribution_{house,senate,governor}.csv`,
`outputs/backtest_scores.csv`, `outputs/backtest_calibration.csv`, and the
figures in `outputs/figures/` (generic-ballot trend, rating calibration,
stacking weights, seat distributions, backtest reliability).
