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

## Quick start

### Google Colab (easiest, nothing to install)

Colab runs Linux with a C compiler, so it avoids both the slow-backend and
the Windows multiprocessing problems. Open a new notebook at
[colab.research.google.com](https://colab.research.google.com) and paste this
into the first cell:

```python
import urllib.request
exec(urllib.request.urlopen(
    "https://raw.githubusercontent.com/ajnutt24/research-papers/"
    "claude/2026-midterms-forecast-model-jj4m98/midterms-2026/START_HERE.py"
).read())
```

It detects Colab and installs whatever is missing automatically. Then
`%run run_pipeline.py --full`.

Colab's filesystem is wiped when the session ends. To keep results and any
polls you enter, mount Drive *before* the setup cell:

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive
```

### Local machine



Paste the contents of `START_HERE.py` into one Jupyter cell and run it. It
downloads the project, moves into it, checks your packages, and prints the
next command. Then run, in a new cell:

    %run run_pipeline.py --full

Do not paste the numbered scripts into cells. They import each other, so they
have to exist as files, which the download handles.

Expected runtime: about 5 minutes. Stages 09, 11, 12 and 14 run MCMC
sampling and dominate the total. `--fast` skips the backtest (stage 14) and
still produces the forecast.

**Sampling runs sequentially by design.** Each stage already runs inside a
subprocess whose stdout is a pipe; letting PyMC spawn its own worker
processes for parallel chains on top of that deadlocks on Windows, where
multiprocessing uses spawn and the workers inherit that pipe. The symptom is
a stage that produces no output and never finishes. `config.MCMC_CORES`
defaults to 1 to avoid it; set `MCMC_CORES=2` to opt back in on Linux/macOS.
Sequential sampling costs roughly 2x wall time for 2 chains.

**Compute backend.** PyMC compiles each model before sampling. `perf.py`
picks the fastest backend available and is imported before pymc everywhere
that samples: the default C backend when PyTensor finds a C compiler,
otherwise numba. Measured on this project's national model fit:

| backend | time |
|---|---|
| default C backend, compiler present | 5.6s |
| numba, no C compiler | 10.9s |
| interpreted fallback, neither | 67.5s |

numba is therefore a listed dependency: `pip install numba` needs no admin
rights and no toolchain, and it keeps a compiler-less machine usable.
`%run speed_check.py` reports which backend you are on and estimates
runtimes. A C toolchain is still slightly faster if you want it
(`conda install -c conda-forge m2w64-toolchain`).

### Getting real polls in

The 2026 poll fetchers try, in order:

1. a CSV endpoint named by `NYT_POLLS_URL` (unset by default),
2. **Wikipedia polling tables** (the default working source),
3. a RealClearPolitics scrape,
4. `data_store/manual/polls_2026.csv`.

If all four come up empty the model builds clearly-labelled placeholder polls
so the pipeline still runs, and every output says `provenance=fixture`.

**How the articles are found.** Guessing titles alone is fragile: Wikipedia
renames and redirects articles, and the polling for a race often lives one
click away from the hub page rather than at the title you would guess. Stage
03 therefore does both. It crawls the three hub articles (Senate, gubernatorial
and House elections) for real links to race articles, maps each link back to a
`race_id`, and merges that with the guessed titles, so a naming change on
either side does not lose a race.

**How each article is read.** Tables are parsed section by section rather than
in bulk, because one statewide House article holds a separate polling table per
district, and a Senate article can hold general-election, primary and
hypothetical-matchup tables side by side. Each table stays attached to its
nearest heading, which lets the parser assign district polls to the right seat
and drop primary and hypothetical sections (a ballot no voter will see).

**Why Wikipedia.** FiveThirtyEight's poll database was discontinued in 2025.
The commercial aggregators (RealClearPolitics, Silver Bulletin, FiftyPlusOne)
render their tables in JavaScript and restrict reuse, so a plain HTTP fetch
returns no rows from them. Wikipedia's per-race "Polling" sections are plain
wikitables that `pandas.read_html` reads directly, each row cites its original
pollster, and the licence permits reuse. `wikipolls.py` does the parsing and
`test_wikipolls.py` covers it with fixtures (dates spanning months, comma
sample sizes, partisan asterisks, non-poll tables that must be skipped).

### The Excel workbook (the preferred poll source)

`MidtermPolls2026.xlsx` is the primary input. When the pipeline finds it, the
workbook supplies the polls and the scrapers are skipped entirely.

```
%run workbook.py        # find it, read it, report what it holds
%run check_data.py      # also prints the workbook's path and how stale it is
```

Where the pipeline looks, in order:

1. `$MIDTERM_WORKBOOK`, if set
2. `config.POLL_WORKBOOK` (the OneDrive path on the author's machine)
3. `data_store/manual/MidtermPolls2026.xlsx` (the copy that makes this work in
   Colab, in a container, and on a second machine)
4. `config.WORKBOOK_EXTRA_PATHS`

Set the environment variable to use a copy elsewhere without editing anything:

```
import os; os.environ["MIDTERM_WORKBOOK"] = r"D:\path\to\MidtermPolls2026.xlsx"
```

**Sheets read.** Only the four input sheets: `Polls`, `Races`, `Ratings` and
`Rating_Scale`. The per-state tabs are generated views that say "Do not edit
here", so reading them would mean reading the same polls twice.

**The workbook replaces the scrapers rather than joining them.** Two reasons.
Its `include_in_model` column records a judgement a scraper cannot make,
separating live general-election matchups from pre-primary ballot tests,
withdrawn candidates and superseded duplicates; re-scraping would put those
excluded rows back. And the workbook was built partly from the same Wikipedia
articles the scraper reads, where the same poll carries a different identifier
in each source, so it would survive de-duplication twice and be counted twice.

**What the model takes from it.** Polls (stage 03), presidential approval from
the `US-APPROVAL` rows (stage 04), and current Cook / Inside Elections /
Sabato ratings from the `Ratings` sheet (stage 05). The last two are
controlled by `config.WORKBOOK_FEEDS_APPROVAL` and
`config.WORKBOOK_FEEDS_RATINGS`.

**Conventions it relies on**, all handled in `workbook.py`:

| Workbook | Model |
| --- | --- |
| `US-GB` | `GENERIC` |
| `AL-SEN`, `OH-SEN-S` | `S-AL`, `S-OH-special` |
| `AL-GOV` | `G-AL` |
| `AL-02`, `AK-AL` | `H-AL-02`, `H-AK-01` |
| shares as fractions (0.51) | percentage points (51.0) |
| option A = D, or the independent where there is no Democrat | main candidate |
| `LV` / `RV` / `A` / `V` | `lv` / `rv` / `a` / `unknown` |

The percentage scale is detected rather than assumed, so switching the sheet to
whole numbers later will not silently divide the forecast by 100. Pollster names
are normalised before house effects are fitted, because the workbook's own notes
warn that its two sources spell the same pollster differently; left alone, one
pollster becomes two and its house effect is estimated twice from half the data.
Sample size is recovered from the margin of error where it is missing.

**Weekly refresh.** Save the workbook, then:

```
%run run_pipeline.py --from 03
```

Nothing about the workbook is cached, so a re-run always reads the file on
disk. `check_data.py` prints when it was last saved and warns if that was more
than 10 days ago, because a stale workbook is the easiest way to get a
confident forecast built on old polls.

### Scraping Wikipedia (the fallback)

Used only when no workbook is found. Check it works from your machine before a
full run:

```
%run test_wikipedia_live.py          # a few representative races
%run test_wikipedia_live.py --all    # all 72 race articles
```

It prints polls found per race and the most recent ones. If Wikipedia is
blocked it says so and stops quickly rather than retrying 72 articles.

**Manual entry** remains available for polls Wikipedia lacks, or to override:

```
python3 add_polls.py --example     # show the format
python3 add_polls.py --file my_polls.txt
python3 add_polls.py --check       # what is loaded now
```

Each line is `race_id, pollster, end_date, sample, population, dem_pct, rep_pct`.
Entries are validated and de-duplicated. Manual polls are merged with scraped
ones, so both can be used together.

Note that polls alone do not clear the `fixture` label: presidential approval
(stage 04) and race ratings (stage 05) have their own placeholders. `%run
check_data.py` lists exactly which inputs are still placeholders.

### Watchlist

`data_store/manual/watchlist.csv` holds the races you are tracking (race_id,
office, rating_at_entry, note). It does not change what is modelled: all 506
races are always forecast. It is a lens for reporting and a priority list for
poll entry.

```
%run show_results.py --watchlist
```

shows only those races, least to most Democratic, with expected seats for the
subset. Without the flag you get the competitive races by probability.

### Supplying race ratings

`data_store/manual/race_ratings_2026.csv` (race_id, source, rating) takes
ratings you enter by hand, which is the reliable path since the rating sites
are JavaScript-rendered. Accepted values: Safe/Likely/Lean D or R, and
Toss-up. A hand-entered list usually covers only competitive races, so stage
05 fills the remaining seats from partisan lean and labels each row with its
origin (`user_provided` vs `fixture_pvi_derived`). `%run check_data.py` and
the stage metadata report the mix.

### Reading the forecast

There are two views, for two audiences.

For a plain-English briefing with no jargon, which is the right starting point
and the thing to share with anyone who does not build models:

    %run briefing.py

It says who is favored and by how much in words, pairs every probability with
the frequency it implies (so "38%" is never misread as "no"), names the
tipping-point Senate race, lists the genuinely contested seats with their
margins in plain language, and closes with what the model does not know. It
also writes `outputs/forecast_briefing.md`, which is shareable as-is.

For the numbers themselves, with ratings, blend weights and margins:

    %run show_results.py

To see where each input actually came from on your machine:

    %run check_data.py

It marks every input as real (live / cache / mirror / manual) or as a
placeholder (fixture), and names the fix for each placeholder.

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
polls. `data_store/processed/` and `outputs/` are not committed: they are
regenerated by every run. The repository carries code, the hand-maintained
input templates, and the cached historical mirrors in `data_store/raw/`.
The environment this was first built in blocked every non-GitHub host, so
the 2026 fetchers (03, 04, 05, FEC in 06) were validated against fixtures
and manual files, not live markup. **Run the pipeline from a machine with
network access, or fill in the manual CSVs, before reading anything into the
2026 numbers.**

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
pip install pandas numpy scipy pyarrow requests beautifulsoup4 lxml html5lib openpyxl pyyaml matplotlib pymc arviz
export FRED_API_KEY=...   FEC_API_KEY=...        # optional but recommended
for s in data/01 data/02 data/03 data/04 data/05 data/07 data/06 \
         model/08 model/09 model/10 model/11 model/12 simulation/13 validation/14; do
  python3 ${s}_*.py || break
done
```

### Running in Jupyter

The scripts are modules that import each other, so they must exist as **files
on disk**. Pasting the contents of `01_fetch_economic_data.py` into a cell
fails with `ModuleNotFoundError: No module named 'config'`, because the paste
never created `config.py` as a file.

The supported way:

1. Download or clone the whole `midterms-2026` folder.
2. Open `forecast.ipynb`, which lives inside it. Each cell `%run`s one stage.
3. Run the first cell, `%run setup_check.py`. It verifies the working
   directory, every package, that PyMC can compile and sample, and prints
   which 2026 inputs are real versus placeholder. Fix anything it flags
   before going further.

Pasting a *whole* stage script into a cell does work, but only once
`config.py` and `utils.py` exist as files and the notebook can see them: each
script searches the working directory, its parents, and any `midterms-2026`
subfolder for them, and raises a message naming the fix if it cannot.

Shortcut runner: `python3 run_pipeline.py --update` re-runs only what new
polls can change (03 -> 08 -> 11 -> 12 -> 13, about a minute);
`--full` runs everything; `--fast` skips the backtest; a list of stage numbers
runs just those; `--from 09` resumes there.

**Missing prerequisites are pulled in automatically.** Each stage reads the
cached parquet outputs of the stages it depends on. On a fresh machine, or a
new Colab session (which starts with an empty filesystem), those files do not
exist yet, so `--update` alone would fail inside stage 08 looking for output
that stage 06 produces. The runner checks what is actually on disk, adds any
missing producers, and reorders so dependencies run first. It prints which
stages it added.

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

## The economic variables: a specification test, and a null result

`validation/15_state_economy_test.py`, output in `outputs/state_economy_test.txt`

The model carries one economic regressor: year-over-year CPI inflation at
October, entering the national fundamentals model signed by the president's
party. It does not earn its place, and this section records the evidence
because a null result that took work is worth as much as a finding.

### The national model cannot identify an economic term

The national model trains on 19 midterms (1950-2022). On that sample the
minimum detectable effect for the inflation coefficient, at 80% power and 5%
two-sided, is **1.21 points of House margin per point of CPI**. Across the
observed range of signed inflation in the training data (a span of 20.8 points)
that implies a **25-point swing in the national House vote**. No real economic
effect is that large. The design can only detect effects too big to exist, so
the fitted coefficient (-0.198, standard error 0.432, t = -0.46) is what noise
looks like.

Leave-one-out cross-validation agrees. Root mean squared error across the 19
cycles, lower is better:

| Specification | LOO RMSE |
| --- | --- |
| no economic term | **3.80** |
| unemployment, change | 3.88 |
| CPI, change | 4.13 |
| unemployment, level | 4.15 |
| CPI, level (what the model runs) | 4.68 |

A Hibbs-style real disposable income growth term (FRED `A229RX0`, available for
16 of the cycles) scores 4.28 against 4.06 for the null **with the wrong sign**.
No specification produces |t| above 1.

The term was also confounded until recently. The raw correlation between signed
CPI and House margin is -0.557, which looks like signal; the partial correlation,
holding approval and the presidential-party sign constant, is **+0.065**. The
cause is structural: `corr(s x CPI, s x midterm) = +0.763`, because CPI is almost
always positive, so `s x CPI` is nearly `s x constant`, which *is* the midterm
penalty regressor. Centering CPI on its training mean breaks that (correlation
-0.002) and tightens the midterm penalty `b_mid` by 17%, from +/-1.33 to +/-1.11.
Centering does not change `b_cpi`, because it is a linear reparameterization, and
it does not move the 2026 forecast.

### The seat model has the power, and still finds nothing

The seat model is the only place in the project with enough degrees of freedom
to adjudicate, so the local-economy hypothesis was tested there: after
conditioning on a race's partisan lean, its incumbency and the national
environment of its cycle, does a state whose unemployment is unusually high (or
deteriorating unusually fast) relative to the nation swing against the party
held responsible?

* **Data.** State unemployment rates for all 50 states, monthly from January
  1976, from FRED (`CAUR`, `IAUR`, ...), read at October of each election year
  alongside the 12- and 24-month changes, as deviations from the national
  reading. Levels alone are structural (West Virginia runs above the nation in
  good years and bad) and the partisan lean already encodes that.
* **Sample.** 966 contested Senate and Governor races, 22 even-year cycles,
  1982-2024.
* **Referents.** Two, tested separately: the president's party (the referendum
  story) and the party holding the seat (the accountability story). Both signed
  so the predicted coefficient is **negative**.
* **Specifications.** Cycle fixed effects; cycle plus state fixed effects
  (within-state movement only, the strictest design); and the national margin as
  a regressor, which is the shape `modellib.fit_seat_model` actually uses.
* **Criterion.** Leave-one-cycle-out prediction, not in-sample fit. Races in one
  election share a national environment, so leaving out a single race would leak
  that cycle into its own prediction. Standard errors are clustered by cycle for
  the same reason.

Nothing clears the bar. 24 terms were tested, at which point roughly 1.2 false
positives at |t| >= 2 are expected by chance, so the family-wise threshold is
|t| >= 3.08. The largest |t| anywhere is 2.08, and no term improves
leave-one-cycle-out error by more than 0.04 points of RMSE on a baseline of 16.

One pattern looked real: `seat_dev_lv`, the state's unemployment level relative
to the nation signed by the party holding the seat, is correctly signed and
almost identical in size in all three specifications (-0.70, -0.79, -0.71),
about -1.0 points of margin per standard deviation. That is what a small true
effect looks like, and also what three correlated views of the same noise look
like.

The way to tell them apart is a sample that did not choose the term. House races
are that sample: 4,027 contested races, 12 cycles 1978-2024, a much stronger
lean (the same district two years earlier rather than the same state six years
earlier), and the same state-level economic reading. Pre-specifying the term
makes it a single test with no multiple-comparison penalty:

| Specification | n | coef | se | t | minimum detectable effect |
| --- | --- | --- | --- | --- | --- |
| cycle FE | 4,027 | -0.045 | 0.134 | -0.34 | 0.44 pts/sd |
| cycle + state FE | 4,027 | +0.054 | 0.133 | +0.41 | 0.44 pts/sd |
| national margin | 4,027 | -0.085 | 0.260 | -0.33 | 0.85 pts/sd |

This is an **informative** null, not an underpowered one. The House sample can
detect an effect of 0.44 points per standard deviation at 80% power, less than
half the -1.0 the Senate and Governor sample suggested, and finds -0.05. The
earlier pattern was noise. Restricting to midterm cycles only (551 state races,
1,355 House races) changes nothing.

### What this does and does not establish

It rules out **state unemployment** as a seat-level regressor. It does not rule
out the local economy. Unemployment is the least elastic indicator available: it
moves slowly and is measured for a state, not for the people who vote in a given
race. Three things would make the hypothesis testable again rather than merely
unproven:

1. **District-level conditions.** A county-to-district crosswalk would let the
   economic reading match the electorate instead of averaging a whole state.
   That is the only extension with a real prospect of finding what this test
   could not.
2. **A faster indicator.** Gas prices, grocery prices or local house prices move
   on an election timescale in a way unemployment does not.
3. **Perceptions rather than conditions.** The economic-voting literature finds
   that *perceived* conditions predict votes far better than measured ones, and
   perceptions are increasingly partisan rather than local.

Until one of those is in the data, the supported position is that the model's
economic content is already carried by presidential approval, which voters
report *after* they have formed a view of the economy, and that a separate
economic regressor adds confounding rather than information.

## Expert race ratings: where they come from and what they are worth

`fetch_historical_ratings.py` -> `data_store/manual/historical_ratings.csv` -> stages 05, 10, 12

Qualitative ratings ("Lean R", "Toss-up") are the only usable signal for the
roughly 350 of 506 races that no pollster will touch. Stage 10 turns a rating
into a win probability and a margin band from the raters' empirical track
record, so that mapping is only as good as the record behind it.

### The problem

That record used to be a single cycle: FiveThirtyEight's 2018 forecast-review
categories. Because 435 of its 506 races are House districts, it was
overwhelmingly safe seats, and the tiers that decide a forecast were thin:

| tier | before | after |
| --- | --- | --- |
| Safe | 376 | 511 |
| Likely | 82 | 136 |
| Lean | 33 | **97** |
| Toss-up | 15 | **43** |

With 15 Toss-ups the Beta-binomial posterior sat on its prior, so the model was
not learning the raters' accuracy at all where it mattered most.

### The source

Wikipedia's per-cycle election articles carry a final pre-election predictions
table listing every major rater's last call with its as-of date. Cook's and
Inside Elections' own archives are paywalled and Sabato's are spread across
individual newsletter posts, so this is the only public source covering several
cycles in one consistent shape. Four midterms were parsed (2006, 2010, 2014,
2022), giving 283 consensus races, 281 of which have a result on file.

Three decisions worth knowing about:

* **Only Cook, Sabato's Crystal Ball and Inside Elections.** The tables also
  carry RealClearPolitics, FiveThirtyEight, the New York Times, Daily Kos, CQ,
  Politico, Fox and others, but those are poll-driven models rather than expert
  ratings and none of them is in the 2026 feed. A calibration is only
  meaningful for the raters it will be applied to. Inside Elections is treated
  as the continuation of the Rothenberg Political Report, which it is.
* **One consensus row per race, not three rater rows.** Three raters looking at
  one race are three correlated judgements, not three observations. Counting
  them separately would inflate the sample threefold and understate the
  uncertainty. The per-rater rows are kept in the file for provenance and for
  anyone who wants to score raters against each other; stage 10 reads only the
  consensus.
* **Races where any rater named an independent are excluded entirely.** Rhode
  Island 2010 is why. Two of three raters said "Lean I" and Lincoln Chafee duly
  won, but an "I" rating has no place on a D-positive tier axis, so taking the
  median of the surviving raters left Cook's lone "Toss-up" and the race would
  have entered the calibration as a toss-up the Democrat lost. The raters were
  right and the record would have said they were wrong. Alaska 2014 is the same
  case.

**The House is not available this way.** Those articles have an "Election
ratings" heading but it carries prose about the number of competitive seats, not
a district-by-district table. District ratings for past cycles would need Daily
Kos Elections' spreadsheets or a Cook subscription, so the added cycles are
Senate and Governor only.

### Does pooling offices still hold?

Stage 10 pools tiers across offices, on the stated grounds that rating
vocabularies are shared and per-office samples are thin. That was an assumption
rather than a finding, and with 281 more races it became testable.

It holds, for the win probability. Adding an office term to a logistic model of
"did the favourite win" on tier gives a likelihood ratio of **0.03 on 1 degree
of freedom, p = 0.87**: no improvement whatsoever. Within the expert data alone,
Senate and Governor are indistinguishable at every tier (all p >= 0.52).

An earlier reading of the same data looked like evidence against pooling: in
2018, statewide "Lean" races went to the favourite 63.6% of the time (n=11)
against 95.5% for House districts (n=22), p = 0.033. That is confounded with
source, not office. Holding office fixed and varying only the source, statewide
Lean races are 63.6% under FiveThirtyEight's 2018 categories and **96.9%**
(n=64) under expert ratings. The difference is the rating system, not the
chamber.

The **margin spread** does differ by office: at Lean, House sd 2.5 against
statewide 6.7 (Levene p = 0.012), and similarly at Likely and Safe. This is not
acted on, for two reasons. It is confounded with source the same way, and House
is represented by one cycle so the two cannot be separated. And it is mostly
moot in practice: the sd floors (8 points at Lean and Toss-up, 10 at Likely)
bind at exactly the tiers where the gap appears, so the pooled number is never
used there. Where the empirical sd is used, at Safe and Toss-up, pooling errs
toward wider bands, which is the safe direction.

### What it changed

The tier mapping moved most where the old sample was thinnest:

| tier | P(favourite wins) before | after | 90% interval after |
| --- | --- | --- | --- |
| Safe | 0.9997 | 1.000 | 0.999 to 1.000 |
| Likely | 0.937 | 0.959 | 0.930 to 0.981 |
| Lean | **0.823** | **0.903** | 0.854 to 0.943 |
| Toss-up | 0.500 (pinned) | 0.500 (pinned) | 0.436 to 0.642 |

Toss-up stays pinned at 0.500 by design, since a toss-up with a favoured side
would not be a toss-up, but its margin sd rose from the 8-point floor to an
empirical 9.07, so toss-up races now get a wider band.

In the backtest the ratings component is, by itself, the **strongest single
component** in most statewide cells: Brier 0.027 against 0.050 for the blend in
the 2010 Senate, 0.027 against 0.039 in 2014, 0.024 against 0.041 in 2022.
Read that with care. These are final pre-election ratings, so they already
contain the polls; the raters are not an independent source so much as a
well-calibrated human blend, which is why stacking gives them a minority weight
rather than a majority one.

Every cycle's blend Brier improved, slightly: 2010 0.0562 -> 0.0559,
2014 0.03319 -> 0.03317, 2018 0.0460 -> 0.0458, 2022 0.0561 -> 0.0558. The
2026 forecast moved from 97.9 / 67.4 / 61.8 to **96.8 / 65.7 / 61.1**, slightly
less confident, which is the expected direction: 359 unpolled races now take
15% of their estimate from a component with a five-cycle track record instead of
almost entirely from fundamentals.

## Three defects in the blend, found by asking why a number moved

`config.STACK_MIN_ROWS_PER_COMPONENT`, `config.POLLED_RATING_SHARE_MODE`, `config.RATING_BAND_PROJECTION`

Adding archived ratings moved the Senate from 67.4% to 65.7% at a moment when the
polls were moving the other way (the generic ballot went from D+5.9 in August to
D+7.1). The seat medians did not move at all, which was the clue: a probability
that falls while the central estimate holds is a dispersion or weighting story,
not a polling story. Tracing it found three defects, none of which was the
ratings data itself. Removing the ratings component entirely gave 65.5%, lower
than keeping it, so it was never the culprit.

### 1. A 7% slice of weight flipping on a boundary solution

Weights for polled races were assembled from two different stacks: the SIZE of
the non-poll slice from the two-way `polled_all_cycles` fit, and only its
fund-versus-rating RATIO from the three-way `polled_rated` fit, as `r / (f + r)`.

The three-way fit puts the fundamentals at exactly 0.000 on polled races at every
horizon, so that expression returns exactly 1.0, and it carries no information
about how well determined it is. Before archived ratings existed, the rating
weight was the one pinned at 0.000 and the same expression returned exactly 0.0.
So the entire slice flipped from one component to the other because of which of
two boundary-clipped weights happened to be the zero. Sweeping that one number
reproduces the whole move:

| fund:rating split on polled races | House | Senate | Governor |
| --- | --- | --- | --- |
| all to fundamentals (old) | 98.0% | 67.4% | 61.8% |
| even split | 97.9% | 66.6% | 61.6% |
| all to ratings (new) | 97.9% | 65.7% | 61.3% |

The fix is to stop mixing two stacks. `POLLED_RATING_SHARE_MODE="stack"` takes
the three-way fit's weights as fitted, so the size and the split come from the
same regression on the same races. A `"ratio_shrunk"` mode keeps the old
construction but shrinks the proportion toward an even split, Beta-style, so a
pinned ratio cannot swing the answer by itself.

### 2. A tier mean used as a point estimate

The calibration maps "Safe" to the mean margin of historically safe races, 35.6
points. But safe races run from +14 to +64 at the 10th and 90th percentiles. The
blend is a mixture, so its variance carries a `(component - mean)^2` term, and
mixing a 35-point constant into a race the fundamentals put at R+12 manufactures
an 18-point standard deviation out of nothing but the gap between two numbers.
That is how the Oklahoma governor's race came to show a 26% Democratic win
probability and Colorado 69%.

What a rating claims is that the race lies in its tier's band, not that it sits
at the band's mean. `RATING_BAND_PROJECTION` projects the component's mean onto
that band from wherever the other components put the race:

* already at or beyond the band, so the rating is redundant: its mean and its sd
  are both set to the other components', and the mixture returns them unchanged.
  "Safe R" tells you nothing you did not already know about a race at R+18.
* short of the band: the rating shifts the estimate to the band's inner edge and
  keeps its tier sd. Polls at R+2 against a Safe R rating is real disagreement
  and still registers, but as "at least R+14" rather than "R+35".

The projection is **one-sided** for any rating with a favoured party, which was
a bug in the first version of this fix. "Safe D" means at least D+14, not between
D+14 and D+64, so a district the components put at D+83 agrees with a Safe D
rating rather than conflicting with it. Clipping it down to the band's far edge
invented a 12-point standard deviation from two components that agreed, and
dropped six safe Democratic House seats from 100% to 88%. A Toss-up is the
exception and is clipped on both sides, since it does assert the race is near
zero.

Effect: 399 of 506 rated races are now redundant and contribute nothing. Mean
idiosyncratic sd in the Senate falls from 8.09 to 5.26.

### 3. Extrapolating from the thinnest point in the archive

The poll archive stops at 21 days and the forecast horizon is 28, so the weights
extrapolate past the end of the curve. That end is where the archive is
thinnest: 65 rated polled races at 21 days against 238 at 14. A simplex fitted
on 65 races lands on its boundary, and the `polled_all_cycles` `hier` weight was
exactly 1.000 on 82 races where 364 races gave 0.982. Both became the anchor for
the extrapolation.

`STACK_MIN_ROWS_PER_COMPONENT` (50) requires that many races per fitted weight,
so a three-way stack needs 150 and a two-way needs 100. Horizons that cannot
support their stack are dropped from the curve and the extrapolation anchors on
the last one that can. Here that drops the 21-day `polled_all_cycles` (82) and
`polled_rated` (65) rows, and keeps 21-day `unpolled_rated` (592 races), where
the rising rating weight is real: the further out, the sparser the polling and
the more a rating is worth. The file previously used a flat minimum of 40 for a
three-way stack, which is the same idea applied too loosely.

### What the three fixes did

| | House | Senate (median) | Governor |
| --- | --- | --- | --- |
| before archived ratings | 97.9% | 67.4% (52) | 61.8% |
| archived ratings, with these defects | 96.8% | 65.7% (52) | 61.1% |
| archived ratings, fixed | **98.3%** | **67.7% (53)** | **62.7%** |

The backtest is unchanged, which is the point: these removed artifacts rather
than bought a score. Mean blend Brier 0.04768 to 0.04773 (+0.1%, inside noise),
log loss improved in three of the four cycles, and the Senate improved in three.
The backtest now applies the same band projection the forecast does, which it
did not before; a backtest scoring a different blend from the one that ships is
the one way to make it actively misleading.

### A diagnostic that came free

With the band projection in place, a race whose other components fall well short
of their rating's band is a flag. Twenty do by more than 10 points, and the worst
are unmistakable: `H-MO-05` (components D+29, rated Safe R), `H-UT-01` (D+26,
Safe R), `H-CA-41` and `H-CA-03` (D+16, Lean R), `H-NC-06` (R+14, Likely D),
`H-LA-06`. Those are all states with new or contested 2026 lines, so the ratings
describe the new map and the lean still describes the old one. The ratings are
right and the lean is stale. See the redistricting section; this is a concrete
to-do list rather than a general warning.

A second pattern in that list is worth a different kind of attention: ten House
districts rated Safe R where thin district polling says only R+2 or R+3. There
the raters are more likely to be right than one or two low-quality district
polls, and the model currently splits the difference.

## Fixture ratings were built from the old maps

`data/05_fetch_race_ratings.py`, `fixture_ratings()`

Only 204 of 506 races carry a real Cook / Inside Elections / Sabato rating. The
other 302 get a rating derived from their partisan lean and labelled
`fixture_pvi_derived`. That derivation read FiveThirtyEight's **2022** lean and
nothing else, so for the 173 districts on new 2026 maps it rated a district that
no longer exists. 117 of those 173 have no real rater rating, so the fixture was
the only one they had.

Where redistricting flipped a seat, the fixture pointed the wrong way:

| District | 2022 lean | 2026 lean | fixture rating was | now |
| --- | --- | --- | --- | --- |
| H-UT-01 | R+24.5 | **D+23.8** | Safe R | Safe D |
| H-NC-06 | D+9.2 | **R+17.9** | Likely D | Safe R |
| H-CA-41 | R+3 | **D+14.2** | Lean R | Likely D |
| H-CA-03 | R+6 | **D+10.0** | Lean R | Likely D |

The function now reads `pvi_manual.csv` first, exactly as stage 06 does, with the
2022 lean as the fallback for the 262 districts whose lines did not change. It is
read directly rather than from `fundamentals_2026`, because stage 06 runs after
stage 05. 18 consensus ratings changed, all of them on new maps, including those
four sign flips. The House moved from 98.3% to 98.4%; the Senate and Governor did
not move, since neither has a fixture-rated competitive race.

The same function also silently turned a missing lean into 0.0, which
`tier_from_lean` reads as a Toss-up: the most consequential rating in the table,
invented from an absent lean. It now warns and names the races. Currently none.

Note how this interacts with the band projection. A fixture rating carries no
information the fundamentals do not already have, because both are the same lean.
The projection recognises that and marks the rating redundant, so it contributes
nothing instead of double-counting the lean. That only works when the two agree,
which is exactly what this fix restores: all four races above now show
`rating_redundant = True`.

### Validating the lean file itself

Before concluding the ratings were at fault, the leans were checked, because the
opposite diagnosis was equally plausible. The Downballot sheet carries 2020
results only for districts whose lines did NOT change, which makes it
self-auditing: the states it reports as redrawn are exactly the nine the model
flags `new_map`, 173 districts, with no mismatch in either direction. Missouri is
NOT among them; `fetch_pvi.py` lists ten states in its docstring and should say
nine. The sheet gives Missouri's 5th as D+23 with Cleaver still its incumbent and
2020 numbers present, consistent with the model's D+19.8.

So the leans are right and MO-05's unanimous Safe R from all three raters is the
one case this cannot settle. Either Missouri's lines changed after the sheet was
published on 2026-07-09, which would make the sheet stale for Missouri and the
raters right, or the rating is in error. It is flagged rather than guessed at.

## Known limitations and what to fix first

1. **The shared polling error is the biggest single lever, and the default
   setting has a known cost.** `config.POLL_BIAS_MODE` is `symmetric`: the
   model makes no claim about which direction polls will miss, which is the
   defensible reading of the record (the last three midterms missed by +3.9,
   -0.2 and +0.1, while the big misses of +4.3 in 2016 and +6.9 in 2020 both
   came with Trump on the ballot). The cost is that pinning the shared term to
   zero also removes the *correlation* between polls, so the fit treats 58
   correlated statewide polls as 58 independent readings of the national mood
   and over-determines it. National uncertainty comes out at sd 1.96, below
   the 4.03-point polling miss the model itself measures, and the House
   probability prints as ~100%. Read that as ">99% under this specification",
   not as certainty, and note the same over-determination makes the Senate and
   Governor figures more confident than the evidence strictly supports.

   The alternative, `POLL_BIAS_MODE=estimated`, fixes the correlation but then
   infers a polling correction from the gap between polls and fundamentals,
   which is confounded because the fundamentals are badly wrong in several
   polled races for unrelated reasons (they cannot see independent
   candidates). Sensitivity, 28 days out, same polls in all three:

   | Setting | House | Senate | Governor | median D seats | fitted `poll_bias` | national sd |
   | --- | --- | --- | --- | --- | --- | --- |
   | `symmetric` (default) | 98% | 67% | 62% | 239 / 52 / 27 | 0 (fixed) | 1.10 |
   | `estimated`, midterm drift prior | 97% | 62% | 52% | 234 / 51 / 26 | +1.05 +/- 1.60 | 2.14 |
   | `estimated`, pooled drift prior | 90% | 53% | 45% | 231 / 51 / 25 | +1.58 +/- 1.97 | 2.50 |

   Reproduce any row with
   `POLL_BIAS_MODE=estimated POLL_BIAS_PRIOR=pooled python3 run_pipeline.py 08 11 12 13`.

   **The specifications have converged**, and that is the main thing to know
   about this parameter now. At 36 days out the Senate ran 62% / 38% / 24%
   across these rows, a 38-point spread that made the assumption the dominant
   fact about the forecast. It is now 67% / 62% / 53%, a 14-point spread,
   because the fitted `poll_bias` is inferred from the gap between polls and
   fundamentals and a thousand polls have closed most of that gap. The Senate
   is a modest Democratic favourite under every specification, which was not
   true in September.

   Where the assumption now bites hardest is the **governors**: 62% under the
   default against 45% under the pooled prior, which crosses from favourite to
   underdog. That is the number to quote with its assumption attached.

   The backtest does not settle the choice: its calibration table is unchanged
   across all three, because it scores the blend components at historical
   cycles and never re-fits this cycle's shared polling error. The case for the
   midterm drift prior is the reference-class argument in
   `config.POLL_BIAS_PRIOR_CYCLES`, not backtest evidence.
2. **Long-horizon stacking weights are extrapolated.** The public poll
   archive covers the final 21 days of each cycle only. Two months out, the
   polls-vs-fundamentals split comes from `config.STACK_EXTRAPOLATION`, not
   from data. Supplying a long-horizon archive (or accumulating this cycle's
   polls) lets `12` estimate the full curve.
   `config.STACK_MIN_ROWS_PER_COMPONENT` now stops the extrapolation anchoring
   on a horizon too thin to fit its own stack, which is a guard rather than a
   cure: the forecast at 28 days is still outside the archive.
3. **Ratings are still calibrated in-sample in the backtest**, though the
   sample is now five cycles rather than one (see the ratings section above).
   The leak was measured by refitting the tier mapping without each cycle and
   re-scoring it: +0.0005 Brier, 1.5% of the in-sample score. It is small
   because the mapping is stable across held-out cycles (Lean ranges 0.886 to
   0.924), which it could not have been on one cycle. The remaining gap is
   House ratings, which exist for 2018 only.
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
8. **House district polling is dominated by Democratic internals, and the
   model makes no adjustment for sponsorship.** 56% of the 183 House district
   polls are Democratic internals against 13% Republican, and **36 of the 90
   polled districts have nothing but Democratic internals**. Nothing in stage 08
   or 11 reads the `partisan` column, so a campaign's own poll is weighted like
   a university's. The deeper problem is publication bias rather than house
   effects: campaigns release internals where the numbers are good, which a
   house-effect term cannot correct. Polls also linger: 45 of 183 are more than
   120 days old and the oldest is from January, yet `H-WA-04` gives 48% weight
   to a single February poll showing D+18 in a district at R+24.6, which is
   almost certainly bad data. This is the biggest open problem in the House
   model. The Senate and Governor are not affected: they run 51% and 64%
   non-partisan with roughly balanced sponsorship, and only three and one race
   respectively rely on one side's internals.
9. **The CPI term is retained but unsupported.** See the section above: on
   leave-one-out cross-validation the national model predicts better with no
   economic term (RMSE 3.80) than with the CPI level it currently carries
   (4.68), and a seat-level local-economy term does not survive testing either.
   The term is centered, so it no longer confounds the midterm penalty, and it
   barely moves the 2026 forecast because approval absorbs most of the economy's
   political effect. But it is there on the strength of the literature, not of
   this model's own evidence, and a write-up should say so.

## Backtest results (election-eve, leave-one-cycle-out weights)

From `outputs/backtest_scores.csv` on the committed run (real historical
inputs; the backtest does not use any fixture):

| cycle | races | Brier | log loss | accuracy | expected D seats (all offices) | actual |
|---|---|---|---|---|---|---|
| 2010 | 421 | 0.056 | 0.178 | 93.1% | 189 | 178 |
| 2014 | 392 | 0.033 | 0.108 | 95.7% | 163 | 157 |
| 2018 | 457 | 0.046 | 0.155 | 93.4% | 224 | 234 |
| 2022 | 464 | 0.056 | 0.176 | 92.0% | 198 | 232 |

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

Two stacking findings worth knowing: (1) the rating component used to receive
roughly zero weight, and the suspicion recorded here was that this reflected
the only available historical ratings being FiveThirtyEight's 2018 categories,
which are themselves model output. That suspicion was right. With archived
Cook / Sabato / Inside Elections ratings for 2006, 2010, 2014 and 2022 the
component earns 7% of the weight on polled races and 15% on unpolled ones at
a 28-day horizon, and 17% on both at 21 days; (2) for unpolled races the plain
fundamentals still beat the state-pooled estimate, so the state random effect
mostly matters through the PyMC model's shared terms rather than as a
point-estimate shift.

## Outputs

`outputs/chamber_control.json`, `outputs/race_probabilities.csv`,
`outputs/seat_distribution_{house,senate,governor}.csv`,
`outputs/backtest_scores.csv`, `outputs/backtest_calibration.csv`, and the
figures in `outputs/figures/` (generic-ballot trend, rating calibration,
stacking weights, seat distributions, backtest reliability).
