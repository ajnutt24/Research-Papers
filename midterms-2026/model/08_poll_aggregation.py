"""
08_poll_aggregation.py
======================
Polls-only estimate for every polled race, a national generic-ballot trend
line, an implied national environment from seat-level polls, and the
historical correlated-polling-error prior.

Methodology
-----------
1. **House effects.** Each poll is modelled as
       margin = race level + pollster house effect + population effect + noise.
   House effects are estimated jointly across all 2026 races (a pollster that
   polls twenty races reveals its lean even if no single race is polled twice)
   and shrunk toward the pollster's mean-reverted historical bias from
   FiveThirtyEight's 1998-2022 archive, with a prior sd of 2 points. RV and
   adult samples are adjusted toward the LV reference. This is a penalised
   backfitting approximation to the multilevel model; it runs in milliseconds.
2. **State-space filter.** Adjusted margins feed a local-level Kalman filter
   per race (random-walk drift of 0.18 pts/day for races, 0.10 for the generic
   ballot). Recency and sample-size weighting fall out of the filter: newer
   polls carry more weight because the state has drifted since older ones, and
   larger polls carry more weight through their smaller observation variance.
   The posterior is projected to Election Day, which widens the band for races
   whose last poll is stale.
3. **National environment.** Two readings: (a) the generic-ballot filter and
   (b) the precision-weighted mean of (race estimate minus partisan lean minus
   incumbency effect) over all polled House races, i.e. the national swing the
   seat polls imply. Both are reported; their precision-weighted combination is
   the polls-only national environment used downstream.
4. **Correlated polling error.** The per-cycle mean bias of final-three-week
   polls in 2014-2022 (including the 2016 and 2020 presidential-year misses)
   gives an RMS miss that becomes the prior sd of the shared polling shock in
   the simulation. This is the piece that stops the model from treating 400
   independent polls as 400 independent draws.

Outputs
-------
poll_estimates_2026.parquet    race_id, poll_margin, poll_sd, poll_sd_election, n_polls, ...
generic_ballot_trend.parquet   daily smoothed generic-ballot margin
national_environment.parquet   one row: generic, seat-implied, combined (mean, sd)
polling_error_history.parquet  cycle x office mean bias; meta carries the RMS prior
house_effects_2026.parquet     pollster, house_effect, n_polls, historical_bias_prior
figures/generic_ballot_trend.png
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from utils import get_logger, load_meta, load_stage, save_stage, worst_provenance  # noqa: E402
from modellib import (aggregate_polls, historical_polling_error, kalman_daily_path,  # noqa: E402
                      load_raw_polls_history, poll_margin_variance, pollster_bias_prior)

log = get_logger("08_polls")
INCUMBENCY_PTS = 3.0   # rough incumbency effect used only to back out the national swing


def main():
    polls = load_stage("polls_2026")
    fund = load_stage("fundamentals_2026")
    p_prov = load_meta("polls_2026").get("provenance", "unknown")
    asof, election = config.FORECAST_ASOF, config.ELECTION_DATE

    # --- historical archive: pollster priors and correlated error -------------
    try:
        rp = load_raw_polls_history()
        prior_bias = pollster_bias_prior(rp)
        err = historical_polling_error(rp)
        cyc = err[err.office == "All"]
        poll_shock_sd = float(np.sqrt(np.mean(cyc["mean_bias"] ** 2)))
        err_prov = "mirror"
        log.info("historical polling miss by cycle (All offices):\n%s", cyc.to_string(index=False))
    except Exception as e:
        log.warning("historical poll archive unavailable (%s); using config fallback", e)
        prior_bias, err, poll_shock_sd, err_prov = {}, pd.DataFrame(), config.POLL_SHOCK_SD_FALLBACK, "fixture"
    log.info("correlated polling-shock prior sd = %.2f pts", poll_shock_sd)

    # --- house effects + Kalman per race ---------------------------------------
    est, h, g = aggregate_polls(polls, asof, election, prior_bias=prior_bias)
    he = pd.DataFrame({"pollster": h.index, "house_effect": h.values})
    he["n_polls"] = he.pollster.map(polls.groupby("pollster").size())
    he["historical_bias_prior"] = he.pollster.map(prior_bias).fillna(0.0)
    log.info("population effects (vs LV): %s", g.round(2).to_dict())

    # --- generic ballot trend --------------------------------------------------
    gb = polls[polls.race_id == "GENERIC"].copy()
    gb_row = est[est.race_id == "GENERIC"]
    if len(gb):
        gb["var"] = [poll_margin_variance(n, d) for n, d in zip(gb.sample_size, gb.dem_pct)]
        gb["adj"] = gb.margin - gb.pollster.map(h).fillna(0) - gb.population.map(g).fillna(0)
        trend = kalman_daily_path(pd.to_datetime(gb.end_date).dt.date.values, gb.adj.values, gb["var"].values,
                                  asof - timedelta(days=200), asof, kind="generic")
        gb_mean, gb_sd = float(gb_row.poll_margin.iloc[0]), float(gb_row.poll_sd.iloc[0])
    else:
        trend = pd.DataFrame(columns=["date", "mean", "sd"])
        gb_mean, gb_sd = np.nan, np.nan

    # --- implied national environment from seat polls --------------------------
    races = est[est.race_id != "GENERIC"].merge(fund[["race_id", "office", "lean", "incumbency", "lean_source"]], on="race_id")
    hs = races[(races.office == "House") & races.lean.notna() & (races.lean_source != "state_fallback_new_map")]
    if len(hs):
        swing = hs.poll_margin - hs.lean - INCUMBENCY_PTS * hs.incumbency
        w = 1 / (hs.poll_sd ** 2 + 6.0 ** 2)      # 6 pts of district-level idiosyncrasy
        imp_mean = float(np.sum(w * swing) / np.sum(w))
        imp_sd = float(np.sqrt(1 / np.sum(w)))
    else:
        imp_mean, imp_sd = np.nan, np.nan
    parts = [(m, s) for m, s in [(gb_mean, gb_sd), (imp_mean, imp_sd)] if m == m]
    if parts:
        wts = np.array([1 / s ** 2 for _, s in parts])
        comb_mean = float(np.sum(wts * np.array([m for m, _ in parts])) / wts.sum())
        comb_sd = float(np.sqrt(1 / wts.sum()))
    else:
        comb_mean, comb_sd = np.nan, np.nan
    nat = pd.DataFrame([{"asof": asof, "generic_mean": gb_mean, "generic_sd": gb_sd,
                         "seat_implied_mean": imp_mean, "seat_implied_sd": imp_sd,
                         "n_house_races_polled": int(len(hs)),
                         "combined_mean": comb_mean, "combined_sd": comb_sd,
                         "poll_shock_sd": poll_shock_sd}])
    log.info("national environment: generic %.1f±%.1f, seat-implied %.1f±%.1f, combined %.1f±%.1f",
             gb_mean, gb_sd, imp_mean, imp_sd, comb_mean, comb_sd)

    prov = worst_provenance(p_prov)
    save_stage(est, "poll_estimates_2026", prov, {"n_races": int((est.race_id != "GENERIC").sum())})
    save_stage(trend, "generic_ballot_trend", prov)
    save_stage(nat, "national_environment", prov, {"poll_shock_sd": poll_shock_sd})
    save_stage(err, "polling_error_history", err_prov, {"poll_shock_sd_rms": poll_shock_sd})
    save_stage(he, "house_effects_2026", prov)

    # --- figure ------------------------------------------------------------------
    if len(trend):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.fill_between(pd.to_datetime(trend.date), trend["mean"] - 1.96 * trend.sd, trend["mean"] + 1.96 * trend.sd, alpha=.2)
        ax.plot(pd.to_datetime(trend.date), trend["mean"], lw=2, label="smoothed generic ballot (D-R)")
        ax.scatter(pd.to_datetime(gb.end_date), gb.adj, s=12, alpha=.5, label="house-adjusted polls")
        ax.axhline(0, color="k", lw=.5)
        ax.set_title(f"Generic ballot, D minus R (provenance: {prov})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(config.FIGURES / "generic_ballot_trend.png", dpi=120)
    print(est.sort_values("n_polls", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
