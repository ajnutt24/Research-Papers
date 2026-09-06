"""
09_fundamentals_model.py
========================
Fundamentals-only forecast: a national "referendum" model of the House vote
plus a seat-level model that maps the national environment, partisan lean,
incumbency and fundraising onto each race.

Methodology
-----------
National level (20 midterms, 1946-2022). Y = national House margin (D-R).
    Y = a + s*(b_mid + b_app*(approval-50) + b_cpi*CPI_yoy + b_war*war) + e
with s = +1 under a Democratic president, -1 under a Republican. Informative
priors from the time-for-change / referendum literature (Abramowitz; Tufte;
Hibbs's bread-and-peace model) compensate for the small sample:
b_mid ~ N(-4, 3): the in-party's structural midterm penalty;
b_app ~ N(0.3, 0.2): vote share per net-approval point;
b_cpi ~ N(-0.5, 0.5): inflation hurts the in-party;
b_war ~ N(0, 3): the war-salience coefficient is centred on zero with a wide
prior, so its sign and size are estimated, not assumed (the spec's requirement);
b_post94 ~ N(-3, 3): a post-1994 realignment shift that removes the structural
Democratic House lean of the 1946-1990 era from the 2026 prediction.
CPI is the only economic regressor unless script 01 found |corr(CPI, unemp)|
< 0.5, in which case unemployment could be added by hand; it is not by default.

Seat level (2018, 2020, 2022 races with actual results; add MIT data for more).
    margin = c[o] + b_nat[o]*national + b_lean[o]*lean + b_inc[o]*inc + b_fund*fund + e
Office-specific coefficients let Senate and Governor races swing less than
one-for-one with the House environment. Fundraising enters as the log ratio of
D to R individual contributions; with no historical FEC series in the training
table the coefficient stays at its prior N(2, 1.5) and only shapes 2026 races
where the FEC pull succeeded.

Outputs
-------
fundamentals_national.parquet  2026 national margin: mean, sd of mean, predictive sd; coefficient table
fundamentals_estimates_2026.parquet  race_id, fund_margin, fund_sd_idio, fund_nat_loading, fund_sd_total
national_model_idata.nc / seat_model_idata.nc  (ArviZ InferenceData for scripts 12 and 14)
"""
from __future__ import annotations

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from utils import get_logger, load_meta, load_stage, save_stage, worst_provenance  # noqa: E402
from modellib import (fit_national_model, fit_seat_model, national_training_table,  # noqa: E402
                      predict_national, predict_seats, seat_features)

log = get_logger("09_fund")
TRAIN_CYCLES = [2018, 2020, 2022]


def main():
    # ---------------- national model ----------------
    nat_tab = national_training_table()
    train = nat_tab[nat_tab.cycle < config.CYCLE]
    log.info("national training table: %d midterms", len(train))
    idata_n, _ = fit_national_model(train)
    summ = az.summary(idata_n, var_names=["a", "b_post94", "b_mid", "b_app", "b_cpi", "b_war", "sigma"])
    log.info("national coefficients:\n%s", summ[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]].to_string())
    cur = nat_tab[nat_tab.cycle == config.CYCLE].iloc[0]
    if pd.isna(cur.cpi_yoy_oct):
        raise RuntimeError("2026 CPI feature missing - rerun 01_fetch_economic_data.py")
    m, sd_mean, sd_pred = predict_national(idata_n, cur.s, cur.approval_c, cur.cpi_yoy_oct, cur.war_salience)
    log.info("2026 national fundamentals: D%+.1f (sd of mean %.1f, predictive sd %.1f) with approval=%.0f, CPI=%.1f, war=%.2f",
             m, sd_mean, sd_pred, cur.approval, cur.cpi_yoy_oct, cur.war_salience)
    az.to_netcdf(idata_n, config.DATA_PROCESSED / "national_model_idata.nc")

    # ---------------- seat model ----------------
    hist = load_stage("historical_results")
    natres = load_stage("historical_national")
    feats = pd.concat([seat_features(c, hist, natres) for c in TRAIN_CYCLES], ignore_index=True)
    log.info("seat training rows: %s", feats.groupby("office").size().to_dict())
    idata_s, _ = fit_seat_model(feats)
    ssum = az.summary(idata_s, var_names=["c", "b_nat", "b_lean", "b_inc", "b_fund", "sigma"])
    log.info("seat coefficients:\n%s", ssum[["mean", "sd", "r_hat"]].to_string())
    az.to_netcdf(idata_s, config.DATA_PROCESSED / "seat_model_idata.nc")

    fund = load_stage("fundamentals_2026")
    pred = predict_seats(idata_s, fund, nat_mean=m, nat_sd=sd_pred)
    provs = [load_meta(n).get("provenance", "unknown") for n in ["approval_history", "economic_cycle_features", "fundamentals_2026"]]
    prov = worst_provenance(*provs)
    nat_out = pd.DataFrame([{"cycle": config.CYCLE, "nat_fund_mean": m, "nat_fund_sd_mean": sd_mean,
                             "nat_fund_sd_pred": sd_pred, "approval": cur.approval, "cpi_yoy": cur.cpi_yoy_oct,
                             "war_salience": cur.war_salience, "pres_party": cur.pres_party}])
    save_stage(nat_out, "fundamentals_national", prov, {"coefficients": summ["mean"].round(3).to_dict()})
    save_stage(pred, "fundamentals_estimates_2026", prov, {"seat_coefficients": ssum["mean"].round(3).to_dict()})
    print(pred.groupby("office")[["fund_margin", "fund_sd_idio"]].describe().T.round(2))


if __name__ == "__main__":
    main()
