"""
14_backtest.py
==============
Backtest the full pipeline against the 2010, 2014, 2018 and 2022 midterms and
score it with proper scoring rules, not just "was the winner right".

Procedure
---------
For each cycle, every component is rebuilt as of election eve (horizon 1
day) using only information available then: polls up to that date through
the same house-effect + Kalman code; the national fundamentals model fitted
on earlier midterms only; the seat model fitted on the other cycles; ratings
where the historical table has them (2018). Stacking weights come from the
other three cycles (leave-one-cycle-out) so no cycle scores its own weights.
Race probabilities use the explicit variance budget in
`modellib.marginal_win_prob` (idiosyncratic + national + polling + state),
the same structure the simulation draws from.

Scores
------
* Brier score (mean squared error of the probability) and log loss, per cycle,
  per office, and per component so you can see whether the blend beats its
  parts. Lower is better; a coin flip scores Brier 0.25 / log loss 0.693.
* Calibration: races binned by forecast probability (10 bins); a
  well-calibrated model wins about 70% of the races it gives 70% to. The
  table and reliability plot are written out; the "70% bucket" line is
  printed explicitly.
* Seat totals: expected Democratic seats vs actual per chamber.

Caveats printed with the results: 2010/2014 leans are lagged-result proxies
(538 leans start in 2018); ratings are calibrated in-sample for 2018; the
poll archive covers polled races only, so unpolled races are scored on
fundamentals/ratings alone, exactly as the live model would treat them.

Outputs (outputs/)
------------------
backtest_scores.csv, backtest_calibration.csv, backtest_races.csv,
figures/backtest_calibration.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
import config  # noqa: E402
from utils import get_logger, load_stage  # noqa: E402
from modellib import BacktestContext, blend_rows, marginal_win_prob, stacking_weights, win_prob  # noqa: E402

log = get_logger("14_backtest")
EVE = 1


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def loo_weights(all_comp: dict[int, pd.DataFrame], test_cycle: int) -> dict[str, dict[str, float]]:
    """Stacking weights fitted on the other cycles at election eve."""
    others = pd.concat([v for c, v in all_comp.items() if c != test_cycle], ignore_index=True)
    out = {}
    polled = others[others.poll_margin.notna()]
    w, _ = stacking_weights(polled, {"hier": ("hier_margin", "hier_sd"), "fund": ("fund_margin", "fund_sd_total")})
    out["polled"] = w
    rated = others[others.rating_margin.notna()]
    comps = {"hier": ("hier_margin", "hier_sd"), "fund": ("fund_margin", "fund_sd_total"), "rating": ("rating_margin", "rating_sd")}
    for name, sub in [("polled_rated", rated[rated.poll_margin.notna()]), ("unpolled_rated", rated[rated.poll_margin.isna()])]:
        if len(sub) >= 40:
            out[name], _ = stacking_weights(sub, comps)
    return out


def apply_weights(df: pd.DataFrame, w: dict) -> pd.DataFrame:
    d = df.copy()
    polled = d.poll_margin.notna().values
    rated = d.rating_margin.notna().values
    wp = w["polled"]
    d["w_hier"] = np.where(polled, wp["hier"], 0.0)
    d["w_fund"] = np.where(polled, wp["fund"], 1.0)
    d["w_rating"] = 0.0
    if "polled_rated" in w:
        share = w["polled_rated"]["rating"] / max(w["polled_rated"]["rating"] + w["polled_rated"]["fund"], 1e-9)
        m = polled & rated
        d.loc[m, "w_rating"] = (1 - wp["hier"]) * share
        d.loc[m, "w_fund"] = (1 - wp["hier"]) * (1 - share)
    if "unpolled_rated" in w:
        m = (~polled) & rated
        u = w["unpolled_rated"]
        d.loc[m, ["w_hier", "w_fund", "w_rating"]] = [u["hier"], u["fund"], u["rating"]]
    return d


def main():
    ctx = BacktestContext()
    poll_shock = load_stage("national_environment").iloc[0].poll_shock_sd
    comps = {c: ctx.components(c, EVE) for c in config.BACKTEST_CYCLES}
    scored = []
    for c, df in comps.items():
        w = loo_weights(comps, c)
        d = blend_rows(apply_weights(df, w))
        d["p_blend"] = marginal_win_prob(d, poll_shock_sd=poll_shock)
        # component-only probabilities with the same variance budget
        d["p_fund"] = win_prob(d.fund_margin, np.sqrt(d.fund_sd_total ** 2 + config.STATE_SHOCK_SD ** 2 + config.RACE_NOISE_FLOOR_SD ** 2))
        d["p_polls"] = win_prob(d.poll_margin, np.sqrt(d.poll_sd ** 2 + poll_shock ** 2 + config.RACE_NOISE_FLOOR_SD ** 2))
        d["p_hier"] = win_prob(d.hier_margin, np.sqrt(d.hier_sd ** 2 + (d.poll_weight_in_hier * poll_shock) ** 2
                                                       + ((1 - d.poll_weight_in_hier) * d.fund_nat_loading * d.nat_fund_sd) ** 2
                                                       + config.RACE_NOISE_FLOOR_SD ** 2))
        d["p_rating"] = d.rating_pwin
        scored.append(d)
        log.info("%d: %d races, %d polled, weights %s", c, len(d), int(d.poll_margin.notna().sum()),
                 {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in w.items()})
    res = pd.concat(scored, ignore_index=True)
    res.to_csv(config.OUTPUTS / "backtest_races.csv", index=False)

    rows = []
    for (c, office), g in res.groupby(["cycle", "office"]):
        for comp in ["blend", "hier", "polls", "fund", "rating"]:
            gg = g.dropna(subset=[f"p_{comp}"])
            if len(gg) < 5:
                continue
            rows.append({"cycle": c, "office": office, "component": comp, "n": len(gg),
                         "brier": brier(gg[f"p_{comp}"], gg.dem_won), "log_loss": logloss(gg[f"p_{comp}"], gg.dem_won),
                         "accuracy": float(((gg[f"p_{comp}"] > .5) == (gg.dem_won == 1)).mean()),
                         "expected_dem_seats": float(gg[f"p_{comp}"].sum()), "actual_dem_seats": float(gg.dem_won.sum())})
    for c, g in res.groupby("cycle"):
        rows.append({"cycle": c, "office": "All", "component": "blend", "n": len(g), "brier": brier(g.p_blend, g.dem_won),
                     "log_loss": logloss(g.p_blend, g.dem_won), "accuracy": float(((g.p_blend > .5) == (g.dem_won == 1)).mean()),
                     "expected_dem_seats": float(g.p_blend.sum()), "actual_dem_seats": float(g.dem_won.sum())})
    scores = pd.DataFrame(rows)
    scores.to_csv(config.OUTPUTS / "backtest_scores.csv", index=False)

    bins = np.linspace(0, 1, 11)
    res["bin"] = pd.cut(res.p_blend, bins, include_lowest=True)
    cal = res.groupby("bin", observed=True).agg(n=("dem_won", "size"), forecast=("p_blend", "mean"), observed=("dem_won", "mean")).reset_index()
    cal["bin"] = cal["bin"].astype(str)
    cal.to_csv(config.OUTPUTS / "backtest_calibration.csv", index=False)
    b70 = res[(res.p_blend >= .6) & (res.p_blend < .8)]
    log.info("calibration check, races given 60-80%% (mean %.0f%%): %d races, D won %.0f%%",
             100 * b70.p_blend.mean(), len(b70), 100 * b70.dem_won.mean())
    # symmetric version (favourite wins?) so R-favoured races count too
    fav_p = np.where(res.p_blend >= .5, res.p_blend, 1 - res.p_blend)
    fav_won = np.where(res.p_blend >= .5, res.dem_won, 1 - res.dem_won)
    m70 = (fav_p >= .6) & (fav_p < .8)
    log.info("favourite-side check, 60-80%% bucket (mean %.0f%%): %d races, favourite won %.0f%%",
             100 * fav_p[m70].mean(), int(m70.sum()), 100 * fav_won[m70].mean())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.scatter(cal.forecast, cal.observed, s=np.sqrt(cal.n) * 6, alpha=.8)
    for r in cal.itertuples():
        ax.annotate(str(r.n), (r.forecast, r.observed), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("forecast P(D win)")
    ax.set_ylabel("observed D win rate")
    ax.set_title("Reliability, blended model, 2010/2014/2018/2022 eve")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "backtest_calibration.png", dpi=120)

    pd.set_option("display.width", 160)
    print(scores[scores.office == "All"].round(3).to_string(index=False))
    print(scores[scores.component.isin(["blend", "fund", "hier"]) & (scores.office != "All")]
          .pivot_table(index=["cycle", "office"], columns="component", values="brier").round(3))
    print(cal.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
