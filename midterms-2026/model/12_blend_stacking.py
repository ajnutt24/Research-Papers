"""
12_blend_stacking.py
====================
Combine the three component forecasts for every race:
  * hierarchical (polls pooled with fundamentals, script 11),
  * fundamentals-only (script 09),
  * expert-rating-based (script 10),
with weights learned by leave-one-out (LOO) stacking on the backtested
midterms rather than a hand-picked formula, and made time-varying so that
fundamentals dominate early and polls dominate near Election Day.

Methodology
-----------
1. For each backtest cycle (2010, 2014, 2018, 2022) and each horizon in
   HORIZONS (days before the election) rebuild every component *as it would
   have stood on that date*: polls filtered to that date and run through the
   same house-effect + Kalman code as 2026; the national and seat fundamentals
   fitted with the test cycle held out; ratings from the historical table.
2. ArviZ stacking (`az.compare(..., method="stacking")`) on the pooled races
   with all components present gives, per horizon, the mixture weights that
   maximise expected log predictive density. Two stacks are run:
      polled races:   {hier, fund}  across all four cycles  -> w_hier(h)
      rated races:    {hier, fund, rating}  (2018, the cycle with ratings)
                      separately for polled and unpolled races
   Ratings exist for only one backtest cycle here, so their weight is treated
   as horizon-invariant; add archived Cook/Sabato/Inside ratings via
   data_store/manual/historical_ratings.csv to relax that.
3. The 2026 weights for a race are interpolated at the current horizon
   (DAYS_TO_ELECTION): for polled races w_hier(h) comes from the all-cycle
   curve and the remainder is split between fundamentals and ratings in the
   ratio the 2018 three-way stack found; unpolled races use the 2018
   unpolled stack directly. Races with no rating give the rating weight to
   fundamentals.
   LIMITATION: the public poll archive (FiveThirtyEight raw-polls) covers only
   the final 21 days of each cycle, so weights are estimated at 1-21 days and
   extrapolated beyond with an exponential decay toward a fundamentals-heavy
   floor (config.STACK_EXTRAPOLATION). Two months out, that extrapolation, not
   data, sets the polls-vs-fundamentals split; the curve is plotted so the
   assumption is visible.
4. The blended distribution is the stacking mixture: mean = sum w_k m_k,
   idiosyncratic variance = sum w_k (s_k^2 + (m_k - mean)^2). Shared errors
   (national, polling, state) are NOT folded in here; script 13 draws them
   once per simulation so they stay correlated across races.

Run with --refit to recompute the stacking table (otherwise a cached
stacking_weights.parquet is reused; the refit takes a few minutes).

Outputs
-------
stacking_weights.parquet  horizon_days, subset, component, weight, n_rows
blend_2026.parquet        per race: component means/sds, weights, blend_margin, blend_sd_idio, p_dem_marginal
figures/stacking_weights.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from utils import get_logger, load_meta, load_stage, save_stage, worst_provenance  # noqa: E402
from modellib import BacktestContext, blend_rows, marginal_win_prob, stacking_weights  # noqa: E402

log = get_logger("12_blend")
HORIZONS = config.STACK_HORIZONS_ESTIMATED
STACK_FILE = config.DATA_PROCESSED / "stacking_weights.parquet"


def fit_stacks() -> pd.DataFrame:
    ctx = BacktestContext()
    rows = []
    comps_all = {}
    for h in HORIZONS:
        frames = [ctx.components(c, h) for c in config.BACKTEST_CYCLES]
        df = pd.concat(frames, ignore_index=True)
        comps_all[h] = df
        polled = df[df.poll_margin.notna()]
        w, tab = stacking_weights(polled, {"hier": ("hier_margin", "hier_sd"), "fund": ("fund_margin", "fund_sd_total")})
        log.info("h=%3d polled (n=%d): %s", h, len(polled), {k: round(v, 3) for k, v in w.items()})
        for k, v in w.items():
            rows.append({"horizon_days": h, "subset": "polled_all_cycles", "component": k, "weight": v, "n_rows": len(polled)})
        rated = df[df.rating_margin.notna()]
        if len(rated) >= 40:
            for name, sub in [("polled_rated", rated[rated.poll_margin.notna()]), ("unpolled_rated", rated[rated.poll_margin.isna()])]:
                comps = {"hier": ("hier_margin", "hier_sd"), "fund": ("fund_margin", "fund_sd_total"), "rating": ("rating_margin", "rating_sd")}
                if len(sub) < 40:
                    continue
                w3, _ = stacking_weights(sub, comps)
                log.info("h=%3d %s (n=%d): %s", h, name, len(sub), {k: round(v, 3) for k, v in w3.items()})
                for k, v in w3.items():
                    rows.append({"horizon_days": h, "subset": name, "component": k, "weight": v, "n_rows": len(sub)})
    return pd.DataFrame(rows)


def weights_for_2026(stack: pd.DataFrame, horizon: int, polled: np.ndarray, has_rating: np.ndarray) -> pd.DataFrame:
    def curve(subset, comp):
        s = stack[(stack.subset == subset) & (stack.component == comp)].sort_values("horizon_days")
        if s.empty:
            return None
        hmax = s.horizon_days.max()
        if horizon <= hmax:
            return float(np.interp(horizon, s.horizon_days, s.weight))
        # beyond the archive: decay the poll-informed weight toward the floor
        w_last = float(s.weight.iloc[-1])
        ex = config.STACK_EXTRAPOLATION
        if comp == "hier":
            return ex["floor_hier"] + (w_last - ex["floor_hier"]) * np.exp(-(horizon - hmax) / ex["tau_days"])
        return w_last   # non-poll components keep their ratio; renormalised later
    w_hier_polled = curve("polled_all_cycles", "hier")
    w_hier_polled = 0.7 if w_hier_polled is None else w_hier_polled
    # ratio fund:rating from the 2018 three-way stack (horizon-invariant)
    f3, r3 = curve("polled_rated", "fund"), curve("polled_rated", "rating")
    if f3 is None or r3 is None or (f3 + r3) == 0:
        rating_share_polled = 0.5
    else:
        rating_share_polled = r3 / (f3 + r3)
    uh, uf, ur = curve("unpolled_rated", "hier"), curve("unpolled_rated", "fund"), curve("unpolled_rated", "rating")
    if uh is None:
        uh, uf, ur = 0.3, 0.3, 0.4
    out = pd.DataFrame(index=range(len(polled)))
    out["w_hier"] = np.where(polled, w_hier_polled, uh)
    rest = 1 - out["w_hier"]
    out["w_rating"] = np.where(polled, rest * rating_share_polled, ur)
    out["w_fund"] = np.where(polled, rest * (1 - rating_share_polled), uf)
    # no rating -> its weight goes to fundamentals
    out.loc[~has_rating, "w_fund"] += out.loc[~has_rating, "w_rating"]
    out.loc[~has_rating, "w_rating"] = 0.0
    tot = out[["w_hier", "w_fund", "w_rating"]].sum(1)
    out[["w_hier", "w_fund", "w_rating"]] = out[["w_hier", "w_fund", "w_rating"]].div(tot, axis=0)
    return out


def main(refit: bool = False):
    if refit or not STACK_FILE.exists():
        stack = fit_stacks()
        save_stage(stack, "stacking_weights", "mirror", {"horizons": HORIZONS})
    else:
        stack = load_stage("stacking_weights")
        log.info("using cached stacking weights (pass --refit to recompute)")

    hier = load_stage("hierarchical_estimates_2026")
    fund = load_stage("fundamentals_estimates_2026")
    rat = load_stage("ratings_estimates_2026")
    natf = load_stage("fundamentals_national").iloc[0]
    nat = load_stage("national_environment").iloc[0]
    df = hier.merge(fund.drop(columns=["office"]), on="race_id").merge(rat, on="race_id", how="left")
    df["nat_fund_sd"] = float(natf.nat_fund_sd_pred)
    h = config.DAYS_TO_ELECTION
    w = weights_for_2026(stack, h, df.polled.values, df.rating_margin.notna().values)
    df = pd.concat([df.reset_index(drop=True), w], axis=1)
    df = blend_rows(df)
    df["p_dem_marginal"] = marginal_win_prob(df, poll_shock_sd=float(nat.poll_shock_sd))
    log.info("horizon %d days: mean weights polled hier/fund/rating = %.2f/%.2f/%.2f; unpolled = %.2f/%.2f/%.2f",
             h, *df[df.polled][["w_hier", "w_fund", "w_rating"]].mean(), *df[~df.polled][["w_hier", "w_fund", "w_rating"]].mean())
    prov = worst_provenance(*[load_meta(n).get("provenance", "unknown") for n in
                              ["hierarchical_estimates_2026", "fundamentals_estimates_2026", "ratings_estimates_2026"]])
    keep = ["race_id", "office", "state", "polled", "n_polls", "weak_lean", "hier_margin", "hier_sd", "hier_sd_idio",
            "poll_weight_in_hier", "fund_margin", "fund_sd_idio", "fund_nat_loading", "nat_fund_sd", "rating",
            "rating_margin", "rating_sd", "rating_pwin", "w_hier", "w_fund", "w_rating", "blend_margin",
            "blend_sd_idio", "p_dem_marginal"]
    save_stage(df[keep], "blend_2026", prov, {"horizon_days": h})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for subset, ls in [("polled_all_cycles", "-"), ("polled_rated", "--"), ("unpolled_rated", ":")]:
        s = stack[stack.subset == subset]
        for comp in s.component.unique():
            c = s[s.component == comp].sort_values("horizon_days")
            ax.plot(c.horizon_days, c.weight, ls, marker="o", label=f"{subset}: {comp}")
    # extrapolated poll-informed weight beyond the archive (assumption, not data)
    s_h = stack[(stack.subset == "polled_all_cycles") & (stack.component == "hier")].sort_values("horizon_days")
    if len(s_h):
        hmax, w_last = int(s_h.horizon_days.max()), float(s_h.weight.iloc[-1])
        ex = config.STACK_EXTRAPOLATION
        hh = np.arange(hmax, 151)
        ax.plot(hh, ex["floor_hier"] + (w_last - ex["floor_hier"]) * np.exp(-(hh - hmax) / ex["tau_days"]),
                "k--", lw=1, label="polled hier: extrapolated (assumption)")
    ax.axvline(h, color="k", lw=.8, alpha=.5)
    ax.set_xlabel("days before election")
    ax.set_ylabel("stacking weight")
    ax.set_title("LOO stacking weights by horizon (backtest cycles)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "stacking_weights.png", dpi=120)
    print(df.groupby("office")[["blend_margin", "blend_sd_idio", "p_dem_marginal"]].describe().T.round(2))


if __name__ == "__main__":
    main(refit="--refit" in sys.argv)
