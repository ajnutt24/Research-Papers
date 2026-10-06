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

# Select the fastest available PyTensor backend BEFORE pymc is imported.
import sys as _sys
from pathlib import Path as _Path
try:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
except NameError:
    _sys.path.insert(0, str(_Path.cwd()))
import perf as _perf  # noqa: E402,F401

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
sys.path.insert(0, str(_ROOT / "model"))
import config  # noqa: E402
from utils import get_logger, load_meta, load_stage, save_stage, worst_provenance  # noqa: E402
from modellib import (BacktestContext, blend_rows, marginal_win_prob,  # noqa: E402
                      project_rating_onto_band, stacking_weights)

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
        # A stack is only recorded when the horizon has enough races to support
        # the number of weights it is fitting. See
        # config.STACK_MIN_ROWS_PER_COMPONENT: the archive thins out sharply with
        # the horizon, and a simplex fitted on too few races lands on its
        # boundary, which then becomes the anchor for the beyond-archive
        # extrapolation. A dropped horizon simply ends the curve earlier.
        min_per = int(getattr(config, "STACK_MIN_ROWS_PER_COMPONENT", 50))

        def record(name, sub, comps):
            need = min_per * len(comps)
            if len(sub) < need:
                log.info("h=%3d %s: %d race(s), below the %d needed for %d weight(s); "
                         "this horizon is left out of the curve",
                         h, name, len(sub), need, len(comps))
                return
            ww, _ = stacking_weights(sub, comps)
            log.info("h=%3d %s (n=%d): %s", h, name, len(sub),
                     {k: round(v, 3) for k, v in ww.items()})
            for k, v in ww.items():
                rows.append({"horizon_days": h, "subset": name, "component": k,
                             "weight": v, "n_rows": len(sub)})

        two = {"hier": ("hier_margin", "hier_sd"), "fund": ("fund_margin", "fund_sd_total")}
        three = {**two, "rating": ("rating_margin", "rating_sd")}
        record("polled_all_cycles", df[df.poll_margin.notna()], two)
        rated = df[df.rating_margin.notna()]
        record("polled_rated", rated[rated.poll_margin.notna()], three)
        record("unpolled_rated", rated[rated.poll_margin.isna()], three)
    return pd.DataFrame(rows)


def weights_for_2026(stack: pd.DataFrame, horizon: int, polled: np.ndarray, has_rating: np.ndarray,
                     lean_state_fallback: np.ndarray | None = None) -> pd.DataFrame:
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
    f3, r3 = curve("polled_rated", "fund"), curve("polled_rated", "rating")
    h3 = curve("polled_rated", "hier")
    mode = getattr(config, "POLLED_RATING_SHARE_MODE", "stack")

    # How the non-poll slice on polled races is split between fundamentals and
    # ratings. The old construction took the slice's SIZE from the two-way
    # polled_all_cycles stack and only its RATIO from the three-way one, as
    # r/(f+r). The three-way fit pins the fundamentals at exactly 0.000 on
    # polled races at every horizon, so that expression returns exactly 1.0 and
    # says nothing about how well determined it is; before archived ratings
    # existed the rating weight was the pinned one and it returned exactly 0.0.
    # The whole slice therefore flipped from one component to the other, worth
    # 1.7 points of Senate probability, on which of two boundary-clipped weights
    # happened to be the zero. See config.POLLED_RATING_SHARE_MODE.
    if mode == "stack" and None not in (h3, f3, r3) and (h3 + f3 + r3) > 0:
        # Use the three-way stack as fitted. Internally consistent: the size and
        # the split come from the same fit, on the same races.
        tot3 = h3 + f3 + r3
        wh_p, wf_p, wr_p = h3 / tot3, f3 / tot3, r3 / tot3
        log.info("polled weights from the three-way polled_rated stack: "
                 "hier/fund/rating = %.3f/%.3f/%.3f", wh_p, wf_p, wr_p)
    else:
        if None in (f3, r3) or (f3 + r3) == 0:
            share = 0.5
        else:
            share = r3 / (f3 + r3)
            n3 = stack.loc[(stack.subset == "polled_rated") & (stack.component == "rating"),
                           "n_rows"]
            n3 = float(n3.iloc[-1]) if len(n3) else 0.0
            k = float(getattr(config, "POLLED_RATING_SHARE_PRIOR_ROWS", 100))
            # Shrink the proportion toward an even split, Beta-style, so a ratio
            # pinned at a boundary cannot swing the answer on its own.
            share = (share * n3 + 0.5 * k) / (n3 + k)
            log.info("fund:rating split on polled races shrunk toward 0.5: "
                     "raw %.2f, n=%.0f, prior %.0f rows -> %.3f",
                     r3 / (f3 + r3), n3, k, share)
        wh_p = w_hier_polled
        wr_p = (1 - wh_p) * share
        wf_p = (1 - wh_p) * (1 - share)
    uh, uf, ur = curve("unpolled_rated", "hier"), curve("unpolled_rated", "fund"), curve("unpolled_rated", "rating")
    if uh is None:
        uh, uf, ur = 0.3, 0.3, 0.4
    out = pd.DataFrame(index=range(len(polled)))
    out["w_hier"] = np.where(polled, wh_p, uh)
    out["w_rating"] = np.where(polled, wr_p, ur)
    out["w_fund"] = np.where(polled, wf_p, uf)
    # no rating -> its weight goes to fundamentals
    out.loc[~has_rating, "w_fund"] += out.loc[~has_rating, "w_rating"]
    out.loc[~has_rating, "w_rating"] = 0.0
    # A state-fallback lean makes the fundamentals component district-blind, so
    # most of its weight belongs to the rating, which is not. The stacking
    # weights are fitted on cycles where the fundamentals did carry
    # district-level information, so they cannot know about this case.
    # See config.WEAK_LEAN_FUND_TO_RATING.
    if lean_state_fallback is not None:
        move = np.asarray(lean_state_fallback, dtype=bool) & np.asarray(has_rating, dtype=bool)
        if move.any():
            frac = float(config.WEAK_LEAN_FUND_TO_RATING)
            out.loc[move, "w_rating"] += out.loc[move, "w_fund"] * frac
            out.loc[move, "w_fund"] *= (1.0 - frac)
            log.info("weak-lean reweighting: %d race(s) move %.0f%% of the fundamentals weight "
                     "to their rating", int(move.sum()), 100 * frac)
    tot = out[["w_hier", "w_fund", "w_rating"]].sum(1)
    out[["w_hier", "w_fund", "w_rating"]] = out[["w_hier", "w_fund", "w_rating"]].div(tot, axis=0)
    return out


def main(refit: bool = False):
    REFERENCE = config.DATA_MANUAL / "reference" / "stacking_weights.parquet"
    if refit:
        stack = fit_stacks()
        save_stage(stack, "stacking_weights", "mirror", {"horizons": HORIZONS})
    elif STACK_FILE.exists():
        stack = load_stage("stacking_weights")
        log.info("using cached stacking weights (pass --refit to recompute)")
    elif REFERENCE.exists():
        # Shipped precomputed weights: these depend only on the historical
        # backtest cycles, not on this cycle's polls, so re-fitting them on
        # every fresh install would cost ~37 PyMC fits for an identical answer.
        stack = pd.read_parquet(REFERENCE)
        save_stage(stack, "stacking_weights", "mirror",
                   {"horizons": HORIZONS, "source": "shipped reference table"})
        log.info("seeded stacking weights from the shipped reference table "
                 "(pass --refit to recompute from scratch)")
    else:
        stack = fit_stacks()
        save_stage(stack, "stacking_weights", "mirror", {"horizons": HORIZONS})

    hier = load_stage("hierarchical_estimates_2026")
    fund = load_stage("fundamentals_estimates_2026")
    rat = load_stage("ratings_estimates_2026")
    natf = load_stage("fundamentals_national").iloc[0]
    nat = load_stage("national_environment").iloc[0]
    df = hier.merge(fund.drop(columns=["office"]), on="race_id").merge(rat, on="race_id", how="left")
    df["nat_fund_sd"] = float(natf.nat_fund_sd_pred)
    h = config.DAYS_TO_ELECTION
    w = weights_for_2026(stack, h, df.polled.values, df.rating_margin.notna().values,
                         lean_state_fallback=(df.lean_state_fallback.values
                                              if "lean_state_fallback" in df else None))
    df = pd.concat([df.reset_index(drop=True), w], axis=1)
    df = project_rating_onto_band(df)
    if "rating_redundant" in df:
        log.info("rating band projection: %d of %d rated race(s) are redundant (the other "
                 "components already sit inside the rating's tier band, so the component "
                 "contributes nothing)", int(df.rating_redundant.sum()),
                 int(df.rating_margin.notna().sum()))
    df = blend_rows(df)
    df["p_dem_marginal"] = marginal_win_prob(df, poll_shock_sd=float(nat.poll_shock_sd))
    log.info("horizon %d days: mean weights polled hier/fund/rating = %.2f/%.2f/%.2f; unpolled = %.2f/%.2f/%.2f",
             h, *df[df.polled][["w_hier", "w_fund", "w_rating"]].mean(), *df[~df.polled][["w_hier", "w_fund", "w_rating"]].mean())
    prov = worst_provenance(*[load_meta(n).get("provenance", "unknown") for n in
                              ["hierarchical_estimates_2026", "fundamentals_estimates_2026", "ratings_estimates_2026"]])
    keep = ["race_id", "office", "state", "polled", "n_polls", "weak_lean",
            "lean_state_fallback", "hier_margin", "hier_sd", "hier_sd_idio",
            "poll_weight_in_hier", "fund_margin", "fund_sd_idio", "fund_nat_loading", "nat_fund_sd", "rating",
            "rating_margin", "rating_sd", "rating_pwin", "w_hier", "w_fund", "w_rating", "blend_margin",
            "blend_sd_idio", "p_dem_marginal", "exp_edge_pts", "main_party", "three_way",
            "caucus_prob_dem",
            # Kept so the band projection is auditable from the output alone:
            # what the rating said before projection, and whether it ended up
            # contributing anything at all.
            "rating_margin_raw", "rating_sd_raw", "rating_redundant", "rating_lo", "rating_hi"]
    for c in ["exp_edge_pts", "main_party", "three_way", "caucus_prob_dem",
              "rating_margin_raw", "rating_sd_raw", "rating_redundant", "rating_lo", "rating_hi"]:
        if c not in df:
            df[c] = {"exp_edge_pts": 0.0, "main_party": "D",
                     "three_way": False, "caucus_prob_dem": 1.0,
                     "rating_margin_raw": np.nan, "rating_sd_raw": np.nan,
                     "rating_redundant": False,
                     "rating_lo": np.nan, "rating_hi": np.nan}[c]
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
