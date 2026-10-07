"""
16_noise_floor_sweep.py
=======================
Is RACE_NOISE_FLOOR_SD = 4.0 the right value, or just inherited?

The floor is the one term in the race-level variance budget that is an
assumption rather than an estimate. Every other term is measured:
`blend_sd_idio` from the components, `nat_fund_sd` from the fundamentals
regression, `poll_shock_sd` from the 1998-2022 archive of final-poll misses,
`STATE_SHOCK_SD` from state-level residual correlation. The floor is a
hard-coded statement that no race is ever more certain than +/- 4 points, and
in well-polled competitive races it is frequently the largest single term.

Method
------
The floor enters the pipeline in exactly one place, `marginal_win_prob`:

    var = blend_sd_idio^2
        + (w_nat * fund_nat_loading * nat_fund_sd)^2
        + (w_poll * poll_shock_sd)^2
        + state_sd^2
        + noise_floor^2

Nothing upstream depends on it: `stacking_weights` and `blend_rows` never see
it, so `blend_margin` and `blend_sd_idio` are floor-invariant. That means saved
per-race rows can be re-scored at any floor value without refitting anything,
and the sweep is exact rather than approximate.

Brier score is a strictly proper scoring rule, so it is the correct target: it
is minimised by the honestly calibrated probability and penalises both
overconfidence (floor too small) and underconfidence (floor too large).

THE HORIZON IS THE WHOLE STORY. The floor represents irreducible uncertainty
that has not yet resolved, so what it should be depends on how far out the
forecast is. Script 14 scores election eve (horizon 1), the point of MINIMUM
unresolved uncertainty, which biases an eve-only sweep toward a low floor. Set
FLOOR_SWEEP_HORIZONS to rebuild the components at several horizons and sweep
each one; the answer flips between horizon 7 and horizon 14.

The live forecast cannot be tested directly. FiveThirtyEight's raw-polls
archive is a FINAL-polls archive, built to rate pollsters on their last poll,
so race coverage collapses with horizon: 95-187 races per cycle at horizon 1,
but 3-1-3-3 races at horizon 27. Horizon 14 is the longest horizon with enough
coverage to fit stacking weights, and the live forecast runs at 27 days. The
floor therefore cannot be calibrated at the horizon it is actually used at, and
the defensible inference is directional, from how the optimum MOVES with
horizon, not from any single optimum.

Two optima are reported and the difference matters:

* **In-sample.** The floor minimising Brier over all four cycles. Optimistic by
  construction.
* **Leave-one-cycle-out (LOCO).** For each cycle, pick the floor on the other
  three and score the held-out cycle at that floor. The only number that
  answers "would changing the floor have helped out of sample". Four cycles
  make this a weak test and the script says so.

A boundary optimum (floor = 0, the hard lower bound) is reported as a boundary
solution, not as an estimate: it means the metric wants maximum sharpness at
that horizon, not that the parameter has been identified.

Outputs (outputs/)
------------------
noise_floor_sweep.csv              floor x (cycle, office) grid, horizon 1
noise_floor_summary.csv            headline table: in-sample and LOCO optima
noise_floor_loco.csv               per-cycle leave-one-cycle-out detail
noise_floor_by_office.csv          per-office optima
noise_floor_sweep_by_horizon.csv   floor x cycle x horizon grid (rebuild mode)
figures/noise_floor_sweep.png      Brier vs floor, overall and per office

Usage
-----
    python3 validation/16_noise_floor_sweep.py
        Fast path: re-score outputs/backtest_races.csv (horizon 1 only).

    FLOOR_SWEEP_HORIZONS=1,7,14 python3 validation/16_noise_floor_sweep.py
        Rebuild components at each horizon and sweep each. Refits the national
        and seat models (cached across horizons) and the stacking weights per
        horizon, so it takes a few minutes. This is the mode that answers the
        question; the fast path alone is misleading.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _find_project_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        pass
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        for d in (cand, cand / "midterms-2026"):
            if (d / "config.py").is_file() and (d / "utils.py").is_file():
                return d
    raise RuntimeError("Cannot find the project folder (config.py + utils.py).")


_ROOT = _find_project_root()
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "model"))
import config  # noqa: E402
from utils import get_logger, load_stage  # noqa: E402

log = get_logger("noise_floor_sweep")

GRID = np.round(np.arange(0.0, 8.01, 0.25), 2)
EPS = 1e-6


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def logloss(p, y) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def variance_parts(df: pd.DataFrame, poll_shock_sd: float,
                   state_sd: float = config.STATE_SHOCK_SD) -> np.ndarray:
    """Everything in the variance budget EXCEPT the floor.

    Mirrors modellib.marginal_win_prob term for term; the floor is added by the
    caller so one pass over the data serves the whole grid.
    """
    w_poll = (df["w_hier"] * df["poll_weight_in_hier"]).fillna(0.0).to_numpy(float)
    w_nat = 1.0 - w_poll
    return (df["blend_sd_idio"].to_numpy(float) ** 2
            + (w_nat * df["fund_nat_loading"].to_numpy(float) * df["nat_fund_sd"].to_numpy(float)) ** 2
            + (w_poll * float(poll_shock_sd)) ** 2
            + float(state_sd) ** 2)


def p_at_floor(margin: np.ndarray, var_ex_floor: np.ndarray, floor: float) -> np.ndarray:
    from scipy import stats
    return stats.norm.cdf(margin / np.sqrt(var_ex_floor + floor ** 2))


def calibration_slope(p: np.ndarray, y: np.ndarray) -> float:
    """Logistic recalibration slope on the forecast logit.

    Slope 1 = well calibrated. Slope < 1 = overconfident (probabilities too
    extreme for the evidence). Slope > 1 = underconfident. Reported as a
    diagnostic only; the floor is chosen on Brier, which is proper.
    """
    from scipy.optimize import minimize
    z = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))

    def nll(th):
        a, b = th
        q = 1 / (1 + np.exp(-(a + b * z)))
        q = np.clip(q, EPS, 1 - EPS)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))

    r = minimize(nll, [0.0, 1.0], method="Nelder-Mead")
    return float(r.x[1])


def argmin_floor(sub: pd.DataFrame, metric: str = "brier") -> float:
    """Floor minimising `metric`, averaged over cycles so no cycle dominates
    on race count alone (House 2022 has ~435 rows, Senate 2014 has ~36)."""
    per_cycle = sub.groupby(["floor", "cycle"])[metric].mean().reset_index()
    curve = per_cycle.groupby("floor")[metric].mean()
    return float(curve.idxmin())



def sweep_horizons(horizons: list[int]) -> None:
    """Rebuild the backtest components at each horizon and sweep the floor there.

    This is the mode that answers the question. The eve-only sweep cannot: the
    floor covers uncertainty that has not yet resolved, and election eve is the
    point where the least of it remains.
    """
    import importlib.util
    from modellib import (BacktestContext, blend_rows,  # noqa: F401
                          project_rating_onto_band)

    spec = importlib.util.spec_from_file_location("bt14", _ROOT / "validation" / "14_backtest.py")
    bt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt)

    ps = float(load_stage("national_environment").iloc[0].poll_shock_sd)
    ctx = BacktestContext()
    cur = float(config.RACE_NOISE_FLOOR_SD)
    out, optima = [], []
    for h in horizons:
        comps = {c: ctx.components(c, h) for c in config.BACKTEST_CYCLES}
        scored = []
        for c, df in comps.items():
            try:
                w = bt.loo_weights(comps, c)
            except ValueError as e:
                log.error("horizon %d, cycle %d: cannot fit stacking weights (%s). "
                          "The poll archive is too thin this far out; skipping horizon.", h, c, e)
                scored = []
                break
            scored.append(blend_rows(project_rating_onto_band(bt.apply_weights(df, w))))
        if not scored:
            continue
        res = pd.concat(scored, ignore_index=True).dropna(
            subset=["blend_margin", "blend_sd_idio", "dem_won"])
        v = variance_parts(res, ps)
        m = res.blend_margin.to_numpy(float)
        y = res.dem_won.to_numpy(float)
        for f in GRID:
            p = p_at_floor(m, v, f)
            res["_p"] = p
            for cyc, g in res.groupby("cycle"):
                out.append({"horizon": h, "floor": f, "cycle": int(cyc), "n": len(g),
                            "brier": brier(g["_p"], g.dem_won), "log_loss": logloss(g["_p"], g.dem_won)})
        sub = pd.DataFrame([r for r in out if r["horizon"] == h])
        curve = sub.groupby(["floor", "cycle"]).brier.mean().reset_index().groupby("floor").brier.mean()
        best = float(curve.idxmin())
        boundary = best <= GRID[0] + 1e-9 or best >= GRID[-1] - 1e-9
        # LOCO at this horizon
        loco = []
        for cyc in sorted(sub.cycle.unique()):
            tr = (sub[sub.cycle != cyc].groupby(["floor", "cycle"]).brier.mean()
                  .reset_index().groupby("floor").brier.mean())
            te = sub[sub.cycle == cyc].set_index("floor").brier
            pick = float(tr.idxmin())
            loco.append({"cycle": int(cyc), "pick": pick, "b_pick": float(te.loc[pick]),
                         "b_cur": float(te.loc[cur]), "b_zero": float(te.loc[0.0]),
                         "own_best": float(te.idxmin())})
        L = pd.DataFrame(loco)
        optima.append({"horizon": h, "n_races": len(res), "best_floor": best,
                       "boundary_solution": boundary, "brier_at_best": float(curve.loc[best]),
                       "brier_at_current": float(curve.loc[cur]), "brier_at_zero": float(curve.loc[0.0]),
                       "loco_brier_picked": float(L.b_pick.mean()),
                       "loco_brier_current": float(L.b_cur.mean()),
                       "loco_brier_zero": float(L.b_zero.mean()),
                       "own_best_min": float(L.own_best.min()), "own_best_max": float(L.own_best.max())})
        log.info("=" * 72)
        log.info("horizon %d days: %d races, %d polled", h, len(res), int(res.poll_margin.notna().sum()))
        log.info("  in-sample best floor %.2f (brier %.5f)%s | current %.2f (brier %.5f) | "
                 "zero (brier %.5f)", best, curve.loc[best],
                 "  [BOUNDARY SOLUTION, not an estimate]" if boundary else "",
                 cur, curve.loc[cur], curve.loc[0.0])
        log.info("  per-cycle own-best floors %.2f to %.2f", L.own_best.min(), L.own_best.max())
        log.info("  LOCO brier: picked %.5f | current %.5f | zero %.5f  -> dropping to zero costs %+.5f",
                 L.b_pick.mean(), L.b_cur.mean(), L.b_zero.mean(), L.b_zero.mean() - L.b_cur.mean())

    if not out:
        raise SystemExit("no horizon produced a scorable set; check archive coverage")
    pd.DataFrame(out).to_csv(config.OUTPUTS / "noise_floor_sweep_by_horizon.csv", index=False)
    O = pd.DataFrame(optima)
    O.to_csv(config.OUTPUTS / "noise_floor_optima_by_horizon.csv", index=False)
    log.info("=" * 72)
    log.info("VERDICT: the optimum moves with horizon:")
    for r in O.itertuples():
        log.info("  h=%2dd -> best floor %.2f%s", r.horizon, r.best_floor,
                 "  (boundary)" if r.boundary_solution else "")
    if len(O) > 1 and O.best_floor.iloc[-1] > O.best_floor.iloc[0]:
        log.info("  The optimum RISES with horizon. The live forecast runs at %d days, beyond",
                 (config.ELECTION_DATE - config.FORECAST_ASOF).days)
        log.info("  the longest testable horizon, so a low floor fitted on election eve does")
        log.info("  not transfer. Keeping RACE_NOISE_FLOOR_SD = %.1f is the supported choice.", cur)
    log.info("wrote noise_floor_sweep_by_horizon.csv, noise_floor_optima_by_horizon.csv")


def main():
    env = os.environ.get("FLOOR_SWEEP_HORIZONS", "").strip()
    if env:
        hs = sorted({int(x) for x in env.replace(" ", "").split(",") if x})
        log.info("rebuild mode: sweeping the floor at horizons %s", hs)
        return sweep_horizons(hs)

    races_path = config.OUTPUTS / "backtest_races.csv"
    if not races_path.exists():
        raise SystemExit(f"{races_path} not found. Run validation/14_backtest.py first.")
    res = pd.read_csv(races_path)
    poll_shock = float(load_stage("national_environment").iloc[0].poll_shock_sd)

    need = ["blend_margin", "blend_sd_idio", "w_hier", "poll_weight_in_hier",
            "fund_nat_loading", "nat_fund_sd", "dem_won", "cycle", "office"]
    missing = [c for c in need if c not in res.columns]
    if missing:
        raise SystemExit(f"backtest_races.csv is missing {missing}")
    res = res.dropna(subset=["blend_margin", "blend_sd_idio", "dem_won"]).copy()

    # Reproduce the shipped probability as a control: at floor = the configured
    # value this must match p_blend from script 14 to floating-point tolerance.
    var_ex = variance_parts(res, poll_shock)
    margin = res["blend_margin"].to_numpy(float)
    y = res["dem_won"].to_numpy(float)
    if "p_blend" in res.columns:
        check = p_at_floor(margin, var_ex, config.RACE_NOISE_FLOOR_SD)
        gap = float(np.nanmax(np.abs(check - res["p_blend"].to_numpy(float))))
        log.info("control: reproduced shipped p_blend to max abs diff %.2e", gap)
        if gap > 1e-8:
            log.warning("re-scoring does NOT reproduce script 14's probabilities "
                        "(max diff %.2e); the variance budget here has drifted "
                        "from modellib.marginal_win_prob", gap)

    rows = []
    for floor in GRID:
        p = p_at_floor(margin, var_ex, floor)
        res["_p"] = p
        for (c, office), g in res.groupby(["cycle", "office"]):
            rows.append({"floor": floor, "cycle": int(c), "office": office, "n": len(g),
                         "brier": brier(g["_p"], g.dem_won), "log_loss": logloss(g["_p"], g.dem_won)})
        for c, g in res.groupby("cycle"):
            rows.append({"floor": floor, "cycle": int(c), "office": "All", "n": len(g),
                         "brier": brier(g["_p"], g.dem_won), "log_loss": logloss(g["_p"], g.dem_won)})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(config.OUTPUTS / "noise_floor_sweep.csv", index=False)

    allsub = sweep[sweep.office == "All"]
    cur = float(config.RACE_NOISE_FLOOR_SD)
    curve = allsub.groupby("floor")["brier"].mean()
    best_in = argmin_floor(allsub, "brier")
    best_ll = argmin_floor(allsub, "log_loss")

    log.info("=" * 72)
    log.info("mean Brier across %d cycles, all offices", allsub.cycle.nunique())
    for f in [0.0, 1.0, 2.0, 3.0, 3.5, cur, 4.5, 5.0, 6.0, 7.0, 8.0]:
        if f in curve.index:
            mark = "  <= current" if f == cur else ("  <= in-sample best" if f == best_in else "")
            log.info("  floor %4.2f   brier %.5f%s", f, curve.loc[f], mark)
    log.info("in-sample optimum: floor %.2f (brier %.5f) vs current %.2f (brier %.5f), "
             "improvement %.5f", best_in, curve.loc[best_in], cur, curve.loc[cur],
             curve.loc[cur] - curve.loc[best_in])
    log.info("in-sample optimum on log loss: floor %.2f", best_ll)

    # Leave-one-cycle-out: choose the floor without seeing the cycle it scores.
    loco = []
    for c in sorted(allsub.cycle.unique()):
        train = allsub[allsub.cycle != c]
        pick = argmin_floor(train, "brier")
        test = allsub[allsub.cycle == c].set_index("floor")
        loco.append({"held_out_cycle": int(c), "floor_from_other_cycles": pick,
                     "brier_at_picked": float(test.loc[pick, "brier"]),
                     "brier_at_current": float(test.loc[cur, "brier"]),
                     "brier_at_own_best": float(test["brier"].min()),
                     "own_best_floor": float(test["brier"].idxmin())})
    loco = pd.DataFrame(loco)
    log.info("=" * 72)
    log.info("leave-one-cycle-out (floor chosen on the other three):")
    for r in loco.itertuples():
        log.info("  %d: picked %.2f -> brier %.5f | current %.2f -> %.5f | "
                 "own best %.2f -> %.5f", r.held_out_cycle, r.floor_from_other_cycles,
                 r.brier_at_picked, cur, r.brier_at_current, r.own_best_floor, r.brier_at_own_best)
    d = float((loco.brier_at_current - loco.brier_at_picked).mean())
    log.info("LOCO mean Brier: picked %.5f vs current %.5f (out-of-sample gain %.5f)",
             loco.brier_at_picked.mean(), loco.brier_at_current.mean(), d)
    log.info("per-cycle own-best floors: %s -> spread %.2f pts",
             list(loco.own_best_floor), loco.own_best_floor.max() - loco.own_best_floor.min())

    # Per office: is one global floor even the right shape?
    log.info("=" * 72)
    log.info("per-office optima (a wide spread means the floor should not be global):")
    off_rows = []
    for office, g in sweep[sweep.office != "All"].groupby("office"):
        b = argmin_floor(g, "brier")
        gc = g.groupby("floor")["brier"].mean()
        off_rows.append({"office": office, "n_races": int(g[g.floor == b]["n"].sum()),
                         "best_floor": b, "brier_at_best": float(gc.loc[b]),
                         "brier_at_current": float(gc.loc[cur])})
        log.info("  %-9s n=%4d  best floor %4.2f (brier %.5f)  current %.2f (brier %.5f)",
                 office, int(g[g.floor == b]["n"].sum()), b, gc.loc[b], cur, gc.loc[cur])
    off = pd.DataFrame(off_rows)

    # Calibration diagnostic at the two candidate floors.
    log.info("=" * 72)
    for f in sorted({cur, best_in}):
        p = p_at_floor(margin, var_ex, f)
        fav_p = np.where(p >= .5, p, 1 - p)
        fav_won = np.where(p >= .5, y, 1 - y)
        m = (fav_p >= .6) & (fav_p < .8)
        log.info("floor %.2f: calibration slope %.3f | favourite-side 60-80%% bucket "
                 "(mean %.0f%%): %d races, favourite won %.0f%%",
                 f, calibration_slope(p, y), 100 * fav_p[m].mean(), int(m.sum()),
                 100 * fav_won[m].mean())

    summary = pd.DataFrame([{"current_floor": cur, "in_sample_best_floor": best_in,
                             "in_sample_brier_current": float(curve.loc[cur]),
                             "in_sample_brier_best": float(curve.loc[best_in]),
                             "loco_mean_brier_current": float(loco.brier_at_current.mean()),
                             "loco_mean_brier_picked": float(loco.brier_at_picked.mean()),
                             "loco_out_of_sample_gain": d,
                             "per_cycle_best_floor_spread": float(loco.own_best_floor.max() - loco.own_best_floor.min()),
                             "n_cycles": int(allsub.cycle.nunique()),
                             "n_races": int(len(res))}])
    summary.to_csv(config.OUTPUTS / "noise_floor_summary.csv", index=False)
    loco.to_csv(config.OUTPUTS / "noise_floor_loco.csv", index=False)
    off.to_csv(config.OUTPUTS / "noise_floor_by_office.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot(curve.index, curve.values, lw=2, color="#2b6cb0")
    ax.axvline(cur, ls="--", color="#718096", label=f"current {cur:.1f}")
    ax.axvline(best_in, ls=":", color="#c53030", label=f"in-sample best {best_in:.2f}")
    ax.set_xlabel("RACE_NOISE_FLOOR_SD (pts)"); ax.set_ylabel("mean Brier (cycle-averaged)")
    ax.set_title("All offices"); ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for office, g in sweep[sweep.office != "All"].groupby("office"):
        gc = g.groupby("floor")["brier"].mean()
        ax.plot(gc.index, gc.values / gc.max(), lw=1.6, label=office)
    ax.axvline(cur, ls="--", color="#718096")
    ax.set_xlabel("RACE_NOISE_FLOOR_SD (pts)"); ax.set_ylabel("Brier / own max (scaled)")
    ax.set_title("By office (scaled to compare shapes)"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    (config.OUTPUTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(config.OUTPUTS / "figures" / "noise_floor_sweep.png", dpi=150)
    log.info("wrote noise_floor_sweep.csv, noise_floor_summary.csv, "
             "noise_floor_loco.csv, noise_floor_by_office.csv, figures/noise_floor_sweep.png")


if __name__ == "__main__":
    main()
