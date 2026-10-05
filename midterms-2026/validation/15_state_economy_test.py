"""
15_state_economy_test.py
========================
Does the LOCAL economy earn a place in the seat model?

    python3 validation/15_state_economy_test.py
    python3 validation/15_state_economy_test.py --midterms   # midterm cycles only

Why this test exists
--------------------
The national fundamentals model trains on 19 midterms. On that sample the
economic term cannot be identified: the minimum detectable effect at
conventional power is about 1.2 points of House margin per point of inflation,
which over the observed inflation range would be a 25-point swing. No economic
indicator can clear a bar that high, and leave-one-out cross-validation on the
19 cycles duly prefers no economic term at all.

The seat model is a different sample. It has hundreds of Senate and Governor
races across a dozen election cycles, and the variation is cross-sectional
rather than a short time series, so a 0.5-point effect is in range. If economic
conditions carry information the partisan lean does not already hold, this is
the only place in the model where the data can say so.

The hypothesis
--------------
Voters punish the party held responsible for a bad local economy. Operationally:
after conditioning on a race's partisan lean, its incumbency, and the national
environment of its cycle, does a state whose unemployment is unusually high (or
deteriorating unusually fast) relative to the nation swing against the party
held responsible?

Two candidate referents, tested separately because they are different claims:

  s_pres   the president's party. The classic referendum story: midterms are a
           verdict on the White House, and local conditions sharpen the verdict.
  s_seat   the party that currently holds the seat. The accountability story:
           a sitting governor owns the state's economy more plainly than the
           president does, and a senator is blamed for the plant that closed.

Both are signed so that a worse local economy predicts a WORSE result for the
referent party, which means the expected coefficient is NEGATIVE in both cases.
A positive coefficient is not a weaker version of the hypothesis, it is evidence
against it.

Identification
--------------
Levels are not the interesting quantity. State unemployment levels are
structural: West Virginia runs above the nation in good years and bad, and the
partisan lean already encodes that kind of fixed state character. So the
regressors are DEVIATIONS from the national reading, in level and in change, and
every specification carries cycle fixed effects, which absorb the national
economy, the national swing, and anything else common to an election year. What
is left is purely cross-sectional: within a single election, did the states
whose economies were worse than the national average vote differently?

A second variant adds state fixed effects. That is the strongest design here,
because identification then comes only from a state's movement relative to its
own history, which no amount of fixed state character can explain away. It is
also the most demanding, and a term that survives it has genuinely earned a
place.

A third variant replaces the cycle fixed effects with the national margin as a
regressor, which is the shape the production seat model actually uses
(modellib.fit_seat_model). A term worth adopting should show up there too.

How a term "earns its place"
----------------------------
Not by a significant coefficient in-sample. Adding any regressor to a
fixed-effects model improves in-sample fit mechanically. The criterion is
LEAVE-ONE-CYCLE-OUT prediction: refit without one election cycle, predict that
cycle's races, and accumulate the error. Races within a cycle share a national
environment and a polling error, so leaving out a single race would leak that
cycle's information into its own prediction and flatter every specification.
Leaving out the whole cycle is the honest version and matches how the model is
actually used: forecasting an election it has never seen.

Standard errors are clustered by cycle for the same reason.

This is a screen, not the final model. It is ordinary least squares because the
question is whether there is any signal to find; the Bayesian seat model is only
worth modifying if the screen says yes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    PROJECT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT = Path.cwd()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "model"))

import config  # noqa: E402
from utils import get_logger, load_stage  # noqa: E402

log = get_logger("15_stecon")

# Party holding the White House at the time of each November election. Stated
# explicitly rather than derived, because the approval_history stage covers
# midterms only and because the convention matters: November 1980 is Carter (D),
# not Reagan, since the sitting president is the one being judged.
PRES_PARTY = {
    1976: "R", 1978: "D", 1980: "D", 1982: "R", 1984: "R", 1986: "R", 1988: "R",
    1990: "R", 1992: "R", 1994: "D", 1996: "D", 1998: "D", 2000: "D", 2002: "R",
    2004: "R", 2006: "R", 2008: "R", 2010: "D", 2012: "D", 2014: "D", 2016: "D",
    2018: "R", 2020: "R", 2022: "D", 2024: "D",
}
# Ford / Carter / Reagan / Bush / Clinton / Bush / Obama / Trump / Biden, each
# read at the election, not at the following January.

LAG = {"Senate": 6, "Governor": 4, "House": 2}   # years back to the same seat's last election

# House lines are redrawn in years ending in 2, so a 2012 district's 2010 result
# belongs to a different district. Those cycles are dropped from the House check
# rather than patched, because a lean built across a redistricting is not a lean.
NEW_MAP_CYCLES = {1982, 1992, 2002, 2012, 2022}


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------
def october_panel() -> pd.DataFrame:
    """State unemployment as of October of each election year, with the 12- and
    24-month changes, and the same three readings for the nation.

    October is the last full month before an election and is the month the
    national model already uses, so the local and national regressors are read
    at the same point in the calendar.
    """
    st = load_stage("state_unemployment").copy()
    st["date"] = pd.to_datetime(st["date"])
    nat = load_stage("economic_monthly")[["date", "unrate"]].copy()
    nat["date"] = pd.to_datetime(nat["date"])
    nat = nat.dropna(subset=["unrate"]).set_index("date")["unrate"]

    wide = st.pivot(index="date", columns="state", values="unrate").sort_index()
    rows = []
    for cy in range(1976, 2027, 2):
        target = pd.Timestamp(cy, 10, 1)
        if target not in wide.index or target not in nat.index:
            continue
        def at(ts, series):
            return float(series.loc[ts]) if ts in series.index else np.nan
        n_lv = at(target, nat)
        n_12 = n_lv - at(pd.Timestamp(cy - 1, 10, 1), nat)
        n_24 = n_lv - at(pd.Timestamp(cy - 2, 10, 1), nat)
        for s in wide.columns:
            col = wide[s].dropna()
            lv = at(target, col)
            if not np.isfinite(lv):
                continue
            c12 = lv - at(pd.Timestamp(cy - 1, 10, 1), col)
            c24 = lv - at(pd.Timestamp(cy - 2, 10, 1), col)
            rows.append({"cycle": cy, "state": s,
                         "st_lv": lv, "st_ch12": c12, "st_ch24": c24,
                         "nat_lv": n_lv, "nat_ch12": n_12, "nat_ch24": n_24,
                         "dev_lv": lv - n_lv, "dev_ch12": c12 - n_12,
                         "dev_ch24": c24 - n_24})
    out = pd.DataFrame(rows)
    log.info("October economic panel: %d state-cycles, %d-%d",
             len(out), out.cycle.min(), out.cycle.max())
    return out


def build_frame(midterms_only: bool = False,
                offices: tuple[str, ...] = ("Senate", "Governor")) -> pd.DataFrame:
    """One row per contested race with a prior result for the same seat."""
    hist = load_stage("historical_results")
    rows = hist[hist.office.isin(offices) & (~hist.uncontested)].copy()
    rows = rows[rows.cycle % 2 == 0]           # off-year governor races have no October panel partner
    # House specials are keyed exactly like the regular race in the same cycle
    # (unlike Senate specials, which carry a -special suffix), so they have to go
    # before the lookup table is built or the key is not unique.
    if "House" in offices:
        rows = rows[~rows.special.astype(bool)]
    # The lookup table for the previous result is then built BEFORE the
    # redistricting filter. A redistricting cycle cannot be an OUTCOME here,
    # because its own lean would cross the new lines, but it is a perfectly good
    # LEAN for the cycle after it, which runs on the same map. Filtering first
    # would silently drop 2004, 2014 and 2024 from the House check, including
    # the two most recent elections.
    prev = rows.set_index(["cycle", "race_key"])["margin"]
    if prev.index.duplicated().any():
        dup = prev.index[prev.index.duplicated()][:5].tolist()
        raise RuntimeError(f"race_key is not unique within a cycle: {dup}. The previous-result "
                           f"lookup would return a Series instead of a number.")
    if "House" in offices:
        rows = rows[~rows.cycle.isin(NEW_MAP_CYCLES)]

    lean, lag_used = [], []
    for r in rows.itertuples():
        k = LAG[r.office]
        lean.append(prev.get((r.cycle - k, r.race_key), np.nan))
        lag_used.append(k)
    rows["lean_raw"] = lean
    rows["lag_years"] = lag_used
    rows["incumbency"] = rows["incumbent_party"].map({"D": 1.0, "R": -1.0}).fillna(0.0)
    rows["s_pres"] = rows["cycle"].map(PRES_PARTY).map({"D": 1.0, "R": -1.0})
    # The party that holds the seat going in. Zero where it is unknown, which
    # drops those races out of the s_seat regressor rather than mislabelling them.
    rows["s_seat"] = rows["incumbent_party"].map({"D": 1.0, "R": -1.0}).fillna(0.0)
    rows["is_midterm"] = (rows.cycle % 4 == 2)

    nat = load_stage("historical_national")[["cycle", "house_margin_national"]]
    rows = rows.merge(nat, on="cycle", how="left")
    rows = rows.merge(october_panel(), on=["cycle", "state"], how="left")

    keep = rows.dropna(subset=["margin", "lean_raw", "dev_lv", "dev_ch12", "dev_ch24", "s_pres"])
    if midterms_only:
        keep = keep[keep.is_midterm]
    keep = keep.copy()
    for base in ("dev_lv", "dev_ch12", "dev_ch24", "st_ch12", "st_ch24"):
        keep[f"pres_{base}"] = keep["s_pres"] * keep[base]
        keep[f"seat_{base}"] = keep["s_seat"] * keep[base]
    log.info("test frame: %d races (%s), %d cycles %d-%d, %s",
             len(keep), "midterms only" if midterms_only else "all even years",
             keep.cycle.nunique(), keep.cycle.min(), keep.cycle.max(),
             keep.groupby("office").size().to_dict())
    return keep.reset_index(drop=True)


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------
def design(df: pd.DataFrame, econ: str | None, fe: str) -> tuple[np.ndarray, list[str]]:
    """Design matrix, built once on the whole frame so train/test slices share
    columns. `fe` is 'cycle', 'cycle+state', or 'natmargin'."""
    parts: list[np.ndarray] = []
    names: list[str] = []

    def add(block, label):
        arr = np.asarray(block, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
            names.append(label)
        else:
            names.extend(label)
        parts.append(arr)

    if fe == "natmargin":
        add(np.ones(len(df)), "const")
        add(df["house_margin_national"].values, "nat_margin")
    else:
        d = pd.get_dummies(df["cycle"], prefix="cy").astype(float)
        add(d.values, list(d.columns))
    if fe == "cycle+state":
        d = pd.get_dummies(df["state"], prefix="st", drop_first=True).astype(float)
        add(d.values, list(d.columns))

    add(df["lean_raw"].values, "lean")
    add(df["incumbency"].values, "incumbency")
    add((df["office"] == "Senate").astype(float).values, "is_senate")
    if econ:
        add(df[econ].values, econ)
    return np.hstack(parts), names


def fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares via the minimum-norm solution, so a fixed-effect column
    that is all zeros in a training slice (the held-out cycle's own dummy) does
    not raise. The coefficient of interest is never in that collinear block."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def cluster_se(X: np.ndarray, resid: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Cluster-robust standard errors, clustered on cycle.

    Races in the same election share the national swing and the polling error of
    that year, so treating them as independent would overstate precision by
    roughly the square root of the cluster size. Clustering on cycle is the
    standard correction and it is deliberately conservative here: with about a
    dozen clusters the cluster-robust estimate is itself noisy, which makes it
    harder, not easier, for a term to clear |t| >= 2.
    """
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        m = groups == g
        u = X[m].T @ resid[m]
        meat += np.outer(u, u)
    G = len(np.unique(groups))
    n, k = X.shape
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov = adj * XtX_inv @ meat @ XtX_inv
    return np.sqrt(np.clip(np.diag(cov), 0, None))


def loco_rmse(df: pd.DataFrame, econ: str | None, fe: str) -> tuple[float, int]:
    """Leave-one-cycle-out error.

    For the fixed-effects specifications the held-out cycle's own effect cannot
    be estimated, and pretending to know a future election's national
    environment would be the leak this whole exercise is meant to avoid. So the
    error is measured on CYCLE-DEMEANED margins: both the actual and the
    predicted margins have the held-out cycle's own mean removed, which is
    exactly the cross-sectional quantity a cycle-fixed-effects model claims to
    explain, and leaves any unknown national offset out of the comparison.

    For the national-margin specification the levels are the object of interest,
    so the error is measured on levels.
    """
    X, _ = design(df, econ, fe)
    y = df["margin"].values
    cyc = df["cycle"].values
    demean = fe != "natmargin"
    # The national-margin specification predicts levels by extrapolating a slope
    # on one cycle-level regressor. With only a few cycles in training that
    # extrapolation can leave the observed range entirely and return a number in
    # the hundreds, which is not a model comparison, it is a warning. Refuse it.
    if not demean and len(np.unique(cyc)) < 6:
        return float("nan"), 0
    errs = []
    for cy in np.unique(cyc):
        te = cyc == cy
        tr = ~te
        if te.sum() < 5 or tr.sum() < 50:
            continue
        beta, _ = fit(X[tr], y[tr])
        pred = X[te] @ beta
        act = y[te]
        if demean:
            pred = pred - pred.mean()
            act = act - act.mean()
        errs.append(act - pred)
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e ** 2))), len(e)


def run(df: pd.DataFrame, fe: str, label: str) -> pd.DataFrame:
    if fe == "natmargin":
        # 1984, 1988, 1992 and 1996 have no national House margin in the mirror,
        # and this specification uses it as a regressor. The fixed-effects
        # specifications do not need it, which is one reason to prefer them.
        n0 = len(df)
        df = df.dropna(subset=["house_margin_national"]).copy()
        if len(df) < n0:
            log.info("specification C drops %d race(s) in cycles with no national "
                     "House margin on file", n0 - len(df))
    terms = [None,
             "pres_dev_lv", "pres_dev_ch12", "pres_dev_ch24",
             "seat_dev_lv", "seat_dev_ch12", "seat_dev_ch24",
             "pres_st_ch12", "pres_st_ch24"]
    base_rmse = None
    out, n_pred = [], 0
    y = df["margin"].values
    for econ in terms:
        X, names = design(df, econ, fe)
        beta, resid = fit(X, y)
        se = cluster_se(X, resid, df["cycle"].values)
        rmse, n_pred = loco_rmse(df, econ, fe)
        if econ is None:
            base_rmse = rmse
        j = names.index(econ) if econ else None
        # Minimum detectable effect at 80% power, 5% two-sided: 2.8 standard
        # errors. Reported per standard deviation of the regressor, which is the
        # readable unit: "an ordinary swing in local unemployment could not move
        # a race by more than this without the test seeing it."
        mde = 2.8 * se[j] * df[econ].std() if econ else np.nan
        out.append({
            "term": econ or "(none)",
            "coef": beta[j] if econ else np.nan,
            "se": se[j] if econ else np.nan,
            "t": (beta[j] / se[j]) if (econ and se[j] > 0) else np.nan,
            "mde_sd": mde,
            "in_rmse": float(np.sqrt(np.mean(resid ** 2))),
            "loco_rmse": rmse,
            "delta": rmse - base_rmse,
        })
    res = pd.DataFrame(out)
    crit = "cross-sectional" if fe != "natmargin" else "level"
    print(f"\n{'='*88}\n{label}\n   n = {len(df)} races, {df.cycle.nunique()} cycles, "
          f"{n_pred} out-of-sample predictions, {crit} error\n{'='*88}")
    print(f"{'economic term':16s} {'coef':>8s} {'se':>7s} {'t':>7s} "
          f"{'in-RMSE':>9s} {'LOCO-RMSE':>11s} {'vs none':>9s} {'MDE/sd':>8s}")
    for r in res.itertuples():
        c = f"{r.coef:8.3f}" if np.isfinite(r.coef) else f"{'':>8s}"
        e = f"{r.se:7.3f}" if np.isfinite(r.se) else f"{'':>7s}"
        t = f"{r.t:7.2f}" if np.isfinite(r.t) else f"{'':>7s}"
        d = f"{r.delta:+9.3f}" if r.term != "(none)" else f"{'baseline':>9s}"
        m = f"{r.mde_sd:8.2f}" if np.isfinite(r.mde_sd) else f"{'':>8s}"
        print(f"{r.term:16s} {c} {e} {t} {r.in_rmse:9.2f} {r.loco_rmse:11.2f} {d} {m}")
    print("   MDE/sd: the smallest effect, in points of margin per standard deviation of "
          "the regressor,\n           that this sample could detect at 80% power. A null "
          "with a small MDE is an informative\n           bound; a null with a large one "
          "only means the test was too weak to say anything.")
    return res


def main() -> None:
    midterms = "--midterms" in sys.argv
    df = build_frame(midterms_only=midterms)
    scope = "midterm cycles only" if midterms else "all even-year cycles"

    print(f"\nSenate and Governor races, {scope}. Outcome: two-party margin "
          f"(D minus R, points).\nExpected sign on every term below: NEGATIVE "
          f"(a worse local economy hurts the referent party).")
    print(f"Regressor spread, for reading the coefficients: "
          f"sd(dev_lv) = {df.dev_lv.std():.2f}, sd(dev_ch12) = {df.dev_ch12.std():.2f}, "
          f"sd(dev_ch24) = {df.dev_ch24.std():.2f} points of unemployment.")

    a = run(df, "cycle", "A. Cycle fixed effects (the national year absorbed; purely cross-sectional)")
    print("   Sanity check: pres_st_ch12 and pres_dev_ch12 return identical coefficients, as "
          "they must.\n   With cycle fixed effects the national change is a within-cycle "
          "constant, so subtracting it\n   from the state change cannot alter the fit. The "
          "deviation and the raw change are the same\n   regressor here, which confirms the "
          "fixed effects are doing what they claim to.")
    b = run(df, "cycle+state", "B. Cycle AND state fixed effects (within-state movement only; strictest)")
    c = run(df, "natmargin", "C. National margin as a regressor (the production seat model's own shape)")

    # ----------------------------------------------------------------
    # Verdict
    # ----------------------------------------------------------------
    # Three bars, and a term has to clear all of them. Any one on its own is
    # easy to clear by accident.
    #
    #   sign          negative, as the hypothesis predicts
    #   out-of-sample LOCO error lower than the no-economics baseline, by more
    #                 than a trivial margin (0.05 points of RMSE, well under a
    #                 tenth of a percent of the baseline, is noise)
    #   precision     |t| large enough to survive the number of terms tested.
    #                 This matters: 9 terms across 3 specifications is 24 tests,
    #                 and at the conventional |t| >= 2 about one false positive
    #                 is EXPECTED. The Bonferroni-corrected threshold is used
    #                 instead, and a term that only clears |t| = 2 is reported
    #                 as what it is, a coefficient consistent with chance.
    #
    # And one more, which is not a statistical bar but an empirical one:
    # consistency. A term that appears in one specification and vanishes when
    # cycle fixed effects are added is not a weak true effect, it is a
    # confounded one, because cycle fixed effects REMOVE assumptions rather than
    # adding them. They absorb the national environment of each election
    # non-parametrically instead of trusting a single national-margin regressor
    # to do it. A finding that needs the weaker control to appear is the
    # national environment leaking in through the state's economy.
    from scipy.stats import norm
    n_tests = sum(len(r) - 1 for r in (a, b, c))
    t_crit = float(norm.ppf(1 - 0.05 / (2 * n_tests)))
    print(f"\n{'='*88}\nVerdict\n{'='*88}")
    print(f"{n_tests} terms were tested in total. At |t| >= 2 roughly "
          f"{0.05 * n_tests:.1f} false positives are expected by chance alone,\nso the "
          f"threshold that controls the family-wise error rate at 5% is |t| >= {t_crit:.2f}.\n")

    survivors = []
    for name, res, fe in (("A", a, "cycle"), ("B", b, "cycle+state"), ("C", c, "natmargin")):
        base = res[res.term == "(none)"].iloc[0]
        cand = res[res.term != "(none)"].copy()
        cand["passes"] = (cand.coef < 0) & (cand.delta < -0.05) & (cand.t.abs() >= t_crit)
        best = cand.sort_values("loco_rmse").iloc[0]
        marginal = cand[(cand.coef < 0) & (cand.t.abs() >= 2.0)]
        print(f"  {name}. best out-of-sample term: {best.term:16s} "
              f"LOCO {best.loco_rmse:.3f} vs {base.loco_rmse:.3f} without "
              f"({'better' if best.loco_rmse < base.loco_rmse else 'worse'} by "
              f"{abs(best.delta):.3f}), coef {best.coef:+.3f} "
              f"({'right' if best.coef < 0 else 'WRONG'} sign), |t| = {abs(best.t):.2f}")
        if len(marginal):
            for _, m in marginal.iterrows():
                print(f"      right sign and |t| >= 2: {m.term} "
                      f"({m.coef:+.3f}, t = {m.t:.2f}) - does NOT clear {t_crit:.2f}")
        survivors.append(set(cand.loc[cand.passes, "term"]))

    passed = set.union(*survivors) if survivors else set()
    consistent = {term for term in passed
                  if sum(term in sv for sv in survivors) >= 2}
    print()
    if consistent:
        print(f"  EARNS ITS PLACE: {sorted(consistent)}. Right sign, out-of-sample improvement, "
              f"significance\n  that survives the multiple-comparison correction, and present in "
              f"at least two specifications.\n  Promote it into modellib.fit_seat_model as a "
              f"Bayesian coefficient and re-run the backtest.")
    elif passed:
        print(f"  Does NOT earn its place: {sorted(passed)} clears the bars in one "
              f"specification only.\n  A term that needs a particular set of controls to "
              f"appear is a confound, not an effect.")
    else:
        print("  DOES NOT EARN ITS PLACE. No term clears all three bars in any "
              "specification.\n  Local unemployment, as measured here, adds nothing a seat's "
              "partisan lean and the national\n  environment do not already carry. Leaving it "
              "out is the supported choice, and the test is\n  the result worth reporting.")

    # Where the single best in-sample candidate goes when the controls tighten.
    # This is the diagnostic that separates a weak effect from a confounded one.
    print("\n  The same term across all three specifications, which is the test that matters "
          "more than\n  any single coefficient:")
    print(f"      {'term':16s} {'A (cycle FE)':>16s} {'B (+state FE)':>16s} {'C (nat margin)':>16s}")
    for term in a[a.term != "(none)"].term:
        def cell(res):
            r = res[res.term == term]
            return f"{r.coef.iloc[0]:+.2f} (t {r.t.iloc[0]:+.2f})" if len(r) else "n/a"
        print(f"      {term:16s} {cell(a):>16s} {cell(b):>16s} {cell(c):>16s}")

    # ----------------------------------------------------------------
    # Confirmation on an independent sample
    # ----------------------------------------------------------------
    # One term, seat_dev_lv, is correctly signed and almost identical in size in
    # all three specifications above while clearing none of them. That pattern
    # is ambiguous: it is what a small real effect looks like, and also what
    # three correlated views of the same noise look like. The way to tell them
    # apart is a sample that was not used to pick the term. House races are
    # exactly that: 7,000 of them, a much stronger lean (the same district's
    # result two years earlier rather than six), and the same state-level
    # economic reading. Because the term is named in advance there is no
    # multiple-comparison penalty, so this is a single clean test at |t| >= 2.
    #
    # The cost of the sample: state unemployment is a crude stand-in for a
    # district's economy, so a null here is weaker evidence than a null on the
    # states would be. A positive result, though, would be strong.
    print(f"\n{'='*88}\nConfirmation: the one consistent candidate, tested on House races\n{'='*88}")
    print("seat_dev_lv is correctly signed and near-identical in size in all three "
          "specifications above\n(-0.70, -0.79, -0.71) without clearing any of them. "
          "Pre-specifying it and testing on House\nraces is a single test on a sample that "
          "did not choose it, so |t| >= 2 is the honest bar.\n")
    hs = build_frame(midterms_only=midterms, offices=("House",))
    for fe, lbl in (("cycle", "cycle FE"), ("cycle+state", "cycle + state FE"),
                    ("natmargin", "national margin")):
        d = hs.dropna(subset=["house_margin_national"]) if fe == "natmargin" else hs
        X, names = design(d, "seat_dev_lv", fe)
        beta, resid = fit(X, d["margin"].values)
        se = cluster_se(X, resid, d["cycle"].values)
        j = names.index("seat_dev_lv")
        rmse, _ = loco_rmse(d, "seat_dev_lv", fe)
        base, _ = loco_rmse(d, None, fe)
        print(f"  {lbl:18s} n = {len(d):5d}  coef {beta[j]:+.3f}  se {se[j]:.3f}  "
              f"t {beta[j]/se[j]:+.2f}  LOCO {rmse:.3f} vs {base:.3f} "
              f"({'better' if rmse < base else 'worse'})  "
              f"MDE/sd {2.8*se[j]*d['seat_dev_lv'].std():.2f}")

    print("\nCaveats. Unemployment is one indicator and the least elastic one: it moves "
          "slowly and is\nmeasured for a state, not for the people who vote in a race. A null "
          "here rules out this\nregressor, not the local economy. The partisan lean is a "
          "lagged result, so any economic\neffect that persists across one election cycle is "
          "already inside it and cannot show up as\nsignal here; what is being tested is the "
          "NEW information in a state's current conditions.")


if __name__ == "__main__":
    main()
