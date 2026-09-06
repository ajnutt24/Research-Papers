"""
model/modellib.py
=================
Numerical building blocks shared by the modelling and validation scripts, kept
in one place so that the live forecast (08-13) and the backtest (14) use
*identical* code paths:

* `kalman_margin`      local-level state-space filter for one race's polls
* `house_effects`      pollster / population-type adjustment by penalised backfitting
* `aggregate_polls`    house effects + Kalman for every race in a poll table
* `load_raw_polls_history`  FiveThirtyEight's 1998-2022 poll archive with actual results
* `historical_polling_error`  cycle-level correlated polling miss (RMS)
* `fast_pool`          closed-form normal-normal pooling used by the backtest
                       as a fast stand-in for the PyMC hierarchical model

All margins are Democratic minus Republican, percentage points.
"""
from __future__ import annotations

import io
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from utils import GH, fetch_text, get_logger  # noqa: E402

log = get_logger("modellib")

RAW_POLLS_URL = f"{GH}/fivethirtyeight/data/master/pollster-ratings/2023/raw-polls.csv"
POLLSTER_RATINGS_URL = f"{GH}/fivethirtyeight/data/master/pollster-ratings/2023/pollster-ratings.csv"

# --------------------------------------------------------------------------
# Observation-noise model for a single poll of the margin
# --------------------------------------------------------------------------
DESIGN_EFFECT = 1.6          # weighting / clustering inflation of sampling variance
POLLSTER_EXCESS_SD = 2.5     # non-sampling error beyond the house effect (pts)
RANDOM_WALK_SD_PER_DAY = {"race": 0.18, "generic": 0.10}   # latent drift
POP_EFFECT_PRIOR = {"lv": 0.0, "rv": 0.0, "a": 0.0, "unknown": 0.0}


def poll_margin_variance(n: float, dem_pct: float | None = None) -> float:
    """Variance (pts^2) of a margin estimate from a poll of size n."""
    n = float(n) if n and n == n else 500.0
    p = 0.5 if dem_pct is None or dem_pct != dem_pct else min(max(dem_pct / 100, 0.2), 0.8)
    var_margin = 4 * p * (1 - p) / n * 100 ** 2 * DESIGN_EFFECT   # margin = 2p-1 scaled to pts
    return var_margin + POLLSTER_EXCESS_SD ** 2


# --------------------------------------------------------------------------
# Kalman filter (local level model)
# --------------------------------------------------------------------------
def kalman_margin(dates: np.ndarray, y: np.ndarray, var: np.ndarray, asof: date,
                  election: date, kind: str = "race",
                  prior_mean: float = 0.0, prior_var: float = 25.0 ** 2) -> dict:
    """Filter poll margins observed on `dates` and return the latent margin at
    `asof` plus its projection to `election`.

    State: m_t = m_{t-1} + w_t,  w_t ~ N(0, q * dt)
    Obs:   y_i = m_{t(i)} + v_i,  v_i ~ N(0, var_i)
    Multiple polls on one day are absorbed sequentially (dt = 0).
    A diffuse prior (sd 25) makes this a polls-only estimate; the hierarchical
    model (script 11) is where fundamentals shrink it.
    """
    q = RANDOM_WALK_SD_PER_DAY[kind] ** 2
    order = np.argsort(dates)
    dates, y, var = dates[order], y[order], var[order]
    m, P = prior_mean, prior_var
    t_prev = dates[0] if len(dates) else asof
    path = []
    for t, yi, vi in zip(dates, y, var):
        dt = max((t - t_prev).days, 0)
        P = P + q * dt
        K = P / (P + vi)
        m = m + K * (yi - m)
        P = (1 - K) * P
        t_prev = t
        path.append((t, m, np.sqrt(P)))
    # project to as-of date and to election day
    dt_asof = max((asof - t_prev).days, 0)
    P_asof = P + q * dt_asof
    dt_el = max((election - t_prev).days, 0)
    P_el = P + q * dt_el
    return {"mean": float(m), "sd_asof": float(np.sqrt(P_asof)), "sd_election": float(np.sqrt(P_el)),
            "n": int(len(y)), "last_date": t_prev, "path": path}


def kalman_daily_path(dates, y, var, start: date, end: date, kind="generic") -> pd.DataFrame:
    """Forward-filter + RTS smoother on a daily grid (for the trend-line plot)."""
    q = RANDOM_WALK_SD_PER_DAY[kind] ** 2
    days = pd.date_range(start, end, freq="D").date
    obs = {}
    for t, yi, vi in zip(dates, y, var):
        obs.setdefault(t, []).append((yi, vi))
    m, P = 0.0, 25.0 ** 2
    m_f, P_f, m_p, P_p = [], [], [], []
    for d in days:
        P_pred = P + q
        m_pred = m
        m_p.append(m_pred); P_p.append(P_pred)
        for yi, vi in obs.get(d, []):
            K = P_pred / (P_pred + vi)
            m_pred = m_pred + K * (yi - m_pred)
            P_pred = (1 - K) * P_pred
        m, P = m_pred, P_pred
        m_f.append(m); P_f.append(P)
    # RTS smoother
    n = len(days)
    m_s, P_s = m_f.copy(), P_f.copy()
    for k in range(n - 2, -1, -1):
        C = P_f[k] / P_p[k + 1]
        m_s[k] = m_f[k] + C * (m_s[k + 1] - m_p[k + 1])
        P_s[k] = P_f[k] + C ** 2 * (P_s[k + 1] - P_p[k + 1])
    return pd.DataFrame({"date": days, "mean": m_s, "sd": np.sqrt(P_s),
                         "filtered_mean": m_f, "filtered_sd": np.sqrt(P_f)})


# --------------------------------------------------------------------------
# House effects
# --------------------------------------------------------------------------
def house_effects(polls: pd.DataFrame, prior_bias: dict[str, float] | None = None,
                  prior_sd: float = 2.0, n_iter: int = 30) -> tuple[pd.Series, pd.Series]:
    """Estimate pollster house effects h_j and population effects g_k in
        margin_i = mu_race(i) + h_j(i) + g_k(i) + e_i
    by penalised backfitting (a fast approximation to a multilevel model):
    race means are free; pollster effects are shrunk toward `prior_bias`
    (historical bias, default 0) with prior sd `prior_sd`; population effects
    are shrunk toward 0 with sd 1.5 ('lv' is the reference).
    Returns (house_effect by pollster, population effect by population).
    """
    df = polls.copy()
    df["w"] = 1.0 / df["var"]
    prior_bias = prior_bias or {}
    h = pd.Series(0.0, index=df["pollster"].unique())
    g = pd.Series(0.0, index=df["population"].unique())
    for _ in range(n_iter):
        adj = df["margin"] - df["pollster"].map(h) - df["population"].map(g)
        mu = (adj * df["w"]).groupby(df["race_id"]).sum() / df["w"].groupby(df["race_id"]).sum()
        resid = df["margin"] - df["race_id"].map(mu) - df["population"].map(g)
        # only races with >1 pollster identify house effects; others get the prior
        for j in h.index:
            m = df["pollster"] == j
            w = df.loc[m, "w"]
            prior_m = prior_bias.get(j, 0.0)
            h[j] = (np.sum(w * resid[m]) + prior_m / prior_sd ** 2) / (np.sum(w) + 1 / prior_sd ** 2)
        h -= np.average(h, weights=df.groupby("pollster")["w"].sum().reindex(h.index).values)
        resid2 = df["margin"] - df["race_id"].map(mu) - df["pollster"].map(h)
        for k in g.index:
            if k == "lv":
                g[k] = 0.0
                continue
            m = df["population"] == k
            w = df.loc[m, "w"]
            g[k] = np.sum(w * resid2[m]) / (np.sum(w) + 1 / 1.5 ** 2)
    return h, g


def aggregate_polls(polls: pd.DataFrame, asof: date, election: date,
                    prior_bias: dict[str, float] | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """House-effect-adjusted Kalman estimate for each race_id in `polls`.
    `polls` needs race_id, pollster, population, end_date, sample_size, dem_pct, margin."""
    df = polls.copy()
    df = df[pd.to_datetime(df["end_date"]).dt.date <= asof]
    if df.empty:
        return pd.DataFrame(columns=["race_id", "poll_margin", "poll_sd", "poll_sd_election", "n_polls", "last_poll_date"]), pd.Series(dtype=float), pd.Series(dtype=float)
    df["var"] = [poll_margin_variance(n, d) for n, d in zip(df["sample_size"], df["dem_pct"])]
    df["population"] = df["population"].fillna("unknown")
    h, g = house_effects(df, prior_bias=prior_bias)
    df["margin_adj"] = df["margin"] - df["pollster"].map(h) - df["population"].map(g)
    rows = []
    for rid, grp in df.groupby("race_id"):
        kind = "generic" if rid == "GENERIC" else "race"
        d = pd.to_datetime(grp["end_date"]).dt.date.values
        k = kalman_margin(d, grp["margin_adj"].values, grp["var"].values, asof, election, kind=kind)
        rows.append({"race_id": rid, "poll_margin": k["mean"], "poll_sd": k["sd_asof"],
                     "poll_sd_election": k["sd_election"], "n_polls": k["n"], "last_poll_date": k["last_date"],
                     "days_since_last_poll": (asof - k["last_date"]).days})
    return pd.DataFrame(rows), h, g


# --------------------------------------------------------------------------
# Historical polls (FiveThirtyEight archive) and polling-error magnitude
# --------------------------------------------------------------------------
def load_raw_polls_history() -> pd.DataFrame:
    txt, _ = fetch_text(RAW_POLLS_URL, "fivethirtyeight_raw_polls.csv", max_age_hours=24 * 30)
    rp = pd.read_csv(io.StringIO(txt))
    rp = rp[rp["type_simple"].isin(["House-G", "Sen-G", "Gov-G"])].copy()
    rp["office"] = rp["type_simple"].map({"House-G": "House", "Sen-G": "Senate", "Gov-G": "Governor"})
    rp["end_date"] = pd.to_datetime(rp["polldate"]).dt.date
    rp["election_date"] = pd.to_datetime(rp["electiondate"]).dt.date
    # race_id in this project's convention
    def rid(r):
        loc = str(r["location"])
        if r["office"] == "House":
            if loc == "US":                      # generic-ballot polls are filed as House-G / US
                return "GENERIC"
            st, num = loc.split("-")
            return f"H-{st}-{int(num):02d}"
        if r["office"] == "Senate":              # 'Sen-GS' = special election
            return f"S-{loc}-special" if "Sen-GS" in str(r["race"]) else f"S-{loc}"
        return f"G-{loc}"
    rp["race_id"] = rp.apply(rid, axis=1)
    rp["cycle"] = rp["year"].astype(int)
    rp.loc[rp["race_id"] == "GENERIC", "office"] = "Generic"
    rp = rp[rp["cand1_party"].eq("DEM") & rp["cand2_party"].eq("REP")]   # D vs R two-party polls only
    rp["dem_pct"] = rp["cand1_pct"]
    rp["rep_pct"] = rp["cand2_pct"]
    rp["margin"] = rp["margin_poll"]
    rp["margin_actual"] = rp["margin_actual"]
    rp["sample_size"] = pd.to_numeric(rp["samplesize"], errors="coerce")
    rp["population"] = "unknown"      # archive does not record LV/RV consistently
    rp["partisan"] = rp["partisan"].fillna("")
    return rp[["cycle", "office", "race_id", "pollster", "pollster_rating_id", "methodology", "partisan",
               "end_date", "election_date", "sample_size", "population", "dem_pct", "rep_pct", "margin",
               "margin_actual", "bias"]].reset_index(drop=True)


def historical_polling_error(rp: pd.DataFrame, cycles=config.POLL_MISS_CYCLES, window_days: int = 21) -> pd.DataFrame:
    """Cycle-level mean poll bias (poll margin minus result, final `window_days`)
    for each office and overall. The RMS across cycles is the prior sd for the
    shared polling shock: the sign of the miss is unknowable in advance, so we
    do not centre it."""
    df = rp[(rp["cycle"].isin(cycles))].copy()
    df["days_out"] = (pd.to_datetime(df["election_date"]) - pd.to_datetime(df["end_date"])).dt.days
    df = df[df["days_out"] <= window_days]
    # bias per race first (so heavily polled races do not dominate), then per cycle
    race = df.groupby(["cycle", "office", "race_id"])["bias"].mean().reset_index()
    out = race.groupby(["cycle", "office"])["bias"].agg(["mean", "count"]).reset_index()
    allo = race.groupby("cycle")["bias"].agg(["mean", "count"]).reset_index().assign(office="All")
    return pd.concat([out, allo], ignore_index=True).rename(columns={"mean": "mean_bias", "count": "n_races"})


def pollster_bias_prior(rp: pd.DataFrame, shrink_n: float = 15.0) -> dict[str, float]:
    """Mean-reverted historical bias per pollster (positive = too Democratic)."""
    g = rp.groupby("pollster")["bias"].agg(["mean", "count"])
    return (g["mean"] * g["count"] / (g["count"] + shrink_n)).to_dict()


# --------------------------------------------------------------------------
# Fast closed-form pooling (backtest stand-in for the PyMC hierarchical model)
# --------------------------------------------------------------------------
def fast_pool(df: pd.DataFrame, tau_state: float = config.STATE_SHOCK_SD,
              tau_race: float = 6.0) -> pd.DataFrame:
    """Empirical-Bayes partial pooling.
    Inputs per race: fund_margin (prior mean), fund_sd (prior sd), poll_margin,
    poll_sd (NaN when unpolled), state, new_map.
    Step 1: state offset = shrunk mean of (poll - fund) residuals in the state.
    Step 2: race posterior = precision-weighted combination of prior
            (fund + state offset, with sd sqrt(fund_sd^2 + tau_race^2)) and poll.
    Newly redrawn districts get tau_race * NEW_MAP_SHRINK_MULTIPLIER, i.e. a
    *wider* prior, so a poll moves them more and the fundamentals anchor less.
    """
    d = df.copy()
    resid = d["poll_margin"] - d["fund_margin"]
    w = 1.0 / (d["poll_sd"] ** 2 + tau_race ** 2)
    has = resid.notna()
    num = (resid * w).where(has, 0).groupby(d["state"]).sum()
    den = w.where(has, 0).groupby(d["state"]).sum()
    state_off = num / (den + 1 / tau_state ** 2)
    state_sd = np.sqrt(1 / (den + 1 / tau_state ** 2))
    d["state_offset"] = d["state"].map(state_off).fillna(0.0)
    d["state_offset_sd"] = d["state"].map(state_sd).fillna(tau_state)
    mult = np.where(d["new_map"].astype(bool), config.NEW_MAP_SHRINK_MULTIPLIER, 1.0)
    prior_mean = d["fund_margin"] + d["state_offset"]
    prior_var = d["fund_sd"] ** 2 + (tau_race * mult) ** 2 + d["state_offset_sd"] ** 2
    poll_var = d["poll_sd"] ** 2
    post_var = 1 / (1 / prior_var + (1 / poll_var).fillna(0))
    post_mean = post_var * (prior_mean / prior_var + (d["poll_margin"] / poll_var).fillna(0))
    d["hier_margin"] = post_mean
    d["hier_sd"] = np.sqrt(post_var)
    d["hier_sd_idio"] = d["hier_sd"]      # no shared terms inside the closed-form posterior
    d["poll_weight_in_hier"] = (post_var / poll_var).fillna(0.0)
    return d


def win_prob(mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return stats.norm.cdf(np.asarray(mean) / np.asarray(sd))


# --------------------------------------------------------------------------
# Fundamentals: shared by 09 (live) and 14 (backtest)
# --------------------------------------------------------------------------
OFFICES = ["House", "Senate", "Governor"]
OFFICE_IDX = {o: i for i, o in enumerate(OFFICES)}


def national_training_table() -> pd.DataFrame:
    """One row per midterm 1946-2022 (+2026 features): national House margin,
    president's party sign, approval, CPI inflation, war salience."""
    from utils import load_stage
    appr = load_stage("approval_history")
    econ = load_stage("economic_cycle_features")
    nat = load_stage("historical_national")[["cycle", "house_margin_national"]]
    seed_p = config.DATA_MANUAL / "historical_house_vote.csv"
    if seed_p.exists():
        seed = pd.read_csv(seed_p)[["cycle", "house_margin_national"]]
        nat = pd.concat([seed[~seed.cycle.isin(nat.cycle)], nat], ignore_index=True)
    df = appr.merge(econ[["cycle", "cpi_yoy_oct", "unrate_oct"]], on="cycle", how="left") \
             .merge(nat, on="cycle", how="left")
    df = df[df.cycle % 4 == 2].copy()            # midterms only
    df["s"] = df["pres_party"].map({"D": 1.0, "R": -1.0})
    df["approval_c"] = df["approval"] - 50.0
    # Post-1994 realignment: the structural Democratic lean of the House vote
    # (Southern Democrats) disappears; without this dummy the intercept from
    # 1946-1990 biases 2026 toward D by ~3-4 points.
    df["post94"] = (df["cycle"] >= 1994).astype(float)
    df["war_salience"] = df["war_salience"].fillna(0.0)
    return df.sort_values("cycle").reset_index(drop=True)


def fit_national_model(df: pd.DataFrame, draws=None, tune=None, chains=None, seed=config.RANDOM_SEED):
    """Bayesian 'referendum' model of the national House margin (D-R):
        y = a + s * (b_mid + b_app*(approval-50) + b_cpi*cpi + b_war*war) + e
    s = +1 with a Democratic president, -1 Republican. Priors follow the
    Abramowitz / Hibbs literature: a midterm penalty of roughly 4 points on
    the president's party, ~0.3 pts of vote per net-approval point, inflation
    hurting the in-party, and a war coefficient centred on ZERO with a wide sd
    so the data decide its sign. Returns (idata, model)."""
    import pymc as pm
    draws = draws or config.MCMC_DRAWS
    tune = tune or config.MCMC_TUNE
    chains = chains or config.MCMC_CHAINS
    train = df.dropna(subset=["house_margin_national", "approval_c", "cpi_yoy_oct"])
    with pm.Model() as model:
        a = pm.Normal("a", 0.0, 3.0)
        b_post94 = pm.Normal("b_post94", -3.0, 3.0)
        b_mid = pm.Normal("b_mid", -4.0, 3.0)
        b_app = pm.Normal("b_app", 0.3, 0.2)
        b_cpi = pm.Normal("b_cpi", -0.5, 0.5)
        b_war = pm.Normal("b_war", 0.0, 3.0)
        sigma = pm.HalfNormal("sigma", 4.0)
        s = pm.Data("s", train["s"].values)
        app = pm.Data("app", train["approval_c"].values)
        cpi = pm.Data("cpi", train["cpi_yoy_oct"].values)
        war = pm.Data("war", train["war_salience"].values)
        post94 = pm.Data("post94", train["post94"].values)
        mu = a + b_post94 * post94 + s * (b_mid + b_app * app + b_cpi * cpi + b_war * war)
        pm.Normal("y", mu, sigma, observed=train["house_margin_national"].values)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                          target_accept=config.MCMC_TARGET_ACCEPT, progressbar=False, idata_kwargs={"log_likelihood": True})
    return idata, model


def predict_national(idata, s: float, approval_c: float, cpi: float, war: float, post94: float = 1.0) -> tuple[float, float, float]:
    """Posterior predictive mean, sd of the mean, and sd of a new observation."""
    post = idata.posterior
    mu = (post["a"] + post["b_post94"] * post94
          + s * (post["b_mid"] + post["b_app"] * approval_c + post["b_cpi"] * cpi + post["b_war"] * war)).values.ravel()
    sig = post["sigma"].values.ravel()
    return float(mu.mean()), float(mu.std()), float(np.sqrt(mu.var() + np.mean(sig ** 2)))


def seat_features(cycle: int, hist: pd.DataFrame, nat: pd.DataFrame) -> pd.DataFrame:
    """Per-race feature frame for a historical cycle: lean (538 vintage when
    it exists, else previous-cycle margin minus previous national margin),
    incumbency (+1 D / -1 R incumbent party, 0 unknown), actual margin,
    national margin that cycle."""
    from utils import load_partisan_lean
    rows = hist[(hist.cycle == cycle) & (~hist.uncontested)].copy()
    rows = rows[(rows.office != "House") | (~rows.special.astype(bool))]
    vint = {2018: "2018", 2020: "2020", 2022: "2022"}.get(cycle)
    natmap = nat.set_index("cycle")["house_margin_national"]
    if vint:
        lean, _ = load_partisan_lean(vint)
        lean = lean.set_index("race_key")["lean"]
        key = np.where(rows.office == "House", rows.state + "-" + rows.district.map("{:02d}".format), rows.state)
        rows["lean"] = [lean.get(k, np.nan) for k in key]
        rows["lean_source"] = f"538_{vint}"
    else:
        prev = cycle - 2
        # House lines change in years ending in 2 -> a 2010 seat's previous result (2008) is on the same map
        pv = hist[(hist.cycle == prev) & (~hist.uncontested)].groupby("race_key")["margin"].first()
        prev_nat = natmap.get(prev, 0.0)
        pv_sen = hist[(hist.cycle == cycle - 6) & (~hist.uncontested)].groupby("race_key")["margin"].first()
        pv_gov = hist[(hist.cycle == cycle - 4) & (~hist.uncontested)].groupby("race_key")["margin"].first()
        lean = []
        for r in rows.itertuples():
            if r.office == "House":
                lean.append(pv.get(r.race_key, np.nan) - prev_nat)
            elif r.office == "Senate":
                lean.append(pv_sen.get(r.race_key, np.nan) - natmap.get(cycle - 6, 0.0))
            else:
                lean.append(pv_gov.get(r.race_key, np.nan) - natmap.get(cycle - 4, 0.0))
        rows["lean"] = lean
        rows["lean_source"] = "lagged_proxy"
    rows["incumbency"] = rows["incumbent_party"].map({"D": 1.0, "R": -1.0}).fillna(0.0)
    rows["nat_margin"] = natmap.get(cycle, np.nan)
    rows["fund_logratio"] = np.nan
    rows["new_map"] = (cycle % 10 == 2)   # first cycle on new lines
    return rows.dropna(subset=["lean"]).reset_index(drop=True)


def fit_seat_model(train: pd.DataFrame, draws=None, tune=None, chains=None, seed=config.RANDOM_SEED):
    """Seat-level fundamentals with office-specific coefficients:
        margin = c[o] + b_nat[o]*national + b_lean[o]*lean + b_inc[o]*incumbency
                 + b_fund*fund_logratio + e,  e ~ N(0, sigma[o])
    Priors: b_nat ~ N(1, .3) (House swings one-for-one with the nation, Senate
    and Governor less), b_lean ~ N(1, .2), b_inc ~ N(3, 2) pts, b_fund ~ N(2, 1.5)
    per log unit of D/R individual contributions (prior-only unless historical
    FEC data are supplied). Returns (idata, model)."""
    import pymc as pm
    draws = draws or config.MCMC_DRAWS
    tune = tune or config.MCMC_TUNE
    chains = chains or config.MCMC_CHAINS
    t = train.dropna(subset=["margin", "lean", "nat_margin"]).copy()
    t["fund"] = t["fund_logratio"].fillna(0.0)
    o = t["office"].map(OFFICE_IDX).values
    with pm.Model(coords={"office": OFFICES}) as model:
        c = pm.Normal("c", 0.0, 3.0, dims="office")
        b_nat = pm.Normal("b_nat", 1.0, 0.3, dims="office")
        b_lean = pm.Normal("b_lean", 1.0, 0.2, dims="office")
        b_inc = pm.Normal("b_inc", 3.0, 2.0, dims="office")
        b_fund = pm.Normal("b_fund", 2.0, 1.5)
        sigma = pm.HalfNormal("sigma", 10.0, dims="office")
        mu = (c[o] + b_nat[o] * t["nat_margin"].values + b_lean[o] * t["lean"].values
              + b_inc[o] * t["incumbency"].values + b_fund * t["fund"].values)
        pm.Normal("y", mu, sigma[o], observed=t["margin"].values)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                          target_accept=config.MCMC_TARGET_ACCEPT, progressbar=False)
    return idata, model


def predict_seats(idata, df: pd.DataFrame, nat_mean: float, nat_sd: float,
                  lean_penalty_sd: float = 6.0) -> pd.DataFrame:
    """Fundamentals prediction per race given a national-environment prior.
    Returns fund_margin (mean), fund_sd_idio (seat-level, excludes the shared
    national error) and fund_sd_total. Races whose lean is a state-level
    fallback get `lean_penalty_sd` added in quadrature."""
    post = idata.posterior
    def pm_(name):
        return post[name].mean(("chain", "draw")).values
    c, b_nat, b_lean, b_inc, sig = pm_("c"), pm_("b_nat"), pm_("b_lean"), pm_("b_inc"), pm_("sigma")
    b_fund = float(post["b_fund"].mean())
    o = df["office"].map(OFFICE_IDX).values
    fund = df["fund_logratio"].fillna(0.0).values if "fund_logratio" in df else 0.0
    inc = df["incumbency"].fillna(0.0).values
    mean = c[o] + b_nat[o] * nat_mean + b_lean[o] * df["lean"].values + b_inc[o] * inc + b_fund * fund
    weak = (df.get("lean_source", pd.Series("", index=df.index)) == "state_fallback_new_map").values
    sd_idio = np.sqrt(sig[o] ** 2 + np.where(weak, lean_penalty_sd ** 2, 0.0))
    out = df[["race_id", "office"]].copy()
    out["fund_margin"] = mean
    out["fund_sd_idio"] = sd_idio
    out["fund_nat_loading"] = b_nat[o]
    out["fund_sd_total"] = np.sqrt(sd_idio ** 2 + (b_nat[o] * nat_sd) ** 2)
    return out


# --------------------------------------------------------------------------
# Backtest components (shared by 12 and 14)
# --------------------------------------------------------------------------
SEAT_TRAIN_CYCLES = [2018, 2020, 2022]
RATING_LEVEL = {"Safe D": ("Safe", 1), "Likely D": ("Likely", 1), "Lean D": ("Lean", 1), "Toss-up": ("Toss-up", 0),
                "Lean R": ("Lean", -1), "Likely R": ("Likely", -1), "Safe R": ("Safe", -1)}


def election_day(cycle: int) -> date:
    d = date(cycle, 11, 1)
    while d.weekday() != 0:      # first Monday
        d += timedelta(days=1)
    return d + timedelta(days=1)  # Tuesday after


class BacktestContext:
    """Loads every historical input once and caches the per-cycle model fits
    so that components can be rebuilt cheaply at many horizons."""

    def __init__(self):
        from utils import load_stage
        self.hist = load_stage("historical_results")
        self.natres = load_stage("historical_national")
        self.nat_table = national_training_table()
        self.rp = load_raw_polls_history()
        self.ratings = load_stage("race_ratings_historical", required=False)
        self.calib = load_stage("rating_calibration", required=False)
        self._nat_fit: dict[int, object] = {}
        self._seat_fit: dict[int, object] = {}
        self._feats: dict[int, pd.DataFrame] = {}

    def national_pred(self, cycle: int) -> tuple[float, float, float]:
        """Leave-future-out: fit on midterms strictly before `cycle`."""
        if cycle not in self._nat_fit:
            train = self.nat_table[self.nat_table.cycle < cycle]
            self._nat_fit[cycle], _ = fit_national_model(train, draws=600, tune=600, chains=2)
        row = self.nat_table[self.nat_table.cycle == cycle].iloc[0]
        return predict_national(self._nat_fit[cycle], row.s, row.approval_c, row.cpi_yoy_oct, row.war_salience, row.post94)

    def seat_fit(self, cycle: int):
        """Seat model trained on the other cycles (leave-one-cycle-out)."""
        train_cycles = [c for c in SEAT_TRAIN_CYCLES if c != cycle]
        key = tuple(train_cycles)
        if key not in self._seat_fit:
            feats = pd.concat([self.features(c) for c in train_cycles], ignore_index=True)
            self._seat_fit[key], _ = fit_seat_model(feats, draws=600, tune=600, chains=2)
        return self._seat_fit[key]

    def features(self, cycle: int) -> pd.DataFrame:
        if cycle not in self._feats:
            self._feats[cycle] = seat_features(cycle, self.hist, self.natres)
        return self._feats[cycle]

    def rating_component(self, cycle: int) -> pd.DataFrame | None:
        if self.ratings is None or self.calib is None or cycle not in set(self.ratings.cycle):
            return None
        r = self.ratings[self.ratings.cycle == cycle][["race_id", "rating"]].drop_duplicates("race_id").copy()
        cal = self.calib.set_index("level")
        lvl = r.rating.map(lambda x: RATING_LEVEL[x][0])
        side = r.rating.map(lambda x: RATING_LEVEL[x][1])
        r["rating_margin"] = side * lvl.map(cal["margin_mean"])
        r["rating_sd"] = lvl.map(cal["margin_sd"])
        p = lvl.map(cal["p_fav_win"])
        r["rating_pwin"] = np.where(side > 0, p, np.where(side < 0, 1 - p, 0.5))
        return r

    def components(self, cycle: int, horizon_days: int) -> pd.DataFrame:
        """Per-race table for `cycle` as of `horizon_days` before the election:
        fund_*, poll_*, hier_*, rating_*, actual margin and dem_won."""
        el = election_day(cycle)
        asof = el - timedelta(days=horizon_days)
        feats = self.features(cycle)
        nat_m, _, nat_sd = self.national_pred(cycle)
        fund = predict_seats(self.seat_fit(cycle), feats.rename(columns={"race_key": "race_id"}), nat_m, nat_sd)
        base = feats.rename(columns={"race_key": "race_id"})[["race_id", "office", "state", "margin", "winner_party", "new_map"]]
        df = base.merge(fund.drop(columns=["office"]), on="race_id")
        df["nat_fund_mean"], df["nat_fund_sd"] = nat_m, nat_sd
        polls = self.rp[(self.rp.cycle == cycle) & (self.rp.race_id != "GENERIC")]
        prior_bias = pollster_bias_prior(self.rp[self.rp.cycle < cycle])
        est, _, _ = aggregate_polls(polls, asof, el, prior_bias=prior_bias)
        df = df.merge(est[["race_id", "poll_margin", "poll_sd_election", "n_polls"]], on="race_id", how="left")
        df = df.rename(columns={"poll_sd_election": "poll_sd"})
        df = fast_pool(df.assign(fund_sd=df["fund_sd_idio"]))
        rc = self.rating_component(cycle)
        if rc is not None:
            df = df.merge(rc, on="race_id", how="left")
        else:
            df["rating_margin"] = np.nan
            df["rating_sd"] = np.nan
            df["rating_pwin"] = np.nan
        df["dem_won"] = (df["winner_party"] == "D").astype(float)
        df["cycle"], df["horizon_days"] = cycle, horizon_days
        return df


# --------------------------------------------------------------------------
# LOO stacking of component models (ArviZ)
# --------------------------------------------------------------------------
def stacking_weights(df: pd.DataFrame, components: dict[str, tuple[str, str]], y_col: str = "margin",
                     seed: int = config.RANDOM_SEED) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit one small calibration model per component,
        y ~ N(m_k + b_k, sqrt(s_k^2 + sigma_k^2)),
    on the same rows, then let ArviZ's LOO-based stacking (Yao et al. 2018)
    choose the mixture weights that maximise out-of-sample predictive density.
    Returns ({component: weight}, az.compare table)."""
    import arviz as az
    import pymc as pm
    cols = [c for pair in components.values() for c in pair]
    d = df.dropna(subset=cols + [y_col]).reset_index(drop=True)
    if len(d) < 20:
        raise ValueError(f"only {len(d)} rows with all components present")
    idatas = {}
    for name, (mcol, scol) in components.items():
        with pm.Model():
            b = pm.Normal("b", 0.0, 3.0)
            extra = pm.HalfNormal("extra", 8.0)
            sd = pm.math.sqrt(d[scol].values ** 2 + extra ** 2)
            pm.Normal("y", d[mcol].values + b, sd, observed=d[y_col].values)
            idatas[name] = pm.sample(draws=800, tune=800, chains=2, random_seed=seed, progressbar=False,
                                     target_accept=0.95, idata_kwargs={"log_likelihood": True})
    comp = az.compare(idatas, ic="loo", method="stacking", scale="log")
    w = comp["weight"].to_dict()
    return w, comp


def marginal_win_prob(df: pd.DataFrame, poll_shock_sd: float, state_sd: float = config.STATE_SHOCK_SD,
                      noise_floor: float = config.RACE_NOISE_FLOOR_SD) -> np.ndarray:
    """Race-level P(D win) from a blended mean and an explicit variance budget:
    idiosyncratic + national (fundamentals share) + polling shock (poll share)
    + state. Used by the backtest, which does not run the PyMC hierarchy."""
    w_poll = (df["w_hier"] * df["poll_weight_in_hier"]).fillna(0.0)
    w_nat = 1.0 - w_poll
    var = (df["blend_sd_idio"] ** 2 + (w_nat * df["fund_nat_loading"] * df["nat_fund_sd"]) ** 2
           + (w_poll * poll_shock_sd) ** 2 + state_sd ** 2 + noise_floor ** 2)
    return stats.norm.cdf(df["blend_margin"] / np.sqrt(var))


def blend_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Mixture mean / idiosyncratic sd from component columns and weights
    w_hier, w_fund, w_rating (rows already carry the weights)."""
    d = df.copy()
    comps = [("hier_margin", "hier_sd_idio", "w_hier"), ("fund_margin", "fund_sd_idio", "w_fund"),
             ("rating_margin", "rating_sd", "w_rating")]
    m = sum(d[w].fillna(0) * d[c].fillna(0) for c, _, w in comps)
    v = sum(d[w].fillna(0) * (d[s].fillna(0) ** 2 + (d[c].fillna(0) - m) ** 2) for c, s, w in comps)
    d["blend_margin"] = m
    d["blend_sd_idio"] = np.sqrt(v)
    return d
