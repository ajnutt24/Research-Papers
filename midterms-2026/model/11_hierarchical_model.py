"""
11_hierarchical_model.py
========================
Bayesian hierarchical model that pools polls and fundamentals across the
national, state and district levels, so data-sparse races borrow strength
from the races around them.

Model (PyMC)
------------
    nat_dev     ~ N(0, sd_pred of the fundamentals national model)   # shared national error
    poll_bias   ~ N(0, historical RMS polling miss)                  # shared polling error
    tau_state   ~ HalfNormal(1.5 * STATE_SHOCK_SD)
    alpha[s]    ~ N(0, tau_state)                                    # state swing vs fundamentals
    delta[r]    ~ N(0, fund_sd_idio[r] * shrink_mult[r])             # race offset vs fundamentals
    theta[r]    = fund_margin[r] + loading[r]*nat_dev + alpha[state r] + delta[r]
    poll_est[r] ~ N(theta[r] + poll_bias, poll_sd_election[r])       # for polled races
    generic     ~ N(nat_fund_mean + nat_dev, generic_sd_forecast)    # national level only

* `fund_margin`, `fund_sd_idio`, `loading` come from script 09 (fundamentals
  prior); `poll_est` from script 08 (the house-adjusted Kalman estimate,
  projected to Election Day); `generic` from the generic-ballot filter.
* Why the generic ballot loads on `nat_dev` alone, and not on
  `nat_dev + poll_bias`. Writing it the second way is conceptually tempting,
  since the generic ballot is itself a poll and so carries the polling bias.
  But it makes the model pathological, because one national observation cannot
  separate two national latents of similar prior width: it identifies only
  their sum. Measured on the 2026 data, the two came out at
  corr(nat_dev, poll_bias) = -0.975 with sd(nat_dev + poll_bias) = 0.56
  against priors of 4.38 and 4.03. The pair became a rigid see-saw, and because
  `nat_dev` shifts all 506 races through `loading` while `poll_bias` shifts only
  the 59 poll observations, the split had large consequences. State polls being
  less Democratic than a D+8 fundamentals prediction was then resolved as
  "polls overstate Democrats by 3.4 points and the environment is 4.3 points
  worse for them", rather than the simpler "the fundamentals national
  prediction is too Democratic". The tight likelihood made that second reading
  arithmetically impossible. Loading the generic ballot on `nat_dev` alone
  keeps the two error sources the specification calls for genuinely distinct:
  `nat_dev` is the national environment, measured by the generic ballot, and
  `poll_bias` stays prior-driven from the historical record, still informed by
  the common component of state-poll deviation but no longer able to drag the
  national environment with it.
* Why `generic_sd_forecast` is not the standard error of the average.
  `generic_sd` (0.58 pts here) is how precisely 397 polls pin TODAY'S average.
  As a predictor of the November environment it must also carry the latent
  drift between now and then (`generic_sd_election`, 0.85) and the generic
  ballot's own capacity to be wrong, which is the historical RMS polling miss
  the same script estimates (`poll_shock_sd`, 4.03). The likelihood therefore
  uses sqrt(generic_sd_election^2 + poll_shock_sd^2). Using 0.58 instead
  asserted that 397 polls fix the national vote to within a point, which
  contradicts the model's own measurement of how far final averages have
  missed.
* Partial pooling: alpha[s] is learned from every polled race in a state, so
  an unpolled district in Michigan inherits Michigan's observed swing; delta[r]
  is shrunk toward zero with a race-specific scale.
* Redistricting: `shrink_mult` = NEW_MAP_SHRINK_MULTIPLIER (1.75) for districts
  on a newly drawn map or whose lean is a state-level fallback. A wider delta
  prior means the fundamentals anchor is weaker there: a new district's lean is
  a mapmaker's assumption, so the model leans harder on polls and on the
  state/national trend for those seats and carries more residual uncertainty.
* The two shared error terms are deliberately separate (national/fundamentals
  error vs correlated polling error). Their sum is what the generic ballot
  identifies; the split is prior-driven, which is the honest statement of what
  polls can and cannot tell us about their own bias.

Outputs
-------
hierarchical_estimates_2026.parquet  race_id, hier_margin, hier_sd, hier_sd_idio, poll_weight_in_hier
hier_draws.npz    theta draws (n_draws x n_races), nat_dev, poll_bias, alpha_state draws, race_id order
hierarchical_idata.nc
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

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


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
import config  # noqa: E402
from utils import get_logger, load_meta, load_stage, save_stage, worst_provenance  # noqa: E402

log = get_logger("11_hier")


def build_frame() -> tuple[pd.DataFrame, dict]:
    uni = load_stage("fundamentals_2026")[["race_id", "office", "state", "district", "new_map", "lean_source"]]
    fund = load_stage("fundamentals_estimates_2026")[["race_id", "fund_margin", "fund_sd_idio", "fund_nat_loading"]]
    polls = load_stage("poll_estimates_2026")[["race_id", "poll_margin", "poll_sd_election", "n_polls"]]
    nat = load_stage("national_environment").iloc[0]
    natf = load_stage("fundamentals_national").iloc[0]
    df = uni.merge(fund, on="race_id").merge(polls, on="race_id", how="left")
    df["weak_lean"] = df["new_map"].astype(bool) | (df["lean_source"] == "state_fallback_new_map")
    df["shrink_mult"] = np.where(df["weak_lean"], config.NEW_MAP_SHRINK_MULTIPLIER, 1.0)
    df["state_idx"] = pd.Categorical(df["state"], categories=config.STATES).codes
    df["polled"] = df["poll_margin"].notna()
    poll_shock_sd = float(nat.poll_shock_sd)
    # Fall back to generic_sd only for parquet written before generic_sd_election
    # existed, so an old cache degrades loudly rather than crashing.
    gb_sd_el = float(nat.get("generic_sd_election", nat.generic_sd))
    if gb_sd_el != gb_sd_el:
        gb_sd_el = float(nat.generic_sd)
    shared = {"nat_fund_mean": float(natf.nat_fund_mean), "nat_fund_sd": float(natf.nat_fund_sd_pred),
              "poll_shock_sd": poll_shock_sd,
              "generic_mean": float(nat.generic_mean),
              "generic_sd_forecast": float(np.sqrt(gb_sd_el ** 2 + poll_shock_sd ** 2))}
    return df.reset_index(drop=True), shared


def fit(df: pd.DataFrame, shared: dict):
    polled = df.index[df.polled].values
    with pm.Model() as model:
        nat_dev = pm.Normal("nat_dev", 0.0, shared["nat_fund_sd"])
        # See config.POLL_BIAS_MODE for why this is a switch. Under "symmetric"
        # the shared polling error is held at 0 here and enters only as
        # symmetric uncertainty in the simulation, which is both what
        # modellib.historical_polling_error says the project intends and what
        # avoids counting the same shock twice.
        if config.POLL_BIAS_MODE == "estimated":
            poll_bias = pm.Normal("poll_bias", 0.0, shared["poll_shock_sd"])
        else:
            poll_bias = pm.Deterministic("poll_bias", pt.constant(0.0))
        tau_state = pm.HalfNormal("tau_state", 1.5 * config.STATE_SHOCK_SD)
        z_state = pm.Normal("z_state", 0.0, 1.0, shape=len(config.STATES))
        alpha = pm.Deterministic("alpha_state", tau_state * z_state)
        z_race = pm.Normal("z_race", 0.0, 1.0, shape=len(df))
        delta = pm.Deterministic("delta", z_race * (df["fund_sd_idio"].values * df["shrink_mult"].values))
        theta = pm.Deterministic("theta", df["fund_margin"].values + df["fund_nat_loading"].values * nat_dev
                                 + alpha[df["state_idx"].values] + delta)
        pm.Normal("poll_obs", theta[polled] + poll_bias, df.loc[polled, "poll_sd_election"].values,
                  observed=df.loc[polled, "poll_margin"].values)
        if shared["generic_mean"] == shared["generic_mean"]:
            pm.Normal("generic_obs", shared["nat_fund_mean"] + nat_dev,
                      shared["generic_sd_forecast"], observed=shared["generic_mean"])
        idata = pm.sample(draws=config.MCMC_DRAWS, tune=config.MCMC_TUNE, chains=config.MCMC_CHAINS,
                          cores=config.MCMC_CORES, random_seed=config.RANDOM_SEED,
                          target_accept=config.MCMC_TARGET_ACCEPT, progressbar=False)
    return idata



def _save_idata(idata, filename: str) -> None:
    """Write posterior draws for later inspection. This is a convenience dump:
    no later stage reads it, so a missing netCDF backend must not fail the run."""
    try:
        az.to_netcdf(idata, config.DATA_PROCESSED / filename)
    except Exception as e:  # noqa: BLE001
        log.warning("could not write %s (%s: %s); continuing", filename, type(e).__name__, e)


def main():
    df, shared = build_frame()
    log.info("%d races, %d polled; shared priors: %s", len(df), int(df.polled.sum()),
             {k: round(v, 2) for k, v in shared.items()})
    log.info("sampling the hierarchical model (%d races, %d chains, cores=%d) ...",
             len(df), config.MCMC_CHAINS, config.MCMC_CORES)
    idata = fit(df, shared)
    log.info("hierarchical sampling done")
    summ = az.summary(idata, var_names=["nat_dev", "poll_bias", "tau_state"])
    log.info("shared terms:\n%s", summ[["mean", "sd", "r_hat", "ess_bulk"]].to_string())
    post = idata.posterior
    theta = post["theta"].stack(sample=("chain", "draw")).values.T          # (n_draws, n_races)
    nat_dev = post["nat_dev"].values.ravel()
    poll_bias = post["poll_bias"].values.ravel()
    alpha = post["alpha_state"].stack(sample=("chain", "draw")).values.T
    delta = post["delta"].stack(sample=("chain", "draw")).values.T
    out = df[["race_id", "office", "state", "polled", "n_polls", "weak_lean"]].copy()
    out["hier_margin"] = theta.mean(0)
    out["hier_sd"] = theta.std(0)
    idio = alpha[:, df["state_idx"].values] + delta
    out["hier_sd_idio"] = idio.std(0)
    # share of the posterior precision contributed by the race's own polls
    prior_var = (df["fund_sd_idio"] * df["shrink_mult"]) ** 2
    out["poll_weight_in_hier"] = np.where(df.polled, 1 - out["hier_sd"] ** 2 / (prior_var + 1e-9), 0.0).clip(0, 1)
    out["nat_dev_mean"] = nat_dev.mean()
    prov = worst_provenance(*[load_meta(n).get("provenance", "unknown") for n in
                              ["poll_estimates_2026", "fundamentals_estimates_2026"]])
    save_stage(out, "hierarchical_estimates_2026", prov,
               {"nat_dev": summ.loc["nat_dev", ["mean", "sd"]].round(2).to_dict(),
                "poll_bias": summ.loc["poll_bias", ["mean", "sd"]].round(2).to_dict(),
                "max_rhat": float(az.rhat(idata, var_names=["nat_dev", "poll_bias", "tau_state", "z_race"]).to_array().max())})
    np.savez_compressed(config.DATA_PROCESSED / "hier_draws.npz", theta=theta.astype(np.float32),
                        nat_dev=nat_dev, poll_bias=poll_bias, alpha_state=alpha.astype(np.float32),
                        race_id=out["race_id"].values.astype(str))
    _save_idata(idata, "hierarchical_idata.nc")
    print(out.sort_values("hier_sd").head(3).to_string(index=False))
    print(out[out.polled].sort_values("hier_margin").iloc[::20].to_string(index=False))


if __name__ == "__main__":
    main()
