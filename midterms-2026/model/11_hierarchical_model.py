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
    generic     ~ N(nat_fund_mean + nat_dev + poll_bias, generic_sd)

* `fund_margin`, `fund_sd_idio`, `loading` come from script 09 (fundamentals
  prior); `poll_est` from script 08 (the house-adjusted Kalman estimate,
  projected to Election Day); `generic` from the generic-ballot filter.
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

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
    shared = {"nat_fund_mean": float(natf.nat_fund_mean), "nat_fund_sd": float(natf.nat_fund_sd_pred),
              "poll_shock_sd": float(nat.poll_shock_sd),
              "generic_mean": float(nat.generic_mean), "generic_sd": float(nat.generic_sd)}
    return df.reset_index(drop=True), shared


def fit(df: pd.DataFrame, shared: dict):
    polled = df.index[df.polled].values
    with pm.Model() as model:
        nat_dev = pm.Normal("nat_dev", 0.0, shared["nat_fund_sd"])
        poll_bias = pm.Normal("poll_bias", 0.0, shared["poll_shock_sd"])
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
            pm.Normal("generic_obs", shared["nat_fund_mean"] + nat_dev + poll_bias, shared["generic_sd"],
                      observed=shared["generic_mean"])
        idata = pm.sample(draws=config.MCMC_DRAWS, tune=config.MCMC_TUNE, chains=config.MCMC_CHAINS,
                          random_seed=config.RANDOM_SEED, target_accept=config.MCMC_TARGET_ACCEPT, progressbar=False)
    return idata


def main():
    df, shared = build_frame()
    log.info("%d races, %d polled; shared priors: %s", len(df), int(df.polled.sum()),
             {k: round(v, 2) for k, v in shared.items()})
    idata = fit(df, shared)
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
    az.to_netcdf(idata, config.DATA_PROCESSED / "hierarchical_idata.nc")
    print(out.sort_values("hier_sd").head(3).to_string(index=False))
    print(out[out.polled].sort_values("hier_margin").iloc[::20].to_string(index=False))


if __name__ == "__main__":
    main()
