"""
13_monte_carlo.py
=================
Monte Carlo simulation of all 506 races that turns the blended per-race
distributions into seat-count distributions and chamber-control
probabilities.

Error structure (per simulation draw)
-------------------------------------
1. One posterior draw of the hierarchical model is selected. It carries a
   single national deviation `nat_dev` (shared national/fundamentals error), a
   single `poll_bias` (shared polling error), one `alpha_state` per state, and
   each race's offset `delta`. That draw's `theta` is applied consistently to
   every race, which is what makes the seat total fat-tailed: 435 races do not
   miss independently.
2. For each race the stacking mixture picks a component with probability
   (w_hier, w_fund, w_rating). The fundamentals and rating components share the
   same national draw (scaled by the office's national loading) and a fresh
   state shock, plus their own idiosyncratic sd.
3. An idiosyncratic election-day noise floor (RACE_NOISE_FLOOR_SD) is added to
   every race so that no race is ever a mathematical certainty.

Implementation
--------------
Vectorised numpy on (chunk x races) matrices; no Python loop over races. The
run is benchmarked on N_SIMS_BENCHMARK draws first; if the extrapolated time
for N_SIMS_DEFAULT (69,420) exceeds SIM_RUNTIME_LIMIT_SECONDS the script falls
back to N_SIMS_FALLBACK (42,069) and logs the reason.

Outputs (outputs/)
------------------
race_probabilities.csv      race_id, p_dem, mean margin, 10/90 pct margin
seat_distribution_house.csv / _senate.csv / _governor.csv
chamber_control.json        P(D House), P(D Senate), P(D majority of governors), medians, n_sims used
figures/seat_distributions.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from utils import get_logger, load_meta, load_stage  # noqa: E402

log = get_logger("13_sim")
CHUNK = 5_000


class Simulator:
    def __init__(self):
        b = load_stage("blend_2026").reset_index(drop=True)
        d = np.load(config.DATA_PROCESSED / "hier_draws.npz", allow_pickle=True)
        order = pd.Index(d["race_id"].astype(str)).get_indexer(b.race_id)
        assert (order >= 0).all(), "race_id mismatch between blend and hierarchical draws"
        self.b = b
        self.theta = d["theta"][:, order].astype(np.float32)        # (n_draws, R)
        self.nat_dev = d["nat_dev"].astype(np.float32)
        self.n_draws, self.R = self.theta.shape
        self.state_idx = pd.Categorical(b.state, categories=config.STATES).codes
        self.office = b.office.values
        self.w_hier = b.w_hier.values.astype(np.float32)
        self.w_fund = b.w_fund.values.astype(np.float32)
        self.fund_m = b.fund_margin.values.astype(np.float32)
        self.fund_sd = b.fund_sd_idio.values.astype(np.float32)
        self.loading = b.fund_nat_loading.values.astype(np.float32)
        self.rat_m = b.rating_margin.fillna(b.fund_margin).values.astype(np.float32)
        self.rat_sd = b.rating_sd.fillna(b.fund_sd_idio).values.astype(np.float32)
        self.nat_scale = np.float32(1.0)   # nat_dev draws already carry the fundamentals sd
        self.floor = np.float32(config.RACE_NOISE_FLOOR_SD)
        self.state_sd = np.float32(config.STATE_SHOCK_SD)

    def run_chunk(self, n: int, rng: np.random.Generator) -> np.ndarray:
        d = rng.integers(self.n_draws, size=n)
        theta = self.theta[d]                                  # (n, R) shared structure inside each draw
        nat = self.nat_dev[d][:, None] * self.loading[None, :]  # one national error per draw
        state = rng.normal(0, self.state_sd, size=(n, len(config.STATES))).astype(np.float32)[:, self.state_idx]
        u = rng.random((n, self.R), dtype=np.float32)
        pick_hier = u < self.w_hier
        pick_fund = (~pick_hier) & (u < self.w_hier + self.w_fund)
        z = rng.standard_normal((n, self.R), dtype=np.float32)
        fund_val = self.fund_m + nat + state + self.fund_sd * z
        rat_val = self.rat_m + nat + state + self.rat_sd * z
        margin = np.where(pick_hier, theta, np.where(pick_fund, fund_val, rat_val))
        margin += self.floor * rng.standard_normal((n, self.R), dtype=np.float32)
        return margin

    def run(self, n_sims: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        out = np.empty((n_sims, self.R), dtype=np.float32)
        for start in range(0, n_sims, CHUNK):
            stop = min(start + CHUNK, n_sims)
            out[start:stop] = self.run_chunk(stop - start, rng)
        return out


def tally(margins: np.ndarray, office: np.ndarray) -> dict[str, np.ndarray]:
    wins = margins > 0
    return {"House": wins[:, office == "House"].sum(1),
            "Senate": wins[:, office == "Senate"].sum(1) + config.SENATE_SEATS_NOT_UP["D"],
            "Governor": wins[:, office == "Governor"].sum(1) + config.GOVERNORS_NOT_UP["D"]}


def main():
    sim = Simulator()
    t0 = time.perf_counter()
    sim.run(config.N_SIMS_BENCHMARK, seed=1)
    bench = time.perf_counter() - t0
    projected = bench * config.N_SIMS_DEFAULT / config.N_SIMS_BENCHMARK
    if projected > config.SIM_RUNTIME_LIMIT_SECONDS:
        n_sims, why = config.N_SIMS_FALLBACK, f"projected {projected:.0f}s for {config.N_SIMS_DEFAULT} exceeds {config.SIM_RUNTIME_LIMIT_SECONDS}s"
    else:
        n_sims, why = config.N_SIMS_DEFAULT, f"projected {projected:.1f}s for {config.N_SIMS_DEFAULT} is within the {config.SIM_RUNTIME_LIMIT_SECONDS}s limit"
    log.info("benchmark: %d draws in %.2fs -> using %d simulations (%s)", config.N_SIMS_BENCHMARK, bench, n_sims, why)

    t0 = time.perf_counter()
    margins = sim.run(n_sims, seed=config.RANDOM_SEED)
    elapsed = time.perf_counter() - t0
    log.info("ran %d simulations x %d races in %.1fs", n_sims, sim.R, elapsed)

    seats = tally(margins, sim.office)
    p_dem = (margins > 0).mean(0)
    races = sim.b[["race_id", "office", "state", "polled", "rating", "blend_margin", "w_hier", "w_fund", "w_rating"]].copy()
    races["p_dem"] = p_dem
    races["margin_mean"] = margins.mean(0)
    races["margin_p10"] = np.percentile(margins, 10, axis=0)
    races["margin_p90"] = np.percentile(margins, 90, axis=0)
    races.to_csv(config.OUTPUTS / "race_probabilities.csv", index=False)

    control = {"n_sims": int(n_sims), "n_sims_reason": why, "benchmark_seconds": round(bench, 3),
               "elapsed_seconds": round(elapsed, 2), "asof": config.FORECAST_ASOF.isoformat(),
               "days_to_election": config.DAYS_TO_ELECTION,
               "provenance": load_meta("blend_2026").get("provenance", "unknown")}
    thresholds = {"House": config.HOUSE_MAJORITY, "Senate": config.SENATE_MAJORITY, "Governor": 26}
    for ch, s in seats.items():
        dist = pd.Series(s).value_counts().sort_index()
        pd.DataFrame({"dem_seats": dist.index, "count": dist.values, "prob": dist.values / n_sims}) \
            .to_csv(config.OUTPUTS / f"seat_distribution_{ch.lower()}.csv", index=False)
        control[ch] = {"p_dem_control": float((s >= thresholds[ch]).mean()),
                       "dem_seats_median": float(np.median(s)), "dem_seats_mean": float(s.mean()),
                       "dem_seats_p10": float(np.percentile(s, 10)), "dem_seats_p90": float(np.percentile(s, 90)),
                       "majority_threshold": thresholds[ch]}
        if ch == "Senate":
            control[ch]["p_dem_50_seats_tie"] = float((s == 50).mean())
    (config.OUTPUTS / "chamber_control.json").write_text(json.dumps(control, indent=2))
    log.info("chamber control: %s", {k: round(v["p_dem_control"], 3) for k, v in control.items() if isinstance(v, dict)})
    if control["provenance"] == "fixture":
        log.warning("INPUTS INCLUDE FIXTURE DATA - these numbers are a pipeline test, not a forecast")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (ch, s) in zip(axes, seats.items()):
        ax.hist(s, bins=np.arange(s.min() - .5, s.max() + 1.5), density=True, alpha=.8)
        ax.axvline(thresholds[ch], color="k", ls="--", lw=1)
        ax.set_title(f"{ch}: P(D control)={control[ch]['p_dem_control']:.0%}")
        ax.set_xlabel("Democratic seats")
    fig.suptitle(f"{n_sims:,} simulations, as of {config.FORECAST_ASOF} (provenance: {control['provenance']})")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "seat_distributions.png", dpi=120)
    print(json.dumps({k: v for k, v in control.items() if k in ("House", "Senate", "Governor", "n_sims")}, indent=1))


if __name__ == "__main__":
    main()
