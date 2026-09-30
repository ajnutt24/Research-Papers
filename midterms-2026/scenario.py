"""
scenario.py
===========
Ask the simulations conditional questions.

    python3 scenario.py                 # the default ladder: 51, 53, 55, 57
    python3 scenario.py --seats 57      # one target
    python3 scenario.py --office House

The chamber forecast answers "how likely is control". This answers the question
underneath it: WHICH races have to fall a particular way for a given seat total,
and which of them are doing the work.

Method
------
Re-runs the same simulator script 13 uses, with the same seed, and keeps the
per-simulation win/loss for every race instead of collapsing straight to a seat
count. Then, for a target total N, it takes the subset of simulations that land
on exactly N and reports each race's win rate inside that subset.

Two numbers per race make this readable:

  P(win)            the race's ordinary marginal probability
  P(win | total=N)  its win rate in the simulations that reached N

The difference between them is what matters. A race already at 95% contributes
nothing to the story of how a party got to N, because it was won in almost every
simulation regardless. A race whose probability jumps from 20% to 70% inside the
subset is one of the seats that HAS to break your way, and is where to watch.

A caution the output repeats: conditioning is not causation. These are the races
that tend to be won together, because they share a national environment and a
polling error, not a list of what "causes" a seat total. A scenario at the far
tail of the distribution is also rare by construction, so the seats it names are
rare events, not sleepers that are secretly close.

Independents are handled as script 13 handles them: a win in Idaho, Nebraska or
South Dakota counts toward the Democratic caucus only if that independent
caucuses with them, drawn per simulation.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    PROJECT = Path(__file__).resolve().parent
except NameError:
    PROJECT = Path.cwd()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "model"))

import config  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("scenario")

STATE = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}


def _load_sim_module():
    """Import simulation/13_monte_carlo.py, whose name is not a valid identifier."""
    path = PROJECT / "simulation" / "13_monte_carlo.py"
    spec = importlib.util.spec_from_file_location("sim13", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sim13"] = mod
    spec.loader.exec_module(mod)
    return mod


def name_of(race_id: str) -> str:
    p = race_id.split("-")
    st = STATE.get(p[1], p[1]) if len(p) > 1 else race_id
    return st + (" (special)" if "special" in p else "")


def run(n_sims: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (race frame, per-sim Senate caucus wins, per-sim seat totals by office)."""
    mod = _load_sim_module()
    sim = mod.Simulator()
    margins = sim.run(n_sims, seed=seed)
    b = sim.b

    rng = np.random.default_rng(seed + 7)
    wins = margins > 0
    sen = (b.office == "Senate").values
    is_ind = (b.main_party.values[sen] == "I")
    sen_wins = wins[:, sen].copy()
    counted = sen_wins.astype(np.int8)
    if is_ind.any():
        draw = rng.random((n_sims, int(is_ind.sum()))) < sim.caucus_prob[sen][is_ind]
        counted[:, is_ind] = (sen_wins[:, is_ind] & draw).astype(np.int8)

    totals = {
        "Senate": counted.sum(1) + config.SENATE_SEATS_NOT_UP["D"],
        "House": wins[:, (b.office == "House").values].sum(1),
        "Governor": wins[:, (b.office == "Governor").values].sum(1) + config.GOVERNORS_NOT_UP["D"],
    }
    return b, counted, totals


def report(b: pd.DataFrame, counted: np.ndarray, totals: dict, office: str, target: int,
           top: int = 14) -> None:
    mask_col = (b.office == office).values
    ids = b.race_id.values[mask_col]
    if office == "Senate":
        per_race = counted
    else:
        raise SystemExit("only Senate is supported for the conditional view right now")

    tot = totals[office]
    sel = tot == target
    n = int(sel.sum())
    if n < 200:
        log.warning("only %d of %d simulations land on exactly %d seats (%.2f%%); the "
                    "conditional rates below are noisy at that sample size",
                    n, len(tot), target, 100 * n / len(tot))
    if n == 0:
        print(f"\nNo simulation reaches exactly {target} seats.")
        return

    marg = per_race.mean(0)
    cond = per_race[sel].mean(0)
    lift = cond - marg
    df = pd.DataFrame({"race_id": ids, "marginal": marg, "conditional": cond, "lift": lift})
    df["name"] = [name_of(r) for r in df.race_id]

    print(f"\n{'='*74}\n{office}: the {n:,} simulations that land on exactly {target} seats "
          f"({100*n/len(tot):.1f}% of all runs)\n{'='*74}")
    print(f"{'race':22s} {'P(win)':>8s} {'P(win | ' + str(target) + ')':>14s} {'change':>9s}")
    for _, x in df.sort_values("lift", ascending=False).head(top).iterrows():
        print(f"{x['name']:22s} {x.marginal:8.0%} {x.conditional:14.0%} {x.lift*100:+8.0f}")

    near = df[(df.conditional > 0.5) & (df.marginal < 0.5)]
    if len(near):
        print(f"\nSeats that are underdogs normally but favourites in this scenario:")
        for _, x in near.sort_values("conditional", ascending=False).iterrows():
            print(f"   {x['name']:22s} {x.marginal:.0%} -> {x.conditional:.0%}")
    always = df[df.conditional > 0.97]
    print(f"\n{len(always)} of the {len(df)} contested seats are won in essentially every "
          f"{target}-seat run, so they carry no information about how the party got there.")


def watch_check(b: pd.DataFrame, counted: np.ndarray, totals: dict, ids: list[str]) -> None:
    """How plausible are specific races, and do they show up in the upside runs?"""
    sen = (b.office == "Senate").values
    all_ids = list(b.race_id.values[sen])
    tot = totals["Senate"]
    print(f"\n{'='*74}\nAre these worth watching?\n{'='*74}")
    print(f"{'race':22s} {'P(win)':>8s} {'| 53':>7s} {'| 55':>7s} {'| 57':>7s}  "
          f"{'rank among the 35':>18s}")
    marg_all = counted.mean(0)
    order = np.argsort(-marg_all)
    rank = {all_ids[i]: k + 1 for k, i in enumerate(order)}
    for rid in ids:
        if rid not in all_ids:
            print(f"{name_of(rid):22s} not a 2026 Senate race")
            continue
        j = all_ids.index(rid)
        row = [counted[tot == t, j].mean() if (tot == t).any() else float("nan")
               for t in (53, 55, 57)]
        print(f"{name_of(rid):22s} {marg_all[j]:8.1%} {row[0]:7.1%} {row[1]:7.1%} {row[2]:7.1%}  "
              f"{rank[rid]:>13d} / 35")


def main() -> None:
    argv = sys.argv[1:]
    n_sims = config.N_SIMS_DEFAULT
    seed = config.RANDOM_SEED
    targets = [51, 53, 55, 57]
    if "--seats" in argv:
        targets = [int(argv[argv.index("--seats") + 1])]

    log.info("re-running %d simulations (seed %d) keeping per-race outcomes", n_sims, seed)
    b, counted, totals = run(n_sims, seed)
    tot = totals["Senate"]
    print(f"\nSenate seat totals across {len(tot):,} simulations: "
          f"median {int(np.median(tot))}, mean {tot.mean():.1f}, "
          f"10th-90th {int(np.percentile(tot,10))}-{int(np.percentile(tot,90))}")
    print(f"P(>= 51) = {np.mean(tot >= 51):.1%}   P(>= 55) = {np.mean(tot >= 55):.1%}   "
          f"P(>= 57) = {np.mean(tot >= 57):.1%}")

    for t in targets:
        report(b, counted, totals, "Senate", t)

    watch_check(b, counted, totals, ["S-FL-special", "S-MS", "S-SC", "S-TX", "S-AK"])
    print("\nConditioning is not causation: these races tend to be won together because "
          "they share a national environment and a polling error, so the list is what a "
          "good night looks like, not a list of causes.")


if __name__ == "__main__":
    main()
