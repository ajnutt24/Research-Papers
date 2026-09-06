"""
10_rating_to_margin_calibration.py
==================================
Turn qualitative expert ratings (Safe / Likely / Lean / Toss-up) into a win
probability and a margin distribution using their *empirical* track record,
not a regression through the rating codes.

Methodology
-----------
* For every historical race with a rating and a result we tabulate, per
  tier: races, Democratic wins, mean margin and the margin's sd. Available out
  of the box: FiveThirtyEight's 2018 forecast-review categories for all 506
  races (script 05). Supply data_store/manual/historical_ratings.csv with Cook
  / Sabato / Inside Elections archives for 2010, 2014 and 2022 to strengthen it.
* Win rate per tier is a Beta-binomial posterior: a soft prior from the
  raters' published long-run accuracy (config.RATING_TIER_PRIOR_PDEM, weight
  20 races) updated by the observed wins. Tiers are pooled across offices
  because rating vocabularies are shared and samples per office are thin.
* Margin per tier: empirical mean shrunk toward a literature centre with the
  same prior weight; the sd is the empirical sd but never below a floor
  (8 pts for Toss-up/Lean, 10 for Likely, 12 for Safe). The floor is the point
  of this script: a "Lean R" seat is not R+6 with a 1-point standard error, it
  is a seat an expert thinks is probably R, and the band must say so.
* Ratings are converted symmetrically (a Lean R tier uses the mirror of Lean D
  evidence) after mirroring margins, which doubles the effective sample and
  removes any cycle-specific partisan wave from the mapping.

Outputs
-------
rating_calibration.parquet   tier, n, dem_wins, p_dem_win, p_lo, p_hi, margin_mean, margin_sd
ratings_estimates_2026.parquet  race_id, rating, rating_margin, rating_sd, rating_pwin
figures/rating_calibration.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from utils import get_logger, load_meta, load_stage, save_stage  # noqa: E402

log = get_logger("10_ratings")
SIDE = {"Safe D": 1, "Likely D": 1, "Lean D": 1, "Toss-up": 0, "Lean R": -1, "Likely R": -1, "Safe R": -1}
LEVEL = {"Safe D": "Safe", "Likely D": "Likely", "Lean D": "Lean", "Toss-up": "Toss-up",
         "Lean R": "Lean", "Likely R": "Likely", "Safe R": "Safe"}
MARGIN_PRIOR = {"Safe": 25.0, "Likely": 12.0, "Lean": 6.0, "Toss-up": 0.0}
SD_FLOOR = {"Safe": 12.0, "Likely": 10.0, "Lean": 8.0, "Toss-up": 8.0}


def calibrate(hist: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    df = hist.merge(results[["cycle", "race_key", "margin", "winner_party"]].rename(columns={"race_key": "race_id"}),
                    on=["cycle", "race_id"], how="left")
    if "dem_won" not in df or df["dem_won"].isna().all():
        df["dem_won"] = (df["winner_party"] == "D").astype(float)
    df["dem_won"] = df["dem_won"].fillna((df["winner_party"] == "D").astype(float))
    df = df.dropna(subset=["margin"])
    # mirror onto the "favoured party wins?" axis
    df["side"] = df["rating"].map(SIDE)
    df["level"] = df["rating"].map(LEVEL)
    df["fav_margin"] = np.where(df["side"] >= 0, df["margin"], -df["margin"])
    df["fav_won"] = np.where(df["side"] > 0, df["dem_won"], np.where(df["side"] < 0, 1 - df["dem_won"], df["dem_won"]))
    rows = []
    W = config.RATING_TIER_PRIOR_WEIGHT
    for lvl in ["Safe", "Likely", "Lean", "Toss-up"]:
        g = df[df.level == lvl]
        prior_p = config.RATING_TIER_PRIOR_PDEM["Safe D" if lvl == "Safe" else "Likely D" if lvl == "Likely" else "Lean D" if lvl == "Lean" else "Toss-up"]
        n, wins = len(g), float(g.fav_won.sum())
        a, b = prior_p * W + wins, (1 - prior_p) * W + (n - wins)
        p = a / (a + b)
        lo, hi = stats.beta.ppf([0.05, 0.95], a, b)
        emp_mean = g.fav_margin.mean() if n else np.nan
        emp_sd = g.fav_margin.std() if n > 2 else np.nan
        m = (MARGIN_PRIOR[lvl] * W + (emp_mean * n if n else 0)) / (W + n)
        sd = max(SD_FLOOR[lvl], emp_sd if emp_sd == emp_sd else 0)
        rows.append({"level": lvl, "n": n, "fav_wins": wins, "p_fav_win": p, "p_lo": lo, "p_hi": hi,
                     "empirical_win_rate": wins / n if n else np.nan, "empirical_margin_mean": emp_mean,
                     "empirical_margin_sd": emp_sd, "margin_mean": m, "margin_sd": sd})
    cal = pd.DataFrame(rows)
    # Toss-up must be symmetric
    cal.loc[cal.level == "Toss-up", ["margin_mean", "p_fav_win"]] = [0.0, 0.5]
    return cal


def main():
    hist = load_stage("race_ratings_historical")
    results = load_stage("historical_results")
    cal = calibrate(hist, results)
    log.info("tier calibration:\n%s", cal.round(3).to_string(index=False))

    ratings = load_stage("race_ratings_2026")
    cons = ratings[ratings.source == "consensus"][["race_id", "rating", "tier_code"]].copy()
    cons["level"] = cons.rating.map(LEVEL)
    cons["side"] = cons.rating.map(SIDE)
    cons = cons.merge(cal[["level", "p_fav_win", "margin_mean", "margin_sd"]], on="level", how="left")
    cons["rating_margin"] = cons.side * cons.margin_mean
    cons["rating_sd"] = cons.margin_sd
    cons["rating_pwin"] = np.where(cons.side > 0, cons.p_fav_win, np.where(cons.side < 0, 1 - cons.p_fav_win, 0.5))
    prov = load_meta("race_ratings_2026").get("provenance", "unknown")
    save_stage(cal, "rating_calibration", load_meta("race_ratings_historical").get("provenance", "unknown"),
               {"cycles": sorted(hist.cycle.unique().tolist())})
    save_stage(cons[["race_id", "rating", "tier_code", "rating_margin", "rating_sd", "rating_pwin"]],
               "ratings_estimates_2026", prov)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(cal))
    ax.bar(x, cal.p_fav_win, yerr=[cal.p_fav_win - cal.p_lo, cal.p_hi - cal.p_fav_win], capsize=4, alpha=.8, label="posterior P(favoured wins)")
    ax.scatter(x, cal.empirical_win_rate, color="k", zorder=3, label="empirical rate")
    ax.set_xticks(x, [f"{l}\n(n={n})" for l, n in zip(cal.level, cal.n)])
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Rating tier calibration (historical ratings vs results)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "rating_calibration.png", dpi=120)
    print(cons.rating.value_counts())


if __name__ == "__main__":
    main()
