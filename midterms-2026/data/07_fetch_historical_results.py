"""
07_fetch_historical_results.py
==============================
Historical House / Senate / Governor general-election results for backtesting
(script 14), rating calibration (script 10) and the lagged-result fundamental
(script 06).

Methodology
-----------
* Preferred source: MIT Election Data + Science Lab (MEDSL) constituency
  returns. Download the Dataverse files by hand (the Dataverse API is often
  blocked by institutional proxies) and drop them into data_store/manual/:
      1976-2022-house.csv     (doi:10.7910/DVN/IG0UN2)
      1976-2020-senate.csv    (doi:10.7910/DVN/PEJ5QU)
  MEDSL has no governor file; governors always come from the mirror below.
* Fallback: FiveThirtyEight's public election-results mirror on GitHub, which
  covers House, Senate and Governor generals 1998-2024 with winner flags and
  the incumbent party (races.csv). Its per-candidate percentages are official
  state results, so for the four backtest cycles the two sources agree to
  rounding; the mirror is labelled provenance="mirror".
* We reduce every race to a two-party D-minus-R margin. Races with only one
  major party (Louisiana jungle winners, California top-two same-party
  contests, unopposed seats) are flagged `uncontested` and excluded from
  model *training*, but counted as certain seats in backtest tallies.
* National House vote margin per cycle is the sum of all D votes minus all R
  votes over total votes; unopposed races are included as reported, which
  slightly understates the winning party's margin (documented limitation).

Outputs
-------
historical_results.parquet   cycle, office, state, district, race_key,
                             dem_pct, rep_pct, margin, winner_party,
                             incumbent_party, uncontested, total_votes
historical_national.parquet  cycle, house_margin_national, dem_seats, rep_seats
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from utils import get_logger, load_races_mirror, load_results_mirror, save_stage  # noqa: E402

log = get_logger("07_hist")
MIT_HOUSE = config.DATA_MANUAL / "1976-2022-house.csv"
MIT_SENATE = config.DATA_MANUAL / "1976-2020-senate.csv"


def race_key(office: str, state: str, district: int | None, special: bool = False) -> str:
    if office == "House":
        return f"H-{state}-{int(district):02d}"
    if office == "Senate":
        return f"S-{state}-special" if special else f"S-{state}"
    return f"G-{state}"


# --------------------------------------------------------------------------
# MIT Election Lab parser
# --------------------------------------------------------------------------
def parse_mit(path: Path, office: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    df = df[(df["stage"].str.upper() == "GEN")]
    if "special" in df.columns:
        df = df[df["special"].astype(str).str.lower() != "true"]
    if "unofficial" in df.columns:
        df = df[df["unofficial"].astype(str).str.lower() != "true"]
    df["party_simple"] = np.where(df["party"].str.upper().str.startswith("DEMOCRAT"), "D",
                          np.where(df["party"].str.upper().str.startswith("REPUBLICAN"), "R", "O"))
    if office == "House":
        df["district"] = df["district"].astype(int).clip(lower=1)   # at-large=0 -> 1
        keys = ["year", "state_po", "district"]
    else:
        df["district"] = 0
        keys = ["year", "state_po", "district"]
    g = df.groupby(keys + ["party_simple"])["candidatevotes"].sum().unstack(fill_value=0)
    tot = df.groupby(keys)["totalvotes"].max()
    out = g.reset_index()
    out["total_votes"] = tot.values
    for p in ["D", "R"]:
        if p not in out:
            out[p] = 0
    out["dem_pct"] = 100 * out["D"] / out["total_votes"]
    out["rep_pct"] = 100 * out["R"] / out["total_votes"]
    out = out.rename(columns={"year": "cycle", "state_po": "state"})
    out["office"] = office
    out["winner_party"] = np.where(out["D"] > out["R"], "D", "R")
    out["incumbent_party"] = np.nan
    out["special"] = False
    return out[["cycle", "office", "state", "district", "special", "dem_pct", "rep_pct",
                "winner_party", "incumbent_party", "total_votes"]]


# --------------------------------------------------------------------------
# FiveThirtyEight mirror parser
# --------------------------------------------------------------------------
def parse_mirror(office: str) -> tuple[pd.DataFrame, str]:
    raw, prov = load_results_mirror(office)
    races = load_races_mirror()[["id", "incumbent_party"]].rename(columns={"id": "race_id"})
    df = raw[raw["stage"] == "general"].copy()
    df["special"] = df["special"].astype(str).str.lower() == "true"
    df["percent"] = pd.to_numeric(df["percent"], errors="coerce")
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce")
    if office == "House":
        df["district"] = df["office_seat_name"].str.extract(r"(\d+)").astype(float).fillna(1).astype(int)
    else:
        df["district"] = 0
    # ranked-choice: keep first round only
    if "ranked_choice_round" in df:
        rc = pd.to_numeric(df["ranked_choice_round"], errors="coerce")
        df = df[rc.isna() | (rc == 1)]
    df["party_simple"] = df["ballot_party"].map({"DEM": "D", "REP": "R"}).fillna("O")
    keys = ["cycle", "state_abbrev", "district", "special", "race_id"]
    # Fusion voting (NY, CT, ...): one candidate appears on several ballot lines.
    # Aggregate votes per candidate across lines and assign the candidate to D/R
    # if any of their lines is DEM/REP.
    df["cand_key"] = df["candidate_id"].fillna(df["candidate_name"]).astype(str)
    df["is_winner"] = df["winner"].astype(str).str.lower() == "true"
    df["is_unopp"] = df["unopposed"].astype(str).str.lower() == "true"
    cand = df.groupby(keys + ["cand_key"]).agg(
        votes=("votes", "sum"), percent=("percent", "sum"),
        is_winner=("is_winner", "any"), is_unopp=("is_unopp", "any"),
        party=("party_simple", lambda s: "D" if (s == "D").any() else ("R" if (s == "R").any() else "O")),
    ).reset_index()
    best = cand.groupby(keys + ["party"])["percent"].max().unstack()
    votes = cand.groupby(keys + ["party"])["votes"].sum().unstack()
    for p in ["D", "R"]:
        if p not in best:
            best[p] = np.nan
        if p not in votes:
            votes[p] = np.nan
    tot = cand.groupby(keys)["votes"].sum()
    win = cand[cand["is_winner"]].groupby(keys)["party"].agg(
        lambda s: "D" if (s == "D").any() else ("R" if (s == "R").any() else "O"))
    unopp = cand.groupby(keys)["is_unopp"].any()
    idx = best.index
    out = best.reset_index().rename(columns={"D": "dem_pct", "R": "rep_pct", "state_abbrev": "state"})
    out["dem_votes"] = votes["D"].reindex(idx).values
    out["rep_votes"] = votes["R"].reindex(idx).values
    out["total_votes"] = tot.reindex(idx).values
    out["winner_party"] = win.reindex(idx).values
    out["unopposed"] = unopp.reindex(idx).fillna(False).values
    # a district with two general-election races on the same day (a special to
    # fill a vacancy plus the regular election): keep the larger contest
    out = out.sort_values("total_votes", ascending=False).drop_duplicates(
        ["cycle", "state", "district", "special"]).sort_values(keys[:4] if False else ["cycle", "state", "district"])
    out = out.merge(races, on="race_id", how="left")
    out["incumbent_party"] = out["incumbent_party"].map({"DEM": "D", "REP": "R"})
    out["office"] = office
    out["winner_party"] = out["winner_party"].where(out["winner_party"].isin(["D", "R"]), "O")
    return out, prov


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["uncontested"] = df["dem_pct"].isna() | df["rep_pct"].isna() | df.get("unopposed", False).fillna(False).astype(bool)
    # give uncontested races a nominal margin so seat tallies still work
    df["dem_pct"] = df["dem_pct"].fillna(0.0)
    df["rep_pct"] = df["rep_pct"].fillna(0.0)
    df["margin"] = df["dem_pct"] - df["rep_pct"]
    df.loc[df["uncontested"] & (df["winner_party"] == "D"), "margin"] = df["margin"].clip(lower=50)
    df.loc[df["uncontested"] & (df["winner_party"] == "R"), "margin"] = df["margin"].clip(upper=-50)
    df["race_key"] = [race_key(o, s, d, sp) for o, s, d, sp in zip(df["office"], df["state"], df["district"], df["special"])]
    df["cycle"] = df["cycle"].astype(int)
    keep = ["cycle", "office", "state", "district", "special", "race_key", "dem_pct", "rep_pct", "margin",
            "winner_party", "incumbent_party", "uncontested", "total_votes", "dem_votes", "rep_votes"]
    for k in keep:
        if k not in df:
            df[k] = np.nan
    return df[keep].sort_values(["office", "cycle", "state", "district"]).reset_index(drop=True)


def main():
    frames, provs = [], []
    used_mit = False
    if MIT_HOUSE.exists():
        log.info("using MIT Election Lab House file")
        frames.append(parse_mit(MIT_HOUSE, "House")); provs.append("manual"); used_mit = True
    if MIT_SENATE.exists():
        log.info("using MIT Election Lab Senate file")
        frames.append(parse_mit(MIT_SENATE, "Senate")); provs.append("manual"); used_mit = True
    mirror_house = None
    for office in ["House", "Senate", "Governor"]:
        if (office == "House" and MIT_HOUSE.exists()) or (office == "Senate" and MIT_SENATE.exists()):
            continue
        df, prov = parse_mirror(office)
        if office == "House":
            mirror_house = df
        frames.append(df); provs.append(prov)
        log.info("%s mirror: %d races, cycles %d-%d", office, len(df), df.cycle.min(), df.cycle.max())
    hist = finalize(pd.concat(frames, ignore_index=True))
    # attach incumbent party from the mirror when MIT lacks it
    prov = "manual" if used_mit else "mirror"
    save_stage(hist, "historical_results", prov, {"used_mit": used_mit})

    h = hist[(hist.office == "House") & (~hist.special)]
    nat = h.groupby("cycle").apply(lambda g: pd.Series({
        "house_margin_national": 100 * (g.dem_votes.sum() - g.rep_votes.sum()) / g.total_votes.sum(),
        "dem_seats": int((g.winner_party == "D").sum()), "rep_seats": int((g.winner_party == "R").sum()),
        "n_races": len(g)}), include_groups=False).reset_index()
    nat = nat[nat.n_races >= 400]
    save_stage(nat, "historical_national", prov)
    print(nat.to_string(index=False))


if __name__ == "__main__":
    main()
