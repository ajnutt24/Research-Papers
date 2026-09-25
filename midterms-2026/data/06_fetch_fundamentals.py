"""
06_fetch_fundamentals.py
========================
Race-level fundamentals for every 2026 race: partisan lean (PVI), the lagged
result in the same seat, incumbency status, and the Democratic share of
individual campaign contributions from the FEC.

Methodology
-----------
* Partisan lean. Preferred: Daily Kos Elections' PVI for the *maps in effect
  in November 2026*, dropped in as data_store/manual/pvi_manual.csv
  (race_id, pvi). Daily Kos publishes new-map PVIs for Texas, California,
  North Carolina, Ohio and Utah; the 2022 file is wrong for those states.
  Fallback: FiveThirtyEight's 2022 partisan lean (public GitHub mirror) for
  districts on the 2020-census map. Districts on a NEW map with no manual PVI
  get their *state's* lean and lean_source="state_fallback_new_map"; the
  hierarchical model (script 11) treats that as a weak prior. Missouri uses
  the 2022 (census-map) file because the general election runs on that map.
* Lagged result. House: 2024 margin in the same district (census-map states
  only; new-map districts get NaN because the lines changed). Senate: the
  seat's last election (2020 for Class II; the 2022 election for the OH/FL
  special seats). Governor: 2022 (2024 for NH/VT-style two-year terms is not
  relevant here; every 2026 governor race was last held in 2022).
* Incumbency. Current House members from the `unitedstates/congress-legislators`
  project (GitHub) give incumbent party by district; retirements, primary
  losses and open seats come from data_store/manual/incumbency_overrides_2026.csv
  (race_id, status, incumbent_party). Senate and Governor incumbency lives in
  config.py. Status vocabulary: incumbent_running | open | incumbent_lost_primary.
  `incumbent_running` for a House member is assumed unless overridden, so the
  overrides file must be maintained (see README).
* Fundraising. OpenFEC /candidates/totals/ for cycle 2026 gives
  `individual_contributions` per candidate. We take the D and R candidates
  with the largest totals per race and compute the Democratic share
  dem/(dem+rep); a missing race gets NaN, which script 09 treats as
  "no information" (share 0.5, zero weight) rather than as a zero.

Output: fundamentals_2026.parquet (one row per race_id)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
import config  # noqa: E402
from utils import GH, SESSION, cache_is_fresh, cache_path, fetch_text, get_logger, load_partisan_lean, load_stage, save_stage, worst_provenance  # noqa: E402

log = get_logger("06_fund")
LEGISLATORS_URL = f"{GH}/unitedstates/congress-legislators/main/legislators-current.yaml"
MANUAL_PVI = config.DATA_MANUAL / "pvi_manual.csv"
MANUAL_INC = config.DATA_MANUAL / "incumbency_overrides_2026.csv"
MANUAL_FEC = config.DATA_MANUAL / "fec_totals_manual.csv"
MANUAL_CANDIDATES = config.DATA_MANUAL / "candidates_2026.csv"
AUTO_CANDIDATES = config.DATA_MANUAL / "candidates_auto.csv"
FEC_BASE = "https://api.open.fec.gov/v1/candidates/totals/"


# --------------------------------------------------------------------------
def partisan_lean(uni: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    lean, prov = load_partisan_lean("2022")
    lean = lean.set_index("race_key")["lean"]
    key = np.where(uni.office == "House", uni.state + "-" + uni.district.map("{:02d}".format), uni.state)
    uni = uni.copy()
    uni["lean"] = [lean.get(k, np.nan) for k in key]
    uni["lean_source"] = "538_2022"
    st_lean = uni.state.map(lean)
    newmap = uni.new_map.astype(bool)
    uni.loc[newmap, "lean"] = st_lean[newmap]
    uni.loc[newmap, "lean_source"] = "state_fallback_new_map"
    if MANUAL_PVI.exists():
        m = pd.read_csv(MANUAL_PVI, comment="#").dropna(subset=["pvi"]).set_index("race_id")["pvi"]
        hit = uni.race_id.isin(m.index)
        uni.loc[hit, "lean"] = uni.loc[hit, "race_id"].map(m)
        uni.loc[hit, "lean_source"] = "dailykos_manual"
        prov = "manual"
        log.info("manual PVI applied to %d races", int(hit.sum()))
    else:
        log.warning("pvi_manual.csv missing: %d new-map districts use their STATE lean",
                    int(newmap.sum()))
    return uni, prov


def lagged_results(uni: pd.DataFrame) -> pd.DataFrame:
    hist = load_stage("historical_results", required=False)
    uni = uni.copy()
    uni["lag_margin"] = np.nan
    uni["lag_cycle"] = np.nan
    if hist is None:
        log.warning("historical_results missing (run 07 first); lag_margin left NaN")
        return uni
    hist = hist[~hist.uncontested]
    # a House special held the same day as the regular election shares a race_key:
    # keep the regular contest; Senate specials keep their own "-special" key
    hist = hist[(hist.office != "House") | (~hist.special.astype(bool))]
    lookup = hist.groupby(["cycle", "race_key"])["margin"].first()
    for i, r in uni.iterrows():
        if r.office == "House":
            if r.new_map:
                continue
            cy, key = 2024, r.race_id
        elif r.office == "Senate":
            cy, key = (2022, f"S-{r.state}") if r.special else (2020, r.race_id)
        else:
            cy, key = 2022, r.race_id
        if (cy, key) in lookup.index:
            uni.at[i, "lag_margin"] = lookup[(cy, key)]
            uni.at[i, "lag_cycle"] = cy
    log.info("lagged results attached for %d/%d races", int(uni.lag_margin.notna().sum()), len(uni))
    return uni


def incumbency(uni: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    uni = uni.copy()
    prov = "config"
    house_inc = {}
    try:
        import yaml
        txt, p = fetch_text(LEGISLATORS_URL, "legislators-current.yaml", max_age_hours=24 * 7)
        prov = "mirror" if p == "live" else p
        for leg in yaml.safe_load(txt):
            term = leg["terms"][-1]
            if term["type"] == "rep":
                party = {"Democrat": "D", "Republican": "R"}.get(term["party"], "I")
                house_inc[f"H-{term['state']}-{max(int(term['district']), 1):02d}"] = party
        log.info("current House members loaded: %d", len(house_inc))
    except Exception as e:
        log.warning("legislators file unavailable (%s); House incumbency from overrides only", e)
    is_house = uni.office == "House"
    uni.loc[is_house, "incumbent_party"] = uni.loc[is_house, "race_id"].map(house_inc)
    uni["inc_status"] = np.where(uni.get("open_seat", pd.Series(False, index=uni.index)).fillna(False).astype(bool),
                                 "open", "incumbent_running")
    uni.loc[is_house & uni.incumbent_party.isna(), "inc_status"] = "open"
    # new-map districts: the member's old district number may not correspond;
    # treat as incumbent only if an override says so
    uni.loc[is_house & uni.new_map.astype(bool) & ~uni.race_id.isin(house_inc.keys()), "inc_status"] = "open"
    if MANUAL_INC.exists():
        ov = pd.read_csv(MANUAL_INC, comment="#").dropna(subset=["status"])
        ov = ov[ov.race_id.isin(uni.race_id)]
        for r in ov.itertuples():
            m = uni.race_id == r.race_id
            uni.loc[m, "inc_status"] = r.status
            if isinstance(r.incumbent_party, str):
                uni.loc[m, "incumbent_party"] = r.incumbent_party
        log.info("applied %d incumbency overrides", len(ov))
    else:
        log.warning("incumbency_overrides_2026.csv missing: retirements/primary losses NOT reflected")
    # numeric incumbency: +1 D incumbent running, -1 R incumbent running, 0 open
    uni["incumbency"] = np.where(uni.inc_status == "incumbent_running",
                                 uni.incumbent_party.map({"D": 1.0, "R": -1.0}).fillna(0.0), 0.0)
    return uni, prov


def fec_totals(uni: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    uni = uni.copy()
    uni["fund_dem"] = np.nan
    uni["fund_rep"] = np.nan
    cache = cache_path("fec_totals_2026.json")
    records = None
    prov = "missing"
    if cache_is_fresh(cache, 24 * 3):
        records = json.loads(cache.read_text())
        prov = "cache"
    else:
        try:
            records = []
            for office in ["H", "S"]:
                page = 1
                while True:
                    url = (f"{FEC_BASE}?api_key={config.FEC_API_KEY}&cycle={config.CYCLE}&office={office}"
                           f"&election_full=true&is_active_candidate=true&per_page=100&page={page}&sort=-receipts")
                    r = SESSION.get(url)
                    js = r.json()
                    records += js.get("results", [])
                    if page >= js.get("pagination", {}).get("pages", 1):
                        break
                    page += 1
            cache.write_text(json.dumps(records))
            prov = "live"
            log.info("FEC: %d candidate totals", len(records))
        except Exception as e:
            log.warning("FEC API unavailable: %s", e)
            records = None
    if records is None and MANUAL_FEC.exists():
        df = pd.read_csv(MANUAL_FEC)   # race_id, party, individual_contributions
        prov = "manual"
    elif records is not None:
        rows = []
        for c in records:
            party = str(c.get("party", ""))[:1]
            if party not in ("D", "R"):
                continue
            st = c.get("state")
            if c.get("office") == "H":
                rid = f"H-{st}-{int(c.get('district') or 1):02d}"
            else:
                rid = f"S-{st}"
            rows.append({"race_id": rid, "party": party,
                         "individual_contributions": float(c.get("individual_contributions") or 0)})
        df = pd.DataFrame(rows)
    else:
        log.warning("no fundraising data; fund_share_dem left NaN (treated as uninformative)")
        return uni, "missing"
    best = df.groupby(["race_id", "party"])["individual_contributions"].max().unstack()
    for p in ["D", "R"]:
        if p not in best:
            best[p] = np.nan
    uni["fund_dem"] = uni.race_id.map(best["D"])
    uni["fund_rep"] = uni.race_id.map(best["R"])
    return uni, prov



def candidate_experience(uni: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Attach the candidates' experience edge and independent-race flags.

    `exp_edge_pts` is (main non-Republican's experience) minus (Republican's),
    in percentage points, using Hummel & Rothschild's published values. It is
    deliberately an offset rather than a fitted coefficient: the 2018-2022
    training set carries no experience coding, so there is nothing here to
    estimate it from. Incumbency stays a separate fitted term, and the
    experience scale excludes the seat being contested, so they do not
    double-count.
    """
    uni = uni.copy()
    uni["exp_edge_pts"] = 0.0
    uni["independent_bonus"] = 0.0
    uni["main_party"] = "D"
    uni["three_way"] = False
    uni["has_democrat"] = True
    uni["caucus_prob_dem"] = 1.0

    for rid, cfg in config.INDEPENDENT_RACES_2026.items():
        m = uni.race_id == rid
        if not m.any():
            continue
        uni.loc[m, "three_way"] = bool(cfg.get("three_way", False))
        uni.loc[m, "has_democrat"] = bool(cfg.get("has_democrat", True))
        if not cfg.get("has_democrat", True):
            uni.loc[m, "main_party"] = "I"
        # if an independent wins, how likely are they to caucus with the Democrats
        uni.loc[m, "caucus_prob_dem"] = float(cfg.get("caucus_prob_dem", 0.5))
        uni.loc[m, "independent_bonus"] = float(config.INDEPENDENT_BONUS_PTS.get(rid, 0.0))

    # Two sources: the auto-built table from 06b (FEC names + a derived
    # experience index) and the hand-maintained file. The hand-maintained file
    # always wins, because automated name matching is imperfect and a
    # correction entered by hand should never be undone by a later fetch.
    frames = []
    if AUTO_CANDIDATES.exists():
        auto = pd.read_csv(AUTO_CANDIDATES, comment="#")
        auto["_src"] = "auto"
        frames.append(auto)
    if MANUAL_CANDIDATES.exists():
        man = pd.read_csv(MANUAL_CANDIDATES, comment="#")
        man["_src"] = "manual"
        frames.append(man)
    if not frames:
        log.warning("no candidate table (neither candidates_auto.csv nor candidates_2026.csv); "
                    "no experience adjustment applied")
        return uni, "missing"

    cand = pd.concat(frames, ignore_index=True)
    # manual last, so drop_duplicates(keep="last") lets it override
    cand = cand.sort_values("_src", ascending=False).drop_duplicates("race_id", keep="last")
    n_auto = int((cand._src == "auto").sum())
    n_man = int((cand._src == "manual").sum())
    log.info("candidate table: %d rows (%d auto, %d hand-entered and overriding)",
             len(cand), n_auto, n_man)
    applied = 0
    for r in cand.itertuples():
        m = uni.race_id == r.race_id
        if not m.any():
            continue
        office = uni.loc[m, "office"].iloc[0]
        scale = config.EXPERIENCE_POINTS_GOVERNOR if office == "Governor" else config.EXPERIENCE_POINTS_SENATE
        main = scale.get(str(getattr(r, "main_experience", "unknown")))
        rep = scale.get(str(getattr(r, "rep_experience", "unknown")))
        # Only score an edge when BOTH candidates are known. If either side is
        # unidentified, an edge would be an artefact of missing data rather
        # than a real advantage.
        edge = 0.0 if (main is None or rep is None) else config.EXPERIENCE_WEIGHT * (main - rep)
        uni.loc[m, "exp_edge_pts"] = edge
        if bool(getattr(r, "three_way", False)):
            uni.loc[m, "three_way"] = True
        if edge:
            applied += 1
    log.info("candidate experience: %d race(s) with a non-zero edge (max %+.1f pts)",
             applied, uni.exp_edge_pts.abs().max() if applied else 0.0)
    n_ind = int((uni.main_party == "I").sum())
    if n_ind:
        log.info("independent-led races (no Democrat on the ballot): %s",
                 ", ".join(sorted(uni.loc[uni.main_party == "I", "race_id"])))
    if uni.three_way.any():
        log.info("three-way races (anti-R vote may split): %s",
                 ", ".join(sorted(uni.loc[uni.three_way, "race_id"])))
    return uni, "manual"


def main():
    uni = config.race_universe()
    uni, p_lean = partisan_lean(uni)
    uni = lagged_results(uni)
    uni, p_inc = incumbency(uni)
    uni, p_fec = fec_totals(uni)
    uni, p_exp = candidate_experience(uni)
    tot = uni.fund_dem.fillna(0) + uni.fund_rep.fillna(0)
    uni["fund_share_dem"] = np.where(tot > 0, uni.fund_dem.fillna(0) / tot.replace(0, np.nan), np.nan)
    uni["fund_logratio"] = np.log((uni.fund_dem.fillna(0) + 1e4) / (uni.fund_rep.fillna(0) + 1e4)).where(tot > 0)
    prov = worst_provenance(p_lean, p_inc) if p_fec != "missing" else worst_provenance(p_lean, p_inc)
    save_stage(uni, "fundamentals_2026", prov,
               {"lean": p_lean, "incumbency": p_inc, "fec": p_fec, "experience": p_exp,
                "n_experience_edge": int((uni.exp_edge_pts != 0).sum()),
                "n_independent_led": int((uni.main_party == "I").sum()),
                "n_three_way": int(uni.three_way.sum()),
                "n_state_fallback_lean": int((uni.lean_source == "state_fallback_new_map").sum())})
    print(uni.groupby(["office", "inc_status"]).size())
    print(uni[["office", "lean", "lag_margin", "incumbency", "fund_share_dem"]].describe().T)


if __name__ == "__main__":
    main()
