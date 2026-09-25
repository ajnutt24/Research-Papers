"""
06b_fetch_candidates.py
=======================
Builds the candidate table the experience adjustment needs: who is running in
each 2026 race, and what office each of them held before.

Why this is split into two very different halves
------------------------------------------------
NAMES cannot be derived. Who won a 2026 primary is a fact about the world that
has to come from a live source. This script reads them from the FEC's OpenFEC
API, which is official, free and lists every filed candidate with party, office
and district. It needs network access to api.open.fec.gov, so it does nothing
useful on a machine that cannot reach the FEC.

EXPERIENCE can largely be derived, and is, from sources that are public and
stable:
  * legislators-current.yaml   -> sitting senators and House members
  * legislators-historical.yaml -> former senators and House members
                                   (this is how a returning candidate such as a
                                   defeated senator is recognised)
  * the election-results mirror -> winners of recent gubernatorial elections,
                                   which identifies sitting and former governors
Anything not matched is left "unknown", which contributes NO adjustment. That
is deliberate: a candidate we failed to identify must not be silently scored as
having no political experience, because those are very different claims.

Output
------
data_store/manual/candidates_auto.csv

Stage 06 merges this with the hand-maintained candidates_2026.csv, and the
hand-maintained file always wins. Automated name matching is imperfect, so
corrections you make by hand are never overwritten by a later run.

    python3 data/06b_fetch_candidates.py            # all races
    python3 data/06b_fetch_candidates.py --senate   # Senate only (faster)
"""
from __future__ import annotations

import io
import json
import sys
import unicodedata
from pathlib import Path


def _find_project_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        here = Path.cwd().resolve()
        for cand in [here, *here.parents]:
            for d in (cand, cand / "midterms-2026"):
                if (d / "config.py").is_file():
                    return d
        return Path.cwd()


_ROOT = _find_project_root()
sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import config  # noqa: E402
from utils import GH, SESSION, cache_is_fresh, cache_path, fetch_text, get_logger  # noqa: E402

log = get_logger("06b_cands")

LEG_CURRENT = f"{GH}/unitedstates/congress-legislators/main/legislators-current.yaml"
LEG_HISTORICAL = f"{GH}/unitedstates/congress-legislators/main/legislators-historical.yaml"
FEC_CANDIDATES = "https://api.open.fec.gov/v1/candidates/"
AUTO_OUT = config.DATA_MANUAL / "candidates_auto.csv"
INDEX_CACHE = config.DATA_PROCESSED / "experience_index.json"

# Which office outranks which, when a candidate has held several.
RANK = ["governor", "senator", "us_house", "statewide", "lt_governor", "local",
        "state_legislator", "none"]


def norm(name: str) -> str:
    """Normalise a name for matching: lowercase, no accents, no punctuation."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()


def key(last: str, state: str, first: str = "") -> str:
    """Match key. Including a first initial cuts collisions sharply: there are
    many Browns in Ohio. Entries are stored under both the precise key and a
    last-name-only key, and lookups try the precise one first."""
    fi = norm(first)[:1]
    return f"{norm(last)}|{fi}|{state.upper()}" if fi else f"{norm(last)}||{state.upper()}"


# --------------------------------------------------------------------------
# Experience index
# --------------------------------------------------------------------------
def build_experience_index(force: bool = False) -> dict[str, str]:
    """{last name|state: highest office held} from public rosters."""
    if not force and INDEX_CACHE.exists() and cache_is_fresh(INDEX_CACHE, 24 * 7):
        return json.loads(INDEX_CACHE.read_text())

    import yaml
    idx: dict[str, str] = {}

    def add(k: str, office: str):
        if k not in idx or RANK.index(office) < RANK.index(idx[k]):
            idx[k] = office

    for url, cache, label in [(LEG_CURRENT, "legislators-current.yaml", "current"),
                              (LEG_HISTORICAL, "legislators-historical.yaml", "historical")]:
        try:
            txt, _ = fetch_text(url, cache, max_age_hours=24 * 7)
            people = yaml.safe_load(txt)
        except Exception as e:
            log.warning("%s legislators unavailable (%s)", label, e)
            continue
        for p in people:
            nm = p.get("name", {})
            last, first = nm.get("last", ""), nm.get("first", "")
            if not last:
                continue
            for t in p.get("terms", []):
                st = t.get("state", "")
                if not st:
                    continue
                office = "senator" if t.get("type") == "sen" else ("us_house" if t.get("type") == "rep" else None)
                if office:
                    add(key(last, st, first), office)
                    add(key(last, st), office)      # fallback for a bare surname
        log.info("%s legislators: %d people indexed", label, len(people))

    # Governors: winners of recent gubernatorial elections, from the results mirror
    try:
        txt, _ = fetch_text(f"{GH}/fivethirtyeight/election-results/main/election_results_gubernatorial.csv",
                            "results_governor.csv", max_age_hours=24 * 7)
        g = pd.read_csv(io.StringIO(txt), low_memory=False)
        g = g[(g.stage == "general") & (g.winner.astype(str).str.lower() == "true") & (g.cycle >= 2010)]
        for r in g.itertuples():
            nm = str(getattr(r, "candidate_name", "") or "")
            st = str(getattr(r, "state_abbrev", "") or "")
            parts = nm.split()
            if len(parts) >= 2 and st:
                add(key(parts[-1], st, parts[0]), "governor")
                add(key(parts[-1], st), "governor")
        log.info("governors indexed from %d election winners since 2010", len(g))
    except Exception as e:
        log.warning("gubernatorial winners unavailable (%s)", e)

    INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_CACHE.write_text(json.dumps(idx))
    log.info("experience index: %d name/state entries", len(idx))
    return idx


# --------------------------------------------------------------------------
# Candidate names from the FEC
# --------------------------------------------------------------------------
def fetch_fec_candidates(offices=("S", "H")) -> pd.DataFrame:
    rows = []
    for office in offices:
        page = 1
        while True:
            url = (f"{FEC_CANDIDATES}?api_key={config.FEC_API_KEY}&cycle={config.CYCLE}"
                   f"&office={office}&candidate_status=C&per_page=100&page={page}")
            try:
                r = SESSION.get(url)
                js = r.json()
            except Exception as e:
                log.warning("FEC unreachable for office %s: %s", office, e)
                return pd.DataFrame(rows)
            for c in js.get("results", []):
                party = str(c.get("party") or "")[:3]
                rows.append({
                    "office": office, "state": c.get("state"),
                    "district": c.get("district"),
                    "name": c.get("name"), "party": party,
                    "incumbent": str(c.get("incumbent_challenge") or "") == "I",
                })
            pages = js.get("pagination", {}).get("pages", 1)
            if page >= pages:
                break
            page += 1
        log.info("FEC office %s: %d candidates so far", office, len(rows))
    return pd.DataFrame(rows)


def fec_name_to_first(name: str) -> str:
    n = str(name)
    return n.split(",")[1].strip().split()[0] if "," in n and len(n.split(",")) > 1 else ""


def fec_name_to_last(name: str) -> str:
    """FEC stores 'LASTNAME, FIRSTNAME MIDDLE'."""
    n = str(name)
    return n.split(",")[0].strip() if "," in n else n.split()[-1]


def to_race_id(office: str, state: str, district) -> str | None:
    if office == "S":
        return f"S-{state}"
    try:
        d = int(district)
    except (TypeError, ValueError):
        return None
    return f"H-{state}-{max(d, 1):02d}"


def main(senate_only: bool = False, force: bool = False):
    idx = build_experience_index(force=force)
    fec = fetch_fec_candidates(("S",) if senate_only else ("S", "H"))
    if fec.empty:
        log.warning("=" * 70)
        log.warning("No candidate names retrieved. This needs api.open.fec.gov, which")
        log.warning("this machine cannot reach. Run it where the FEC is reachable, or")
        log.warning("fill data_store/manual/candidates_2026.csv by hand.")
        log.warning("Set FEC_API_KEY for a higher rate limit than DEMO_KEY allows.")
        log.warning("=" * 70)
        return

    uni = set(config.race_universe().race_id)
    fec["race_id"] = [to_race_id(o, s, d) for o, s, d in zip(fec.office, fec.state, fec.district)]
    fec = fec[fec.race_id.isin(uni)]
    fec["last"] = fec.name.map(fec_name_to_last)
    fec["first"] = fec.name.map(fec_name_to_first)

    def lookup(last, first, state, incumbent):
        # An incumbent's hold on THIS seat is already priced by the separate
        # incumbency coefficient. Crediting it again as "experience" would
        # double-count, and is exactly how an appointed senator defending the
        # seat ends up scored as a veteran senator.
        if incumbent:
            return "unknown"
        return idx.get(key(last, state, first)) or idx.get(key(last, state)) or "unknown"

    fec["experience"] = [lookup(l, f, s, inc)
                         for l, f, s, inc in zip(fec.last, fec.first, fec.state, fec.incumbent)]

    rows = []
    for rid, g in fec.groupby("race_id"):
        dem = g[g.party == "DEM"]
        rep = g[g.party == "REP"]
        oth = g[~g.party.isin(["DEM", "REP"])]
        main_row = dem.iloc[0] if len(dem) else (oth.iloc[0] if len(oth) else None)
        rep_row = rep.iloc[0] if len(rep) else None
        rows.append({
            "race_id": rid,
            "main_party": "D" if len(dem) else ("I" if len(oth) else "D"),
            "main_name": main_row["name"] if main_row is not None else "",
            "main_experience": main_row["experience"] if main_row is not None else "unknown",
            "main_is_incumbent": bool(main_row["incumbent"]) if main_row is not None else False,
            "rep_name": rep_row["name"] if rep_row is not None else "",
            "rep_experience": rep_row["experience"] if rep_row is not None else "unknown",
            "rep_is_incumbent": bool(rep_row["incumbent"]) if rep_row is not None else False,
            "n_dem_filed": len(dem), "n_rep_filed": len(rep), "n_other_filed": len(oth),
            "source": "fec_auto",
        })
    out = pd.DataFrame(rows).sort_values("race_id")
    out.to_csv(AUTO_OUT, index=False)
    known = ((out.main_experience != "unknown") & (out.rep_experience != "unknown")).sum()
    log.info("wrote %s: %d races, %d with BOTH candidates identified",
             AUTO_OUT.name, len(out), int(known))
    log.info("experience found for %d of %d listed candidates",
             int((out.main_experience != "unknown").sum() + (out.rep_experience != "unknown").sum()),
             2 * len(out))
    print(out.head(12).to_string(index=False))


if __name__ == "__main__":
    main(senate_only="--senate" in sys.argv, force="--force" in sys.argv)
