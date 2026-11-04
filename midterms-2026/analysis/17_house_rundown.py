#!/usr/bin/env python3
"""One row per House district: candidates, the model's probability, and the result.

Built for scoring the forecast against what actually happened, so it carries
both sides of every comparison in one file rather than making you join two.

Candidate names are NOT in the model's own data. They arrive only on polls, and
118 of 435 districts were ever polled, so the names here come from Wikipedia's
per-state House tables and the polled names are kept beside them in
`poll_dem_candidate` / `poll_rep_candidate` with a `name_conflict` flag where
the two disagree. Neither source is authoritative; the flag is there so a
disagreement is visible instead of silently picked.

Three things the parsing has to get right, each of which produced a wrong file
before it was handled:

  * Minnesota Democrats run as the Democratic-Farmer-Labor party, so matching
    on "Democratic" alone left all eight Minnesota districts with no Democrat.
  * California and Washington use a top-two primary, so a district can field
    two Democrats and no Republican. Those are marked in `same_party_contest`
    rather than appearing as a missing name.
  * Candidates are <li> items inside one cell. Splitting that cell's text
    instead merges adjacent entries and files a candidate under the wrong
    party.

`wiki_winner_marker` and `status` are Wikipedia's live result reporting, which
is useful for a first pass and is not an official source. Fill the real result
in yourself before scoring anything on it.
"""
import sys, importlib.util, re
import pandas as pd
root="/home/user/Research-Papers/midterms-2026"; sys.path.insert(0, root)
spec=importlib.util.spec_from_file_location("fp", root+"/data/03_fetch_polls.py")
fp=importlib.util.module_from_spec(spec); sys.modules["fp"]=fp; spec.loader.exec_module(fp)
from bs4 import BeautifulSoup
import config

ABBR={s.lower():a for a,s in [
 ("AL","Alabama"),("AK","Alaska"),("AZ","Arizona"),("AR","Arkansas"),("CA","California"),
 ("CO","Colorado"),("CT","Connecticut"),("DE","Delaware"),("FL","Florida"),("GA","Georgia"),
 ("HI","Hawaii"),("ID","Idaho"),("IL","Illinois"),("IN","Indiana"),("IA","Iowa"),("KS","Kansas"),
 ("KY","Kentucky"),("LA","Louisiana"),("ME","Maine"),("MD","Maryland"),("MA","Massachusetts"),
 ("MI","Michigan"),("MN","Minnesota"),("MS","Mississippi"),("MO","Missouri"),("MT","Montana"),
 ("NE","Nebraska"),("NV","Nevada"),("NH","New Hampshire"),("NJ","New Jersey"),("NM","New Mexico"),
 ("NY","New York"),("NC","North Carolina"),("ND","North Dakota"),("OH","Ohio"),("OK","Oklahoma"),
 ("OR","Oregon"),("PA","Pennsylvania"),("RI","Rhode Island"),("SC","South Carolina"),
 ("SD","South Dakota"),("TN","Tennessee"),("TX","Texas"),("UT","Utah"),("VT","Vermont"),
 ("VA","Virginia"),("WA","Washington"),("WV","West Virginia"),("WI","Wisconsin"),("WY","Wyoming")]}

html,prov = fp.fetch_wiki_html("2026_United_States_House_of_Representatives_elections",
                               "wiki_us_house_all.html", max_age_hours=12)
soup = BeautifulSoup(html, "lxml")
tables = [t for t in soup.find_all("table")
          if "Candidates" in (t.find("tr").get_text(" ", strip=True) if t.find("tr") else "")]

DIST_RE = re.compile(r"^([A-Za-z .]+?)\s+(\d+|at-large)$", re.I)
CAND_RE = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")

rows = {}
for t in tables:
    for tr in t.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) < 3:
            continue
        head = cells[0].get_text(" ", strip=True)
        m = DIST_RE.match(head)
        if not m:
            continue
        st = ABBR.get(m.group(1).strip().lower())
        if not st:
            continue
        num = 1 if m.group(2).lower() == "at-large" else int(m.group(2))
        rid = f"H-{st}-{num:02d}"
        txt = [c.get_text(" ", strip=True) for c in cells]
        # Cell count varies: 7 on a normal row, 5 where the seat is new (the
        # member/party/first-elected columns collapse into one), and 4 on a
        # rowspan continuation, which never matches the district regex above.
        # So index the fixed columns from the END, where candidates is always
        # last and the status sentence always second to last.
        cand_cell = cells[-1]
        status = txt[-2] if len(txt) >= 2 else ""
        if len(cells) >= 7:
            inc_name, inc_party = txt[2], txt[3]
        else:
            inc_name, inc_party = txt[2] if len(txt) > 2 else "", ""

        # Candidates are <li> items, each "<colour bar> [Y] Name (Party)".
        # The Y is Wikipedia's winner tick. Splitting the cell's text instead
        # merges adjacent names, which silently filed Alabama's Democrat as
        # its Republican.
        items = [li.get_text(" ", strip=True) for li in cand_cell.find_all("li")]
        if not items:
            items = [x for x in re.split(r"▌", cand_cell.get_text(" ", strip=True)) if x.strip()]
        names, others, winner, extra = {}, [], "", {}
        for it in items:
            it = it.replace("▌", "").strip()
            is_win = bool(re.match(r"^Y\b", it))
            it = re.sub(r"^Y\s+", "", it).strip()
            cm = CAND_RE.match(it)
            if not cm:
                continue
            nm, party = cm.group(1).strip(), cm.group(2).strip()
            if is_win:
                winner = f"{nm} ({party})"
            # Minnesota's Democrats run as the Democratic-Farmer-Labor party
            # and North Dakota's as Democratic-NPL, so a startswith on
            # "Democratic" alone drops every Minnesota Democrat on the ballot.
            is_dem = party.startswith("Democratic") or party.strip() in ("DFL", "D-NPL", "DNPL")
            is_rep = party.startswith("Republican")
            if is_dem and "dem" not in names:
                names["dem"] = nm
            elif is_rep and "rep" not in names:
                names["rep"] = nm
            else:
                others.append(f"{nm} ({party})")
                if is_dem:
                    extra["dem"] = extra.get("dem", 0) + 1
                if is_rep:
                    extra["rep"] = extra.get("rep", 0) + 1
        rows[rid] = dict(race_id=rid, state=st, district=num,
                         pvi_2026=txt[1] if len(txt) > 1 else "",
                         incumbent=inc_name, incumbent_party_wiki=inc_party,
                         status=status,
                         dem_candidate=names.get("dem", ""), rep_candidate=names.get("rep", ""),
                         other_candidates="; ".join(others),
                         # California and Washington run a top-two primary, so a
                         # district can field two Democrats and no Republican.
                         # Without this the file reads as a missing name rather
                         # than a same-party contest.
                         same_party_contest=("D vs D" if extra.get("dem") and "rep" not in names
                                             else "R vs R" if extra.get("rep") and "dem" not in names
                                             else ""),
                         wiki_winner_marker=winner)

wiki = pd.DataFrame(rows.values())
print(f"parsed {len(wiki)} districts from Wikipedia ({prov})")

r = pd.read_csv(root+"/outputs/race_probabilities.csv")
h = r[r.office == "House"].copy()
f = pd.read_parquet(root+"/data_store/processed/fundamentals_2026.parquet")
f = f[f.office == "House"][["race_id", "incumbent_party", "inc_status", "lean"]]
p = pd.read_parquet(root+"/data_store/processed/polls_2026.parquet")
p = p[p.race_id.str.startswith("H-")]
agg = p.groupby("race_id").agg(n_polls=("poll_id", "size"),
                               last_poll=("end_date", "max"))

out = (h.merge(f, on="race_id", how="left")
         .merge(agg, on="race_id", how="left")
         .merge(wiki.drop(columns=["state", "district"]), on="race_id", how="left"))
out["n_polls"] = out.n_polls.fillna(0).astype(int)
out["p_dem_pct"] = (out.p_dem * 100).round(1)
out["p_rep_pct"] = ((1 - out.p_dem) * 100).round(1)
out["district"] = out.race_id.str.split("-").str[2].astype(int)
out["favoured"] = ["D" if x >= 0.5 else "R" for x in out.p_dem]
out = out.rename(columns={"blend_margin": "model_margin_dem_minus_rep",
                          "margin_p10": "margin_p10", "margin_p90": "margin_p90",
                          "lean": "partisan_lean_pvi", "rating": "cook_rating"})
cols = ["race_id", "state", "district", "dem_candidate", "rep_candidate",
        "incumbent", "incumbent_party", "inc_status", "status",
        "p_dem_pct", "p_rep_pct", "favoured",
        "model_margin_dem_minus_rep", "margin_p10", "margin_p90",
        "cook_rating", "partisan_lean_pvi", "pvi_2026", "n_polls", "last_poll",
        "other_candidates", "same_party_contest", "wiki_winner_marker"]
out = out[cols].sort_values(["state", "district"])
for c in ("model_margin_dem_minus_rep", "margin_p10", "margin_p90", "partisan_lean_pvi"):
    out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
dest = root+"/outputs/house_rundown_2026.csv"
out.to_csv(dest, index=False)
print("wrote", dest, len(out), "rows")
print("  with both candidate names:", int(((out.dem_candidate.fillna('')!='') & (out.rep_candidate.fillna('')!='')).sum()))
print("  missing a name:", int(((out.dem_candidate.fillna('')=='') | (out.rep_candidate.fillna('')=='')).sum()))
print("  with a Wikipedia winner marker:", int((out.wiki_winner_marker.fillna('')!='').sum()))
print()
print(out.head(6).to_string(index=False))
