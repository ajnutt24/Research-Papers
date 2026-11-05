#!/usr/bin/env python3
"""Scrape Senate and Governor candidate names from Wikipedia for the scorecard.

The model carries no candidate names. They arrive only attached to polls, so an
unpolled race has none, and even a polled one only has whoever the pollster
named. Wikipedia's per-race tables are the one source covering all 506 seats.

Writes two CSVs the workbook builder (19) joins onto the model's output.

Parsing notes, each one a bug that produced a wrong file first:
  * Candidates are <li> items inside a single cell. Splitting that cell's text
    merges adjacent entries and files a candidate under the wrong party.
  * Minnesota Democrats run as the Democratic-Farmer-Labor party, so matching
    on "Democratic" alone drops every Minnesota Democrat.
  * Florida and Ohio hold Class 3 special elections, headed "Florida (Class 3)",
    which have to map to S-FL-special and S-OH-special rather than S-FL / S-OH.
  * The gubernatorial page carries a second table for DC, Guam and the Northern
    Mariana Islands, which are not among the 36 races on our ballot.
"""
import sys, importlib.util, re
import pandas as pd
ROOT="/home/user/Research-Papers/midterms-2026"; sys.path.insert(0, ROOT)
spec=importlib.util.spec_from_file_location("fp", ROOT+"/data/03_fetch_polls.py")
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
CAND_RE=re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")

def parse_candidate_cell(cell):
    items=[li.get_text(" ",strip=True) for li in cell.find_all("li")]
    if not items:
        items=[x for x in re.split(r"▌", cell.get_text(" ",strip=True)) if x.strip()]
    names,others,winner,extra={}, [], "", {}
    for it in items:
        it=re.sub(r"\[\s*\d+\s*\]","",it.replace("▌","")).strip()
        it=re.sub(r"\s+\d+\.?\d*%\s*$","",it).strip()
        is_win=bool(re.match(r"^Y\b",it))
        it=re.sub(r"^Y\s+","",it).strip()
        m=CAND_RE.match(it)
        if not m: continue
        nm,party=m.group(1).strip(), m.group(2).strip()
        if is_win: winner=f"{nm} ({party})"
        is_dem=party.startswith("Democratic") or party.strip() in ("DFL","D-NPL","DNPL")
        is_rep=party.startswith("Republican")
        if is_dem and "dem" not in names: names["dem"]=nm
        elif is_rep and "rep" not in names: names["rep"]=nm
        else:
            others.append(f"{nm} ({party})")
            if is_dem: extra["dem"]=extra.get("dem",0)+1
            if is_rep: extra["rep"]=extra.get("rep",0)+1
    return names,others,winner,extra

def scrape(title,cache,kind):
    html,prov=fp.fetch_wiki_html(title,cache,max_age_hours=12)
    soup=BeautifulSoup(html,"lxml")
    tables=[t for t in soup.find_all("table")
            if "Candidates" in (t.find("tr").get_text(" ",strip=True) if t.find("tr") else "")]
    out={}
    for t in tables:
        hdr=t.find("tr").get_text(" ",strip=True)
        if kind=="gov" and hdr.startswith("Territory"):   # DC, Guam, NMI are not our races
            continue
        for tr in t.find_all("tr"):
            cells=tr.find_all(["th","td"])
            if len(cells)<4: continue
            head=cells[0].get_text(" ",strip=True)
            m=re.match(r"^([A-Za-z .]+?)(?:\s*\(Class\s*\d+\))?$", head.strip())
            if not m: continue
            st=ABBR.get(m.group(1).strip().lower())
            if not st: continue
            txt=[c.get_text(" ",strip=True) for c in cells]
            names,others,winner,extra=parse_candidate_cell(cells[-1])
            if kind=="sen":
                rid=f"S-{st}-special" if "Class 3" in head and st in ("FL","OH") else f"S-{st}"
            else:
                rid=f"G-{st}"
            out[rid]=dict(race_id=rid, state=st,
                          incumbent=txt[2] if len(txt)>3 else "",
                          status=txt[-2] if len(txt)>=2 else "",
                          dem_candidate=names.get("dem",""), rep_candidate=names.get("rep",""),
                          other_candidates="; ".join(others),
                          same_party_contest=("D vs D" if extra.get("dem") and "rep" not in names
                                              else "R vs R" if extra.get("rep") and "dem" not in names else ""),
                          wiki_winner_marker=winner)
    print(f"  {title}: {len(out)} races ({prov})")
    return pd.DataFrame(out.values())

sen=scrape("2026_United_States_Senate_elections","wiki_us_senate_all.html","sen")
gov=scrape("2026_United_States_gubernatorial_elections","wiki_us_gov_all.html","gov")
sen.to_csv(ROOT + "/outputs/sen_cands.csv", index=False)
gov.to_csv(ROOT + "/outputs/gov_cands.csv", index=False)
exp_s={f"S-{st}-special" if sp else f"S-{st}" for st,_c,sp,*_ in config.SENATE_2026}
exp_g={f"G-{st}" for st,*_ in config.GOVERNOR_2026}
print("  senate  matched %d of %d | missing: %s" % (len(exp_s & set(sen.race_id)), len(exp_s), sorted(exp_s-set(sen.race_id))))
print("  governor matched %d of %d | missing: %s" % (len(exp_g & set(gov.race_id)), len(exp_g), sorted(exp_g-set(gov.race_id))))
print("  senate winners marked:", int((sen.wiki_winner_marker!="").sum()), "| governor:", int((gov.wiki_winner_marker!="").sum()))
