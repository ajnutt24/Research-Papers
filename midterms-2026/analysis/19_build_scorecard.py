#!/usr/bin/env python3
"""Build the three-tab scorecard workbook: House, Senate, Governor.

One row per race with the model's probability beside the result, so scoring the
forecast is reading one file rather than joining several.

The hold/flip classification and the model-agreement column are Excel FORMULAS,
not values computed here, and the row colours are conditional formatting keyed
off the outcome formula. That is deliberate: races are still being called, so
typing a D or an R into the Winner column has to update the classification, the
colour and the scoreboard without anyone re-running this script.

Colours follow the user's scheme: Republican hold red, Republican flip pink,
Democratic hold blue, Democratic flip light blue.

The Winner column is pre-filled from Wikipedia's live reporting, which is
unofficial and incomplete. The legend tab says so on the face of the sheet.
"""
import re, sys
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import FormulaRule
from openpyxl.utils import get_column_letter

ROOT="/home/user/Research-Papers/midterms-2026"; sys.path.insert(0,ROOT)
import config
S="/tmp/claude-0/-home-user-Research-Papers/6856a5ad-3082-5cc4-a8a8-cce34704eb59/scratchpad"
ARIAL="Arial"; thin=Side(style="thin", color="BFBFBF")
FILLS={"R hold":"E06666","R flip":"F4CCCC","D hold":"6FA8DC","D flip":"CFE2F3"}

def winner_party(x):
    x=str(x or "")
    if "(Democratic" in x or "(DFL" in x: return "D"
    if "(Republican" in x: return "R"
    if "(Independent" in x: return "I"
    return ""

r=pd.read_csv(ROOT+"/outputs/race_probabilities.csv")
f=pd.read_parquet(ROOT+"/data_store/processed/fundamentals_2026.parquet")
pl=pd.read_parquet(ROOT+"/data_store/processed/polls_2026.parquet")
agg=pl.groupby("race_id").agg(n_polls=("poll_id","size"), last_poll=("end_date","max"))

house=pd.read_csv(ROOT+"/outputs/house_rundown_2026.csv")
house["winner_party"]=house.wiki_winner_marker.map(winner_party)

def build(office, cands_csv, held_map):
    c=pd.read_csv(cands_csv)
    m=r[r.office==office][["race_id","state","p_dem","p_dem_caucus","blend_margin",
                           "margin_p10","margin_p90","rating","main_party"]]
    out=(m.merge(c.drop(columns=["state"]), on="race_id", how="left")
           .merge(f[["race_id","lean"]], on="race_id", how="left")
           .merge(agg, on="race_id", how="left"))
    out["incumbent_party"]=out.race_id.map(held_map)
    out["winner_party"]=out.wiki_winner_marker.map(winner_party)
    out["p_dem_pct"]=(out.p_dem*100).round(1)
    out["p_rep_pct"]=((1-out.p_dem)*100).round(1)
    out["p_dem_caucus_pct"]=(out.p_dem_caucus*100).round(1)
    out["favoured"]=["D" if x>=0.5 else "R" for x in out.p_dem]
    out["n_polls"]=out.n_polls.fillna(0).astype(int)
    return out.rename(columns={"blend_margin":"model_margin_dem_minus_rep",
                               "rating":"cook_rating","lean":"partisan_lean_pvi"}).sort_values("state")

sen_held={(f"S-{st}-special" if sp else f"S-{st}"): p for st,_c,sp,p,*_ in config.SENATE_2026}
gov_held={f"G-{st}": p for st,p,*_ in config.GOVERNOR_2026}
sen=build("Senate", S+"/sen_cands.csv", sen_held)
gov=build("Governor", S+"/gov_cands.csv", gov_held)

HOUSE_COLS=[("race_id","Race ID",10),("state","State",7),("district","Dist",6),
    ("dem_candidate","Democrat",24),("rep_candidate","Republican",24),
    ("incumbent_party","Held by",8),("winner_party","Winner",9),("outcome","Outcome",11),
    ("model_agree","Model agree",12),("p_dem_pct","P(D) %",9),("p_rep_pct","P(R) %",9),
    ("favoured","Model says",11),("model_margin_dem_minus_rep","Model margin",13),
    ("margin_p10","p10",8),("margin_p90","p90",8),("cook_rating","Cook",10),
    ("partisan_lean_pvi","Lean",8),("pvi_2026","PVI",8),("n_polls","Polls",7),
    ("last_poll","Last poll",11),("incumbent","Incumbent",22),
    ("same_party_contest","Same party",11),("name_conflict","Name conflict",13),
    ("other_candidates","Other candidates",34),
    ("wiki_winner_marker","Wiki winner (unofficial)",28),("status","Wiki status",30)]
SG_COLS=[("race_id","Race ID",14),("state","State",7),
    ("dem_candidate","Democrat / main",24),("rep_candidate","Republican",24),
    ("incumbent_party","Held by",8),("winner_party","Winner",9),("outcome","Outcome",11),
    ("model_agree","Model agree",12),("p_dem_pct","P(D) %",9),("p_rep_pct","P(R) %",9),
    ("p_dem_caucus_pct","P(D caucus) %",14),("favoured","Model says",11),
    ("model_margin_dem_minus_rep","Model margin",13),("margin_p10","p10",8),
    ("margin_p90","p90",8),("cook_rating","Cook",10),("partisan_lean_pvi","Lean",8),
    ("main_party","Main party",10),("n_polls","Polls",7),("last_poll","Last poll",11),
    ("incumbent","Incumbent",22),("same_party_contest","Same party",11),
    ("other_candidates","Other candidates",40),
    ("wiki_winner_marker","Wiki winner (unofficial)",30),("status","Wiki status",34)]

wb=Workbook(); wb.remove(wb.active)
sheets={}
def add_sheet(name, df, COLS):
    ws=wb.create_sheet(name)
    for j,(key,label,width) in enumerate(COLS,start=1):
        c=ws.cell(row=1,column=j,value=label)
        c.font=Font(name=ARIAL,size=10,bold=True,color="FFFFFF")
        c.fill=PatternFill("solid",fgColor="203864")
        c.alignment=Alignment(horizontal="center",vertical="center",wrap_text=True)
        ws.column_dimensions[get_column_letter(j)].width=width
    ws.row_dimensions[1].height=30
    idx={k:i+1 for i,(k,_l,_w) in enumerate(COLS)}
    L={k:get_column_letter(v) for k,v in idx.items()}
    for i,row in enumerate(df.itertuples(),start=2):
        for key,_l,_w in COLS:
            if key=="outcome":
                v=(f'=IF(OR({L["winner_party"]}{i}="",{L["incumbent_party"]}{i}=""),"",'
                   f'IF({L["winner_party"]}{i}={L["incumbent_party"]}{i},'
                   f'{L["winner_party"]}{i}&" hold",{L["winner_party"]}{i}&" flip"))')
            elif key=="model_agree":
                v=(f'=IF({L["winner_party"]}{i}="","",'
                   f'IF({L["winner_party"]}{i}={L["favoured"]}{i},"Y","N"))')
            else:
                v=getattr(row,key,"")
                if isinstance(v,float) and pd.isna(v): v=""
                elif v is None or (isinstance(v,str) and v=="nan"): v=""
            c=ws.cell(row=i,column=idx[key],value=v)
            c.font=Font(name=ARIAL,size=10); c.border=Border(bottom=thin)
            if key in ("state","district","incumbent_party","winner_party","favoured",
                       "model_agree","n_polls","same_party_contest","pvi_2026","main_party"):
                c.alignment=Alignment(horizontal="center")
            if key in ("p_dem_pct","p_rep_pct","p_dem_caucus_pct"): c.number_format="0.0"
            if key in ("model_margin_dem_minus_rep","margin_p10","margin_p90","partisan_lean_pvi"):
                c.number_format="+0.00;-0.00;0.00"
        ws.cell(row=i,column=idx["winner_party"]).font=Font(name=ARIAL,size=10,bold=True)
        ws.cell(row=i,column=idx["model_agree"]).font=Font(name=ARIAL,size=10,bold=True)
    last=len(df)+1
    rng=f"A2:{get_column_letter(len(COLS))}{last}"
    for label,colour in FILLS.items():
        ws.conditional_formatting.add(rng, FormulaRule(formula=[f'${L["outcome"]}2="{label}"'],
            fill=PatternFill("solid",bgColor=colour), stopIfTrue=False))
    ws.conditional_formatting.add(f'{L["model_agree"]}2:{L["model_agree"]}{last}',
        FormulaRule(formula=[f'${L["model_agree"]}2="N"'],
                    font=Font(name=ARIAL,size=10,bold=True,color="9C0006")))
    ws.freeze_panes = "D2" if "district" in idx else "C2"
    ws.auto_filter.ref=f"A1:{get_column_letter(len(COLS))}{last}"
    sheets[name]=(L,last)
    return ws

add_sheet("House", house, HOUSE_COLS)
add_sheet("Senate", sen, SG_COLS)
add_sheet("Governor", gov, SG_COLS)

lg=wb.create_sheet("Legend & score")
for col,w in (("A",26),("B",11),("C",11),("D",11),("E",54)): lg.column_dimensions[col].width=w
def put(rw,cl,val,**kw):
    c=lg.cell(row=rw,column=cl,value=val); c.font=Font(name=ARIAL,**kw); return c
put(1,1,"2026 midterms: model vs result",size=13,bold=True)
put(3,1,"ROW COLOURS",size=11,bold=True)
put(3,5,"set by the Outcome column, which is a formula",size=10,color="555555")
for i,(lab,key) in enumerate([("Republican hold","R hold"),("Republican flip","R flip"),
                              ("Democratic hold","D hold"),("Democratic flip","D flip")],start=4):
    put(i,1,lab,size=11)
    cc=lg.cell(row=i,column=2); cc.fill=PatternFill("solid",fgColor=FILLS[key]); cc.border=Border(bottom=thin)
put(9,1,"HOW TO USE",size=11,bold=True)
put(9,5,"Type D or R in the Winner column. Outcome, Model agree, the row colour and "
        "this scoreboard all follow.",size=10,color="555555")
put(11,1,"SCOREBOARD",size=11,bold=True)
for j,name in enumerate(("House","Senate","Governor"),start=2):
    put(12,j,name,size=11,bold=True).alignment=Alignment(horizontal="center")
metrics=[("Races on the ballot", None),
         ("Called so far", 'COUNTIF({s}!{w}2:{w}{n},"<>")'),
         ("Still outstanding", None),
         ("Democratic wins", 'COUNTIF({s}!{w}2:{w}{n},"D")'),
         ("Republican wins", 'COUNTIF({s}!{w}2:{w}{n},"R")'),
         ("Democratic flips", 'COUNTIF({s}!{o}2:{o}{n},"D flip")'),
         ("Republican flips", 'COUNTIF({s}!{o}2:{o}{n},"R flip")'),
         ("Model correct", 'COUNTIF({s}!{a}2:{a}{n},"Y")'),
         ("Model wrong", 'COUNTIF({s}!{a}2:{a}{n},"N")'),
         ("Model accuracy", None)]
for i,(lab,tpl) in enumerate(metrics,start=13):
    put(i,1,lab,size=11,bold=(lab=="Model accuracy"))
    for j,name in enumerate(("House","Senate","Governor"),start=2):
        L,n=sheets[name]; col=get_column_letter(j)
        if lab=="Races on the ballot": fml=f"={n-1}"
        elif lab=="Still outstanding": fml=f"={col}13-{col}14"
        elif lab=="Model accuracy": fml=f'=IFERROR({col}20/({col}20+{col}21),"")'
        else: fml="="+tpl.format(s=f"'{name}'" if " " in name else name,
                                 w=L["winner_party"], o=L["outcome"], a=L["model_agree"], n=n)
        c=lg.cell(row=i,column=j,value=fml)
        c.font=Font(name=ARIAL,size=11,bold=(lab=="Model accuracy"))
        c.alignment=Alignment(horizontal="center")
        if lab=="Model accuracy": c.number_format="0.0%"
put(24,1,"The Winner column is pre-filled from Wikipedia's live reporting, which is unofficial "
         "and incomplete. Replace it with certified results before relying on the score.",
    size=9,italic=True,color="7F7F7F")
put(25,1,"Senate note: in Nebraska, Idaho, South Dakota and Montana the non-Republican is an "
         "independent, so P(D) is that candidate's chance of winning the seat and P(D caucus) "
         "discounts it by the chance they caucus with Democrats.",size=9,italic=True,color="7F7F7F")

dest=ROOT+"/outputs/midterms_2026_scorecard.xlsx"
wb.save(dest)
print("wrote",dest)
for name,df in (("House",house),("Senate",sen),("Governor",gov)):
    called=(df.winner_party.isin(["D","R"])).sum()
    print(f"  {name:9s} {len(df):3d} races | {called:3d} called")
