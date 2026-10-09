"""
viz/make_cards.py
=================
Build the share cards (1080x1350, Instagram 4:5) from the current run.

    python3 viz/make_cards.py [outdir]          # writes HTML
    python3 viz/make_cards.py [outdir] --shoot  # also renders PNGs

Five cards:
  ig_senate_ladder    the Senate watchlist, ranked by P(D win)
  ig_governor_ladder  all 36 governor races, ranked by P(D win)
  ig_house            House seat distribution + P(control)
  ig_senate           Senate seat distribution + P(control)
  ig_governor         Governor seat distribution + P(control)

Everything is read from outputs/, so re-running the pipeline and re-running
this script is the whole refresh. Colours are the project's validated
diverging pair (CVD dE 21.6 light / 19.2 dark, both well clear of the 8 floor).

Seat labels say MEDIAN, not "most likely". In a flat distribution the two
differ: at the 9 Oct run the House median is 239 while the mode is 237, and
the Senate median is 53 while the mode is 54. The median is the stabler
statistic and the one the interval is built around, so it is what is shown.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
import config  # noqa: E402

OUTD = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else _ROOT / "outputs" / "cards"

# Names for races whose main candidate is an independent. candidates_2026.csv
# carries the party but not the names for these, so they are listed here and
# must be checked by hand each cycle.
IND_NAMES = {"S-NE": "Osborn", "S-SD": "Bengs", "S-ID": "the independent"}

EXCLUDE_SENATE = {"S-ID"}   # safe enough that it crowds the ladder without informing it

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

CSS_BASE = """
:root{--ink:#121519;--ink-2:#4a5460;--ink-3:#7c8796;--line:#dde2e8;--line-2:#eef1f4;
 --bg:#f7f8fa;--dem:#2a78d6;--rep:#e34948;
 --display:"Zilla Slab",Georgia,serif;--body:"Public Sans",-apple-system,Helvetica,Arial,sans-serif;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--ink:#f2f4f7;--ink-2:#aab4c0;--ink-3:#7f8a97;
 --line:#2b3138;--line-2:#20252b;--bg:#13161a;--dem:#3987e5;--rep:#e66767;color-scheme:dark}}
:root[data-theme="dark"]{--ink:#f2f4f7;--ink-2:#aab4c0;--ink-3:#7f8a97;--line:#2b3138;--line-2:#20252b;
 --bg:#13161a;--dem:#3987e5;--rep:#e66767;color-scheme:dark}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:var(--body)}
.card{width:1080px;height:1350px;display:flex;flex-direction:column;background:var(--bg)}
.eyebrow{font-size:19px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-3);margin:0}
.foot{margin-top:auto;padding-top:18px;border-top:1px solid var(--line);display:flex;
 justify-content:space-between;gap:20px;font-size:18px;color:var(--ink-3);align-items:flex-end}
.key{display:flex;gap:22px;flex-wrap:wrap}
.k{display:inline-flex;align-items:center;gap:8px}
.sw{width:15px;height:15px;border-radius:3px;display:inline-block}
.sw-d{background:var(--dem)} .sw-r{background:var(--rep)}
"""

HEAD = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zilla+Slab:wght@500;700&family=Public+Sans:wght@400;600;800&display=swap">
<style>{css}</style></head><body>"""


# --------------------------------------------------------------------------
# Ladder
# --------------------------------------------------------------------------
LADDER_CSS = """
.card{padding:50px 60px 40px}
h1{font-family:var(--display);font-size:56px;font-weight:700;line-height:1.03;letter-spacing:-.02em;margin:14px 0 0}
.sub{margin:14px 0 0;font-size:21px;color:var(--ink-2);line-height:1.38;max-width:58ch}
.sub b{color:var(--ink);font-weight:600}
table{width:100%;border-collapse:collapse;margin-top:20px}
tr{border-bottom:1px solid var(--line-2)} tr:last-child{border-bottom:0}
th{text-align:left;padding:var(--rowpad) 0;width:286px;vertical-align:middle}
.st{display:block;font-size:var(--stsize);font-weight:600;line-height:1.1}
.hd{display:inline-block;font-size:12.5px;font-weight:800;letter-spacing:.07em;margin-top:3px;
 padding:2px 7px;border-radius:3px}
.h-d{color:var(--dem);background:color-mix(in srgb,var(--dem) 13%,transparent)}
.h-r{color:var(--rep);background:color-mix(in srgb,var(--rep) 13%,transparent)}
.h-i{color:var(--ink-2);background:color-mix(in srgb,var(--ink-3) 18%,transparent)}
.mk{display:block;font-size:13px;font-weight:800;color:var(--ink);letter-spacing:.03em;margin-top:5px}
.mk.dim{color:var(--ink-3);font-weight:600}
td.bar{padding:var(--rowpad) 18px}
.track{position:relative;display:block;height:var(--barh);background:var(--line-2);border-radius:3px}
.fifty{position:absolute;left:50%;top:-5px;bottom:-5px;width:2px;background:var(--ink-3);opacity:.85}
.fill{position:absolute;left:0;top:0;bottom:0;border-radius:3px}
.f-dem{background:var(--dem)} .f-rep{background:var(--rep)}
.pc{width:106px;text-align:right;font-size:var(--pcsize);font-weight:800;
 font-variant-numeric:tabular-nums;padding:var(--rowpad) 0}
.u{font-size:19px;margin-left:1px}
.c-dem{color:var(--dem)} .c-rep{color:var(--rep)}
/* Compact: badge inline with the state, so a 36-row ladder still fits. */
.compact h1{font-size:50px}
.compact .sub{font-size:19px;margin-top:12px}
.compact table{margin-top:14px}
.compact .st{display:inline}
.compact .hd{margin-top:0;margin-left:9px;vertical-align:2px;font-size:11.5px;padding:1px 6px}
.compact .mk{display:inline;margin-left:9px;font-size:12px}
"""


def ladder_rows(df: pd.DataFrame, fund: pd.DataFrame, compact: bool) -> str:
    inc = dict(zip(fund.race_id, fund.incumbent_party))
    out = []
    for r in df.itertuples():
        state = STATE.get(r.state, r.state)
        p = float(r.p_dem)
        dem = p >= .5
        if r.main_party == "I":
            tag, cls = "IND", "h-i"
            name = IND_NAMES.get(r.race_id, "the independent")
            note = f'<span class="mk dim">{name} &rarr; counts only if he caucuses</span>'
        else:
            held = inc.get(r.race_id, "R")
            tag, cls = ("D hold", "h-d") if held == "D" else ("FLIP", "h-r")
            note = ""
        out.append(
            f'<tr><th scope="row"><span class="st">{state}</span>'
            f'<span class="hd {cls}">{tag}</span>{note}</th>'
            f'<td class="bar"><span class="track"><span class="fifty"></span>'
            f'<span class="fill {"f-dem" if dem else "f-rep"}" style="width:{p*100:.1f}%"></span></span></td>'
            f'<td class="pc {"c-dem" if dem else "c-rep"}">{round(p*100)}<span class="u">%</span></td></tr>')
    return "".join(out)


def ladder_card(title, eyebrow, h1, sub, df, fund, asof_line) -> str:
    n = len(df)
    compact = n > 20
    vars_ = (":root{--rowpad:2px;--stsize:19px;--barh:15px;--pcsize:21px}" if compact
             else ":root{--rowpad:5px;--stsize:24px;--barh:25px;--pcsize:30px}")
    css = CSS_BASE + LADDER_CSS + vars_
    return (HEAD.format(title=title, css=css) +
            f'<div class="card{" compact" if compact else ""}">'
            f'<p class="eyebrow">{eyebrow}</p><h1>{h1}</h1>'
            f'<p class="sub">{sub}</p>'
            f'<table><tbody>{ladder_rows(df, fund, compact)}</tbody></table>'
            f'<div class="foot"><span class="key">'
            f'<span class="k"><span class="sw sw-d"></span>Democrat favoured</span>'
            f'<span class="k"><span class="sw sw-r"></span>Republican favoured</span></span>'
            f'<span>{asof_line}</span></div></div></body></html>')


# --------------------------------------------------------------------------
# Chamber distribution card
# --------------------------------------------------------------------------
CHAMBER_CSS = """
.card{padding:70px 64px 52px}
h1{font-family:var(--display);font-size:104px;font-weight:700;line-height:.98;letter-spacing:-.025em;margin:18px 0 0}
.sub{margin:18px 0 0;font-size:27px;color:var(--ink-2);line-height:1.35}
.hero{display:flex;align-items:center;gap:34px;margin:54px 0 0}
.pct{font-family:var(--display);font-weight:700;font-size:186px;line-height:.84;color:var(--dem);
 font-variant-numeric:tabular-nums;letter-spacing:-.035em}
.pct i{font-style:normal;font-size:76px;letter-spacing:0}
.heroL{font-size:30px;color:var(--ink-2);line-height:1.28}
.rule{border:0;border-top:1px solid var(--line);margin:50px 0 0}
.range{margin:34px 0 0;font-size:31px;color:var(--ink-2);line-height:1.3}
.range b{color:var(--ink);font-weight:800;font-variant-numeric:tabular-nums}
.range .big{font-family:var(--display);font-size:44px;font-weight:700;letter-spacing:-.01em}
svg{width:100%;height:auto;display:block;margin-top:26px;overflow:visible}
rect.d{fill:var(--dem)} rect.r{fill:var(--rep)}
.tk{font-family:var(--body);font-size:21px;fill:var(--ink-3);font-variant-numeric:tabular-nums}
.lbl{font-family:var(--body);font-size:23px;font-weight:800;fill:var(--ink)}
"""

CH = {
    "House":    dict(thr=218, unit="seats", range_lbl="80% interval seat range",
                     sub="435 seats up &middot; 218 for control",
                     win="chance Democrats<br>win control"),
    "Senate":   dict(thr=51, unit="seats", range_lbl="80% interval seat range",
                     sub="35 seats up &middot; 51 for control",
                     win="chance Democrats<br>win control"),
    "Governor": dict(thr=26, unit="governorships", range_lbl="80% interval range",
                     sub="36 up this year &middot; 26 of 50 for a majority",
                     win="chance Democrats hold<br>26+ of 50 offices"),
}

W, H, PAD_B, TOP = 952.0, 548.0, 42.0, 26.0


def histogram(df: pd.DataFrame, thr: int, office: str, trim: float = 4e-4) -> str:
    d = df.copy()
    d["prob"] = d["count"] / d["count"].sum()
    keep = d[d.prob >= trim]
    lo, hi = int(keep.dem_seats.min()), int(keep.dem_seats.max())
    d = d[(d.dem_seats >= lo) & (d.dem_seats <= hi)].reset_index(drop=True)
    n = len(d)
    pitch = W / n
    bw = max(pitch - 2.0, 1.0)            # 2px surface gap between adjacent bars
    pmax, plot_h, base = d.prob.max(), H - PAD_B - TOP, H - PAD_B
    bars = []
    for r in d.itertuples():
        h = (r.prob / pmax) * plot_h
        bars.append(f'<rect class="{"d" if r.dem_seats >= thr else "r"}" '
                    f'x="{(r.dem_seats - lo) * pitch:.1f}" y="{base - h:.1f}" '
                    f'width="{bw:.1f}" height="{max(h, 1.0):.1f}" rx="4"/>')
    xthr = (thr - lo) * pitch - 1.0
    step = 10 if (hi - lo) > 40 else 5
    ticks = [t for t in range(lo - lo % step + step, hi + 1, step) if abs(t - thr) > step * .6]
    tk = "".join(f'<text class="tk" x="{(t - lo) * pitch + bw / 2:.1f}" y="{base + 30:.0f}" '
                 f'text-anchor="middle">{t}</text>' for t in ticks)
    anchor = "start" if (thr - lo) < n * .72 else "end"
    dx = 12 if anchor == "start" else -12
    med = int(d.dem_seats[(d.prob.cumsum() / d.prob.sum()) >= .5].iloc[0])
    return (f'<svg viewBox="0 0 {W:.0f} {H + 14:.0f}" role="img" aria-label="Distribution of '
            f'Democratic {office} seats across 69,420 simulations, median {med}, majority at {thr}">'
            f'<g>{"".join(bars)}</g>'
            f'<line x1="{xthr:.1f}" y1="{TOP - 18:.0f}" x2="{xthr:.1f}" y2="{base:.1f}" '
            f'stroke="var(--ink)" stroke-width="2.5" stroke-dasharray="7 5"/>'
            f'<text class="lbl" x="{xthr + dx:.1f}" y="{TOP - 20:.0f}" text-anchor="{anchor}">'
            f'{thr} = majority</text>{tk}</svg>')


def chamber_card(office: str, cc: dict, df: pd.DataFrame, asof_line: str) -> str:
    m, c = CH[office], cc[office]
    pct = round(c["p_dem_control"] * 100)
    med, p10, p90 = int(c["dem_seats_median"]), int(c["dem_seats_p10"]), int(c["dem_seats_p90"])
    return (HEAD.format(title=f"{office} Seat Forecast", css=CSS_BASE + CHAMBER_CSS) +
            f'<div class="card"><p class="eyebrow">2026 Midterms &middot; Seat Forecast</p>'
            f'<h1>{office}</h1><p class="sub">{m["sub"]}</p>'
            f'<div class="hero"><span class="pct">{pct}<i>%</i></span>'
            f'<span class="heroL">{m["win"]}</span></div><hr class="rule">'
            f'<p class="range"><span class="big">{med}</span> {m["unit"]} is the median outcome'
            f'&nbsp;&middot;&nbsp; {m["range_lbl"]} <b>{p10}&ndash;{p90}</b></p>'
            f'{histogram(df, m["thr"], office)}'
            f'<div class="foot"><span class="key">'
            f'<span class="k"><span class="sw sw-d"></span>Democratic majority</span>'
            f'<span class="k"><span class="sw sw-r"></span>Republican majority</span></span>'
            f'<span>{asof_line}</span></div></div></body></html>')


# --------------------------------------------------------------------------
def main():
    O = _ROOT / "outputs"
    cc = json.loads((O / "chamber_control.json").read_text())
    rp = pd.read_csv(O / "race_probabilities.csv")
    fund = pd.read_parquet(config.DATA_PROCESSED / "fundamentals_2026.parquet")
    asof = cc.get("asof", str(config.FORECAST_ASOF))
    days = (config.ELECTION_DATE - pd.Timestamp(asof).date()).days
    nice = pd.Timestamp(asof).strftime("%-d %b %Y")
    asof_line = f"{cc['n_sims']:,} simulations &middot; {nice}"
    OUTD.mkdir(parents=True, exist_ok=True)

    # Senate ladder: the watchlist, minus the races that only pad it.
    wl = pd.read_csv(config.DATA_MANUAL / "watchlist.csv", comment="#")
    sen_ids = [r for r in wl.race_id if r.startswith("S-") and r not in EXCLUDE_SENATE]
    sen = rp[rp.race_id.isin(sen_ids)].sort_values("p_dem", ascending=False)
    settled = 35 - len(sen)
    (OUTD / "ig_senate_ladder.html").write_text(ladder_card(
        "Road to 51", f"2026 Senate &middot; road to 51",
        f"{len(sen)} races decide<br>control of the Senate.",
        "Democrats hold <b>34 seats</b> not on the ballot. Each bar is the model's chance "
        "Democrats win that race; the tick is 50%.",
        sen, fund, asof_line))

    # Governor ladder: every race on the ballot.
    gov = rp[rp.office == "Governor"].sort_values("p_dem", ascending=False)
    (OUTD / "ig_governor_ladder.html").write_text(ladder_card(
        "Road to 26", "2026 Governors &middot; road to 26",
        "Democrats get to a majority<br>by holding, not flipping.",
        f"All <b>{len(gov)}</b> races on the ballot, ranked by the model's chance Democrats win. "
        "Democrats hold <b>6</b> governorships that are not up and need <b>26 of 50</b>.",
        gov, fund, asof_line))

    for office, slug in [("House", "house"), ("Senate", "senate"), ("Governor", "governor")]:
        dist = pd.read_csv(O / f"seat_distribution_{slug}.csv")
        (OUTD / f"ig_{slug}.html").write_text(chamber_card(office, cc, dist, asof_line))

    print(f"wrote 5 cards to {OUTD}  (asof {asof}, {days} days out)")
    for f in sorted(OUTD.glob("ig_*.html")):
        print("  ", f.name)


if __name__ == "__main__":
    main()
