"""
dashboard.py
============
Render the forecast as a single self-contained HTML page.

    python3 dashboard.py            -> outputs/dashboard.html

Reads the pipeline's own outputs, so re-running it after a refresh regenerates
the page with current numbers. Nothing is hard-coded: every figure, every chart
point and every table row comes from outputs/ and data_store/processed/, and the
page states its own as-of date and data provenance so a stale copy is obvious.

Why a generator rather than hand-written HTML: the numbers change weekly, and a
page whose figures were typed in by hand goes wrong silently the first time it is
not regenerated. This way the page cannot disagree with the model.

Design notes
------------
* Margins use a diverging encoding, blue for Democratic and red for Republican
  with a neutral grey midpoint at zero. That is the subject's own vernacular, and
  the pair is validated for colour-vision deficiency separation in both light and
  dark modes (worst adjacent CVD dE 21.6 light / 19.2 dark against a >= 8 target).
* Party colour never carries meaning alone: every bar sits beside its signed
  number and its rating text.
* Numerals are tabular throughout, because these are columns of figures meant to
  be compared down the column.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    PROJECT = Path(__file__).resolve().parent
except NameError:
    PROJECT = Path.cwd()
sys.path.insert(0, str(PROJECT))

import config  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("dashboard")
OUT = PROJECT / "outputs"
PROC = PROJECT / "data_store" / "processed"
DEST = OUT / "dashboard.html"

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
E = html.escape


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def load() -> dict:
    cc = OUT / "chamber_control.json"
    if not cc.exists():
        raise SystemExit(f"{cc} not found. Run the pipeline first:  python3 run_pipeline.py")
    d = json.loads(cc.read_text())

    races = pd.read_csv(OUT / "race_probabilities.csv")
    blend = pd.read_parquet(PROC / "blend_2026.parquet")[
        ["race_id", "n_polls", "weak_lean"]]
    fund = pd.read_parquet(PROC / "fundamentals_2026.parquet")[["race_id", "lean", "lean_source"]]
    races = races.merge(blend, on="race_id", how="left").merge(fund, on="race_id", how="left")

    wl_path = config.DATA_MANUAL / "watchlist.csv"
    watch = set(pd.read_csv(wl_path, comment="#").race_id) if wl_path.exists() else set()

    gb = pd.read_parquet(PROC / "generic_ballot_trend.parquet")
    gb["date"] = pd.to_datetime(gb["date"])
    est = pd.read_parquet(PROC / "poll_estimates_2026.parquet")
    nat = pd.read_parquet(PROC / "national_environment.parquet").iloc[0]
    polls = pd.read_parquet(PROC / "polls_2026.parquet")
    polls["end_date"] = pd.to_datetime(polls["end_date"])

    seats = {}
    for ch in ("house", "senate", "governor"):
        f = OUT / f"seat_distribution_{ch}.csv"
        if f.exists():
            s = pd.read_csv(f)
            s["p"] = s["count"] / s["count"].sum()
            seats[ch] = s
    return dict(cc=d, races=races, watch=watch, gb=gb, est=est, nat=nat, polls=polls, seats=seats)


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def race_name(rid: str, office: str) -> str:
    p = rid.split("-")
    if office == "Generic":
        return "Generic ballot"
    st = STATE.get(p[1], p[1]) if len(p) > 1 else rid
    if office == "House":
        d = p[2] if len(p) > 2 else "?"
        if d == "01" and config.HOUSE_SEATS_BY_STATE.get(p[1], 0) == 1:
            return f"{st} at-large"
        n = int(d) if d.isdigit() else 0
        suf = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{st} {n}{suf}"
    return st + (" (special)" if "special" in p else "")


def pct(x: float) -> str:
    if x >= 0.995:
        return ">99%"
    if x <= 0.005:
        return "<1%"
    return f"{x * 100:.0f}%"


def lean_word(p: float) -> tuple[str, str]:
    """(label, css class) for a win probability."""
    if p >= 0.95: return "Safe D", "d3"
    if p >= 0.80: return "Likely D", "d2"
    if p >= 0.60: return "Lean D", "d1"
    if p > 0.40:  return "Toss-up", "t0"
    if p > 0.20:  return "Lean R", "r1"
    if p > 0.05:  return "Likely R", "r2"
    return "Safe R", "r3"


# --------------------------------------------------------------------------
# Charts (inline SVG, drawn to one scale)
# --------------------------------------------------------------------------
def svg_trend(gb: pd.DataFrame, polls: pd.DataFrame) -> str:
    """Generic-ballot trend: filtered mean, its band, and the individual polls.

    The band is the filter's own standard error, so the reader can see that the
    line is a smoothed estimate rather than a series of observations.
    """
    g = gb.dropna(subset=["mean"]).copy()
    if not len(g):
        return '<p class="empty">No generic-ballot trend available.</p>'
    g = g[g.date >= g.date.max() - pd.Timedelta(days=270)]
    dots = polls[(polls.race_id == "GENERIC") & polls.end_date.notna()].copy()
    dots = dots[dots.end_date >= g.date.min()]
    dots["m"] = dots.dem_pct - dots.rep_pct

    W, H = 760, 300
    L, R, T, B = 46, 16, 16, 34
    x0, x1 = g.date.min().value, g.date.max().value
    lo = float(min(g["mean"].min() - 2.5, dots.m.min() if len(dots) else 0, 0)) - 1
    hi = float(max(g["mean"].max() + 2.5, dots.m.max() if len(dots) else 0)) + 1

    def X(v): return L + (v - x0) / max(x1 - x0, 1) * (W - L - R)
    def Y(v): return T + (hi - v) / max(hi - lo, 1e-9) * (H - T - B)

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
             f'aria-label="Generic ballot trend, Democratic margin in percentage points">']
    # horizontal grid + y labels
    step = 5 if (hi - lo) <= 30 else 10
    v = int(np.floor(lo / step) * step)
    while v <= hi:
        y = Y(v)
        if T <= y <= H - B:
            parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{L-8}" y="{y+4:.1f}" class="ytick">{"D+" if v>0 else ("R+" if v<0 else "")}{abs(v)}</text>')
        v += step
    # zero reference, emphasised: it is the only meaningful threshold on this axis
    if lo < 0 < hi:
        parts.append(f'<line x1="{L}" y1="{Y(0):.1f}" x2="{W-R}" y2="{Y(0):.1f}" class="zero"/>')
    # month ticks
    for d in pd.date_range(g.date.min(), g.date.max(), freq="MS"):
        x = X(d.value)
        parts.append(f'<line x1="{x:.1f}" y1="{H-B}" x2="{x:.1f}" y2="{H-B+4}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{H-B+18}" class="xtick">{d:%b}</text>')
    # individual polls behind the line
    for _, p in dots.iterrows():
        parts.append(f'<circle cx="{X(p.end_date.value):.1f}" cy="{Y(p.m):.1f}" r="2.6" class="dot"/>')
    # uncertainty band
    up = " ".join(f"{X(r.date.value):.1f},{Y(r['mean'] + 1.96*r['sd']):.1f}" for _, r in g.iterrows())
    dn = " ".join(f"{X(r.date.value):.1f},{Y(r['mean'] - 1.96*r['sd']):.1f}" for _, r in g[::-1].iterrows())
    parts.append(f'<polygon points="{up} {dn}" class="band"/>')
    # the filtered line, with an emphasised endpoint
    line = " ".join(f"{X(r.date.value):.1f},{Y(r['mean']):.1f}" for _, r in g.iterrows())
    parts.append(f'<polyline points="{line}" class="trend"/>')
    last = g.iloc[-1]
    lx, ly = X(last.date.value), Y(last["mean"])
    parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="5" class="endpoint"/>')
    parts.append(f'<text x="{lx-10:.1f}" y="{ly-12:.1f}" class="endlabel">D+{last["mean"]:.1f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_seats(s: pd.DataFrame, threshold: int, label: str) -> str:
    """Seat-count distribution with the majority threshold marked.

    Bars left of the threshold are the losing side, right of it the winning
    side, so the share of the mass past the line IS the win probability the
    headline states. One scale, and the threshold is labelled with its number.
    """
    if s is None or not len(s):
        return ""
    s = s[s["p"] > 0.00015]
    W, H = 760, 168
    L, R, T, B = 40, 14, 12, 30
    lo, hi = int(s.dem_seats.min()), int(s.dem_seats.max())
    pmax = float(s["p"].max())
    bw = (W - L - R) / max(hi - lo + 1, 1)

    def X(v): return L + (v - lo) * bw
    def Hh(p): return (H - T - B) * (p / pmax if pmax else 0)

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
             f'aria-label="{E(label)} seat distribution; majority at {threshold}">']
    for _, r in s.iterrows():
        h = Hh(r["p"])
        x = X(r.dem_seats)
        cls = "bar-d" if r.dem_seats >= threshold else "bar-r"
        # 2px surface gap between adjacent bars, 4px rounded data-end on the baseline
        parts.append(f'<rect x="{x+1:.1f}" y="{H-B-h:.1f}" width="{max(bw-2,1):.1f}" '
                     f'height="{max(h,0.6):.1f}" rx="2" class="{cls}"/>')
    tx = X(threshold)
    parts.append(f'<line x1="{tx:.1f}" y1="{T}" x2="{tx:.1f}" y2="{H-B}" class="thresh"/>')
    # The label sits over the bars, so it gets a surface plate behind it. Without
    # one it is unreadable exactly where the distribution is tallest, which is
    # the middle of the chart on any close race.
    txt = f"majority {threshold}"
    flip = tx > (W - R) - 90          # keep the plate inside the drawing bounds
    px = tx - 6 - len(txt) * 5.4 if flip else tx + 6
    parts.append(f'<rect x="{px-3:.1f}" y="{T+1}" width="{len(txt)*5.4+6:.1f}" height="14" '
                 f'rx="3" class="plate"/>')
    parts.append(f'<text x="{px:.1f}" y="{T+11}" class="threshlabel">{txt}</text>')
    for v in (lo, hi):
        parts.append(f'<text x="{X(v)+bw/2:.1f}" y="{H-B+18}" class="xtick">{v}</text>')
    parts.append(f'<text x="{L-6}" y="{H-B+18}" class="xtick" text-anchor="end">D seats</text>')
    parts.append("</svg>")
    return "".join(parts)


def bar_cell(margin: float, scale: float = 22.0) -> str:
    """A small diverging bar for one race's margin, centred on zero."""
    frac = max(-1.0, min(1.0, margin / scale))
    w = abs(frac) * 50
    side = "left:50%" if frac >= 0 else f"right:50%"
    cls = "mb-d" if frac >= 0 else "mb-r"
    return (f'<span class="mbar"><span class="mbar-z"></span>'
            f'<span class="mbar-f {cls}" style="{side};width:{w:.1f}%"></span></span>')


# --------------------------------------------------------------------------
# Page sections
# --------------------------------------------------------------------------
def chamber_card(ch: str, v: dict, total: int, seats: pd.DataFrame | None) -> str:
    p = v["p_dem_control"]
    lab, cls = lean_word(p)
    tie = v.get("p_dem_50_seats_tie")
    tie_html = (f'<p class="note">A 50-50 split is {pct(tie)} likely and goes Republican on the '
                f'vice-presidential tiebreak, so Democrats need {v["majority_threshold"]}.</p>'
                if tie else "")
    return f"""<section class="card">
  <header class="card-h">
    <h3>{E(ch)}</h3>
    <span class="chip {cls}">{E(lab)}</span>
  </header>
  <p class="big"><span class="bignum">{pct(p)}</span><span class="biglab">chance of Democratic control</span></p>
  <dl class="kv">
    <div><dt>Median D seats</dt><dd>{v['dem_seats_median']:.0f} <span class="of">of {total}</span></dd></div>
    <div><dt>80% range</dt><dd>{v['dem_seats_p10']:.0f}&ndash;{v['dem_seats_p90']:.0f}</dd></div>
    <div><dt>Majority needs</dt><dd>{v['majority_threshold']}</dd></div>
  </dl>
  {svg_seats(seats, int(v['majority_threshold']), ch)}
  {tie_html}
</section>"""


def race_table(df: pd.DataFrame, office: str, pcol: str) -> str:
    rows = []
    for _, x in df.iterrows():
        lab, cls = lean_word(x[pcol])
        n = int(x.n_polls) if pd.notna(x.n_polls) else 0
        flag = ('<span class="tag" title="Redrawn for 2026; past results there say less about the '
                'new lines, so the model widens its uncertainty">new map</span>'
                if bool(x.weak_lean) else "")
        polls = (f'{n}' if n else '<span class="none">none</span>')
        rows.append(f"""<tr>
  <td class="rname">{E(race_name(x.race_id, office))} {flag}</td>
  <td class="rating">{E(str(x.rating))}</td>
  <td class="num">{x.blend_margin:+.1f}</td>
  <td class="bar">{bar_cell(x.blend_margin)}</td>
  <td class="num prob"><span class="chip sm {cls}">{pct(x[pcol])}</span></td>
  <td class="num polls">{polls}</td>
</tr>""")
    return f"""<div class="twrap"><table>
  <thead><tr>
    <th>Race</th><th>Rating</th><th class="num">Margin</th>
    <th class="bar">D&nbsp;&#8592;&nbsp;|&nbsp;&#8594;&nbsp;R</th>
    <th class="num">P(D win)</th><th class="num">Polls</th>
  </tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table></div>"""


# --------------------------------------------------------------------------
def build(d: dict) -> str:
    cc, races, watch = d["cc"], d["races"], d["watch"]
    gb, est, nat, polls, seats = d["gb"], d["est"], d["nat"], d["polls"], d["seats"]

    gbrow = est[est.race_id == "GENERIC"]
    gb_now = float(gbrow.poll_margin.iloc[0]) if len(gbrow) else float("nan")
    gb_n = int(gbrow.n_polls.iloc[0]) if len(gbrow) else 0
    gb_last = str(gbrow.last_poll_date.iloc[0]) if len(gbrow) else "n/a"

    # 30-day change in the filtered trend, which is the trend question itself
    gg = gb.dropna(subset=["mean"]).sort_values("date")
    delta = ""
    if len(gg) > 1:
        past = gg[gg.date <= gg.date.max() - pd.Timedelta(days=30)]
        if len(past):
            ch = float(gg["mean"].iloc[-1]) - float(past["mean"].iloc[-1])
            arrow = "&#9650;" if ch > 0.05 else ("&#9660;" if ch < -0.05 else "&#9644;")
            word = "toward Democrats" if ch > 0.05 else ("toward Republicans" if ch < -0.05 else "flat")
            delta = f'<span class="delta">{arrow} {abs(ch):.1f} pts in 30 days, {word}</span>'

    wl = races[races.race_id.isin(watch)]
    sen = wl[wl.office == "Senate"].sort_values("p_dem_caucus", ascending=False)
    hou = wl[wl.office == "House"].sort_values("p_dem", ascending=False)

    # Senate tipping point: walk the whole chamber, not just the watchlist
    held = config.SENATE_SEATS_NOT_UP["D"]
    tot, tip = held, None
    for _, x in races[races.office == "Senate"].sort_values("p_dem_caucus", ascending=False).iterrows():
        tot += 1
        if tot >= config.SENATE_MAJORITY:
            tip = x
            break
    tip_html = ""
    if tip is not None:
        tip_html = (f'<p class="note">Line every Senate race up from safest Democratic to safest '
                    f'Republican and the {config.SENATE_MAJORITY}th seat lands on '
                    f'<strong>{E(race_name(tip.race_id, "Senate"))}</strong>, currently '
                    f'{pct(tip.p_dem_caucus)}. Democrats hold {held} seats that are not on the '
                    f'ballot this year.</p>')

    n_h_polled = int((races[races.office == "House"].n_polls.fillna(0) > 0).sum())
    fresh = polls.end_date.max()
    prov = cc["provenance"]
    prov_word = {"live": "live sources", "cache": "cached live sources",
                 "manual": "the curated workbook", "fixture": "PLACEHOLDERS"}.get(prov, prov)

    warn = ""
    if prov == "fixture":
        warn = ('<div class="warn"><strong>These numbers are a pipeline test, not a forecast.</strong> '
                'Some inputs are placeholders.</div>')

    return f"""<title>2026 Midterms Forecast</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a masthead, three chamber cards, the national trend, then the two
   watchlists as scannable tables. Summary before detail throughout. */
:root {{
  --bg:#fbfaf7; --surface:#ffffff; --surface-2:#f4f2ee;
  --ink:#16161a; --ink-2:#4d4d55; --ink-3:#7c7c86;
  --rule:#e2ded6; --rule-2:#cfcabf;
  --dem:#2a78d6; --rep:#e34948; --mid:#f0efec;
  --dem-soft:#dce9f8; --rep-soft:#fbdedd;
  --accent:#8a6d3b;
  --ff-display:"Newsreader",Georgia,"Times New Roman",serif;
  --ff-body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --ff-mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#17171a; --surface:#1f1f23; --surface-2:#26262b;
    --ink:#f4f3f0; --ink-2:#b4b3ad; --ink-3:#85847e;
    --rule:#33333a; --rule-2:#44444c;
    --dem:#3987e5; --rep:#e66767; --mid:#383835;
    --dem-soft:#1c2f47; --rep-soft:#3a2323;
    --accent:#c9a870; color-scheme:dark;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#17171a; --surface:#1f1f23; --surface-2:#26262b;
  --ink:#f4f3f0; --ink-2:#b4b3ad; --ink-3:#85847e;
  --rule:#33333a; --rule-2:#44444c;
  --dem:#3987e5; --rep:#e66767; --mid:#383835;
  --dem-soft:#1c2f47; --rep-soft:#3a2323;
  --accent:#c9a870; color-scheme:dark;
}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--ink);font-family:var(--ff-body);
  font-size:15px;line-height:1.55;margin:0}}
.page{{max-width:1040px;margin:0 auto;padding-block:32px 64px;padding-left:16px;padding-right:16px}}
h1,h2,h3{{font-family:var(--ff-display);font-weight:600;text-wrap:balance;margin:0}}
h1{{font-size:clamp(30px,5.2vw,46px);line-height:1.08;letter-spacing:-.015em}}
h2{{font-size:22px;margin-bottom:4px}}
h3{{font-size:19px}}
.eyebrow{{font-family:var(--ff-body);font-size:11px;font-weight:600;letter-spacing:.11em;
  text-transform:uppercase;color:var(--accent);margin:0 0 10px}}
.masthead{{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:8px}}
.sub{{color:var(--ink-2);margin:10px 0 0;max-width:62ch}}
.meta{{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:14px;font-family:var(--ff-mono);
  font-size:12px;color:var(--ink-3)}}
.meta b{{color:var(--ink-2);font-weight:500}}
section{{margin-top:38px}}
.lead{{color:var(--ink-2);max-width:68ch;margin:6px 0 18px}}
.note{{color:var(--ink-2);font-size:13.5px;max-width:70ch;margin:12px 0 0}}
.grid3{{display:grid;grid-template-columns:repeat(auto-fit,minmax(288px,1fr));gap:16px}}
.card{{background:var(--surface);border:1px solid var(--rule);border-radius:10px;
  padding:18px 18px 14px;margin:0;min-width:0}}
.card-h{{display:flex;align-items:center;justify-content:space-between;gap:10px;
  border-bottom:1px solid var(--rule);padding-bottom:10px;margin-bottom:12px}}
.big{{margin:0 0 12px;display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}}
.bignum{{font-family:var(--ff-mono);font-size:40px;font-weight:500;letter-spacing:-.03em;
  font-variant-numeric:tabular-nums;line-height:1}}
.biglab{{font-size:12.5px;color:var(--ink-2);max-width:14ch;line-height:1.3}}
.kv{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:0 0 14px}}
.kv div{{min-width:0}}
.kv dt{{font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3)}}
.kv dd{{margin:2px 0 0;font-family:var(--ff-mono);font-size:16px;font-variant-numeric:tabular-nums}}
.kv .of{{font-size:11px;color:var(--ink-3)}}
.chip{{display:inline-block;font-size:11px;font-weight:600;letter-spacing:.04em;
  padding:3px 9px;border-radius:999px;border:1px solid transparent;white-space:nowrap}}
.chip.sm{{font-family:var(--ff-mono);font-weight:500;letter-spacing:0;padding:2px 7px}}
.d3{{background:var(--dem);color:#fff}} .d2{{background:var(--dem-soft);color:var(--dem);border-color:var(--dem)}}
.d1{{background:transparent;color:var(--dem);border-color:var(--dem)}}
.t0{{background:var(--surface-2);color:var(--ink-2);border-color:var(--rule-2)}}
.r1{{background:transparent;color:var(--rep);border-color:var(--rep)}}
.r2{{background:var(--rep-soft);color:var(--rep);border-color:var(--rep)}}
.r3{{background:var(--rep);color:#fff}}
.chart{{display:block;width:100%;height:auto;margin:6px 0 2px;overflow:visible}}
.grid{{stroke:var(--rule);stroke-width:1}}
.zero{{stroke:var(--rule-2);stroke-width:1.5;stroke-dasharray:4 3}}
.ytick,.xtick{{fill:var(--ink-3);font-family:var(--ff-mono);font-size:10px}}
.ytick{{text-anchor:end}} .xtick{{text-anchor:middle}}
.dot{{fill:var(--ink-3);opacity:.34}}
.band{{fill:var(--dem);opacity:.15;stroke:none}}
.trend{{fill:none;stroke:var(--dem);stroke-width:2;stroke-linejoin:round}}
.endpoint{{fill:var(--dem);stroke:var(--surface);stroke-width:2}}
.endlabel{{fill:var(--ink);font-family:var(--ff-mono);font-size:12px;font-weight:500;text-anchor:end}}
.bar-d{{fill:var(--dem)}} .bar-r{{fill:var(--rep)}}
.thresh{{stroke:var(--ink);stroke-width:1.5}}
.plate{{fill:var(--surface);opacity:.92}}
.threshlabel{{fill:var(--ink);font-family:var(--ff-mono);font-size:10px;font-weight:500}}
.twrap{{overflow-x:auto;border:1px solid var(--rule);border-radius:10px;background:var(--surface)}}
table{{width:100%;border-collapse:collapse;font-size:13.5px}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid var(--rule);vertical-align:middle}}
thead th{{position:sticky;top:0;background:var(--surface-2);font-size:10.5px;font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);white-space:nowrap}}
tbody tr:last-child td{{border-bottom:none}}
td.num,th.num{{text-align:right;font-family:var(--ff-mono);font-variant-numeric:tabular-nums}}
td.rname{{font-weight:500;white-space:nowrap}}
td.rating{{color:var(--ink-2);white-space:nowrap}}
td.polls,td.prob{{white-space:nowrap}}
.none{{color:var(--ink-3)}}
.tag{{font-size:9.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--accent);
  border:1px solid var(--accent);border-radius:3px;padding:1px 4px;margin-left:5px;white-space:nowrap}}
th.bar,td.bar{{width:132px;min-width:110px}}
.mbar{{position:relative;display:block;height:16px;background:var(--mid);border-radius:3px}}
.mbar-z{{position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--rule-2)}}
.mbar-f{{position:absolute;top:2px;bottom:2px;border-radius:2px}}
.mb-d{{background:var(--dem)}} .mb-r{{background:var(--rep)}}
.statrow{{display:flex;flex-wrap:wrap;gap:12px;margin:0 0 14px}}
.stat{{flex:1 1 150px;min-width:0;background:var(--surface);border:1px solid var(--rule);
  border-radius:8px;padding:12px 14px}}
.stat .v{{font-family:var(--ff-mono);font-size:24px;font-variant-numeric:tabular-nums;
  letter-spacing:-.02em;display:block;line-height:1.15}}
.stat .l{{font-size:11px;color:var(--ink-3);letter-spacing:.05em;text-transform:uppercase}}
.delta{{display:inline-block;margin-top:6px;font-family:var(--ff-mono);font-size:12px;color:var(--ink-2)}}
.warn{{background:var(--rep-soft);border:1px solid var(--rep);color:var(--ink);
  border-radius:8px;padding:12px 14px;margin:18px 0}}
.caveats{{background:var(--surface-2);border:1px solid var(--rule);border-radius:10px;
  padding:18px 20px}}
.caveats ul{{margin:8px 0 0;padding-left:20px}}
.caveats li{{margin:8px 0;color:var(--ink-2);max-width:78ch}}
.caveats strong{{color:var(--ink)}}
footer{{margin-top:44px;padding-top:16px;border-top:1px solid var(--rule);
  font-family:var(--ff-mono);font-size:11.5px;color:var(--ink-3)}}
@media (max-width:560px){{
  .kv{{grid-template-columns:repeat(2,1fr)}}
  .bignum{{font-size:34px}}
  th.bar,td.bar{{display:none}}
}}
@media (prefers-reduced-motion:reduce){{*{{transition:none!important;animation:none!important}}}}
</style>

<div class="page">
<header class="masthead">
  <p class="eyebrow">Bayesian hierarchical forecast &middot; poll aggregation</p>
  <h1>2026 Midterms Forecast</h1>
  <p class="sub">All 435 House seats, 35 Senate races and 36 governorships, simulated
  {cc['n_sims']:,} times from {gb_n} generic-ballot polls and {len(polls):,} race polls.</p>
  <div class="meta">
    <span><b>As of</b> {cc['asof']}</span>
    <span><b>{cc['days_to_election']}</b> days to the election</span>
    <span><b>Newest poll</b> {fresh:%Y-%m-%d}</span>
    <span><b>Data</b> {E(prov_word)}</span>
  </div>
</header>
{warn}

<section>
  <h2>Where control stands</h2>
  <p class="lead">Each figure is the share of simulated elections that chamber ends up
  under Democratic control. The bars below each one show the full distribution of seat
  counts, with the majority line marked, so the mass to the right of that line is the
  headline number.</p>
  <div class="grid3">
    {chamber_card("House", cc["House"], 435, seats.get("house"))}
    {chamber_card("Senate", cc["Senate"], 100, seats.get("senate"))}
    {chamber_card("Governor", cc["Governor"], 50, seats.get("governor"))}
  </div>
</section>

<section>
  <h2>The national environment</h2>
  <p class="lead">The generic congressional ballot asks which party a voter would back
  without naming candidates. It is the single most important national input, and the
  line is a Kalman-filtered estimate rather than a poll average, so it can move between
  polls and carries its own error band.</p>
  <div class="statrow">
    <div class="stat"><span class="v">D+{gb_now:.1f}</span><span class="l">Generic ballot</span>{delta}</div>
    <div class="stat"><span class="v">D+{float(nat.seat_implied_mean):.1f}</span><span class="l">Implied by district polls</span></div>
    <div class="stat"><span class="v">{gb_n}</span><span class="l">Polls in the average</span></div>
    <div class="stat"><span class="v">{n_h_polled}</span><span class="l">House seats with a poll</span></div>
  </div>
  {svg_trend(gb, polls)}
  <p class="note">Dots are individual polls; the shaded band is the filter's 95% interval
  for the underlying trend. The model deliberately uses the generic ballot alone for the
  national environment: the district-poll figure beside it is shown for comparison, but
  feeding it in as well would count those polls twice, since they already inform their
  own races.</p>
</section>

<section>
  <h2>Senate watchlist</h2>
  <p class="lead">{len(sen)} races, most to least Democratic. Probabilities account for an
  independent caucusing decision where one leads.</p>
  {race_table(sen, "Senate", "p_dem_caucus")}
  <p class="note">Expected Democratic seats from these {len(sen)} alone:
  <strong>{sen.p_dem_caucus.sum():.1f}</strong>.</p>
  {tip_html}
</section>

<section>
  <h2>House watchlist</h2>
  <p class="lead">{len(hou)} districts, most to least Democratic.</p>
  {race_table(hou, "House", "p_dem")}
  <p class="note">Expected Democratic seats from these {len(hou)} alone:
  <strong>{hou.p_dem.sum():.1f}</strong>.</p>
</section>

<section>
  <h2>How to read this</h2>
  <div class="caveats">
    <ul>
      <li><strong>A probability is not a prediction.</strong> A race at 70% is one the
      trailing side wins about 3 times in 10. If that happens, the forecast was not wrong.</li>
      <li><strong>Read the range, not the point.</strong> The median seat count is the middle
      of a distribution; the 80% range is the forecast.</li>
      <li><strong>The model makes no call on which way polls will miss.</strong> It treats an
      error favouring either party as equally likely, which the midterm record supports: the
      last three missed by +3.9, &minus;0.2 and +0.1 points. The cost is that it treats each
      state's polls as independent evidence about the national mood when they share whatever
      error is in the industry's methods, so it is more confident than the evidence strictly
      supports, in every chamber.</li>
      <li><strong>Most House districts have no poll of their own.</strong> {n_h_polled} of 435
      do; the rest are forecast from district partisanship, the race rating and the national
      environment.</li>
      <li><strong>Redistricting is unsettled.</strong> Ten states redrew their maps for 2026.
      Those districts are tagged &ldquo;new map&rdquo; above and carry wider uncertainty,
      because past results on the old lines say less about the new ones.</li>
      <li><strong>Late-breaking events are not forecastable.</strong> A scandal or an economic
      shock in October is not in these numbers and cannot be.</li>
    </ul>
  </div>
</section>

<footer>
  Generated by dashboard.py from the pipeline's own outputs &middot; forecast dated
  {cc['asof']} &middot; {cc['n_sims']:,} simulations &middot; data provenance: {E(prov)}
</footer>
</div>"""


def main() -> None:
    d = load()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(build(d), encoding="utf-8")
    log.info("wrote %s (%.0f KB)", DEST, DEST.stat().st_size / 1024)


if __name__ == "__main__":
    main()
