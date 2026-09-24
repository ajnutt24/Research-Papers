"""Unit tests for wikipolls.py. Run: python3 test_wikipolls.py

Self-contained: the fixture below mirrors the structure of a real Wikipedia
"Polling" wikitable (two race sections plus one non-poll table that must be
ignored), so the parser can be checked without network access.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wikipolls as w

SAMPLE_HTML = """<h3>Georgia</h3>
<table class="wikitable sortable">
<tbody>
<tr><th>Poll source</th><th>Date(s)<br/>administered</th><th>Sample<br/>size</th><th>Margin<br/>of error</th>
<th>Jon<br/>Ossoff<br/>(D)</th><th>Buddy<br/>Carter<br/>(R)</th><th>Other</th><th>Undecided</th></tr>
<tr><td>Emerson College</td><td>September 10&#8211;12, 2026</td><td>800 (LV)</td><td>&#177; 3.4%</td>
<td><b>51%</b></td><td>45%</td><td>1%</td><td>3%</td></tr>
<tr><td>Quinnipiac University</td><td>September 4&#8211;8, 2026</td><td>1,203 (RV)</td><td>&#177; 2.8%</td>
<td><b>49%</b></td><td>44%</td><td>2%</td><td>5%</td></tr>
<tr><td>AtlasIntel*</td><td>August 28 &#8211; September 2, 2026</td><td>1,050 (LV)</td><td>&#177; 3%</td>
<td>47%</td><td><b>48%</b></td><td>2%</td><td>3%</td></tr>
</tbody></table>
<h3>North Carolina</h3>
<table class="wikitable sortable">
<tbody>
<tr><th>Poll source</th><th>Date(s) administered</th><th>Sample size</th><th>Margin of error</th>
<th>Roy Cooper (D)</th><th>Michael Whatley (R)</th><th>Undecided</th></tr>
<tr><td>Marist College</td><td>September 15&#8211;18, 2026</td><td>912 (LV)</td><td>&#177; 3.9%</td>
<td><b>50%</b></td><td>46%</td><td>4%</td></tr>
<tr><td>East Carolina University</td><td>September 8&#8211;10, 2026</td><td>1,000 (RV)</td><td>&#177; 3.1%</td>
<td><b>48%</b></td><td>45%</td><td>7%</td></tr>
</tbody></table>
<table class="wikitable">
<tbody>
<tr><th>Not a poll table</th><th>Something else</th></tr>
<tr><td>abc</td><td>def</td></tr>
</tbody></table>
"""

HOUSE_HTML = """<h2>District 1</h2>
<table class="wikitable"><tbody>
<tr><th>Poll source</th><th>Date(s) administered</th><th>Sample size</th><th>Alice Adams (D)</th><th>Bob Brown (R)</th></tr>
<tr><td>Public Policy Polling</td><td>September 5&#8211;7, 2026</td><td>600 (LV)</td><td>47%</td><td>49%</td></tr>
</tbody></table>
<h2>District 2</h2>
<table class="wikitable"><tbody>
<tr><th>Poll source</th><th>Date(s) administered</th><th>Sample size</th><th>Carol Chen (D)</th><th>Dan Davis (R)</th></tr>
<tr><td>Split Ticket</td><td>September 9&#8211;11, 2026</td><td>500 (LV)</td><td>52%</td><td>44%</td></tr>
<tr><td>Cygnal*</td><td>September 1&#8211;3, 2026</td><td>450 (LV)</td><td>48%</td><td>48%</td></tr>
</tbody></table>
<h2>Democratic primary</h2>
<table class="wikitable"><tbody>
<tr><th>Poll source</th><th>Date(s) administered</th><th>Sample size</th><th>Carol Chen (D)</th><th>Eve Evans (D)</th></tr>
<tr><td>Some Pollster</td><td>May 1&#8211;3, 2026</td><td>400 (LV)</td><td>55%</td><td>40%</td></tr>
</tbody></table>
<h2>See also</h2>
<p>Links: <a href="/wiki/2026_United_States_Senate_election_in_Georgia">Georgia</a>
<a href="/wiki/2026_United_States_Senate_election_in_North_Carolina">North Carolina</a>
<a href="/wiki/File:Something.png">an image</a>
<a href="/wiki/2026_Arizona_gubernatorial_election">Arizona governor</a></p>
"""

ok = True
def check(label, got, want):
    global ok
    good = got == want
    ok = ok and good
    print(f"  [{'OK ' if good else 'FAIL'}] {label}: got {got!r}" + ("" if good else f", want {want!r}"))

print("parse_end_date")
check("range within a month", w.parse_end_date("September 10–12, 2026"), date(2026, 9, 12))
check("range across months", w.parse_end_date("August 28 – September 2, 2026"), date(2026, 9, 2))
check("single day", w.parse_end_date("September 12, 2026"), date(2026, 9, 12))
check("with footnote", w.parse_end_date("September 12, 2026[1]"), date(2026, 9, 12))
check("garbage", w.parse_end_date("not a date"), None)

print("\nparse_sample")
check("LV with comma", w.parse_sample("1,203 (RV)"), (1203.0, "rv"))
check("LV", w.parse_sample("800 (LV)"), (800.0, "lv"))
check("no population", w.parse_sample("650"), (650.0, "unknown"))

print("\nparse_pct")
check("percent sign", w.parse_pct("51%"), 51.0)
check("decimal", w.parse_pct("48.5%"), 48.5)
check("bare number", w.parse_pct(47), 47.0)
check("dash", w.parse_pct("–") != w.parse_pct("–"), True)   # NaN

print("\nclean_pollster")
check("asterisk means partisan", w.clean_pollster("AtlasIntel*"), ("AtlasIntel", True))
check("plain", w.clean_pollster("Emerson College"), ("Emerson College", False))

print("\nidentify_columns")
cols = ["Poll source", "Date(s) administered", "Sample size", "Margin of error",
        "Jon Ossoff (D)", "Buddy Carter (R)", "Other", "Undecided"]
roles = w.identify_columns(cols)
check("finds dem column", roles and roles["dem"], "Jon Ossoff (D)")
check("finds rep column", roles and roles["rep"], "Buddy Carter (R)")
check("rejects a non-poll table", w.identify_columns(["Not a poll table", "Something else"]), None)

print("\nfull article parse")
html = SAMPLE_HTML
df = w.parse_html(html, "S-GA")
check("row count (3 GA + 2 NC, junk table skipped)", len(df), 5)
if len(df):
    r = df.iloc[0]
    check("first pollster", r.pollster, "Emerson College")
    check("first end_date", r.end_date, date(2026, 9, 12))
    check("first dem_pct", r.dem_pct, 51.0)
    check("first rep_pct", r.rep_pct, 45.0)
    check("first sample", r.sample_size, 800.0)
    check("first population", r.population, "lv")
    check("partisan flagged", df[df.pollster == "AtlasIntel"].partisan.iloc[0], "partisan")


print("\nsection-aware parsing (statewide House article)")
def _rid(heading):
    d = w.district_from_heading(heading)
    return f"H-NE-{d:02d}" if d else None

hdf = w.parse_html_sections(HOUSE_HTML, _rid)
check("general-election polls only (primary table skipped)", len(hdf), 3)
check("districts assigned", sorted(set(hdf.race_id)), ["H-NE-01", "H-NE-02"])
check("primary pollster excluded", "Some Pollster" in set(hdf.pollster), False)
check("partisan asterisk carried through", set(hdf[hdf.pollster == "Cygnal"].partisan), {"partisan"})

print("\ndistrict_from_heading")
check("District 2", w.district_from_heading("District 2"), 2)
check("2nd congressional district", w.district_from_heading("2nd congressional district"), 2)
check("primary heading", w.district_from_heading("Democratic primary"), None)

print("\nlink discovery")
import re as _re
links = w.discover_links(HOUSE_HTML, _re.compile(r"2026.*(Senate election|gubernatorial election)", _re.I))
check("finds Georgia Senate", "2026_United_States_Senate_election_in_Georgia" in links, True)
check("finds Arizona governor", "2026_Arizona_gubernatorial_election" in links, True)
check("skips File: links", any("File:" in t for t in links), False)

print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
sys.exit(0 if ok else 1)
