"""
wikipolls.py
============
Parse election polling tables from Wikipedia articles.

Why Wikipedia: it is the only comprehensive, free, structured, legally clear
source of 2026 general-election polls. FiveThirtyEight's poll database was
discontinued in 2025, and the commercial aggregators (RealClearPolitics,
Silver Bulletin, FiftyPlusOne) render their tables in JavaScript and/or
restrict reuse, so a plain HTTP fetch returns no rows. Wikipedia's polling
sections are plain wikitables that `pandas.read_html` reads directly, and
each row cites its original pollster, so the data is traceable.

Table shape this handles (the standard "Polling" wikitable):

    | Poll source | Date(s) administered | Sample size | Margin of error |
    | Jon Ossoff (D) | Buddy Carter (R) | Other | Undecided |
    | Emerson College | September 10-12, 2026 | 800 (LV) | +/- 3.4% | 51% | 45% | ... |

A table is treated as a poll table only if it has a pollster-like column, a
date column, and exactly one Democratic and one Republican candidate column.
Anything else is skipped, so infoboxes and results tables are ignored.

Everything here is pure parsing with no network calls, so it is unit-testable
without internet access; `test_wikipolls.py` exercises it against fixtures.
"""
from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd

MONTHS = ("January|February|March|April|May|June|July|August|September|"
          "October|November|December")
# "September 10-12, 2026" / "August 28 - September 2, 2026" / "September 12, 2026"
DATE_RE = re.compile(
    rf"(?:({MONTHS})\s+(\d{{1,2}})\s*[–—-]\s*)?({MONTHS})?\s*(\d{{1,2}}),?\s*(\d{{4}})",
    re.I)
PCT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%?")
SAMPLE_RE = re.compile(r"([\d,]+)\s*(?:\(\s*(LV|RV|A|V)\s*\))?", re.I)
POLLSTER_COL = re.compile(r"poll(ster)?(\s*source)?|source", re.I)
DATE_COL = re.compile(r"date", re.I)
SAMPLE_COL = re.compile(r"sample", re.I)
# Two states do not put "(D)" on the ballot: Minnesota's affiliate is the
# Democratic-Farmer-Labor party (DFL) and North Dakota's is the
# Democratic-Nonpartisan League (DNL). Without them, every Minnesota poll
# is silently discarded.
DEM_COL = re.compile(r"\(\s*(D|DFL|DNL)\s*\)|\bdemocrat", re.I)
# An independent can be the main alternative to the Republican, and in Idaho,
# Nebraska and South Dakota there is no Democrat at all. A parser that insists
# on a (D) column silently discards every poll of the actual contest.
IND_COL = re.compile(r"\(\s*(I|IND|L|G)\s*\)|\bindependent\b", re.I)
REP_COL = re.compile(r"\(\s*R\s*\)|\brepublican", re.I)
IGNORE_COL = re.compile(r"other|undecided|margin|error|lead|spread|none|someone", re.I)
# "Generic Democrat" is a hypothetical, not a candidate on any ballot.
GENERIC_COL = re.compile(r"\bgeneric\b|\bunnamed\b|\bany\s+(democrat|republican)", re.I)
# Tables that summarise other people's averages are not polls.
AGGREGATOR_COL = re.compile(r"aggregat|average", re.I)
AGGREGATOR_ROW = re.compile(r"270towin|race to the wh|realclear|rcp|fivethirtyeight|538|"
                            r"silver bulletin|decision desk|average|aggregat|split ticket average",
                            re.I)


def _flatten(col) -> str:
    """Wikipedia tables sometimes come back with MultiIndex headers."""
    if isinstance(col, tuple):
        seen, parts = set(), []
        for p in col:
            p = str(p).strip()
            if p and not p.lower().startswith("unnamed") and p not in seen:
                seen.add(p)
                parts.append(p)
        return " ".join(parts)
    return str(col).strip()


def parse_end_date(text: str, default_year: int = 2026) -> date | None:
    """Return the LAST day of a poll's field period.

    'September 10-12, 2026'          -> 2026-09-12
    'August 28 - September 2, 2026'  -> 2026-09-02
    'September 12, 2026'             -> 2026-09-12
    """
    if not isinstance(text, str):
        return None
    t = re.sub(r"\[.*?\]", "", text).strip()          # strip footnote markers
    m = DATE_RE.search(t)
    if not m:
        return None
    start_month, _start_day, end_month, end_day, year = m.groups()
    month_name = end_month or start_month
    if not month_name:
        return None
    try:
        return pd.to_datetime(f"{month_name} {end_day}, {year or default_year}").date()
    except Exception:
        return None


def parse_sample(text: str) -> tuple[float, str]:
    """'1,203 (RV)' -> (1203.0, 'rv'); '800 (LV)' -> (800.0, 'lv')."""
    if not isinstance(text, str):
        if isinstance(text, (int, float)) and text == text:
            return float(text), "unknown"
        return np.nan, "unknown"
    m = SAMPLE_RE.search(text.replace(" ", ""))
    if not m:
        return np.nan, "unknown"
    n = float(m.group(1).replace(",", "")) if m.group(1) else np.nan
    pop = (m.group(2) or "unknown").lower()
    pop = {"v": "unknown", "a": "a", "lv": "lv", "rv": "rv"}.get(pop, "unknown")
    return n, pop


def parse_pct(text) -> float:
    """'51%' -> 51.0. Returns NaN for '-', 'â€“', blanks and other non-numbers."""
    if isinstance(text, (int, float)):
        return float(text) if text == text else np.nan
    if not isinstance(text, str):
        return np.nan
    t = re.sub(r"\[.*?\]", "", text).replace(" ", "").strip()
    t = re.sub(r"<[^>]+>", "", t)
    m = PCT_RE.search(t)
    return float(m.group(1)) if m else np.nan


def clean_pollster(text: str) -> tuple[str, bool]:
    """Return (name, is_partisan). Wikipedia marks internal/partisan polls
    with a trailing asterisk or a bracketed note."""
    if not isinstance(text, str):
        return "", False
    t = re.sub(r"\[.*?\]", "", text).strip()
    partisan = "*" in t
    t = t.replace("*", "").strip()
    t = re.sub(r"\s*\(.*?(internal|for |sponsored).*?\)\s*$", "", t, flags=re.I).strip()
    return t, partisan


def identify_columns(columns: list[str]) -> dict | None:
    """Map a table's columns to roles, or return None if it is not a poll table."""
    pollster = date_c = sample = None
    dem = rep = ind = None
    if any(AGGREGATOR_COL.search(c) for c in columns):
        return None          # a table of other aggregators' averages
    for c in columns:
        low = c.lower()
        if pollster is None and POLLSTER_COL.search(low) and not IGNORE_COL.search(low):
            pollster = c
            continue
        if date_c is None and DATE_COL.search(low):
            date_c = c
            continue
        if sample is None and SAMPLE_COL.search(low):
            sample = c
            continue
        if IGNORE_COL.search(low) or GENERIC_COL.search(c):
            continue
        if rep is None and REP_COL.search(c):
            rep = c
            continue
        if dem is None and DEM_COL.search(c):
            dem = c
            continue
        if ind is None and IND_COL.search(c):
            ind = c
            continue
    # The Democrat is the main alternative when one is running; otherwise the
    # independent is. Either way the margin is main-non-Republican minus
    # Republican, which is what the model consumes.
    main = dem or ind
    if pollster is None or date_c is None or main is None or rep is None:
        return None
    return {"pollster": pollster, "date": date_c, "sample": sample,
            "dem": main, "rep": rep, "main_is_independent": dem is None}



def _candidate_name(column_header: str) -> str:
    """'Dan Osborn (I)' -> 'Dan Osborn'. Lets a caller check whether a table
    polls the actual nominees or a match-up that is no longer on the ballot."""
    return re.sub(r"\s*\(\s*[A-Za-z]{1,3}\s*\)\s*$", "", str(column_header)).strip()


def parse_tables(tables: list[pd.DataFrame], race_id: str, default_year: int = 2026) -> pd.DataFrame:
    """Turn every poll-shaped table into rows of the project's poll schema."""
    rows = []
    for t in tables:
        if t.empty or t.shape[1] < 4:
            continue
        t = t.copy()
        t.columns = [_flatten(c) for c in t.columns]
        roles = identify_columns(list(t.columns))
        if roles is None:
            continue
        for _, r in t.iterrows():
            name, partisan = clean_pollster(r[roles["pollster"]])
            if not name or AGGREGATOR_ROW.search(name):
                continue
            end = parse_end_date(r[roles["date"]], default_year)
            dem = parse_pct(r[roles["dem"]])
            rep = parse_pct(r[roles["rep"]])
            if end is None or dem != dem or rep != rep:
                continue
            n, pop = parse_sample(r[roles["sample"]]) if roles["sample"] else (np.nan, "unknown")
            rows.append({"race_id": race_id, "pollster": name, "sponsor": "",
                         "dem_candidate": _candidate_name(roles["dem"]),
                         "rep_candidate": _candidate_name(roles["rep"]),
                         "main_is_independent": bool(roles.get("main_is_independent")),
                         "partisan": "partisan" if partisan else "",
                         "start_date": end, "end_date": end, "sample_size": n,
                         "population": pop, "methodology": "",
                         "dem_pct": dem, "rep_pct": rep, "hypothetical": False})
    return pd.DataFrame(rows)


def parse_html(html: str, race_id: str, default_year: int = 2026) -> pd.DataFrame:
    """Read every table in an article and return the poll rows found."""
    import io
    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except Exception:
        return pd.DataFrame()
    return parse_tables(tables, race_id, default_year)

# --------------------------------------------------------------------------
# Section-aware parsing
# --------------------------------------------------------------------------
# `pandas.read_html` returns tables with no idea which heading they sat under.
# That is fine for a single-race article, but a statewide House article holds
# one polling table per district, and a Senate article can hold a general
# election table plus primary and hypothetical-matchup tables. Walking the DOM
# keeps each table attached to its nearest preceding heading so the caller can
# tell them apart.
HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
DISTRICT_RE = re.compile(r"(?:district\s+(\d{1,2})\b|\b(\d{1,2})(?:st|nd|rd|th)\s+(?:congressional\s+)?district)", re.I)
PRIMARY_RE = re.compile(r"primary|caucus|nomination|runoff", re.I)
# Sections polling match-ups that cannot happen (a candidate who lost a
# primary, or a declared non-candidate). The build spec requires these be
# excluded: they measure a ballot no voter will see.
HYPOTHETICAL_RE = re.compile(r"hypothetical|potential match|if .* were", re.I)


def iter_tables_with_sections(html: str):
    """Yield (section_heading, DataFrame) for every table in the article."""
    from bs4 import BeautifulSoup
    import io

    soup = BeautifulSoup(html, "lxml")
    for tbl in soup.find_all("table"):
        heading = ""
        node = tbl
        # walk backwards through the document for the nearest heading
        while node is not None:
            node = node.find_previous(HEADING_TAGS)
            if node is None:
                break
            text = node.get_text(" ", strip=True)
            if text:
                heading = text
                break
        try:
            # Wikipedia puts header <th> cells inside <tbody>, which pandas does
            # not always recognise; without header=0 every column comes back as
            # 0,1,2,3 and nothing can be identified.
            dfs = pd.read_html(io.StringIO(str(tbl)), header=0, flavor="lxml")
        except Exception:
            # One unreadable table is normal (nested layouts, colspan tricks).
            # It must never abort the article, let alone the run: pandas raises
            # ImportError here, not ValueError, when it falls back to html5lib.
            continue
        for df in dfs:
            yield heading, df


def district_from_heading(heading: str) -> int | None:
    """'District 2' or '2nd congressional district' -> 2."""
    m = DISTRICT_RE.search(heading or "")
    if not m:
        return None
    num = m.group(1) or m.group(2)
    try:
        return int(num)
    except (TypeError, ValueError):
        return None


def parse_html_sections(html: str, race_id_for: callable, default_year: int = 2026) -> pd.DataFrame:
    """Parse an article whose tables belong to different races.

    `race_id_for(heading)` maps a section heading to a race_id, or returns
    None to skip that table. Tables under a primary/nomination heading are
    always skipped: this model forecasts general elections.
    """
    rows = []
    for heading, df in iter_tables_with_sections(html):
        if PRIMARY_RE.search(heading or "") or HYPOTHETICAL_RE.search(heading or ""):
            continue
        rid = race_id_for(heading)
        if rid is None:
            continue
        part = parse_tables([df], rid, default_year)
        if len(part):
            rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# Link discovery
# --------------------------------------------------------------------------
# Wikipedia's hub articles ("2026 United States Senate elections") link out to
# each race's own article. Guessing titles works most of the time but breaks on
# naming variations and redirects, so we also harvest the real links.
def discover_links(html: str, pattern: re.Pattern) -> dict[str, str]:
    """Return {article title: link text} for every /wiki/ link matching pattern."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    out = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith(("/wiki/", "./")):
            continue
        title = href.split("/wiki/")[-1].lstrip("./").split("#")[0]
        if ":" in title:            # skip File:, Category:, Help: ...
            continue
        if pattern.search(title.replace("_", " ")):
            out.setdefault(title, a.get_text(" ", strip=True))
    return out

# --------------------------------------------------------------------------
# Presidential approval
# --------------------------------------------------------------------------
# Wikipedia's approval article lists individual polls in per-month tables
# ("July 2026"), with Approve / Disapprove columns. The row's own date is often
# partial ("July 23-27"), so the section heading supplies the month and year.
APPROVE_COL = re.compile(r"^approve", re.I)
DISAPPROVE_COL = re.compile(r"^disapprove", re.I)
MONTH_YEAR_RE = re.compile(rf"({MONTHS})\s+(\d{{4}})", re.I)
# Demographic and partisan breakdown tables repeat a poll once per subgroup.
# Averaging those in would weight a single poll many times and mix "approval
# among Republicans" into a national figure.
SUBGROUP_HEADING = re.compile(r"\bby (party|gender|race|age|education|region)\b|"
                              r"subgroup|demograph|crosstab|among\b", re.I)


def _approval_date(cell: str, heading: str, default_year: int) -> date | None:
    """Resolve a partial poll date using its section heading for context."""
    my = MONTH_YEAR_RE.search(heading or "")
    month, year = (my.group(1), int(my.group(2))) if my else (None, default_year)
    txt = re.sub(r"\[.*?\]", "", str(cell)).strip()
    full = parse_end_date(txt, year)
    if full:
        return full
    # "July 23-27" or "23-27" -> take the last day, with the heading's month
    m = re.search(rf"(?:({MONTHS})\s+)?(\d{{1,2}})\s*[\u2013\u2014-]\s*(\d{{1,2}})", txt, re.I)
    if m:
        mon = m.group(1) or month
        if mon:
            try:
                return pd.to_datetime(f"{mon} {m.group(3)}, {year}").date()
            except Exception:
                return None
    m2 = re.search(rf"(?:({MONTHS})\s+)?(\d{{1,2}})$", txt, re.I)
    if m2 and (m2.group(1) or month):
        try:
            return pd.to_datetime(f"{m2.group(1) or month} {m2.group(2)}, {year}").date()
        except Exception:
            return None
    return None


def parse_approval(html: str, default_year: int = 2026) -> pd.DataFrame:
    """Individual presidential-approval polls from a Wikipedia article."""
    rows = []
    for heading, df in iter_tables_with_sections(html):
        cols = [_flatten(c) for c in df.columns]
        appr = next((c for c in cols if APPROVE_COL.search(c)), None)
        disa = next((c for c in cols if DISAPPROVE_COL.search(c)), None)
        pollster = next((c for c in cols if POLLSTER_COL.search(c.lower())), None)
        datec = next((c for c in cols if DATE_COL.search(c.lower())), None)
        if not (appr and disa and pollster and datec):
            continue
        if any(AGGREGATOR_COL.search(c) for c in cols):
            continue          # the summary-of-aggregators table
        if SUBGROUP_HEADING.search(heading or ""):
            continue          # a breakdown, not a national poll
        t = df.copy()
        t.columns = cols
        for _, r in t.iterrows():
            name, partisan = clean_pollster(r[pollster])
            if not name or AGGREGATOR_ROW.search(name):
                continue
            d = _approval_date(r[datec], heading, default_year)
            a, dd = parse_pct(r[appr]), parse_pct(r[disa])
            if d is None or a != a or dd != dd:
                continue
            n, pop = (np.nan, "unknown")
            sample_col = next((c for c in cols if SAMPLE_COL.search(c.lower())), None)
            if sample_col:
                n, pop = parse_sample(r[sample_col])
            rows.append({"date": d, "pollster": name, "approve": a, "disapprove": dd,
                         "sample_size": n, "population": pop,
                         "partisan": "partisan" if partisan else ""})
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    # One national number per pollster per field period. Where a table still
    # lists several rows for one poll, the first is the overall figure.
    out = out.sort_values("date").drop_duplicates(["date", "pollster"], keep="first")
    # Approve and disapprove should account for nearly everyone; a pair that
    # does not is a subgroup or a mis-parse.
    total = out.approve + out.disapprove
    return out[(total > 80) & (total <= 105)].reset_index(drop=True)
