"""
utils.py
========
Small shared helpers: a polite HTTP session (rate-limited, retrying,
robots.txt-aware), a download cache, parquet/CSV IO with a provenance column,
and logging.

Design choice: every fetch script follows the same three-tier fallback so the
pipeline never dead-ends on a network failure:

    1. live source (API or scrape)          -> provenance = "live"
    2. cached copy on disk (< CACHE_MAX_AGE) -> provenance = "cache"
    3. clearly-labelled fixture              -> provenance = "fixture"

Downstream stages propagate the worst provenance they consumed so the final
forecast can say, honestly, whether it rests on real polls or on placeholders.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from urllib import robotparser

import pandas as pd
import requests

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:
    sys.path.insert(0, str(Path.cwd()))
import config  # noqa: E402

PROVENANCE_RANK = {"live": 0, "cache": 1, "manual": 1, "mirror": 1, "fixture": 3}


def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log


log = get_logger("utils")


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class PoliteSession:
    """requests.Session wrapper with per-host rate limiting, retries and a
    robots.txt check for scraped (non-API) hosts."""

    def __init__(self, min_interval: float = config.HTTP_MIN_INTERVAL):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": config.USER_AGENT})
        self.min_interval = min_interval
        self._last: dict[str, float] = {}
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}

    def _wait(self, host: str):
        last = self._last.get(host, 0.0)
        gap = time.monotonic() - last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last[host] = time.monotonic()

    def allowed_by_robots(self, url: str) -> bool:
        host = urlparse(url).netloc
        if host not in self._robots:
            rp = robotparser.RobotFileParser()
            try:
                r = self.s.get(f"https://{host}/robots.txt", timeout=config.HTTP_TIMEOUT)
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
                    self._robots[host] = rp
                else:
                    self._robots[host] = None  # no robots file -> allowed
            except requests.RequestException:
                self._robots[host] = None
        rp = self._robots[host]
        return True if rp is None else rp.can_fetch(config.USER_AGENT, url)

    def get(self, url: str, *, check_robots: bool = False, **kw) -> requests.Response:
        if check_robots and not self.allowed_by_robots(url):
            raise PermissionError(f"robots.txt disallows fetching {url}")
        host = urlparse(url).netloc
        kw.setdefault("timeout", config.HTTP_TIMEOUT)
        err: Exception | None = None
        for attempt in range(config.HTTP_RETRIES):
            self._wait(host)
            try:
                r = self.s.get(url, **kw)
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"{r.status_code} from {host}", response=r)
                r.raise_for_status()
                return r
            except requests.RequestException as e:  # includes proxy/TLS failures
                err = e
                # A proxy/network-policy refusal will not heal on retry: fail fast.
                if isinstance(e, requests.exceptions.ProxyError) or "Tunnel connection failed" in str(e):
                    break
                time.sleep(2 ** attempt)
        raise RuntimeError(f"GET {url} failed after {config.HTTP_RETRIES} attempts: {err}")


SESSION = PoliteSession()


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
def cache_path(name: str) -> Path:
    return config.DATA_RAW / name


def cache_is_fresh(path: Path, max_age_hours: float = config.CACHE_MAX_AGE_HOURS) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=max_age_hours)


def fetch_text(url: str, cache_name: str, *, max_age_hours=config.CACHE_MAX_AGE_HOURS,
               check_robots=False, force=False) -> tuple[str, str]:
    """Return (text, provenance). provenance in {"live", "cache"}; raises if neither."""
    p = cache_path(cache_name)
    if not force and cache_is_fresh(p, max_age_hours):
        return p.read_text(encoding="utf-8-sig"), "cache"
    try:
        r = SESSION.get(url, check_robots=check_robots)
        text = r.text
        if text.startswith("\ufeff"):      # drop a byte-order mark before caching
            text = text.lstrip("\ufeff")
        p.write_text(text, encoding="utf-8")
        return text, "live"
    except Exception as e:
        if p.exists():
            log.warning("live fetch failed (%s); using stale cache %s", e, p.name)
            return p.read_text(encoding="utf-8-sig"), "cache"
        raise


def fetch_json(url: str, cache_name: str, **kw) -> tuple[dict, str]:
    txt, prov = fetch_text(url, cache_name, **kw)
    return json.loads(txt), prov


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def save_stage(df: pd.DataFrame, name: str, provenance: str, meta: dict | None = None) -> Path:
    """Write a stage output as parquet (+ CSV twin for eyeballing) and a JSON sidecar."""
    df = df.copy()
    df["provenance"] = provenance
    out = config.DATA_PROCESSED / f"{name}.parquet"
    df.to_parquet(out, index=False)
    df.to_csv(config.DATA_PROCESSED / f"{name}.csv", index=False)
    side = {"stage": name, "provenance": provenance, "rows": int(len(df)),
            "written": datetime.now().isoformat(timespec="seconds"), **(meta or {})}
    (config.DATA_PROCESSED / f"{name}.meta.json").write_text(json.dumps(side, indent=2, default=str))
    log.info("wrote %s (%d rows, provenance=%s)", out.name, len(df), provenance)
    return out


def load_stage(name: str, required: bool = True) -> pd.DataFrame | None:
    p = config.DATA_PROCESSED / f"{name}.parquet"
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Missing upstream output {p}. Run the producing script first.")
        return None
    return pd.read_parquet(p)


def load_meta(name: str) -> dict:
    p = config.DATA_PROCESSED / f"{name}.meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def worst_provenance(*provs: str) -> str:
    provs = [p for p in provs if p]
    return max(provs, key=lambda p: PROVENANCE_RANK.get(p, 2)) if provs else "unknown"


def margin_from_pcts(dem: float, rep: float) -> float:
    return float(dem) - float(rep)


# --------------------------------------------------------------------------
# Shared reference-data loaders (GitHub-hosted mirrors, cached for a week)
# --------------------------------------------------------------------------
GH = "https://raw.githubusercontent.com"
PARTISAN_LEAN_URLS = {
    # vintage -> (districts_url, states_url). 538's lean = D minus R margin
    # relative to the nation; the same sign convention as this project.
    "2022": (f"{GH}/fivethirtyeight/data/master/partisan-lean/fivethirtyeight_partisan_lean_DISTRICTS.csv",
             f"{GH}/fivethirtyeight/data/master/partisan-lean/fivethirtyeight_partisan_lean_STATES.csv"),
    "2020": (f"{GH}/fivethirtyeight/data/master/partisan-lean/2020/fivethirtyeight_partisan_lean_DISTRICTS.csv",
             f"{GH}/fivethirtyeight/data/master/partisan-lean/2020/fivethirtyeight_partisan_lean_STATES.csv"),
    "2018": (f"{GH}/fivethirtyeight/data/master/partisan-lean/2018/fivethirtyeight_partisan_lean_DISTRICTS.csv",
             f"{GH}/fivethirtyeight/data/master/partisan-lean/2018/fivethirtyeight_partisan_lean_STATES.csv"),
}
RESULTS_MIRROR_URLS = {
    "House": f"{GH}/fivethirtyeight/election-results/main/election_results_house.csv",
    "Senate": f"{GH}/fivethirtyeight/election-results/main/election_results_senate.csv",
    "Governor": f"{GH}/fivethirtyeight/election-results/main/election_results_gubernatorial.csv",
    "races": f"{GH}/fivethirtyeight/election-results/main/races.csv",
}
STATE_ABBR = {v: k for k, v in config.STATE_NAMES.items()}


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip byte-order marks and whitespace from column names.

    Several FiveThirtyEight CSVs (the 2018 and 2020 partisan-lean files) are
    saved with a UTF-8 BOM. Some pandas versions strip it on read and some
    keep it, so the first column can arrive as either 'district' or
    '\ufeffdistrict'. Normalising here makes column lookups behave the same
    on every machine.
    """
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def _lean_to_number(v) -> float:
    """Parse a partisan-lean cell.

    Handles 'R+15.21' and 'D+3.4' (2018/2020 vintages) as well as plain
    signed floats (2022 vintage). Republican leans are negative, matching the
    project-wide convention that margins are Democratic minus Republican.
    """
    if v is None or (isinstance(v, float) and v != v):
        return float("nan")
    if not isinstance(v, str):
        return float(v)
    v = v.strip()
    if not v:
        return float("nan")
    side = v[0].upper() if v[0].isalpha() else ""
    body = v[1:] if side else v
    body = body.replace("+", "").strip()
    try:
        num = float(body)
    except ValueError:
        return float("nan")
    return -num if side == "R" else num


def _pick_lean_column(df: pd.DataFrame, key_col: str, source: str) -> str:
    """Return the name of the lean column, and verify it actually holds leans.

    Picking 'the column that is not the key' silently returns the key itself
    when the key's name does not match (for instance because of a BOM), which
    produced a confusing float-conversion error far downstream. This checks
    the choice instead of trusting it.
    """
    candidates = [c for c in df.columns if c != key_col]
    if not candidates:
        raise ValueError(f"{source}: no lean column found; columns are {list(df.columns)}")
    col = candidates[0]
    parsed = df[col].map(_lean_to_number)
    if parsed.notna().mean() < 0.8:
        raise ValueError(
            f"{source}: column {col!r} does not look like partisan lean "
            f"(only {parsed.notna().mean():.0%} of values parsed). "
            f"Columns are {list(df.columns)}."
        )
    return col


def load_partisan_lean(vintage: str = "2022") -> tuple[pd.DataFrame, str]:
    """Return (DataFrame[race_key, lean], provenance). race_key is 'TX-23' for
    districts and 'TX' for states; at-large districts are 'AK-01'."""
    import io
    d_url, s_url = PARTISAN_LEAN_URLS[vintage]
    d_txt, p1 = fetch_text(d_url, f"plean_districts_{vintage}.csv", max_age_hours=24 * 7)
    s_txt, p2 = fetch_text(s_url, f"plean_states_{vintage}.csv", max_age_hours=24 * 7)
    d = _clean_columns(pd.read_csv(io.StringIO(d_txt)))
    s = _clean_columns(pd.read_csv(io.StringIO(s_txt)))

    lean_col_d = _pick_lean_column(d, "district", f"partisan lean {vintage} districts")
    lean_col_s = _pick_lean_column(s, "state", f"partisan lean {vintage} states")

    def norm_district(x: str) -> str:
        st, num = str(x).split("-")
        return f"{st.strip()}-{int(num):02d}"

    d = pd.DataFrame({"race_key": d["district"].map(norm_district),
                      "lean": d[lean_col_d].map(_lean_to_number).astype(float)})
    s = pd.DataFrame({"race_key": s["state"].map(lambda x: STATE_ABBR.get(str(x).strip())),
                      "lean": s[lean_col_s].map(_lean_to_number).astype(float)})
    out = pd.concat([d, s], ignore_index=True).dropna(subset=["race_key"])
    out["vintage"] = vintage
    return out, worst_provenance(p1, p2) if p1 == "live" or p2 == "live" else "mirror"


def load_results_mirror(office: str) -> tuple[pd.DataFrame, str]:
    """Raw FiveThirtyEight election-results mirror for one office (all cycles)."""
    import io
    txt, prov = fetch_text(RESULTS_MIRROR_URLS[office], f"results_{office.lower()}.csv", max_age_hours=24 * 7)
    df = pd.read_csv(io.StringIO(txt), low_memory=False)
    return df, ("mirror" if prov == "live" else prov)


def load_races_mirror() -> pd.DataFrame:
    import io
    txt, _ = fetch_text(RESULTS_MIRROR_URLS["races"], "results_races.csv", max_age_hours=24 * 7)
    return pd.read_csv(io.StringIO(txt), low_memory=False)


def tier_from_lean(lean: float, thresholds=(3.0, 8.0, 15.0)) -> str:
    """Map a D-minus-R margin to a Cook-style tier (used only for fixtures)."""
    a = abs(lean)
    side = "D" if lean > 0 else "R"
    if a < thresholds[0]:
        return "Toss-up"
    if a < thresholds[1]:
        return f"Lean {side}"
    if a < thresholds[2]:
        return f"Likely {side}"
    return f"Safe {side}"

def require(module, *names) -> None:
    """Fail with an actionable message when a stale copy of a module is loaded.

    Refreshing the files on disk does not replace a module Python has already
    imported, so an out-of-date notebook kernel raises AttributeError for
    functions that plainly exist in the file. This turns that into an
    instruction.
    """
    missing = [n for n in names if not hasattr(module, n)]
    if missing:
        raise RuntimeError(
            f"{module.__name__} is out of date: missing {', '.join(missing)}.\n"
            f"Loaded from: {getattr(module, '__file__', '?')}\n"
            "Re-run the setup cell (START_HERE.py) to refresh the code, which also\n"
            "reloads stale modules. If it still fails, restart the kernel and re-run it."
        )
