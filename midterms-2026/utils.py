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

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
        return p.read_text(), "cache"
    try:
        r = SESSION.get(url, check_robots=check_robots)
        p.write_text(r.text)
        return r.text, "live"
    except Exception as e:
        if p.exists():
            log.warning("live fetch failed (%s); using stale cache %s", e, p.name)
            return p.read_text(), "cache"
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


def load_partisan_lean(vintage: str = "2022") -> tuple[pd.DataFrame, str]:
    """Return (DataFrame[race_key, lean], provenance). race_key is 'TX-23' for
    districts and 'TX' for states; at-large districts are 'AK-01'."""
    import io
    d_url, s_url = PARTISAN_LEAN_URLS[vintage]
    d_txt, p1 = fetch_text(d_url, f"plean_districts_{vintage}.csv", max_age_hours=24 * 7)
    s_txt, p2 = fetch_text(s_url, f"plean_states_{vintage}.csv", max_age_hours=24 * 7)
    d = pd.read_csv(io.StringIO(d_txt))
    s = pd.read_csv(io.StringIO(s_txt))
    lean_col_d = [c for c in d.columns if c != "district"][0]
    lean_col_s = [c for c in s.columns if c != "state"][0]

    def to_num(v):
        """Older vintages store 'R+15.21' / 'D+3.4' strings; newer ones store signed floats."""
        if isinstance(v, str):
            v = v.strip()
            m = v[0].upper() if v and v[0].isalpha() else ""
            num = float(v.lstrip("DR+ ").replace("+", "")) if v else float("nan")
            return -num if m == "R" else num
        return float(v)
    d[lean_col_d] = d[lean_col_d].map(to_num)
    s[lean_col_s] = s[lean_col_s].map(to_num)

    def norm_district(x: str) -> str:
        st, num = x.split("-")
        return f"{st}-{int(num):02d}"

    d = pd.DataFrame({"race_key": d["district"].map(norm_district), "lean": d[lean_col_d].astype(float)})
    s = pd.DataFrame({"race_key": s["state"].map(STATE_ABBR), "lean": s[lean_col_s].astype(float)})
    out = pd.concat([d, s], ignore_index=True)
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
