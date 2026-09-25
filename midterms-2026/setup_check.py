"""
setup_check.py
==============
Run this FIRST on a new machine (or as the first Jupyter cell):

    python3 setup_check.py          # terminal
    %run setup_check.py             # notebook

It verifies, in order: the working directory holds the project files, every
required package imports, PyMC can actually compile and sample, the cached
historical data is reachable, and which 2026 inputs are live vs placeholder.
It changes nothing; it only reports.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

REQUIRED = ["pandas", "numpy", "scipy", "pyarrow", "requests", "bs4", "lxml", "html5lib", "yaml",
            "matplotlib", "pymc", "arviz", "numba"]
STAGE_FILES = ["config.py", "utils.py", "run_pipeline.py", "data/01_fetch_economic_data.py",
               "model/modellib.py", "simulation/13_monte_carlo.py"]
ok = True


def check(label: str, passed: bool, detail: str = "") -> bool:
    global ok
    print(f"[{'OK ' if passed else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
    ok = ok and passed
    return passed


print(f"Python {sys.version.split()[0]}\nWorking directory: {Path.cwd()}\n")

# 1. location
here = Path.cwd()
found = (here / "config.py").is_file() and (here / "utils.py").is_file()
if not found:
    alt = next((d for c in [here, *here.parents] for d in (c, c / "midterms-2026")
                if (d / "config.py").is_file()), None)
    check("project folder", False,
          f"config.py not here. Found it at {alt}; run  %cd {alt}" if alt else
          "config.py and utils.py must exist as FILES. Download the midterms-2026 folder from the repo.")
    sys.exit(1)
check("project folder", True, str(here))
for f in STAGE_FILES:
    check(f"file {f}", Path(f).is_file())

# 2. packages
print()
for m in REQUIRED:
    try:
        mod = importlib.import_module(m)
        check(f"import {m}", True, getattr(mod, "__version__", ""))
    except ImportError as e:
        hint = "conda install -c conda-forge pymc" if m in ("pymc", "arviz") else f"pip install {m}"
        check(f"import {m}", False, f"{e}. Try: {hint}")

# 3. PyMC can compile and sample (the usual Windows failure point)
print()
try:
    import pymc as pm
    with pm.Model():
        pm.Normal("x", 0, 1)
        pm.sample(draws=20, tune=20, chains=1, progressbar=False, compute_convergence_checks=False)
    check("PyMC compiles and samples", True)
except Exception as e:
    check("PyMC compiles and samples", False,
          f"{type(e).__name__}: {str(e)[:160]} | needs a C compiler; conda-forge install is the easy fix")

# 4. data
print()
sys.path.insert(0, str(here))
try:
    import config
    from utils import load_meta
    uni = config.race_universe()
    check("race universe", len(uni) == 506, f"{len(uni)} races (435 House + 35 Senate + 36 Governor)")
    raw = list((config.DATA_RAW).glob("*.csv"))
    check("cached historical data", len(raw) > 0, f"{len(raw)} files in data_store/raw")
    print("\n2026 input provenance (fixture = placeholder, not real data):")
    for name in ["polls_2026", "approval_2026", "race_ratings_2026", "fundamentals_2026",
                 "economic_cycle_features", "historical_results"]:
        m = load_meta(name)
        print(f"    {name:26s} {m.get('provenance', 'not yet generated')}")
except Exception as e:
    check("project imports", False, f"{type(e).__name__}: {e}")

print("\n" + ("All checks passed. Run:  python3 run_pipeline.py --full"
               if ok else "Fix the FAIL lines above before running the pipeline."))
