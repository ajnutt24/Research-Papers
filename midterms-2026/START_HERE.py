"""
START_HERE.py
=============
Paste the contents of this file into ONE Jupyter cell and run it. It will:

  1. download the project from GitHub (if it is not already here),
  2. move into the project folder,
  3. check that every package is installed,
  4. tell you the exact next command to run.

You do not need to paste any other script into a cell. Everything else runs
from the files this downloads.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/ajnutt24/research-papers.git"
BRANCH = "claude/2026-midterms-forecast-model-jj4m98"
FOLDER = "midterms-2026"


def find_project():
    """Return the midterms-2026 folder if it already exists nearby."""
    here = Path.cwd().resolve()
    for base in [here, *here.parents]:
        for cand in (base, base / FOLDER, base / "research-papers" / FOLDER):
            if (cand / "config.py").is_file() and (cand / "run_pipeline.py").is_file():
                return cand
    return None


project = find_project()

if project is None:
    print("Downloading the project from GitHub...")
    r = subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO],
                       capture_output=True, text=True)
    if r.returncode != 0 and "already exists" not in r.stderr:
        print("Download failed:\n" + r.stderr)
        print("\nNo git installed? Download the ZIP instead:")
        print(f"  {REPO[:-4]}/archive/refs/heads/{BRANCH}.zip")
        print("unzip it, then re-run this cell from the folder you unzipped into.")
        sys.exit(1)
    project = find_project()

if project is None:
    print("Could not find the project folder after downloading. Current folder:", Path.cwd())
    sys.exit(1)

os.chdir(project)
sys.path.insert(0, str(project))
print(f"Project folder: {project}\n")

missing = []
for mod, install in [("pandas", "pandas"), ("numpy", "numpy"), ("scipy", "scipy"),
                     ("pyarrow", "pyarrow"), ("requests", "requests"), ("bs4", "beautifulsoup4"),
                     ("lxml", "lxml"), ("yaml", "pyyaml"), ("matplotlib", "matplotlib"),
                     ("pymc", "pymc"), ("arviz", "arviz")]:
    try:
        __import__(mod)
    except ImportError:
        missing.append(install)

if missing:
    print("Missing packages:", " ".join(missing))
    print("\nRun this in a cell, restart the kernel, then re-run this cell:")
    print(f"  !pip install {' '.join(missing)}")
    if "pymc" in missing:
        print("\n(On Windows, PyMC installs much more reliably with:")
        print("  !conda install -c conda-forge pymc -y )")
else:
    print("All packages installed.\n")
    print("You are ready. Next, run this in a new cell (takes about 6 minutes):\n")
    print("    %run run_pipeline.py --full\n")
    print("When it finishes it prints the forecast. After that, to refresh with new polls:\n")
    print("    %run run_pipeline.py --update")
