"""
START_HERE.py
=============
Paste the contents of this file into ONE Jupyter cell and run it. It will:

  1. download the project from GitHub (no git required: it uses a ZIP),
  2. move into the project folder,
  3. check that every package is installed,
  4. tell you the exact next command to run.

You do not need to paste any other script into a cell. Everything else runs
from the files this downloads. Works on Windows, macOS and Linux.

If the project is already present this does NOT re-download. To force an
update to the newest code, run `update.py`, or pass --update to this script.
"""
import io
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

OWNER_REPO = "ajnutt24/research-papers"
BRANCH = "claude/2026-midterms-forecast-model-jj4m98"
FOLDER = "midterms-2026"
ZIP_URL = f"https://codeload.github.com/{OWNER_REPO}/zip/refs/heads/{BRANCH}"


def find_project():
    """Return the midterms-2026 folder if it already exists nearby."""
    here = Path.cwd().resolve()
    candidates = []
    for base in [here, *here.parents]:
        candidates += [base, base / FOLDER]
        candidates += list(base.glob(f"*/{FOLDER}"))
    for cand in candidates:
        if (cand / "config.py").is_file() and (cand / "run_pipeline.py").is_file():
            return cand
    return None


FORCE = "--update" in sys.argv
project = None if FORCE else find_project()

if project is None:
    print(f"Downloading the project (about 3 MB) from {OWNER_REPO} ...")
    try:
        with urllib.request.urlopen(ZIP_URL, timeout=120) as resp:
            blob = resp.read()
    except Exception as e:
        print(f"\nDownload failed: {type(e).__name__}: {e}")
        print("\nManual alternative:")
        print(f"  1. Open this link in your browser:\n     https://github.com/{OWNER_REPO}/archive/refs/heads/{BRANCH}.zip")
        print("  2. Unzip it.")
        print("  3. Re-run this cell from the folder you unzipped into.")
        raise SystemExit(1)

    dest = Path.cwd().resolve()
    if FORCE:
        existing = find_project()
        if existing is not None:
            dest = existing.parent
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        members = [n for n in z.namelist() if f"/{FOLDER}/" in n and not n.endswith("/")]
        if not members:
            print("The ZIP did not contain the project folder. Check the branch name.")
            raise SystemExit(1)
        root = members[0].split("/")[0]
        for name in members:
            # strip the "<repo>-<branch>/" prefix so we extract midterms-2026/... directly
            rel = Path(name).relative_to(f"{root}")
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    print(f"Extracted {len(members)} files.")
    project = find_project()

if project is None:
    print("Could not find the project folder after downloading. Current folder:", Path.cwd())
    raise SystemExit(1)

os.chdir(project)
if str(project) not in sys.path:
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
    print("\nRun this in a new cell, restart the kernel, then re-run this cell:\n")
    print(f"    !pip install {' '.join(missing)}")
    if "pymc" in missing:
        print("\n(On Windows, PyMC installs much more reliably with conda:")
        print("     !conda install -c conda-forge pymc -y )")
else:
    print("All packages installed.\n")
    print("You are ready. Run this in a new cell (takes about 5 minutes):\n")
    print("    %run run_pipeline.py --full\n")
    print("When it finishes it prints the forecast. Later, to refresh with new polls:\n")
    print("    %run run_pipeline.py --update")
