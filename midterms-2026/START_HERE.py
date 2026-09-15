"""
START_HERE.py
=============
Run this in the FIRST cell of your notebook, every time you start or restart
the kernel. It:

  1. downloads the project from GitHub if you do not have it,
  2. refreshes the code to the latest version if you do,
  3. moves Python into the project folder (so %run finds the scripts),
  4. checks that every package is installed,
  5. tells you the next command.

Your own files under data_store/manual/ and everything in outputs/ are left
untouched. No git required. Works on Windows, macOS and Linux.
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
    """Look for the project in the current folder and walking upward.

    Deliberately narrow: it checks each folder itself, a `midterms-2026`
    child, and a `research-papers/midterms-2026` child. It does NOT glob
    across sibling folders, which could otherwise latch onto an unrelated
    copy somewhere else on the disk.
    """
    here = Path.cwd().resolve()
    for base in [here, *here.parents]:
        for cand in (base, base / FOLDER, base / "research-papers" / FOLDER):
            if (cand / "config.py").is_file() and (cand / "run_pipeline.py").is_file():
                return cand
    return None


existing = find_project()
target_parent = existing.parent if existing else Path.cwd().resolve()
action = "Refreshing" if existing else "Downloading"
print(f"{action} the project (about 3 MB) from {OWNER_REPO} ...")

try:
    with urllib.request.urlopen(ZIP_URL, timeout=120) as resp:
        blob = resp.read()
except Exception as e:
    if existing:
        print(f"Could not reach GitHub ({type(e).__name__}); using the copy you already have.")
        blob = None
    else:
        print(f"\nDownload failed: {type(e).__name__}: {e}\n")
        print("Manual alternative:")
        print(f"  1. Open: https://github.com/{OWNER_REPO}/archive/refs/heads/{BRANCH}.zip")
        print("  2. Unzip it.  3. Re-run this cell from the folder you unzipped into.")
        raise SystemExit(1)

if blob is not None:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        members = [n for n in z.namelist() if f"/{FOLDER}/" in n and not n.endswith("/")]
        if not members:
            print("The ZIP did not contain the project folder. Check the branch name.")
            raise SystemExit(1)
        root = members[0].split("/")[0]
        for name in members:
            target = target_parent / Path(name).relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    print(f"{len(members)} files written.")

project = find_project() or (target_parent / FOLDER)
if not (project / "config.py").is_file():
    print("Could not find the project folder after downloading. Current folder:", Path.cwd())
    raise SystemExit(1)

os.chdir(project)
if str(project) not in sys.path:
    sys.path.insert(0, str(project))
print(f"Working folder: {project}\n")

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
        print("\n(On Windows, PyMC installs more reliably with conda:")
        print("     !conda install -c conda-forge pymc -y )")
else:
    print("All packages installed. You are ready.\n")
    print("Next cell:\n")
    print("    %run run_pipeline.py --full      (about 5 minutes)\n")
    print("Then:\n")
    print("    %run show_results.py             (the forecast)")
    print("    %run check_data.py               (real data vs placeholders)")
