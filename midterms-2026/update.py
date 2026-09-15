"""
update.py
=========
Re-download the latest project code from GitHub, overwriting the .py files in
place. Your own data files (anything you added under data_store/manual/) and
your generated outputs are left alone.

    %run update.py
"""
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

OWNER_REPO = "ajnutt24/research-papers"
BRANCH = "claude/2026-midterms-forecast-model-jj4m98"
FOLDER = "midterms-2026"
ZIP_URL = f"https://codeload.github.com/{OWNER_REPO}/zip/refs/heads/{BRANCH}"

here = Path.cwd().resolve()
project = None
for base in [here, *here.parents]:
    for cand in [base, base / FOLDER, *base.glob(f"*/{FOLDER}")]:
        if (cand / "config.py").is_file():
            project = cand
            break
    if project:
        break
if project is None:
    raise SystemExit("Could not find the midterms-2026 folder. Run START_HERE.py instead.")

print(f"Updating {project} ...")
with urllib.request.urlopen(ZIP_URL, timeout=120) as r:
    blob = r.read()

updated = 0
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    members = [n for n in z.namelist() if f"/{FOLDER}/" in n and not n.endswith("/")]
    root = members[0].split("/")[0]
    for name in members:
        rel = Path(name).relative_to(root).relative_to(FOLDER)
        target = project / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(name) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
        updated += 1

print(f"Updated {updated} files.\n")
print("Now run:\n    %run run_pipeline.py --from 09")
