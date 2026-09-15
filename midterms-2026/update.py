"""
update.py
=========
Kept for convenience. START_HERE.py already refreshes the code every time it
runs, so this simply calls it.

    %run update.py
"""
from pathlib import Path

here = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
exec(open(here / "START_HERE.py").read())
