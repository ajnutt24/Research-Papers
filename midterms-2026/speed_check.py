"""
speed_check.py
==============
Measures how fast PyMC sampling runs on this machine and estimates how long
the pipeline will take. Also reports whether PyTensor found a C compiler,
which is the single biggest factor: without one it falls back to a much
slower mode and the modelling stages can take 20-50x longer.

    %run speed_check.py

Takes about a minute.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    _ROOT = Path(__file__).resolve().parent
except NameError:
    _ROOT = Path.cwd()
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

# Reference timings (seconds) measured on a machine with a working compiler.
REFERENCE = {"01": 5, "02": 1, "03": 1, "04": 1, "05": 14, "07": 10, "06": 3,
             "08": 14, "09": 47, "10": 2, "11": 18, "12": 23, "13": 6, "14": 75}
REFERENCE_BENCH = 6.0   # seconds the benchmark model takes on that machine

print("Checking PyTensor configuration ...\n")
import pytensor  # noqa: E402

cxx = pytensor.config.cxx
blas = pytensor.config.blas__ldflags
print(f"  C compiler : {cxx if cxx else '(none found)'}")
print(f"  BLAS flags : {blas if blas else '(none - using a slower fallback)'}")

backend = "default (C)"
if not cxx:
    try:
        import numba  # noqa: F401
        import pytensor as _pt
        _pt.config.mode = "NUMBA"
        backend = "NUMBA"
        print("  backend    : NUMBA (no C compiler, using numba instead)")
        print("\n  This is fine: numba recovers most of the speed a C compiler")
        print("  would give (measured: 5.6s with a compiler, 10.9s on numba,")
        print("  67.5s with neither, on the same model).")
    except ImportError:
        backend = "interpreted"
        print("\n  !! No C compiler AND no numba, so every model runs in a slow")
        print("     interpreted mode (about 11x slower). Fix with:")
        print("         !pip install numba")
        print("     then restart the kernel.")
else:
    print("  backend    : default C backend")

print("\nTiming a small model (about a minute) ...")
import pymc as pm  # noqa: E402

rng = np.random.default_rng(0)
x = rng.normal(size=400)
y = 1.5 + 2.0 * x + rng.normal(scale=0.5, size=400)

t0 = time.perf_counter()
with pm.Model():
    a = pm.Normal("a", 0, 5)
    b = pm.Normal("b", 0, 5)
    s = pm.HalfNormal("s", 5)
    pm.Normal("y", a + b * x, s, observed=y)
    pm.sample(draws=500, tune=500, chains=2, progressbar=False,
              compute_convergence_checks=False, random_seed=1)
bench = time.perf_counter() - t0
ratio = bench / REFERENCE_BENCH

print(f"\n  benchmark model: {bench:.1f}s  (reference machine: {REFERENCE_BENCH:.0f}s)")
if ratio >= 1.2:
    print(f"  this machine is about {ratio:.0f}x slower than the reference\n")
elif ratio <= 0.8:
    print(f"  this machine is about {1 / ratio:.0f}x faster than the reference\n")
else:
    print("  this machine is about as fast as the reference\n")

heavy = {"09", "11", "12", "14"}
def est(stage):
    r = REFERENCE[stage]
    return r * ratio if stage in heavy else r * max(ratio ** 0.3, 1.0)

full = sum(est(s) for s in REFERENCE)
fast = sum(est(s) for s in REFERENCE if s != "14")
upd = sum(est(s) for s in ["03", "08", "11", "12", "13"])

print(f"{'MODE':34s} {'ESTIMATE':>12s}")
print("-" * 48)
print(f"{'run_pipeline.py --full  (01-14)':34s} {full/60:>9.0f} min")
print(f"{'run_pipeline.py --fast  (01-13)':34s} {fast/60:>9.0f} min")
print(f"{'run_pipeline.py --update (polls)':34s} {upd/60:>9.0f} min")
print("\nSlowest stages at this speed:")
for s in sorted(heavy, key=lambda s: -est(s)):
    print(f"   stage {s}: about {est(s)/60:.0f} min")

if ratio > 5:
    print("\n" + "=" * 64)
    print("This machine is much slower than expected for this kind of model.")
    if backend == "interpreted":
        print("Cause: no C compiler and no numba. Easiest fix, no admin needed:")
        print("\n    !pip install numba")
        print("\nthen restart the kernel. Measured speedup on this project: ~6x.")
    else:
        print("A C toolchain would help further:")
        print("\n    conda install -c conda-forge m2w64-toolchain -y")
        print("\n(run it in a terminal, then restart the kernel).")
    print("\nIf you would rather not install anything, run with lighter")
    print("sampling (less precise, much faster):")
    print("\n    import os; os.environ['MCMC_DRAWS']='400'; os.environ['MCMC_TUNE']='400'")
    print("    %run run_pipeline.py --fast")
    print("=" * 64)
else:
    print("\nSpeed looks normal for this model.")
