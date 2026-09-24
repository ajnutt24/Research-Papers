"""
perf.py
=======
Picks the fastest PyTensor backend available on this machine.

PyMC compiles each model before sampling. The default backend needs a C
compiler; without one PyTensor falls back to an interpreted mode that is
roughly 11x slower (measured on this project's national model: 5.6s with a
compiler, 67.5s without).

Numba is a pure-Python-installable JIT compiler that PyTensor can use
instead, and it recovers most of that gap (10.9s on the same model, no C
compiler present). So when no C compiler is found and numba is importable,
this module switches PyTensor to the NUMBA backend.

Import it BEFORE pymc, or call `configure()` before any sampling. It is safe
to call more than once and never raises.
"""
from __future__ import annotations

import logging

_log = logging.getLogger("perf")
_STATE: dict[str, str] = {}


def configure(verbose: bool = True) -> dict[str, str]:
    """Select a backend and return a short description of what was chosen."""
    if _STATE:
        return _STATE
    info = {"backend": "default", "cxx": "", "reason": ""}
    try:
        import pytensor
    except ImportError:
        _STATE.update(info | {"reason": "pytensor not installed"})
        return _STATE

    info["cxx"] = pytensor.config.cxx or ""
    if info["cxx"]:
        info["reason"] = "C compiler found; using the default C backend"
    else:
        try:
            import numba  # noqa: F401
            pytensor.config.mode = "NUMBA"
            info["backend"] = "NUMBA"
            info["reason"] = ("no C compiler found; switched to the numba backend "
                              "(about 6x faster than the interpreted fallback)")
        except ImportError:
            info["reason"] = ("no C compiler and numba is not installed, so models run in a "
                              "slow interpreted mode. Install numba (pip install numba) or a "
                              "C toolchain (conda install -c conda-forge m2w64-toolchain).")
    _STATE.update(info)
    if verbose:
        print(f"[perf] {info['reason']}", flush=True)
    return _STATE


configure()
