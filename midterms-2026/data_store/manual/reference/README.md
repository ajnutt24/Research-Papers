# Precomputed reference tables

Files here are model outputs that are expensive to recompute but do not change
with new polls. They are shipped so a fresh install does not have to refit
them.

| file | what it is | recompute with |
|---|---|---|
| stacking_weights.parquet | LOO stacking weights by horizon, fitted on the 2010/2014/2018/2022 backtest cycles (about 37 PyMC fits, 2 minutes on a fast machine and 20+ on a slow one) | `python3 model/12_blend_stacking.py --refit` |

These depend only on historical data, so they stay valid for the whole cycle.
Recompute them if you add archived race ratings for more cycles, change the
component models, or change `config.STACK_HORIZONS_ESTIMATED`.
