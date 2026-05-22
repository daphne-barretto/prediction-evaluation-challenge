"""const_global_mean — slide-29 rung 2: return the global mean pass rate.

The constant is the Bayes-optimal prediction when no per-item information is
available and the test distribution matches the train distribution. From
artifacts/global_mean_acc.json (computed over all non-cold-start training
rows): 0.6450659659207623.

Hardcoded here rather than read from JSON so the ZIP is one file. The value
is documented in the manifest.
"""
from __future__ import annotations

# Train-time global pass rate over non-cold-start rows (see
# artifacts/global_mean_acc.json from the sub-5 artifact set).
GLOBAL_MEAN_ACC = 0.6450659659207623


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    return GLOBAL_MEAN_ACC
