"""const_0p5 — slide-29 rung 1: return 0.5.

This is the absolute minimum predict() that satisfies the Codabench contract
(returns a Python float in [0, 1]). It carries zero predictive information,
so NLL is exactly ln(0.5) = -0.6931 regardless of the hidden label
distribution. AUC is undefined / 0.5 by convention.

Why ship this:
  - Slide 29 explicitly lists this as rung 1 ("smoke test"). The report
    needs the number to show how much our trained model actually adds.
  - Bounds from below: if a complex model scores worse than -0.693 it is
    actively hurting (over-confident in the wrong direction). Sub 5 at
    -0.65 beats this by only ~0.04 NLL, which itself is a finding for the
    report (calibration dominates ranking).
"""
from __future__ import annotations


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    return 0.5
