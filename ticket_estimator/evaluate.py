"""Back-test the model on the most recently resolved tickets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model import EffortModel
from .records import Ticket


@dataclass
class Scores:
    n: int
    median_abs_error_hours: float
    within_50pct: float   # share of tickets where |pred - actual| <= 50% of actual
    within_2x: float      # share where pred is between actual/2 and actual*2

    def row(self, name: str) -> str:
        return (f"{name:<28} n={self.n:<5} median error={self.median_abs_error_hours:6.2f}h   "
                f"within ±50%={self.within_50pct:5.1%}   within 2x={self.within_2x:5.1%}")


def score(pred: Sequence[float], actual: Sequence[float]) -> Scores:
    p, a = np.asarray(pred, float), np.asarray(actual, float)
    ratio = np.maximum(p, 1e-6) / a
    return Scores(
        n=len(a),
        median_abs_error_hours=float(np.median(np.abs(p - a))),
        within_50pct=float(np.mean(np.abs(p - a) <= 0.5 * a)),
        within_2x=float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
    )


def time_split(tickets: Sequence[Ticket], test_fraction: float = 0.2) -> tuple[list[Ticket], list[Ticket]]:
    usable = sorted((t for t in tickets if t.hours_spent and t.hours_spent > 0), key=lambda t: t.resolved)
    cut = int(len(usable) * (1 - test_fraction))
    return usable[:cut], usable[cut:]


def backtest(tickets: Sequence[Ticket], test_fraction: float = 0.2, **model_kwargs) -> tuple[dict[str, Scores], float]:
    """Train on older tickets, score on the newest ones, and compare with simple baselines.

    Returns the scores and the share of test tickets whose actual hours fell in the predicted range.
    """
    train, test = time_split(tickets, test_fraction)
    model = EffortModel(**model_kwargs).fit(train)
    log_pred = model.predict_log(test)[0]
    pred = np.expm1(log_pred)
    actual = [t.hours_spent for t in test]
    results = {"model": score(pred, actual)}

    # Baseline: median hours of past tickets with the same project and type.
    groups: dict[tuple[str, str], list[float]] = {}
    for t in train:
        groups.setdefault((t.project, t.issue_type), []).append(t.hours_spent)
    overall = float(np.median([t.hours_spent for t in train]))
    base = [float(np.median(groups.get((t.project, t.issue_type), [overall]))) for t in test]
    results["median per project+type"] = score(base, actual)

    # The team's own original estimates, where they exist.
    with_est = [(i, t) for i, t in enumerate(test) if t.hours_estimated]
    if with_est:
        idx = [i for i, _ in with_est]
        results["team original estimate"] = score([t.hours_estimated for _, t in with_est],
                                                  [t.hours_spent for _, t in with_est])
        results["model (same tickets)"] = score(pred[idx], [actual[i] for i in idx])
    lo, hi = np.expm1(log_pred + model.resid_lo), np.expm1(log_pred + model.resid_hi)
    coverage = float(np.mean((lo <= actual) & (np.asarray(actual) <= hi)))
    return results, coverage
