"""Estimation and forecasting methods widely used by product and engineering teams.

* Reference class forecasting (Kahneman/Flyvbjerg "outside view"): size a ticket from the
  spread of outcomes of similar past tickets rather than from the details of the plan.
* PERT three-point estimate: combine optimistic / likely / pessimistic into a mean and spread.
* T-shirt sizing: coarse buckets for roadmap conversations, with "split it" for anything over a week.
* Evidence-Based Scheduling (Spolsky, FogBugz): Monte Carlo over each estimator's historical
  actual/estimate ratios to turn a list of estimates into delivery probabilities.
* Throughput forecasting (Kanban / #NoEstimates): Monte Carlo over weekly completed-ticket counts
  to answer "when will these N tickets be done?" without estimating any of them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

import numpy as np

from .records import Ticket

PERCENTILES = (50, 85, 95)

# Working-hour buckets. Anything bigger than a week of work should be broken down before planning.
TSHIRT_SIZES = (("XS", 1.0), ("S", 4.0), ("M", 8.0), ("L", 24.0), ("XL", 40.0))


def tshirt_size(hours: float) -> str:
    for label, limit in TSHIRT_SIZES:
        if hours <= limit:
            return label
    return "XXL (split it)"


@dataclass
class ThreePoint:
    optimistic: float
    likely: float
    pessimistic: float

    @property
    def mean(self) -> float:
        return (self.optimistic + 4 * self.likely + self.pessimistic) / 6

    @property
    def std(self) -> float:
        return (self.pessimistic - self.optimistic) / 6


def reference_class(hours: Sequence[float]) -> ThreePoint:
    """Three-point estimate from the actual hours of a reference class of similar tickets."""
    h = np.asarray(hours, float)
    if h.size == 0:
        raise ValueError("Reference class is empty.")
    p10, p50, p90 = np.percentile(h, [10, 50, 90])
    return ThreePoint(float(p10), float(p50), float(p90))


# ---- Evidence-Based Scheduling ------------------------------------------------------------

def estimate_ratios(tickets: Sequence[Ticket]) -> list[tuple[Ticket, float]]:
    """Actual/estimate ratio for every ticket that has both numbers."""
    return [(t, t.hours_spent / t.hours_estimated) for t in tickets
            if t.hours_spent and t.hours_estimated and t.hours_estimated > 0]


def ratio_history(tickets: Sequence[Ticket], project: str = "", assignee: str = "",
                  min_samples: int = 10) -> tuple[np.ndarray, str]:
    """Pick the most specific history with enough samples: person, then project, then everyone."""
    pairs = estimate_ratios(tickets)
    for scope, keep in ((f"assignee {assignee}", lambda t: assignee and t.assignee == assignee),
                        (f"project {project}", lambda t: project and t.project == project),
                        ("all projects", lambda t: True)):
        r = np.array([ratio for t, ratio in pairs if keep(t)])
        if r.size >= min_samples:
            return r, scope
    raise ValueError("Not enough tickets with both an original estimate and logged time.")


@dataclass
class Forecast:
    percentiles: dict[int, float]
    unit: str
    basis: str

    def describe(self) -> str:
        parts = [f"{p}% chance ≤ {v:,.1f} {self.unit}" for p, v in self.percentiles.items()]
        return "; ".join(parts) + f"  (based on {self.basis})"


def ebs_forecast(estimates: Sequence[float], ratios: np.ndarray, basis: str = "",
                 runs: int = 10_000, seed: int = 0) -> Forecast:
    """Total hours for a set of estimated tickets, sampling a historical ratio for each."""
    rng = np.random.default_rng(seed)
    est = np.asarray(estimates, float)
    sampled = rng.choice(ratios, size=(runs, est.size))
    totals = (sampled * est).sum(axis=1)
    return Forecast({p: float(np.percentile(totals, p)) for p in PERCENTILES}, "hours",
                    basis or f"{ratios.size} past actual/estimate ratios")


# ---- Throughput forecasting ---------------------------------------------------------------

def _resolved_day(t: Ticket) -> date | None:
    try:
        return datetime.fromisoformat(t.resolved[:10]).date()
    except ValueError:
        return None


def weekly_throughput(tickets: Sequence[Ticket], weeks: int = 12, project: str = "") -> np.ndarray:
    """Completed tickets per ISO week over the most recent `weeks` weeks (empty weeks count as 0)."""
    days = [d for t in tickets if (not project or t.project == project) and (d := _resolved_day(t))]
    if not days:
        raise ValueError("No resolved tickets to measure throughput from.")
    last = max(days)
    counts = Counter((last - d).days // 7 for d in days)
    return np.array([counts.get(w, 0) for w in range(weeks)], float)


def throughput_forecast(throughput: np.ndarray, items: int, runs: int = 10_000, seed: int = 0,
                        max_weeks: int = 520) -> Forecast:
    """Weeks needed to finish `items` tickets, sampling past weekly throughput."""
    if items <= 0:
        raise ValueError("items must be positive")
    if throughput.sum() == 0:
        raise ValueError("Throughput history is all zeros.")
    rng = np.random.default_rng(seed)
    draws = rng.choice(throughput, size=(runs, max_weeks)).cumsum(axis=1)
    weeks = (draws < items).sum(axis=1) + 1
    return Forecast({p: float(np.percentile(weeks, p)) for p in PERCENTILES}, "weeks",
                    f"{throughput.size} weeks of throughput, median {np.median(throughput):.0f}/week")


# ---- Calibration --------------------------------------------------------------------------

@dataclass
class Calibration:
    group: str
    n: int
    median_ratio: float   # >1 means work takes longer than estimated
    p25: float
    p75: float
    within_25pct: float   # share of tickets landing within ±25% of their estimate


def calibration(tickets: Sequence[Ticket], by: str = "project", min_samples: int = 5) -> list[Calibration]:
    groups: dict[str, list[float]] = {}
    for t, ratio in estimate_ratios(tickets):
        groups.setdefault(getattr(t, by) or "?", []).append(ratio)
    out = []
    for g, r in groups.items():
        if len(r) >= min_samples:
            a = np.array(r)
            p25, p50, p75 = np.percentile(a, [25, 50, 75])
            out.append(Calibration(g, a.size, float(p50), float(p25), float(p75),
                                   float(np.mean(np.abs(a - 1) <= 0.25))))
    return sorted(out, key=lambda c: -c.n)
