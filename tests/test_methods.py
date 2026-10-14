import numpy as np
import pytest

from ticket_estimator.methods import (ThreePoint, calibration, ebs_forecast, ratio_history, reference_class,
                                      throughput_forecast, tshirt_size, weekly_throughput)
from ticket_estimator.records import Ticket


def ticket(i, project="WEB", spent=4.0, est=4.0, assignee="Ana", resolved="2026-09-01"):
    return Ticket(key=f"{project}-{i}", project=project, issue_type="Task", summary="s", assignee=assignee,
                  hours_spent=spent, hours_estimated=est, resolved=resolved)


@pytest.mark.parametrize("hours,size", [(0.5, "XS"), (1, "XS"), (3, "S"), (8, "M"), (20, "L"), (40, "XL"),
                                        (41, "XXL (split it)")])
def test_tshirt_size(hours, size):
    assert tshirt_size(hours) == size


def test_pert_and_reference_class():
    tp = ThreePoint(2, 5, 14)
    assert tp.mean == pytest.approx(6.0) and tp.std == pytest.approx(2.0)
    rc = reference_class([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert rc.optimistic < rc.likely < rc.pessimistic and rc.likely == pytest.approx(5.5)
    with pytest.raises(ValueError):
        reference_class([])


def test_ratio_history_prefers_the_most_specific_scope():
    tickets = [ticket(i, spent=8, est=4, assignee="Ana") for i in range(12)]
    tickets += [ticket(100 + i, spent=4, est=4, assignee="Bo") for i in range(3)]
    r, scope = ratio_history(tickets, project="WEB", assignee="Ana")
    assert scope == "assignee Ana" and np.all(r == 2.0)
    r, scope = ratio_history(tickets, project="WEB", assignee="Bo")  # Bo has too few tickets
    assert scope == "project WEB" and r.size == 15
    with pytest.raises(ValueError):
        ratio_history(tickets[:3])


def test_ebs_forecast_scales_estimates_by_history():
    f = ebs_forecast([10, 10], np.array([2.0]))
    assert all(v == pytest.approx(40.0) for v in f.percentiles.values())
    f = ebs_forecast([10] * 20, np.array([0.5, 1.0, 2.0]))
    assert f.percentiles[50] < f.percentiles[85] < f.percentiles[95]


def test_throughput_forecast():
    tickets = [ticket(i, resolved=f"2026-09-{1 + i % 28:02d}") for i in range(40)]
    tp = weekly_throughput(tickets, weeks=4)
    assert tp.size == 4 and tp.sum() > 0
    f = throughput_forecast(np.array([5.0]), items=20)
    assert all(v == 4 for v in f.percentiles.values())
    with pytest.raises(ValueError):
        throughput_forecast(np.zeros(4), items=5)


def test_calibration_groups_and_orders_by_size():
    tickets = [ticket(i, project="A", spent=6, est=4) for i in range(8)]
    tickets += [ticket(50 + i, project="B", spent=4, est=4) for i in range(5)]
    tickets += [ticket(90, project="C")]  # below min_samples
    rows = calibration(tickets)
    assert [r.group for r in rows] == ["A", "B"]
    assert rows[0].median_ratio == pytest.approx(1.5) and rows[0].within_25pct == 0
    assert rows[1].within_25pct == 1
