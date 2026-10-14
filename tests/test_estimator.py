import json
import random

import numpy as np
import pytest

from ticket_estimator.cli import main
from ticket_estimator.evaluate import backtest
from ticket_estimator.model import EffortModel
from ticket_estimator.records import Ticket, adf_to_text, load_tickets, ticket_from_issue

# Synthetic history: the kind of work in the summary drives the effort.
WORK = {
    "fix typo in footer copy": 0.5,
    "update button colour on landing page": 1.0,
    "add validation to signup form fields": 4.0,
    "build new reporting dashboard with charts": 24.0,
    "migrate payments service to new provider": 40.0,
}


def raw_issue(i, summary, hours, project="WEB", estimate=None):
    return {
        "key": f"{project}-{i}",
        "fields": {
            "summary": summary,
            "description": {"type": "doc", "content": [{"type": "paragraph", "content": [
                {"type": "text", "text": f"Details for {summary} &amp; more."},
                {"type": "mention", "attrs": {"id": "x"}}]}]},
            "issuetype": {"name": "Task"},
            "project": {"key": project},
            "priority": {"name": "Medium"},
            "labels": ["frontend"],
            "components": [{"name": "web"}],
            "parent": {"fields": {"summary": "Epic A"}},
            "assignee": {"displayName": "Dev One"},
            "timespent": int(hours * 3600),
            "timeoriginalestimate": int(estimate * 3600) if estimate else None,
            "customfield_10016": 3,
            "created": f"2026-01-{1 + i % 28:02d}T10:00:00.000+0000",
            "resolutiondate": f"2026-{1 + i // 40:02d}-{1 + i % 28:02d}T10:00:00.000+0000",
        },
    }


@pytest.fixture
def history():
    rng = random.Random(0)
    issues = []
    for i in range(200):
        summary, hours = rng.choice(list(WORK.items()))
        noisy = hours * rng.uniform(0.7, 1.4)
        issues.append(raw_issue(i, summary, noisy, estimate=hours))
    return issues


def test_adf_to_text_flattens_and_skips_mentions():
    doc = {"type": "doc", "content": [{"type": "paragraph", "content": [
        {"type": "text", "text": "Hello"}, {"type": "mention"}, {"type": "text", "text": "world"}]}]}
    assert adf_to_text(doc).split() == ["Hello", "world"]
    assert adf_to_text("plain") == "plain"
    assert adf_to_text(None) == ""


def test_ticket_from_issue_maps_fields():
    t = ticket_from_issue(raw_issue(1, "Fix **login** bug", 2.5, estimate=2))
    assert t.key == "WEB-1"
    assert t.summary == "Fix login bug"
    assert t.description == "Details for Fix **login** bug & more.".replace("**", "")
    assert (t.hours_spent, t.hours_estimated, t.story_points) == (2.5, 2.0, 3.0)
    assert (t.project, t.issue_type, t.parent_summary, t.assignee) == ("WEB", "Task", "Epic A", "Dev One")
    assert t.components == ["web"] and t.labels == ["frontend"]


@pytest.mark.parametrize("fmt", ["jsonl", "rest", "mcp"])
def test_load_tickets_reads_every_export_format(tmp_path, history, fmt):
    if fmt == "jsonl":
        path = tmp_path / "h.jsonl"
        path.write_text("\n".join(json.dumps(i) for i in history))
    else:
        path = tmp_path / "page.json"
        body = {"issues": history} if fmt == "rest" else {"issues": {"nodes": history}}
        path.write_text(json.dumps(body))
    # Loading the same file twice must not duplicate tickets.
    assert len(load_tickets([path, tmp_path])) == len(history)


def test_model_learns_effort_from_text(history):
    model = EffortModel().fit([ticket_from_issue(i) for i in history])
    small = model.estimate(Ticket(key="N1", project="WEB", issue_type="Task", summary="fix typo in footer copy"))
    big = model.estimate(Ticket(key="N2", project="WEB", issue_type="Task",
                                summary="migrate payments service to new provider"))
    assert small.hours < 2 < 20 < big.hours
    assert small.low <= small.hours <= small.high
    assert big.similar and "migrate payments" in big.similar[0].summary
    assert big.team_estimate_bias == pytest.approx(1.0, abs=0.2)


def test_fit_rejects_too_little_history():
    with pytest.raises(ValueError, match="at least 30"):
        EffortModel().fit([Ticket(key="A-1", project="A", issue_type="Bug", summary="x", hours_spent=1.0)])


def test_backtest_beats_baseline_on_learnable_data(history):
    scores, coverage = backtest([ticket_from_issue(i) for i in history])
    assert scores["model"].within_2x > 0.9
    assert scores["model"].median_abs_error_hours <= scores["median per project+type"].median_abs_error_hours
    assert 0.5 <= coverage <= 1.0


def test_cli_train_then_estimate(tmp_path, history, capsys):
    data = tmp_path / "history.jsonl"
    data.write_text("\n".join(json.dumps(i) for i in history))
    model = tmp_path / "m.joblib"
    main(["train", "--data", str(data), "--model", str(model)])
    main(["estimate", "--model", str(model), "--project", "WEB", "--type", "Task", "--json",
          "--summary", "add validation to signup form fields"])
    out = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert 1 < out["hours"] < 10
    assert np.isfinite(out["low"]) and out["low"] <= out["hours"] <= out["high"]
    assert out["size"] in ("S", "M") and out["reference_class"]["p10"] <= out["reference_class"]["p90"]

    main(["forecast", "--data", str(data), "--items", "30", "--hours", "4", "8"])
    text = capsys.readouterr().out
    assert "Throughput forecast for 30 tickets" in text and "Evidence-based forecast" in text
    main(["calibrate", "--data", str(data)])
    assert "WEB" in capsys.readouterr().out
