"""Command line entry point: fetch history, train, back-test and estimate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

from .evaluate import backtest
from .jira import JiraClient, fetch_history, history_fields
from .model import EffortModel, Estimate
from .records import DEFAULT_STORY_POINTS_FIELD, Ticket, load_tickets, ticket_from_issue

DEFAULT_JQL = "statusCategory = Done AND timespent > 0 AND resolved >= -365d ORDER BY resolved ASC"
DEFAULT_DATA = Path("data/history.jsonl")
DEFAULT_MODEL = Path("models/estimator.joblib")


def _hours(h: float) -> str:
    return f"{h:.1f}h" if h < 8 else f"{h:.1f}h (~{h / 8:.1f}d)"


def format_estimate(est: Estimate, ticket: Ticket) -> str:
    lines = [f"Estimate for: {ticket.summary}",
             f"  Most likely : {_hours(est.hours)}",
             f"  Likely range: {_hours(est.low)} – {_hours(est.high)}  (wide on purpose: see `backtest` for how often it holds)"]
    if est.team_estimate_bias:
        b = est.team_estimate_bias
        verdict = "over-run" if b > 1.1 else "finish under" if b < 0.9 else "land close to"
        lines.append(f"  History     : {ticket.project} tickets typically {verdict} their original estimate "
                     f"(actual/estimate = {b:.2f})")
    if est.similar:
        lines.append("  Most similar past tickets:")
        for s in est.similar:
            lines.append(f"    {s.key:<10} {_hours(s.hours_spent):>14}  sim={s.similarity:.2f}  {s.summary[:70]}")
    return "\n".join(lines)


def cmd_fetch(args: argparse.Namespace) -> None:
    n = fetch_history(JiraClient(), args.jql, args.out, args.story_points_field)
    print(f"Saved {n} issues to {args.out}")


def cmd_train(args: argparse.Namespace) -> None:
    tickets = load_tickets(args.data, args.story_points_field)
    model = EffortModel().fit(tickets)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    print(f"Trained on {len(model.train_meta)} tickets with logged time -> {args.model}")


def cmd_backtest(args: argparse.Namespace) -> None:
    tickets = load_tickets(args.data, args.story_points_field)
    print(f"Training on the oldest {1 - args.test_fraction:.0%} of tickets, scoring the newest {args.test_fraction:.0%}:")
    scores, coverage = backtest(tickets, args.test_fraction)
    for name, s in scores.items():
        print("  " + s.row(name))
    print(f"  Actual hours fell inside the model's likely range for {coverage:.0%} of tickets.")


def cmd_estimate(args: argparse.Namespace) -> None:
    model: EffortModel = joblib.load(args.model)
    if args.issue:
        issue = JiraClient().get_issue(args.issue, history_fields(args.story_points_field))
        ticket = ticket_from_issue(issue, args.story_points_field)
    elif args.summary:
        ticket = Ticket(key="NEW", project=args.project or "", issue_type=args.type or "", summary=args.summary,
                        description=args.description or "", priority=args.priority or "",
                        assignee=args.assignee or "")
    else:
        sys.exit("Pass --issue KEY or --summary TEXT")
    est = model.estimate(ticket)
    if args.json:
        print(json.dumps({"ticket": ticket.key, "hours": est.hours, "low": est.low, "high": est.high,
                          "team_estimate_bias": est.team_estimate_bias,
                          "similar": [s.__dict__ for s in est.similar]}, indent=2))
    else:
        print(format_estimate(est, ticket))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ticket-estimator", description=__doc__)
    p.add_argument("--story-points-field", default=DEFAULT_STORY_POINTS_FIELD)
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="Download resolved tickets from Jira")
    f.add_argument("--jql", default=DEFAULT_JQL)
    f.add_argument("--out", type=Path, default=DEFAULT_DATA)
    f.set_defaults(func=cmd_fetch)

    for name, func, help_ in (("train", cmd_train, "Train and save the model"),
                              ("backtest", cmd_backtest, "Measure accuracy on the newest tickets")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--data", type=Path, nargs="+", default=[DEFAULT_DATA],
                       help="JSONL from `fetch`, saved Jira search responses, or directories of them")
        s.set_defaults(func=func)
        if name == "train":
            s.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        else:
            s.add_argument("--test-fraction", type=float, default=0.2)

    e = sub.add_parser("estimate", help="Estimate a ticket")
    e.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    e.add_argument("--issue", help="Existing Jira key, e.g. MT-1234 (needs Jira credentials)")
    e.add_argument("--summary")
    e.add_argument("--description")
    e.add_argument("--project", help="Project key, e.g. MT")
    e.add_argument("--type", help="Issue type, e.g. Bug, Task, Subtask, Story")
    e.add_argument("--priority")
    e.add_argument("--assignee", help="Display name as shown in Jira")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_estimate)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
