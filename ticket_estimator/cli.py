"""Command line entry point: fetch history, train, back-test, estimate, calibrate and forecast."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

from .evaluate import backtest
from .jira import JiraClient, fetch_history, history_fields
from .methods import (calibration, ebs_forecast, ratio_history, reference_class, throughput_forecast,
                      tshirt_size, weekly_throughput)
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
             f"  Likely range: {_hours(est.low)} – {_hours(est.high)}  (wide on purpose: see `backtest` for how often it holds)",
             f"  T-shirt size: {tshirt_size(est.hours)}"]
    if est.reference_hours:
        rc = reference_class(est.reference_hours)
        lines.append(f"  Outside view: {len(est.reference_hours)} closest past tickets took "
                     f"{_hours(rc.optimistic)} / {_hours(rc.likely)} / {_hours(rc.pessimistic)} (P10/P50/P90); "
                     f"PERT mean {_hours(rc.mean)}")
    if est.team_estimate_bias:
        b = est.team_estimate_bias
        verdict = "over-run" if b > 1.1 else "finish under" if b < 0.9 else "land close to"
        lines.append(f"  History     : {est.bias_scope} tickets typically {verdict} their original estimate "
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
        rc = reference_class(est.reference_hours)
        print(json.dumps({"ticket": ticket.key, "hours": est.hours, "low": est.low, "high": est.high,
                          "size": tshirt_size(est.hours),
                          "reference_class": {"p10": rc.optimistic, "p50": rc.likely, "p90": rc.pessimistic,
                                              "pert_mean": rc.mean},
                          "team_estimate_bias": est.team_estimate_bias,
                          "similar": [s.__dict__ for s in est.similar]}, indent=2))
    else:
        print(format_estimate(est, ticket))


def cmd_calibrate(args: argparse.Namespace) -> None:
    tickets = load_tickets(args.data, args.story_points_field)
    rows = calibration(tickets, by=args.by)
    print(f"How actual time compares with the original estimate, by {args.by.replace('_', ' ')}:")
    print(f"  {'group':<24} {'tickets':>7} {'actual/est':>10} {'middle 50%':>13} {'within ±25%':>12}")
    for c in rows:
        print(f"  {c.group[:24]:<24} {c.n:>7} {c.median_ratio:>10.2f} {c.p25:>6.2f}–{c.p75:<6.2f} {c.within_25pct:>11.0%}")


def cmd_forecast(args: argparse.Namespace) -> None:
    tickets = load_tickets(args.data, args.story_points_field)
    hours = list(args.hours or [])
    items = args.items
    if args.jql:
        model: EffortModel = joblib.load(args.model)
        open_tickets = [ticket_from_issue(i, args.story_points_field)
                        for i in JiraClient().search(args.jql, history_fields(args.story_points_field))]
        guessed = [t for t in open_tickets if not t.hours_estimated]
        hours += [t.hours_estimated for t in open_tickets if t.hours_estimated]
        hours += [model.estimate(t).hours for t in guessed]
        items = items or len(open_tickets)
        print(f"{len(open_tickets)} open tickets; {len(guessed)} had no original estimate, so the model filled it in.")
    if items:
        tp = weekly_throughput(tickets, args.weeks, args.project)
        print(f"Throughput forecast for {items} tickets: " + throughput_forecast(tp, items).describe())
    if hours:
        ratios, scope = ratio_history(tickets, args.project, args.assignee)
        f = ebs_forecast(hours, ratios, basis=f"{ratios.size} actual/estimate ratios from {scope}")
        print(f"Evidence-based forecast for {len(hours)} estimates totalling {sum(hours):,.1f}h: " + f.describe())
    if not (items or hours):
        sys.exit("Pass --items N, --hours H [H ...], or --jql for open tickets")


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

    c = sub.add_parser("calibrate", help="How actual time compares with original estimates")
    c.add_argument("--data", type=Path, nargs="+", default=[DEFAULT_DATA])
    c.add_argument("--by", choices=["project", "issue_type", "assignee"], default="project")
    c.set_defaults(func=cmd_calibrate)

    fc = sub.add_parser("forecast", help="Monte Carlo delivery forecast for a batch of work")
    fc.add_argument("--data", type=Path, nargs="+", default=[DEFAULT_DATA])
    fc.add_argument("--items", type=int, help="Number of tickets to finish (throughput forecast)")
    fc.add_argument("--hours", type=float, nargs="+", help="Original estimates in hours (evidence-based forecast)")
    fc.add_argument("--jql", help="Open tickets to forecast, e.g. 'sprint in openSprints() AND project = MT'")
    fc.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Used for tickets without an estimate")
    fc.add_argument("--project", default="", help="Limit history to one project")
    fc.add_argument("--assignee", default="", help="Use this person's estimate history when there is enough")
    fc.add_argument("--weeks", type=int, default=12, help="Weeks of throughput history to sample")
    fc.set_defaults(func=cmd_forecast)

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
