"""Command line: fetch jobs, match them to your skills, and apply."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from .apply import apply
from .matcher import score_job
from .profile import load_profile
from .sources import SOURCES, fetch_all
from .store import STATUSES, Store

DEFAULT_PROFILE = Path("profile.toml")
DEFAULT_DB = Path("data/jobs.db")
DEFAULT_OUTBOX = Path("outbox")


def cmd_fetch(args, profile, store) -> None:
    jobs, report = fetch_all(profile, args.source)
    new = store.upsert_jobs(jobs)
    for name, msg in report.items():
        print(f"  {name:<11} {msg}")
    print(f"Fetched {len(jobs)} postings, {new} new (cross-posted duplicates dropped)")


def cmd_match(args, profile, store) -> None:
    jobs = store.jobs(unscored_only=not args.rescore)
    matches = [score_job(j, profile) for j in jobs]
    store.save_matches(matches)
    print(f"Scored {len(matches)} jobs: {sum(m.accepted for m in matches)} pass your filters "
          f"(min_score {profile.min_score:g})")


def cmd_list(args, profile, store) -> None:
    rows = store.shortlist(limit=args.limit, include_applied=args.all)
    if args.mode:
        rows = [(j, m) for j, m in rows if j.work_mode == args.mode]
    if args.json:
        print(json.dumps([{**j.to_dict(), "description": j.description[:500], "match": m.to_dict()}
                          for j, m in rows], indent=2))
        return
    if args.csv:
        w = csv.writer(sys.stdout)
        w.writerow(["id", "score", "title", "company", "work_mode", "location", "emails", "matched", "missing", "url"])
        for j, m in rows:
            w.writerow([j.id, m.score, j.title, j.company, j.work_mode, j.location, " ".join(j.emails),
                        "; ".join(m.matched_skills), "; ".join(m.missing_skills), j.apply_url or j.url])
        return
    if not rows:
        print("No matches yet. Run `job-hunter fetch` then `job-hunter match`, or lower min_score.")
    for j, m in rows:
        contact = f"✉ {j.emails[0]}" if j.emails else "form"
        print(f"{m.score:5.1f}  {j.work_mode:<6}  {j.title[:48]:<48}  {j.company[:22]:<22}  {j.location[:24]:<24}  {contact}")
        if args.verbose:
            print(f"       matched: {', '.join(m.matched_skills)}")
            if m.missing_skills:
                print(f"       gaps   : {', '.join(m.missing_skills)}")
            print(f"       {j.apply_url or j.url}\n       id: {j.id}")


def _review(job, to, body) -> bool:
    print("\n" + "=" * 78 + f"\nTo: {to}\nJob: {job.title} @ {job.company}\n" + "-" * 78 + f"\n{body}")
    return input("Send this? [y/N] ").strip().lower() == "y"


def cmd_apply(args, profile, store) -> None:
    if args.send and not args.yes:
        print(f"Live mode: each email is shown for approval (pass --yes to skip review). "
              f"Daily cap {profile.daily_limit}, {store.sent_today()} already sent today.")
    res = apply(store, profile, args.outbox, send=args.send, limit=args.limit,
                confirm=None if (args.yes or not args.send) else _review)
    print(f"Emailed {len(res.emailed)}, drafted {len(res.drafted)} (in {args.outbox}/drafts), "
          f"manual {len(res.manual)} (in {args.outbox}/manual), failed {len(res.failed)}, skipped {len(res.skipped)}")
    if res.drafted and not args.send:
        print("Dry run — nothing was sent. Review the drafts, then run `job-hunter apply --send`.")
    if res.stopped_at_limit:
        print(f"Stopped at the daily limit of {profile.daily_limit}; the rest go out on the next run.")


def cmd_run(args, profile, store) -> None:
    args.rescore = False
    cmd_fetch(args, profile, store)
    cmd_match(args, profile, store)
    cmd_apply(args, profile, store)


def cmd_status(args, profile, store) -> None:
    if args.job_id:
        if not store.set_status(args.job_id, args.job_status, args.note or ""):
            raise SystemExit(f"Unknown job id {args.job_id}")
        print(f"{args.job_id} -> {args.job_status}")
        return
    print("  ".join(f"{k}: {v}" for k, v in store.counts().items()))
    for r in store.applications()[: args.limit or 30]:
        print(f"{r['status']:<9} {r['channel']:<5} {(r['score'] or 0):5.1f}  {r['title'][:44]:<44}  "
              f"{r['company'][:20]:<20}  {r['recipient'][:40]}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="job-hunter", description=__doc__)
    p.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("-v", "--debug", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def apply_args(sp):
        sp.add_argument("--send", action="store_true", help="actually send emails (default: write drafts only)")
        sp.add_argument("--yes", action="store_true", help="with --send, skip the per-email review prompt")
        sp.add_argument("--limit", type=int, help="process at most N jobs this run")
        sp.add_argument("--outbox", type=Path, default=DEFAULT_OUTBOX)

    sp = sub.add_parser("fetch", help="pull openings from every enabled source")
    sp.add_argument("--source", action="append", choices=sorted(SOURCES), help="only this source (repeatable)")
    sp.set_defaults(fn=cmd_fetch)

    sp = sub.add_parser("match", help="score stored jobs against your profile")
    sp.add_argument("--rescore", action="store_true", help="re-score everything (after editing your profile)")
    sp.set_defaults(fn=cmd_match)

    sp = sub.add_parser("list", help="show the best matches")
    sp.add_argument("--limit", type=int, default=40)
    sp.add_argument("--mode", choices=["remote", "hybrid", "onsite"])
    sp.add_argument("--all", action="store_true", help="include jobs already applied to")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--csv", action="store_true")
    sp.add_argument("--verbose", action="store_true", help="show matched skills, gaps and links")
    sp.set_defaults(fn=cmd_list)

    sp = sub.add_parser("apply", help="email recruiters / prepare form applications for matches")
    apply_args(sp)
    sp.set_defaults(fn=cmd_apply)

    sp = sub.add_parser("run", help="fetch + match + apply in one go (for cron)")
    sp.add_argument("--source", action="append", choices=sorted(SOURCES))
    apply_args(sp)
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("status", help="application tracker; or set a job's status")
    sp.add_argument("job_id", nargs="?")
    sp.add_argument("job_status", nargs="?", choices=STATUSES)
    sp.add_argument("--note")
    sp.add_argument("--limit", type=int)
    sp.set_defaults(fn=cmd_status)

    args = p.parse_args(argv)
    if args.cmd == "status" and args.job_id and not args.job_status:
        p.error("status <job_id> needs a new status, e.g. `status lever:abc applied`")
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING, format="%(levelname)s %(message)s")
    if not args.profile.exists():
        raise SystemExit(f"No profile at {args.profile}. Copy profile.example.toml to profile.toml and edit it.")
    args.fn(args, load_profile(args.profile), Store(args.db))


if __name__ == "__main__":
    main()
