"""Apply to matched jobs.

- Posting lists a recruiter email -> a tailored email with your résumé attached (sent only with --send).
- Otherwise (Greenhouse/Lever/Ashby/job-board forms) -> a ready-to-paste cover letter plus the apply
  link in outbox/manual/. Those forms need accounts, CAPTCHAs and per-question answers, and their
  terms forbid bots, so the tool prepares everything and you click submit.
"""

from __future__ import annotations

import mimetypes
import os
import re
import smtplib
import time
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path
from typing import Callable

from .jobs import Job
from .matcher import Match
from .profile import Profile
from .store import Store

DEFAULT_TEMPLATE = """Dear Hiring Team at {company},

I came across the {title} opening ({work_mode}, {location}) and would like to be considered.

{summary}

The role asks for experience I bring directly: {matched_skills}.

My résumé is attached. I'd welcome a short call to discuss how I can help {company}.

Thank you for your time,
{name}
{email}{phone_line}{linkedin_line}

Job link: {url}
"""


class _Blank(dict):
    def __missing__(self, key):  # unknown placeholders render empty instead of crashing a batch
        return ""


def _fields(job: Job, match: Match, profile: Profile) -> dict:
    skills = match.matched_skills[:6]
    return _Blank(
        name=profile.name, email=profile.email, phone=profile.phone, linkedin=profile.linkedin,
        phone_line=f"\n{profile.phone}" if profile.phone else "",
        linkedin_line=f"\n{profile.linkedin}" if profile.linkedin else "",
        summary=profile.summary or f"I have {profile.years_experience:g}+ years of relevant experience.",
        years=f"{profile.years_experience:g}", title=job.title, company=job.company or "your company",
        location=job.location or "location flexible", work_mode=job.work_mode, url=job.url,
        matched_skills=(", ".join(skills[:-1]) + f" and {skills[-1]}") if len(skills) > 1 else "".join(skills),
        score=f"{match.score:g}",
    )


def compose(job: Job, match: Match, profile: Profile) -> tuple[str, str]:
    f = _fields(job, match, profile)
    body = (profile.email_template or DEFAULT_TEMPLATE).format_map(f)
    return profile.email_subject.format_map(f), re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"


def build_message(profile: Profile, to: str, subject: str, body: str) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = f"{profile.name} <{profile.email}>"
    msg["To"] = to
    msg["Subject"] = subject
    msg["Message-ID"] = make_msgid(domain=profile.email.split("@")[-1])
    msg.set_content(body)
    if profile.resume_path:
        p = Path(profile.resume_path)
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        msg.add_attachment(p.read_bytes(), maintype=maintype, subtype=subtype, filename=p.name)
    return msg


def smtp_sender() -> Callable[[EmailMessage], None]:
    """Gmail: SMTP_USER=you@gmail.com, SMTP_PASSWORD=<16-char app password> (myaccount.google.com/apppasswords)."""
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user, password = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASSWORD")
    if not (user and password):
        raise SystemExit("Set SMTP_USER and SMTP_PASSWORD (for Gmail use an app password) to send emails.")

    def send(msg: EmailMessage) -> None:
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.starttls()
            s.login(user, password)
            s.send_message(msg)
    return send


def _safe(job_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", job_id)


@dataclass
class ApplyResult:
    emailed: list[str]
    drafted: list[str]
    manual: list[str]
    failed: list[str]
    skipped: list[str]
    stopped_at_limit: bool = False


def apply(store: Store, profile: Profile, outbox: Path, send: bool = False, limit: int | None = None,
          sender: Callable[[EmailMessage], None] | None = None, sleep: Callable[[float], None] = time.sleep,
          confirm: Callable[[Job, str, str], bool] | None = None) -> ApplyResult:
    if profile.resume_path and not Path(profile.resume_path).exists():
        raise SystemExit(f"Résumé not found: {profile.resume_path}")
    res = ApplyResult([], [], [], [], [])
    candidates = [(j, m) for j, m in store.shortlist(include_applied=True)
                  if store.application_status(j.id) in (None, "drafted")]
    if limit:
        candidates = candidates[:limit]
    if send and sender is None:
        sender = smtp_sender()
    budget = profile.daily_limit - store.sent_today()
    emailed_this_run: set[str] = set()

    for job, match in candidates:
        subject, body = compose(job, match, profile)
        to = next((e for e in job.emails if e not in emailed_this_run and not store.contacted(e)), None)

        if not to:  # no recruiter address -> prepare a manual application
            if job.emails:  # every address on it was already emailed for another opening
                store.record_application(job.id, "skipped", "email", job.emails[0], note="recruiter already contacted")
                res.skipped.append(job.id)
                continue
            path = outbox / "manual" / f"{_safe(job.id)}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {job.title} — {job.company}\n\n- Score: {match.score:g}\n- Mode: {job.work_mode}"
                            f"\n- Location: {job.location}\n- **Apply here:** {job.apply_url or job.url}\n"
                            f"- Matched skills: {', '.join(match.matched_skills)}\n"
                            f"- Skills they want you haven't listed: {', '.join(match.missing_skills) or '—'}\n\n"
                            f"## Cover letter\n\n{body}")
            store.record_application(job.id, "manual", "form", job.apply_url or job.url, subject, body)
            res.manual.append(job.id)
            continue

        if not send:
            path = outbox / "drafts" / f"{_safe(job.id)}.eml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes(build_message(profile, to, subject, body)))
            store.record_application(job.id, "drafted", "email", to, subject, body)
            res.drafted.append(job.id)
            continue

        if budget <= 0:
            res.stopped_at_limit = True
            break
        if confirm and not confirm(job, to, body):
            store.record_application(job.id, "skipped", "email", to, subject, body, note="declined at review")
            res.skipped.append(job.id)
            continue
        try:
            if res.emailed:
                sleep(profile.seconds_between_emails)
            sender(build_message(profile, to, subject, body))
            store.record_application(job.id, "sent", "email", to, subject, body)
            res.emailed.append(job.id)
            emailed_this_run.add(to)
            budget -= 1
        except Exception as e:
            store.record_application(job.id, "failed", "email", to, subject, body, note=str(e)[:300])
            res.failed.append(job.id)
    return res
