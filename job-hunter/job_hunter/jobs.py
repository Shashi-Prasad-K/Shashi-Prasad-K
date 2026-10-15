"""The normalised job record every source produces, plus text helpers shared by sources and matcher."""

from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass, field

REMOTE, HYBRID, ONSITE = "remote", "hybrid", "onsite"

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"[ \t\r\f\v]+")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Addresses that never reach a person, or belong to the job board rather than the employer
_NOT_RECRUITER = re.compile(r"no-?reply|do-?not-?reply|mailer-daemon|privacy|unsubscribe|support@|abuse@|"
                            r"@(?:example\.|remotive\.|remoteok\.|himalayas\.|jobicy\.|arbeitnow\.|"
                            r"greenhouse\.io|lever\.co|ashbyhq\.com|adzuna\.)", re.I)
_HYBRID = re.compile(r"\bhybrid\b", re.I)
_REMOTE = re.compile(r"\b(?:remote|work from home|wfh|anywhere|distributed team)\b", re.I)
_ONSITE = re.compile(r"\b(?:on-?site|in[- ]office|office[- ]based|on location)\b", re.I)


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    text = html.unescape(text)  # Greenhouse double-escapes its HTML
    text = re.sub(r"<\s*(?:br|/p|/li|/h\d|/div)\s*/?>", "\n", text, flags=re.I)
    text = html.unescape(_TAG.sub(" ", text))
    lines = (_SPACE.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def infer_work_mode(*texts: str | None, default: str = ONSITE) -> str:
    """Hybrid wins over remote ("remote-friendly, hybrid 2 days") and remote over the on-site default."""
    blob = " ".join(t for t in texts if t)
    if _HYBRID.search(blob):
        return HYBRID
    if _REMOTE.search(blob):
        return REMOTE
    if _ONSITE.search(blob):
        return ONSITE
    return default


def normalise_work_mode(value: str | None) -> str | None:
    v = (value or "").lower().replace("-", "").replace("_", "").replace(" ", "")
    return {"remote": REMOTE, "hybrid": HYBRID, "onsite": ONSITE, "inoffice": ONSITE, "office": ONSITE}.get(v)


def recruiter_emails(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in _EMAIL.findall(text or ""):
        addr = m.strip(".").lower()
        if not _NOT_RECRUITER.search(addr) and not addr.endswith((".png", ".jpg", ".gif", ".svg")):
            seen.setdefault(addr, None)
    return list(seen)


@dataclass
class Job:
    source: str
    source_id: str
    title: str
    company: str
    location: str
    work_mode: str
    url: str
    description: str = ""
    apply_url: str = ""
    posted_at: str = ""
    salary: str = ""
    emails: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.title = (self.title or "").strip()
        self.company = (self.company or "").strip()
        self.location = (self.location or "").strip()
        if not self.emails:
            self.emails = recruiter_emails(self.description)

    @property
    def id(self) -> str:
        return f"{self.source}:{self.source_id}"

    @property
    def dedupe_key(self) -> str:
        """The same opening is often cross-posted; company + title + mode catches most copies."""
        norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
        return f"{norm(self.company)}|{norm(self.title)}|{self.work_mode}"

    def to_dict(self) -> dict:
        return asdict(self)
