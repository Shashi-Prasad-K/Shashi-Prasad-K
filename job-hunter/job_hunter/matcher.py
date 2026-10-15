"""Score each job 0–100 against your profile and explain why, so you can trust (or tune) the cut-off."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from .jobs import REMOTE, Job
from .profile import Profile

TITLE_WEIGHT, SKILL_WEIGHT = 0.45, 0.55
# Matching this much skill weight counts as a full skill score; jobs rarely list more than ~8 of yours
SKILL_SATURATION = 8.0
_STOP = {"and", "or", "the", "a", "of", "for", "in", "to", "senior", "sr", "junior", "jr", "lead", "i", "ii", "iii"}

# Common skills employers ask for — used only to tell you what a posting wants that your profile lacks
SKILL_VOCAB = [
    "PMP", "PRINCE2", "Agile", "Scrum", "Kanban", "SAFe", "Waterfall", "Jira", "Confluence", "MS Project",
    "Smartsheet", "Asana", "Azure DevOps", "Stakeholder management", "Risk management", "Budgeting",
    "Vendor management", "SDLC", "ITIL", "Change management", "Resource planning", "Roadmap",
    "Python", "Java", "JavaScript", "TypeScript", "SQL", "AWS", "Azure", "GCP", "Kubernetes", "Docker",
    "Salesforce", "SAP", "ServiceNow", "Power BI", "Tableau", "Excel", "CI/CD", "DevOps", "Data analysis",
    "Machine learning", "Cloud migration", "ERP", "CRM", "Fintech", "Banking", "Insurance", "Healthcare",
    "CSM", "PMI-ACP", "Six Sigma", "Lean", "OKRs", "Product management", "UAT", "Business analysis",
]


@lru_cache(maxsize=4096)
def _pattern(term: str) -> re.Pattern:
    # Word boundaries that also work for terms like "C++", "CI/CD" or ".NET"
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", re.I)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9+#]+", text.lower()) if t not in _STOP}


@dataclass
class Match:
    job_id: str
    score: float
    accepted: bool
    title_score: float
    skill_score: float
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def title_score(job_title: str, wanted: list[str]) -> float:
    if not wanted:
        return 1.0
    best = 0.0
    jt = _tokens(job_title)
    for w in wanted:
        if _pattern(w).search(job_title):
            return 1.0
        wt = _tokens(w)
        if wt:
            best = max(best, len(wt & jt) / len(wt))
    return best


def location_ok(job: Job, profile: Profile) -> tuple[bool, str]:
    loc = job.location.lower()
    if job.work_mode == REMOTE:
        if not loc or any(r in loc for r in profile.remote_regions) or any(p.lower() in loc for p in profile.locations):
            return True, ""
        return False, f"remote but restricted to '{job.location}'"
    if not profile.locations or any(p.lower() in loc for p in profile.locations):
        return True, ""
    return False, f"{job.work_mode} in '{job.location}', not one of your locations"


def score_job(job: Job, profile: Profile) -> Match:
    text = f"{job.title}\n{job.description}"
    reasons: list[str] = []

    banned = [k for k in profile.exclude_keywords if _pattern(k).search(job.title)]
    if banned:
        reasons.append(f"title contains excluded keyword(s): {', '.join(banned)}")
    if job.work_mode not in profile.work_modes:
        reasons.append(f"work mode {job.work_mode} not wanted")
    ok, why = location_ok(job, profile)
    if not ok:
        reasons.append(why)

    matched, weight = [], 0.0
    for s in profile.skills:
        if any(_pattern(t).search(text) for t in s.terms):
            matched.append(s.name)
            weight += s.weight
    have = {t.lower() for s in profile.skills for t in s.terms}
    missing = [v for v in SKILL_VOCAB if v.lower() not in have and _pattern(v).search(text)]

    ts = title_score(job.title, profile.titles)
    ss = min(1.0, weight / min(SKILL_SATURATION, sum(s.weight for s in profile.skills)))
    score = round(100 * (TITLE_WEIGHT * ts + SKILL_WEIGHT * ss), 1)
    if score < profile.min_score:
        reasons.append(f"score {score} below min_score {profile.min_score:g}")
    return Match(job.id, score, not reasons, round(ts, 2), round(ss, 2), matched, missing, reasons)
