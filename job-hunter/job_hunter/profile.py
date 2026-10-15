"""Your candidate profile: who you are, what you're good at, what you want, and how far to automate."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skill:
    name: str
    aliases: list[str] = field(default_factory=list)
    weight: float = 1.0

    @property
    def terms(self) -> list[str]:
        return [self.name, *self.aliases]


@dataclass
class Profile:
    name: str
    email: str
    phone: str = ""
    linkedin: str = ""
    resume_path: str = ""
    summary: str = ""
    years_experience: float = 0
    titles: list[str] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)
    work_modes: list[str] = field(default_factory=lambda: ["remote", "hybrid", "onsite"])
    locations: list[str] = field(default_factory=list)
    remote_regions: list[str] = field(default_factory=lambda: ["worldwide", "anywhere", "global"])
    min_score: float = 55
    # Automation guard-rails
    daily_limit: int = 15
    seconds_between_emails: int = 60
    email_subject: str = "Application for {title} – {name}"
    email_template: str = ""
    sources: dict = field(default_factory=dict)


def load_profile(path: Path) -> Profile:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    me = raw.get("me", {})
    want = raw.get("preferences", {})
    auto = raw.get("apply", {})
    skills = []
    for s in raw.get("skills", []):
        skills.append(Skill(s) if isinstance(s, str) else
                      Skill(s["name"], list(s.get("aliases", [])), float(s.get("weight", 1.0))))
    if not me.get("name") or not me.get("email"):
        raise ValueError(f"{path}: [me] needs at least name and email")
    if not skills:
        raise ValueError(f"{path}: add your skills under [[skills]] so jobs can be matched")
    resume = me.get("resume", "")
    if resume and not Path(resume).is_absolute():
        resume = str((path.parent / resume).resolve())
    return Profile(
        name=me["name"], email=me["email"], phone=me.get("phone", ""), linkedin=me.get("linkedin", ""),
        resume_path=resume, summary=me.get("summary", "").strip(),
        years_experience=float(me.get("years_experience", 0)),
        titles=list(want.get("titles", [])), skills=skills,
        exclude_keywords=list(want.get("exclude_keywords", [])),
        work_modes=[m.lower() for m in want.get("work_modes", ["remote", "hybrid", "onsite"])],
        locations=list(want.get("locations", [])),
        remote_regions=[r.lower() for r in want.get("remote_regions", ["worldwide", "anywhere", "global"])],
        min_score=float(want.get("min_score", 55)),
        daily_limit=int(auto.get("daily_limit", 15)),
        seconds_between_emails=int(auto.get("seconds_between_emails", 60)),
        email_subject=auto.get("subject", "Application for {title} – {name}"),
        email_template=auto.get("template", ""),
        sources=raw.get("sources", {}),
    )
