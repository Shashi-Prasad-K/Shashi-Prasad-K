"""Turn raw Jira issue JSON into flat records the model can learn from."""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

SECONDS_PER_HOUR = 3600.0

# Fields requested from Jira. customfield_10016 is the default "Story point estimate"
# field on Jira Cloud; override with --story-points-field if your site differs.
JIRA_FIELDS = [
    "summary",
    "description",
    "issuetype",
    "project",
    "labels",
    "components",
    "priority",
    "timespent",
    "timeoriginalestimate",
    "created",
    "resolutiondate",
    "parent",
    "assignee",
]
DEFAULT_STORY_POINTS_FIELD = "customfield_10016"


@dataclass
class Ticket:
    key: str
    project: str
    issue_type: str
    summary: str
    description: str = ""
    priority: str = ""
    labels: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    parent_summary: str = ""
    assignee: str = ""
    created: str = ""
    resolved: str = ""
    hours_spent: float | None = None
    hours_estimated: float | None = None
    story_points: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def adf_to_text(node: Any) -> str:
    """Flatten an Atlassian Document Format tree (or plain string) into text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return " ".join(adf_to_text(n) for n in node)
    if isinstance(node, dict):
        if node.get("type") == "text":
            return node.get("text", "")
        if node.get("type") == "mention":
            return ""
        return " ".join(adf_to_text(c) for c in node.get("content", []))
    return str(node)


_MARKUP = re.compile(r"<[^>]+>|\[([^\]]*)\]\([^)]*\)|[*_#`>|]+")


def clean_text(text: str) -> str:
    text = _MARKUP.sub(lambda m: m.group(1) or " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _hours(seconds: Any) -> float | None:
    if seconds in (None, ""):
        return None
    return float(seconds) / SECONDS_PER_HOUR


def _name(obj: Any, attr: str = "name") -> str:
    return (obj or {}).get(attr, "") if isinstance(obj, dict) else ""


def ticket_from_issue(issue: dict[str, Any], story_points_field: str = DEFAULT_STORY_POINTS_FIELD) -> Ticket:
    f = issue.get("fields", {})
    parent = f.get("parent") or {}
    sp = f.get(story_points_field)
    return Ticket(
        key=issue.get("key", ""),
        project=_name(f.get("project"), "key"),
        issue_type=_name(f.get("issuetype")),
        summary=clean_text(f.get("summary") or ""),
        description=clean_text(adf_to_text(f.get("description"))),
        priority=_name(f.get("priority")),
        labels=list(f.get("labels") or []),
        components=[c.get("name", "") for c in f.get("components") or []],
        parent_summary=clean_text((parent.get("fields") or {}).get("summary", "")),
        assignee=_name(f.get("assignee"), "displayName"),
        created=f.get("created") or "",
        resolved=f.get("resolutiondate") or "",
        hours_spent=_hours(f.get("timespent")),
        hours_estimated=_hours(f.get("timeoriginalestimate")),
        story_points=float(sp) if isinstance(sp, (int, float)) else None,
    )


def _iter_raw_issues(path: Path) -> Iterator[dict[str, Any]]:
    """Yield raw issues from a JSONL file or a saved Jira search response."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)
        return
    doc = json.loads(text)
    if isinstance(doc, list):
        yield from doc
    elif isinstance(doc.get("issues"), list):  # REST API response
        yield from doc["issues"]
    elif isinstance(doc.get("issues"), dict):  # Atlassian MCP response
        yield from doc["issues"].get("nodes", [])


def load_tickets(paths: Iterable[Path], story_points_field: str = DEFAULT_STORY_POINTS_FIELD) -> list[Ticket]:
    """Load tickets from files or directories, de-duplicated by key."""
    files: list[Path] = []
    for p in map(Path, paths):
        files.extend(sorted(p.glob("*.json*")) if p.is_dir() else [p])
    seen: dict[str, Ticket] = {}
    for fp in files:
        for raw in _iter_raw_issues(fp):
            if "fields" in raw:
                t = ticket_from_issue(raw, story_points_field)
            else:  # already a flattened record
                t = Ticket(**{k: v for k, v in raw.items() if k in Ticket.__dataclass_fields__})
            seen[t.key] = t
    return list(seen.values())
