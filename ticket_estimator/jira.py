"""Minimal Jira Cloud REST client (API token auth)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import requests

from .records import DEFAULT_STORY_POINTS_FIELD, JIRA_FIELDS


class JiraError(RuntimeError):
    pass


class JiraClient:
    """Reads issues from Jira Cloud.

    Credentials come from JIRA_BASE_URL, JIRA_EMAIL and JIRA_API_TOKEN
    (create a token at https://id.atlassian.com/manage-profile/security/api-tokens).
    """

    def __init__(self, base_url: str | None = None, email: str | None = None, token: str | None = None):
        self.base_url = (base_url or os.environ.get("JIRA_BASE_URL", "")).rstrip("/")
        email = email or os.environ.get("JIRA_EMAIL")
        token = token or os.environ.get("JIRA_API_TOKEN")
        if not (self.base_url and email and token):
            raise JiraError("Set JIRA_BASE_URL, JIRA_EMAIL and JIRA_API_TOKEN to talk to Jira.")
        self.session = requests.Session()
        self.session.auth = (email, token)
        self.session.headers["Accept"] = "application/json"

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        resp = self.session.get(f"{self.base_url}{path}", params=params, timeout=60)
        if resp.status_code >= 400:
            raise JiraError(f"Jira {resp.status_code} on {path}: {resp.text[:300]}")
        return resp.json()

    def search(self, jql: str, fields: list[str], page_size: int = 100) -> Iterator[dict[str, Any]]:
        token: str | None = None
        while True:
            params: dict[str, Any] = {"jql": jql, "fields": ",".join(fields), "maxResults": page_size}
            if token:
                params["nextPageToken"] = token
            page = self._get("/rest/api/3/search/jql", params)
            yield from page.get("issues", [])
            token = page.get("nextPageToken")
            if not token or page.get("isLast", True):
                return

    def get_issue(self, key: str, fields: list[str]) -> dict[str, Any]:
        return self._get(f"/rest/api/3/issue/{key}", {"fields": ",".join(fields)})


def history_fields(story_points_field: str = DEFAULT_STORY_POINTS_FIELD) -> list[str]:
    return JIRA_FIELDS + [story_points_field]


def fetch_history(client: JiraClient, jql: str, out: Path, story_points_field: str = DEFAULT_STORY_POINTS_FIELD) -> int:
    """Stream matching issues to a JSONL file; returns the number written."""
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for issue in client.search(jql, history_fields(story_points_field)):
            fh.write(json.dumps(issue) + "\n")
            n += 1
    return n
