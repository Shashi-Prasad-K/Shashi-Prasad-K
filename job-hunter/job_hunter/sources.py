"""Job sources. Every one is a public, documented feed or API — no logged-in scraping.

LinkedIn, Naukri, Indeed and Glassdoor are deliberately absent: their terms forbid automated
scraping and they ban accounts that do it. Company career pages on Greenhouse, Lever and Ashby
publish the same openings through open JSON APIs, and Adzuna aggregates many Indian boards.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Callable, Iterable

import requests

from .jobs import HYBRID, ONSITE, REMOTE, Job, infer_work_mode, normalise_work_mode, strip_html
from .profile import Profile

log = logging.getLogger(__name__)
UA = "job-hunter/0.1 (personal job search; +https://github.com/shashi-prasad-k)"
TIMEOUT = 25


def _get(session: requests.Session, url: str, **params) -> object:
    r = session.get(url, params={k: v for k, v in params.items() if v is not None}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _iso(value) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (int, float)):  # epoch seconds or milliseconds
        return datetime.fromtimestamp(value / 1000 if value > 1e11 else value, timezone.utc).isoformat()
    return str(value)


def _queries(profile: Profile) -> list[str]:
    return profile.titles or [profile.skills[0].name]


# ---------------------------------------------------------------- remote-only boards

def remotive(session, profile: Profile) -> Iterable[Job]:
    for q in _queries(profile):
        for j in _get(session, "https://remotive.com/api/remote-jobs", search=q, limit=100).get("jobs", []):
            yield Job("remotive", str(j["id"]), j["title"], j.get("company_name", ""),
                      j.get("candidate_required_location") or "Worldwide", REMOTE, j["url"],
                      strip_html(j.get("description")), j["url"], j.get("publication_date", ""), j.get("salary", ""))


def remoteok(session, profile: Profile) -> Iterable[Job]:
    data = _get(session, "https://remoteok.com/api")
    for j in data[1:] if isinstance(data, list) else []:  # item 0 is a legal notice
        if not j.get("id"):
            continue
        salary = f"{j['salary_min']}-{j['salary_max']} USD" if j.get("salary_min") else ""
        yield Job("remoteok", str(j["id"]), j.get("position", ""), j.get("company", ""),
                  j.get("location") or "Worldwide", REMOTE, j.get("url", ""), strip_html(j.get("description")),
                  j.get("apply_url") or j.get("url", ""), j.get("date", ""), salary)


def jobicy(session, profile: Profile) -> Iterable[Job]:
    for q in _queries(profile):
        for j in _get(session, "https://jobicy.com/api/v2/remote-jobs", count=50, tag=q).get("jobs", []):
            yield Job("jobicy", str(j["id"]), j.get("jobTitle", ""), j.get("companyName", ""),
                      j.get("jobGeo") or "Worldwide", REMOTE, j.get("url", ""), strip_html(j.get("jobDescription")),
                      j.get("url", ""), j.get("pubDate", ""))


def himalayas(session, profile: Profile) -> Iterable[Job]:
    for q in _queries(profile):
        for j in _get(session, "https://himalayas.app/jobs/api/search", q=q).get("jobs", []):
            regions = ", ".join(j.get("locationRestrictions") or []) or "Worldwide"
            yield Job("himalayas", str(j.get("guid") or j.get("applicationLink")), j.get("title", ""),
                      j.get("companyName", ""), regions, REMOTE, j.get("applicationLink", ""),
                      strip_html(j.get("description")), j.get("applicationLink", ""), _iso(j.get("pubDate")))


# ---------------------------------------------------------------- mixed remote / hybrid / on-site

def arbeitnow(session, profile: Profile) -> Iterable[Job]:
    for page in range(1, int(profile.sources.get("arbeitnow_pages", 3)) + 1):
        data = _get(session, "https://www.arbeitnow.com/api/job-board-api", page=page).get("data", [])
        for j in data:
            desc = strip_html(j.get("description"))
            mode = REMOTE if j.get("remote") else infer_work_mode(j.get("title"), desc)
            yield Job("arbeitnow", j["slug"], j.get("title", ""), j.get("company_name", ""), j.get("location", ""),
                      mode, j.get("url", ""), desc, j.get("url", ""), _iso(j.get("created_at")))
        if not data:
            break


def themuse(session, profile: Profile) -> Iterable[Job]:
    category = profile.sources.get("themuse_category", "Project Management")
    for page in range(int(profile.sources.get("themuse_pages", 3))):
        data = _get(session, "https://www.themuse.com/api/public/jobs", category=category, page=page)
        for j in data.get("results", []):
            loc = ", ".join(l["name"] for l in j.get("locations", []))
            desc = strip_html(j.get("contents"))
            mode = REMOTE if "flexible / remote" in loc.lower() else infer_work_mode(loc, j.get("name"), desc)
            url = j.get("refs", {}).get("landing_page", "")
            yield Job("themuse", str(j["id"]), j.get("name", ""), j.get("company", {}).get("name", ""), loc,
                      mode, url, desc, url, j.get("publication_date", ""))
        if page + 1 >= data.get("page_count", 0):
            break


def adzuna(session, profile: Profile) -> Iterable[Job]:
    """Free key at https://developer.adzuna.com — covers India (country "in") and 15+ other countries."""
    app_id, key = os.environ.get("ADZUNA_APP_ID"), os.environ.get("ADZUNA_APP_KEY")
    if not (app_id and key):
        log.info("adzuna: set ADZUNA_APP_ID and ADZUNA_APP_KEY to enable")
        return
    cfg = profile.sources.get("adzuna", {})
    country = cfg.get("country", "in")
    for q in _queries(profile):
        for where in profile.locations or [None]:
            data = _get(session, f"https://api.adzuna.com/v1/api/jobs/{country}/search/1", app_id=app_id,
                        app_key=key, what=q, where=where, results_per_page=50, max_days_old=cfg.get("max_days_old", 30))
            for j in data.get("results", []):
                desc = strip_html(j.get("description"))
                salary = f"{j['salary_min']:.0f}-{j.get('salary_max', 0):.0f}" if j.get("salary_min") else ""
                yield Job("adzuna", str(j["id"]), strip_html(j.get("title")), j.get("company", {}).get("display_name", ""),
                          j.get("location", {}).get("display_name", ""), infer_work_mode(j.get("title"), desc),
                          j.get("redirect_url", ""), desc, j.get("redirect_url", ""), j.get("created", ""), salary)


# ---------------------------------------------------------------- company career pages (ATS APIs)

def greenhouse(session, profile: Profile) -> Iterable[Job]:
    for board in profile.sources.get("greenhouse", []):
        for j in _get(session, f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs", content="true").get("jobs", []):
            loc = (j.get("location") or {}).get("name", "")
            desc = strip_html(j.get("content"))
            yield Job("greenhouse", f"{board}/{j['id']}", j.get("title", ""), board, loc,
                      infer_work_mode(loc, j.get("title"), desc), j.get("absolute_url", ""), desc,
                      j.get("absolute_url", ""), j.get("updated_at", ""))


def lever(session, profile: Profile) -> Iterable[Job]:
    for company in profile.sources.get("lever", []):
        for j in _get(session, f"https://api.lever.co/v0/postings/{company}", mode="json"):
            loc = (j.get("categories") or {}).get("location", "")
            desc = strip_html(j.get("descriptionPlain") or j.get("description"))
            mode = normalise_work_mode(j.get("workplaceType")) or infer_work_mode(loc, j.get("text"), desc)
            yield Job("lever", j["id"], j.get("text", ""), company, loc, mode, j.get("hostedUrl", ""), desc,
                      j.get("applyUrl", ""), _iso(j.get("createdAt")))


def ashby(session, profile: Profile) -> Iterable[Job]:
    for org in profile.sources.get("ashby", []):
        for j in _get(session, f"https://api.ashbyhq.com/posting-api/job-board/{org}").get("jobs", []):
            loc = j.get("location", "")
            desc = j.get("descriptionPlain") or strip_html(j.get("descriptionHtml"))
            mode = (normalise_work_mode(j.get("workplaceType")) or (REMOTE if j.get("isRemote") else None)
                    or infer_work_mode(loc, j.get("title"), desc))
            yield Job("ashby", j["id"], j.get("title", ""), org, loc, mode, j.get("jobUrl", ""), desc,
                      j.get("applyUrl", ""), j.get("publishedAt", ""))


SOURCES: dict[str, Callable[[requests.Session, Profile], Iterable[Job]]] = {
    "remotive": remotive, "remoteok": remoteok, "jobicy": jobicy, "himalayas": himalayas,
    "arbeitnow": arbeitnow, "themuse": themuse, "adzuna": adzuna,
    "greenhouse": greenhouse, "lever": lever, "ashby": ashby,
}


def fetch_all(profile: Profile, only: list[str] | None = None,
              session: requests.Session | None = None) -> tuple[list[Job], dict[str, str]]:
    """Fetch from every enabled source. One source failing never stops the rest."""
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", UA)
    enabled = only or profile.sources.get("enabled", list(SOURCES))
    jobs: list[Job] = []
    report: dict[str, str] = {}
    for name in enabled:
        fn = SOURCES.get(name)
        if fn is None:
            report[name] = "unknown source"
            continue
        try:
            got = [j for j in fn(session, profile) if j.title and j.url]
            jobs.extend(got)
            report[name] = f"{len(got)} jobs"
        except Exception as e:  # network errors, API shape changes, rate limits
            log.info("%s failed: %s", name, e)
            report[name] = f"failed: {e.__class__.__name__}: {str(e)[:120]}"
    return jobs, report


__all__ = ["SOURCES", "fetch_all", "REMOTE", "HYBRID", "ONSITE"]
