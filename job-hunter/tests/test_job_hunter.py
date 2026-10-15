from email import message_from_bytes
from pathlib import Path

import pytest

from job_hunter import sources
from job_hunter.apply import apply, compose
from job_hunter.jobs import HYBRID, ONSITE, REMOTE, Job, infer_work_mode, recruiter_emails, strip_html
from job_hunter.matcher import score_job, title_score
from job_hunter.profile import load_profile
from job_hunter.store import Store

EXAMPLE = Path(__file__).parent.parent / "profile.example.toml"


@pytest.fixture
def profile():
    p = load_profile(EXAMPLE)
    p.resume_path = ""
    return p


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "jobs.db")


def job(i=1, title="IT Project Manager", company="Acme", location="Bengaluru, India", mode=HYBRID,
        desc="Lead Agile Scrum delivery, Jira, stakeholder management and risk management across the SDLC.",
        emails=None, source="lever"):
    return Job(source, str(i), title, company, location, mode, f"https://x/{i}", desc, f"https://x/{i}/apply",
               emails=emails or [])


# ------------------------------------------------------------ parsing helpers

def test_strip_html_handles_double_escaped_greenhouse_content():
    assert strip_html("&lt;p&gt;Hello &amp;amp; welcome&lt;/p&gt;&lt;ul&gt;&lt;li&gt;Jira&lt;/li&gt;") == "Hello & welcome\nJira"


@pytest.mark.parametrize("text,mode", [
    ("Hybrid - 3 days in Bengaluru office", HYBRID),
    ("Remote-friendly, hybrid 2 days a week", HYBRID),
    ("Fully remote within India", REMOTE),
    ("Work from home", REMOTE),
    ("On-site in Pune", ONSITE),
    ("Pune, Maharashtra", ONSITE),
])
def test_infer_work_mode(text, mode):
    assert infer_work_mode(text) == mode


def test_recruiter_emails_ignores_noreply_and_board_addresses():
    text = "Send CV to Priya.R@acme.in or hr@acme.in. Questions: no-reply@acme.in, jobs@remotive.com, logo@2x.png"
    assert recruiter_emails(text) == ["priya.r@acme.in", "hr@acme.in"]


# ------------------------------------------------------------ sources (recorded API shapes)

class FakeSession:
    def __init__(self, routes):
        self.routes, self.headers, self.calls = routes, {}, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        for prefix, payload in self.routes.items():
            if url.startswith(prefix):
                if isinstance(payload, Exception):
                    raise payload
                return type("R", (), {"raise_for_status": lambda s: None, "json": lambda s: payload})()
        raise AssertionError(f"unexpected url {url}")


def test_sources_normalise_every_api(profile):
    profile.titles = ["Project Manager"]
    profile.sources.update(greenhouse=["acme"], lever=["beta"], ashby=["gamma"], arbeitnow_pages=1, themuse_pages=1)
    routes = {
        "https://remotive.com": {"jobs": [{"id": 1, "url": "https://r/1", "title": "Remote PM", "company_name": "R",
                                           "candidate_required_location": "Worldwide", "description": "<p>Agile</p>",
                                           "publication_date": "2026-10-01"}]},
        "https://remoteok.com": [{"legal": "notice"}, {"id": "9", "position": "PM", "company": "OK", "location": "",
                                                       "description": "Scrum", "url": "https://ok/9"}],
        "https://jobicy.com": {"jobs": [{"id": 3, "url": "https://j/3", "jobTitle": "PM", "companyName": "J",
                                         "jobGeo": "APAC", "jobDescription": "Jira"}]},
        "https://himalayas.app": {"jobs": [{"guid": "h4", "title": "PM", "companyName": "H",
                                            "locationRestrictions": ["India"], "description": "x",
                                            "applicationLink": "https://h/4", "pubDate": 1760000000}]},
        "https://www.arbeitnow.com": {"data": [{"slug": "pm-berlin", "title": "PM", "company_name": "A",
                                                "location": "Berlin", "remote": False, "url": "https://a/1",
                                                "description": "Hybrid working model", "created_at": 1760000000}]},
        "https://www.themuse.com": {"page_count": 1, "results": [
            {"id": 5, "name": "PM", "company": {"name": "Muse"}, "locations": [{"name": "Flexible / Remote"}],
             "contents": "<b>Agile</b>", "refs": {"landing_page": "https://m/5"}}]},
        "https://boards-api.greenhouse.io": {"jobs": [{"id": 7, "title": "PM", "absolute_url": "https://g/7",
                                                       "location": {"name": "Bengaluru (Hybrid)"},
                                                       "content": "&lt;p&gt;mail hr@acme.in&lt;/p&gt;"}]},
        "https://api.lever.co": [{"id": "lv", "text": "PM", "hostedUrl": "https://l/1", "applyUrl": "https://l/1/apply",
                                  "categories": {"location": "Pune"}, "workplaceType": "onsite",
                                  "descriptionPlain": "Scrum", "createdAt": 1760000000000}],
        "https://api.ashbyhq.com": {"jobs": [{"id": "ab", "title": "PM", "location": "India", "isRemote": True,
                                              "descriptionPlain": "Agile", "jobUrl": "https://ab/1",
                                              "applyUrl": "https://ab/1/apply"}]},
    }
    jobs, report = sources.fetch_all(profile, session=FakeSession(routes))
    by_source = {j.source: j for j in jobs}
    assert report["adzuna"] == "0 jobs"  # no API key set -> skipped quietly
    assert set(by_source) == set(sources.SOURCES) - {"adzuna"}
    assert by_source["remotive"].work_mode == REMOTE and by_source["remotive"].description == "Agile"
    assert by_source["remoteok"].source_id == "9"
    assert by_source["arbeitnow"].work_mode == HYBRID
    assert by_source["themuse"].work_mode == REMOTE
    assert by_source["greenhouse"].id == "greenhouse:acme/7" and by_source["greenhouse"].work_mode == HYBRID
    assert by_source["greenhouse"].emails == ["hr@acme.in"]
    assert by_source["lever"].work_mode == ONSITE and by_source["lever"].apply_url.endswith("/apply")
    assert by_source["ashby"].work_mode == REMOTE
    assert by_source["himalayas"].posted_at.startswith("2025-10")


def test_one_failing_source_does_not_stop_the_others(profile):
    session = FakeSession({"https://remotive.com": ConnectionError("down"),
                           "https://remoteok.com": [{}, {"id": "1", "position": "PM", "url": "https://ok/1"}]})
    jobs, report = sources.fetch_all(profile, only=["remotive", "remoteok"], session=session)
    assert report["remotive"].startswith("failed: ConnectionError") and report["remoteok"] == "1 jobs"
    assert len(jobs) == 1


# ------------------------------------------------------------ matching

def test_title_score():
    wanted = ["IT Project Manager", "Scrum Master"]
    assert title_score("Senior IT Project Manager", wanted) == 1.0
    assert title_score("IT Program Manager", wanted) == pytest.approx(2 / 3)
    assert title_score("Backend Engineer", wanted) == 0


def test_good_match_is_accepted_with_reasons(profile):
    m = score_job(job(desc="Agile, Scrum, Jira, Confluence, stakeholder management, risk management, SDLC, "
                           "MS Project. PMP preferred."), profile)
    assert m.accepted and m.score >= 90
    assert {"Agile", "Scrum", "Jira", "Stakeholder management"} <= set(m.matched_skills)
    assert "PMP" in m.missing_skills


def test_filters_reject_with_explanations(profile):
    assert "excluded keyword" in score_job(job(title="Project Manager Intern"), profile).reasons[0]
    off = score_job(job(location="Berlin, Germany", mode=ONSITE), profile)
    assert not off.accepted and "not one of your locations" in off.reasons[0]
    us_only = score_job(job(location="USA only", mode=REMOTE), profile)
    assert not us_only.accepted and "restricted" in us_only.reasons[0]
    assert score_job(job(location="Worldwide", mode=REMOTE), profile).accepted
    weak = score_job(job(title="Backend Engineer", desc="Go and Postgres"), profile)
    assert not weak.accepted and "below min_score" in weak.reasons[-1]


def test_skill_aliases_match_without_false_positives(profile):
    m = score_job(job(desc="Experience in BFSI and Microsoft Project"), profile)
    assert {"Fintech", "MS Project"} <= set(m.matched_skills)
    assert "Excel" not in score_job(job(desc="We excellently deliver"), profile).matched_skills


# ------------------------------------------------------------ store and applying

def test_store_dedupes_cross_posted_jobs(store):
    assert store.upsert_jobs([job(1), job(2, title="IT  project manager!"), job(3, title="Scrum Master")]) == 2
    assert store.upsert_jobs([job(1)]) == 0


def _seed(store, profile, *jobs):
    store.upsert_jobs(list(jobs))
    store.save_matches([score_job(j, profile) for j in jobs])


def test_dry_run_drafts_then_send_respects_limits(store, profile, tmp_path):
    _seed(store, profile,
          job(1, emails=["hr@acme.in"]), job(2, company="Beta", emails=["hr@acme.in"]),  # same recruiter twice
          job(3, company="Gamma", emails=["ta@gamma.io"]), job(4, company="Delta"))      # no email -> manual
    res = apply(store, profile, tmp_path / "outbox")
    assert len(res.drafted) == 3 and res.manual == ["lever:4"] and not res.emailed
    eml = message_from_bytes((tmp_path / "outbox/drafts/lever_1.eml").read_bytes())
    assert eml["To"] == "hr@acme.in" and "IT Project Manager" in eml["Subject"]
    assert "Apply here:** https://x/4/apply" in (tmp_path / "outbox/manual/lever_4.md").read_text()

    sent, naps = [], []
    profile.daily_limit = 5
    res = apply(store, profile, tmp_path / "outbox", send=True, sender=sent.append, sleep=naps.append)
    assert [m["To"] for m in sent] == ["hr@acme.in", "ta@gamma.io"]
    assert res.skipped == ["lever:2"]  # never email the same recruiter for a second opening
    assert naps == [profile.seconds_between_emails]
    assert store.counts()["sent"] == 2

    # Re-running never re-applies
    assert not apply(store, profile, tmp_path / "outbox", send=True, sender=sent.append, sleep=naps.append).emailed


def test_daily_limit_and_failures(store, profile, tmp_path):
    _seed(store, profile, *[job(i, company=f"C{i}", emails=[f"hr@c{i}.in"]) for i in range(4)])
    profile.daily_limit = 2

    def flaky(msg):
        if msg["To"] == "hr@c0.in":
            raise OSError("smtp down")
    res = apply(store, profile, tmp_path, send=True, sender=flaky, sleep=lambda s: None)
    assert len(res.failed) == 1 and len(res.emailed) == 2 and res.stopped_at_limit


def test_review_can_decline(store, profile, tmp_path):
    _seed(store, profile, job(1, emails=["hr@acme.in"]))
    res = apply(store, profile, tmp_path, send=True, sender=lambda m: None, confirm=lambda j, to, b: False)
    assert res.skipped == ["lever:1"] and store.application_status("lever:1") == "skipped"


def test_compose_fills_template(profile):
    j = job(1)
    subject, body = compose(j, score_job(j, profile), profile)
    assert subject == "Application for IT Project Manager – Shashi Prasad K"
    assert "Dear Hiring Team at Acme" in body and "Agile" in body and "{" not in body


def test_resume_is_attached(store, profile, tmp_path):
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"%PDF-1.4 fake")
    profile.resume_path = str(resume)
    _seed(store, profile, job(1, emails=["hr@acme.in"]))
    sent = []
    apply(store, profile, tmp_path, send=True, sender=sent.append)
    att = [p for p in sent[0].iter_attachments()]
    assert att[0].get_filename() == "resume.pdf" and att[0].get_content() == b"%PDF-1.4 fake"


def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    from job_hunter import cli
    prof = tmp_path / "profile.toml"
    prof.write_text(EXAMPLE.read_text().replace('resume = "resume.pdf"', 'resume = ""'))
    monkeypatch.setattr(cli, "fetch_all", lambda p, only=None: (
        [job(1, emails=["hr@acme.in"]), job(2, company="Beta")], {"lever": "2 jobs"}))
    base = ["--profile", str(prof), "--db", str(tmp_path / "db.sqlite")]
    cli.main(base + ["run", "--outbox", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "2 new" in out and "2 pass your filters" in out and "drafted 1" in out and "manual 1" in out
    cli.main(base + ["status", "lever:2", "applied"])
    cli.main(base + ["status"])
    assert "applied: 1" in capsys.readouterr().out
