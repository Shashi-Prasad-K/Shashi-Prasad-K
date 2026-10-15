# Job Hunter

Collects remote, hybrid and on-site openings from many public sources, scores each one against
your skills, and applies:

- **The posting lists a recruiter email:** it emails the recruiter a tailored message with your
  résumé attached.
- **The posting only has an application form** (Greenhouse, Lever, Ashby, job boards): it writes
  a ready-to-paste cover letter and the apply link to `outbox/manual/`, and you click submit.

It never applies to the same job twice, and it emails each recruiter address only once.

## Sources

| Source | Covers | Notes |
|---|---|---|
| Remotive, RemoteOK, Jobicy, Himalayas | Remote | Public APIs, no key |
| Arbeitnow, The Muse | Remote, hybrid, on-site | Public APIs, no key |
| Adzuna | Remote, hybrid, on-site, **India** (plus 15+ other countries) | Free key from [developer.adzuna.com](https://developer.adzuna.com) |
| Greenhouse, Lever, Ashby | Any company career page you list | Public job-board APIs, no key |

LinkedIn, Naukri, Indeed and Glassdoor are left out on purpose. Their terms forbid automated
scraping and applying, and they ban accounts that do it. Many of the same openings come through
Adzuna or the company's own Greenhouse, Lever or Ashby board.

## Setup

```bash
cd job-hunter
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp profile.example.toml profile.toml      # edit: skills, titles, locations, résumé path
cp ~/Downloads/resume.pdf resume.pdf

# Optional but recommended for Indian on-site and hybrid roles
export ADZUNA_APP_ID=... ADZUNA_APP_KEY=...
# Needed only to send email. For Gmail, create an app password at myaccount.google.com/apppasswords
export SMTP_USER=you@gmail.com SMTP_PASSWORD=xxxxxxxxxxxxxxxx   # SMTP_HOST/SMTP_PORT default to Gmail
```

`profile.toml`, your résumé, `data/` and `outbox/` are git-ignored.

## Usage

```bash
job-hunter fetch                 # pull openings from every enabled source
job-hunter match                 # score them (use --rescore after editing your profile)
job-hunter list --verbose        # best matches: score, mode, matched skills, gaps, link
job-hunter list --mode hybrid --csv > hybrid.csv

job-hunter apply                 # DRY RUN: writes emails to outbox/drafts/*.eml, forms to outbox/manual/
job-hunter apply --send          # sends, showing each email for y/N approval first
job-hunter apply --send --yes    # fully automatic, still capped by daily_limit

job-hunter run --send --yes      # fetch + match + apply in one go
job-hunter status                # tracker: sent / drafted / manual / failed / skipped ...
job-hunter status lever:abc123 applied     # after you submit a form; also interview, rejected, offer
```

Run it automatically every morning (cron, 9:05 am):

```
5 9 * * * cd ~/job-hunter && . ./.env && .venv/bin/job-hunter run --send --yes >> data/run.log 2>&1
```

## How matching works

Each job gets a score from 0 to 100:

- **45% title fit.** 1.0 if one of your `titles` appears in the job title. Otherwise, the share of
  that title's words the job title contains ("IT Program Manager" scores 0.67 against "IT Project
  Manager").
- **55% skill fit.** The total `weight` of your skills (or their `aliases`) found in the title and
  description. A job that mentions 8 weight-points of your skills scores the full 55%.

These filters reject a job outright:

- `exclude_keywords` in the job title.
- A work mode that isn't in `work_modes`.
- A hybrid or on-site job outside your `locations`.
- A remote job restricted to regions that aren't in your `locations` or `remote_regions`.

A job also needs a score of at least `min_score`. `list --verbose` shows the matched skills and
the gaps, meaning skills the posting asks for that your profile doesn't list. Gaps are worth
adding to your profile if you have them, or learning if you don't.

## Guard-rails

- `apply` is a dry run unless you pass `--send`.
- `daily_limit` (default 15) and `seconds_between_emails` (default 60) keep a new Gmail account out
  of spam filters.
- Each recruiter address is emailed once, ever. Job-board, noreply and ATS addresses are never
  emailed.
- A source that fails, for example from a timeout or an API change, is reported, and the other
  sources keep running.

## Tests

```bash
.venv/bin/pytest
```

The tests replay recorded response shapes for every source, so they run offline.
