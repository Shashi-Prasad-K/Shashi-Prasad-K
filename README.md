# Ticket Estimator

Estimates how many hours a new Jira ticket will take, learned from your team's resolved tickets.
Each estimate comes with:

- a **most likely** number of hours and a deliberately wide **likely range**,
- the **most similar past tickets** and how long they actually took, and
- how far that project's tickets usually **over- or under-run** their original estimate.

It predicts *logged time* (`timespent`) rather than story points. On the Jira site this was built
against, almost no tickets carry story points, but most closed tickets have work logged and many
have an original estimate.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

export JIRA_BASE_URL=https://your-site.atlassian.net
export JIRA_EMAIL=you@company.com
export JIRA_API_TOKEN=...   # https://id.atlassian.com/manage-profile/security/api-tokens
```

## Usage

```bash
# 1. Pull a year of resolved tickets with logged time (narrow it with --jql, e.g. "project = MT AND ...")
ticket-estimator fetch

# 2. See how accurate it is on your data before trusting it
ticket-estimator backtest

# 3. Train on everything
ticket-estimator train

# 4. Estimate an existing ticket, or describe a new one
ticket-estimator estimate --issue MT-1700
ticket-estimator estimate --project MT --type Bug \
  --summary "Checkout shows wrong tax for EU customers" \
  --description "Tax is calculated on the pre-discount subtotal..."
```

Sample output (illustrative):

```
Estimate for: Checkout shows wrong tax for EU customers
  Most likely : 5.2h
  Likely range: 2.9h – 9.1h (~1.1d)  (wide on purpose: see `backtest` for how often it holds)
  History     : MT tickets typically over-run their original estimate (actual/estimate = 1.38)
  Most similar past tickets:
    MT-101       12.0h (~1.5d)  sim=0.16  Test cart checkout quote and pricing rules
    MT-102                6.2h  sim=0.11  Discount cap applied at checkout
```

Add `--json` for machine-readable output. `train` and `backtest` also accept saved Jira search
responses (REST or Atlassian MCP format) or directories of them via `--data`.

`data/` and `models/` are git-ignored because they contain company ticket data.

## How it works

`ticket_estimator/model.py` blends two learners in log-hours space:

1. **Ridge regression** over TF-IDF of the summary, parent summary, labels, components and
   description, plus one-hot project, issue type, priority and assignee.
2. **Nearest neighbours**: a similarity-weighted median of the hours spent on the most
   similar past tickets.

The blend weight and the likely range (10th–90th percentile of errors) are fitted on 5-fold
out-of-fold predictions. The top 1% of logged hours is capped so that long-running time-bucket
tickets don't skew results.

## How accurate is it?

Back-test on 703 resolved tickets from a real Jira site (Jul–Oct 2026): trained on the oldest 80%, scored on the
newest 141.

| Estimator | Median error | Within ±50% | Within 2× |
|---|---|---|---|
| This model | 2.1h | 29% | 38% |
| Median hours for the same project + type | 3.0h | 26% | 45% |
| Model, on the 55 tickets that had a human estimate | 1.1h | 36% | 47% |
| **Team's original estimate** (same 55 tickets) | **0.5h** | **56%** | **87%** |

Actual hours fell inside the model's likely range for 62% of the newest tickets.

What this means:

- **The model doesn't beat your team's own estimates.** Use it as a first-pass number and a
  sanity check, not as a replacement for the person doing the work. Part of the human advantage
  is anchoring: 17% of tickets log exactly their original estimate.
- **Ticket text is a weak signal here.** A learning curve from 224 to 562 training tickets showed
  no improvement, so pulling more history alone is unlikely to help much. Richer descriptions,
  or an LLM reading each ticket alongside its similar past tickets, are the more promising
  next steps.
- Re-run `ticket-estimator backtest` after each `fetch`, since these numbers will drift.

## Tests

```bash
.venv/bin/python -m pytest
```
