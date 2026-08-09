# Contributing

Thanks for taking an interest in the platform. This document describes how to
get set up and what the review bar is.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                # then set JWT_SECRET_KEY
news-platform init-db
python scripts/seed_demo.py                         # offline demo data
```

## Before opening a pull request

```bash
python scripts/check.py          # lint, format, types, security, tests
python scripts/check.py --fast   # quicker loop while iterating
```

CI runs the same checks plus an integration pass against PostgreSQL, a Docker
build and a container smoke test.

## Expectations for a change

- **Layer boundaries hold.** Source quirks stay in connectors; SQL stays in
  repositories; business rules stay in services; routes stay thin. A change that
  puts a query in a route or a `requests` call in a service will be sent back.
- **New input is validated.** Anything crossing a boundary gets a Pydantic model
  with explicit bounds. `extra="forbid"` unless there is a stated reason.
- **Security-relevant code ships with a test that would fail without it.** The
  existing suite has models for SSRF, XXE, injection, XSS, token forgery and
  privilege escalation - follow them.
- **Tests describe behaviour, not implementation.** Prefer asserting the
  observable outcome over asserting that a mock was called.
- **Comments explain why.** The code already says what it does.

## Adding a news source

Most sources need no code at all - add an entry to `configs/sources.yaml` and
run `news-platform sources sync`. Credentials are referenced by environment
variable name (`api_key_env`), never inlined.

A genuinely new *protocol* needs a connector: subclass `NewsSource`, implement
`_collect()`, and register it with `register_source_type()`. Everything else -
fetching, retries, validation, rate limiting - is inherited.

## Adding an NLP component

`SentimentAnalyzer`, `EntityRecognizer` and `TopicClassifier` are Protocols with
swappable defaults. Implement the Protocol and install it with the matching
`set_default_*` function; no pipeline change is required.

## Commit messages

Imperative mood, one logical change per commit, and a body explaining the
reasoning when the change is not obvious.
