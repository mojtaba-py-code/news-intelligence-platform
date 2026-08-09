# Multi-Source News Intelligence Platform

A production-grade Python platform that collects news from heterogeneous sources,
normalises and validates it, removes duplicates, enriches it with NLP, detects
trends and events, and serves the result through a versioned REST API and an
analytics dashboard.

It is built as a **modular, fault-tolerant data pipeline**, not a scraper:
sources fail in isolation, ingestion is idempotent, every input is validated at
a defined boundary, and security is enforced in the layer that owns it.

```
Sources → Connectors → Ingestion → Normalisation → Deduplication → NLP
       → Intelligence (relevance / trends / events) → PostgreSQL + Redis
       → FastAPI → Dashboard & API
```

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Technology stack](#technology-stack)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Database setup](#database-setup)
- [Running with Docker](#running-with-docker)
- [API documentation](#api-documentation)
- [CLI usage](#cli-usage)
- [Testing](#testing)
- [Security](#security)
- [Performance](#performance)
- [Architecture decisions](#architecture-decisions)
- [Project structure](#project-structure)
- [Future improvements](#future-improvements)

---

## Features

### Ingestion
- **Plugin-based connectors** - RSS/Atom/RDF, generic REST API, JSON Feed, web
  scraper. New sources are YAML entries, not code.
- **Secure fetching** - SSRF guard on every hop, streamed size limits,
  content-type allowlist, timeouts, connection pooling, per-host politeness
  delays, `robots.txt` compliance.
- **Fault isolation** - retries with jittered exponential backoff, a circuit
  breaker per source, and per-source error boundaries. One dead source never
  stops the others.

### Processing
- **Normalisation** - schema mapping, HTML cleaning, boilerplate removal, date
  parsing across a dozen formats, URL canonicalisation with tracking-parameter
  stripping.
- **Four-level deduplication** - canonical URL → SHA-256 content hash → 64-bit
  SimHash (Hamming distance) → TF-IDF cosine similarity, cheapest first.
- **Data quality** - per-article validation with machine-readable issue codes
  and per-run quality metrics.

### Intelligence
- **Language detection** without a model download (script ranges + function-word
  profiles, 12 languages).
- **Sentiment analysis** - a valence-shifted lexicon model handling negation
  scope, intensifiers, contrast markers and emphasis. Not keyword counting.
- **Keyword extraction** (RAKE-style), **named entity recognition** (gazetteer +
  orthographic rules), **topic classification** against a configurable
  vocabulary, **extractive summarisation**.
- **Relevance scoring** - eight independent, individually explainable signals
  with configurable weights; the same code path powers personalised feeds.
- **Trend detection** - window-over-window growth, volume and source breadth,
  with confidence reported separately from the score.
- **Event clustering** - single-pass incremental clustering in TF-IDF space,
  weighted by cross-source corroboration.

### Platform
- REST API with JWT authentication, three-tier RBAC, pagination, filtering,
  sorting, structured errors and OpenAPI documentation.
- Server-rendered analytics dashboard with **no third-party assets**.
- Durable job queue, background worker and scheduler.
- Configurable alert rules (in-app, webhook) with SSRF-checked destinations.
- Structured logging with automatic secret redaction, Prometheus metrics,
  health/readiness probes, and an append-only audit trail.

---

## Architecture

```
                     ┌──────────────────────────┐
                     │  Sources                 │
                     │  RSS · APIs · JSON · Web │
                     └────────────┬─────────────┘
                                  │
                   ┌──────────────▼──────────────┐
                   │  Connectors  (plugin layer) │
                   │  auth · pagination · parse  │
                   └──────────────┬──────────────┘
                                  │  RawArticle  (validated)
                   ┌──────────────▼──────────────┐
                   │  Secure fetcher             │
                   │  SSRF · retry · breaker     │
                   │  robots.txt · rate limits   │
                   └──────────────┬──────────────┘
                                  │
        ┌─────────────────────────▼─────────────────────────┐
        │  Processing                                       │
        │  normalise → clean → validate → deduplicate       │
        └─────────────────────────┬─────────────────────────┘
                                  │  NormalizedArticle
        ┌─────────────────────────▼─────────────────────────┐
        │  Intelligence                                     │
        │  language · sentiment · keywords · entities       │
        │  topics · summary · relevance                     │
        └─────────────────────────┬─────────────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
     ┌──────────────────┐                    ┌──────────────────┐
     │  PostgreSQL      │                    │  Redis           │
     │  articles·events │                    │  cache · limits  │
     │  trends·entities │                    │  (optional)      │
     └────────┬─────────┘                    └────────┬─────────┘
              └──────────────────┬────────────────────┘
                                 ▼
                   ┌─────────────────────────────┐
                   │  FastAPI                    │
                   │  auth · search · analytics  │
                   └──────────────┬──────────────┘
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
             ┌─────────────┐            ┌──────────────┐
             │  Dashboard  │            │  REST API    │
             └─────────────┘            └──────────────┘

   Workers ──▶ job queue (PostgreSQL) ──▶ ingest · process · trends
                                          events · alerts · cleanup
```

### Layer boundaries

| Layer | Owns | Never does |
|---|---|---|
| `app/ingestion` | fetching, source quirks, parsing | database access, business rules |
| `app/processing` | normalisation, dedup, quality | HTTP, persistence |
| `app/intelligence` | NLP and scoring | I/O of any kind |
| `app/database` | queries, transactions | HTTP, NLP |
| `app/services` | business logic | SQL construction, HTTP details |
| `app/api` | HTTP concerns | business logic |

Source-specific behaviour stops at the connector: everything downstream only
ever sees the canonical `RawArticle` → `NormalizedArticle` → `Article` chain.

---

## Technology stack

| Concern | Choice | Why |
|---|---|---|
| API | FastAPI + Pydantic v2 | typed validation at the boundary, OpenAPI for free |
| Database | PostgreSQL (SQLAlchemy 2.0 async) | relational integrity, real constraints, indexes |
| Development DB | SQLite + aiosqlite | zero-setup local runs and CI |
| Migrations | Alembic | reviewable schema history |
| Cache / limits | Redis, with an in-memory fallback | works with or without infrastructure |
| HTTP | httpx | async, connection pooling, testable via respx |
| Parsing | defusedxml + BeautifulSoup (`html.parser`) | XXE-safe by construction |
| Auth | PyJWT (HS256, pinned) + Argon2id | standard, memory-hard hashing |
| NLP | in-house (NumPy) | no model downloads; swappable via Protocols |
| CLI | Typer + Rich | discoverable operator tooling |
| Tests | pytest, pytest-asyncio, respx | unit → integration → API → e2e |

**Dependencies are deliberately few.** TF-IDF, SimHash, sentiment, NER and the
metrics registry are implemented in-house rather than pulling in scikit-learn,
spaCy and `prometheus_client`: it keeps the install small, the behaviour
inspectable, and the supply-chain surface narrow. Every NLP component sits
behind a `Protocol`, so a transformer model can replace it without touching the
pipeline.

---

## Quick start

Requires **Python 3.12+**.

```bash
git clone <repository-url>
cd news-intelligence-platform

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

cp .env.example .env
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(64))" >> .env

news-platform init-db                       # schema + topics + configs/sources.yaml
news-platform create-admin --email you@example.com --username admin

news-platform ingest --all                  # fetch from every enabled source
news-platform trends
news-platform serve --reload
```

Then open:

- <http://localhost:8000/> - dashboard
- <http://localhost:8000/docs> - interactive API documentation
- <http://localhost:8000/health> - liveness probe

---

## Configuration

Everything is environment-driven; see [`.env.example`](.env.example) for the
annotated list. The most important settings:

| Variable | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` enables strict validation |
| `DATABASE_URL` | SQLite file | must use an async driver |
| `REDIS_URL` | *(empty)* | unset → in-process cache and rate limits |
| `JWT_SECRET_KEY` | placeholder | **must** be replaced in production |
| `SSRF_PROTECTION_ENABLED` | `true` | keep it on |
| `RATE_LIMIT_REQUESTS` / `_WINDOW_SECONDS` | `120` / `60` | per client identity |
| `RESPECT_ROBOTS_TXT` | `true` | scraping politeness |
| `DEDUP_TITLE_THRESHOLD` / `_CONTENT_THRESHOLD` | `0.85` / `0.80` | dedup sensitivity |
| `RETENTION_DAYS` | `90` | cleanup job horizon |

**Production is validated at startup.** The process refuses to boot with a
placeholder secret, `DEBUG=true`, `*` in CORS or trusted hosts, SQLite, or SSRF
protection / rate limiting disabled - failing loudly beats running unsafely.

Source credentials are referenced by *variable name* (`api_key_env`), never
stored in the database or in `configs/sources.yaml`.

---

## Database setup

```bash
# Development - create the schema directly
news-platform init-db

# Production - use migrations
alembic upgrade head
alembic revision --autogenerate -m "describe the change"
```

Schema highlights:

- `articles.canonical_url` and `articles.content_hash` are **unique** - the
  database is the last line of defence for idempotent ingestion under
  concurrency.
- Composite indexes back the hot query shapes:
  `(source_id, published_at)`, `(category, published_at)`,
  `(language, published_at)`, `(status, published_at)`, plus single-column
  indexes on `relevance_score` and `sentiment_score`.
- Check constraints keep `relevance_score ∈ [0,1]` and
  `sentiment_score ∈ [-1,1]` true at the storage layer, not just in Python.
- Cascades are declared with `ondelete` **and** `passive_deletes=True`, so
  deleting a source does not pull a million rows into memory.

---

## Running with Docker

```bash
cp .env.example .env                     # then set JWT_SECRET_KEY
docker compose up --build
```

Services: `api` (8000), `worker`, `scheduler`, `postgres`, `redis`, plus a
one-shot `migrate` job that the worker and scheduler wait for.

The image is multi-stage, runs as a non-root user (uid 10001), contains no build
toolchain, and ships a `HEALTHCHECK`. Postgres is **not** published to the host
by default.

---

## API documentation

Base path `/api/v1`. Interactive docs at `/docs` (disabled in production).

| Group | Endpoints |
|---|---|
| Articles | `GET /articles`, `/articles/search`, `/articles/{id}`, `/articles/{id}/similar`, `/articles/stats`, `/articles/feed` |
| Sources | `GET /sources`, `/sources/{slug}`, `/sources/{slug}/health`; `POST/PATCH/DELETE` (admin); `POST /sources/{slug}/ingest` (analyst) |
| Intelligence | `GET /topics`, `/entities`, `/entities/{id}/graph`, `/events`, `/events/breaking`, `/trends`, `/trends/{type}/{key}/history` |
| Analytics | `GET /analytics/overview`, `/sentiment`, `/topics`, `/timeseries`, `/sources` |
| Auth | `POST /auth/register`, `/auth/login`, `/auth/token`, `/auth/refresh`, `/auth/change-password`, `/auth/logout`; `GET /auth/me` |
| Users | `GET/PATCH /users/me`, `GET/PUT /users/me/preferences`, `GET/POST/DELETE /users/me/saved` |
| Alerts | `GET/POST /alerts`, `PATCH/DELETE /alerts/{id}`, `GET /alerts/triggers`, `POST /alerts/{id}/test` |
| Admin | `GET /admin/users`, `PATCH /admin/users/{id}`, `POST /admin/ingest`, `GET /admin/jobs`, `GET /admin/audit` |
| Operations | `GET /health`, `/ready`, `/metrics`, `/circuits` |

Authentication:

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"..."}' | jq -r .access_token)

curl -H "Authorization: Bearer $TOKEN" \
  'http://localhost:8000/api/v1/articles/search?q=artificial+intelligence&sort_by=relevance_score'
```

Every error uses one envelope:

```json
{
  "error": {
    "code": "not_found",
    "message": "Article 42 does not exist.",
    "details": {},
    "request_id": "9f2c1ab34de5"
  }
}
```

---

## CLI usage

```bash
news-platform init-db [--with-topics/--no-with-topics]
news-platform create-admin --email you@example.com --username admin

news-platform sources list          # catalogue with health and reliability
news-platform sources sync          # (re)import configs/sources.yaml
news-platform sources kinds         # registered connector types

news-platform ingest --all
news-platform ingest --source bbc-world --source techcrunch
news-platform process --job deduplicate

news-platform trends --hours 24 --limit 15
news-platform events --hours 48
news-platform stats
news-platform health

news-platform worker                # background job runner
news-platform scheduler             # recurring job scheduler
news-platform serve --host 0.0.0.0 --port 8000
```

---

## Testing

```bash
pytest                                   # everything
pytest tests/unit -q                     # fast, no I/O
pytest -m integration                    # database + mocked HTTP
pytest tests/e2e                         # full pipeline through the API
pytest --cov=app --cov-report=term-missing
```

The suite is layered on purpose:

- **unit** - parsers, normalisers, dedup, NLP, scoring, security primitives.
- **integration** - repositories against a real schema; connectors and the
  fetcher against mocked HTTP (`respx`); the ingestion pipeline end to end.
- **api** - authentication, authorisation, validation, security headers, error
  envelopes, dashboard escaping.
- **e2e** - three publishers, overlapping coverage, ingestion → deduplication →
  NLP → trends → events → API → dashboard, plus a run where two of the three
  sources are broken.

Security-specific coverage includes SSRF (private ranges, cloud metadata,
IPv4-mapped IPv6, redirect hops), XXE and billion-laughs, SQL-injection and
LIKE-wildcard handling, XSS escaping in the dashboard, JWT forgery
(`alg: none`, wrong secret, wrong audience, refresh-as-access), privilege
escalation via mass assignment, and secret redaction in logs.

---

## Security

Security controls, and where each one lives:

| Control | Implementation |
|---|---|
| Secrets management | env-only; `SecretStr`; sources reference variable *names*; production boot validation |
| Password storage | Argon2id (64 MiB, t=3, p=4), transparent rehash on login, length cap |
| Authentication | JWT HS256 with a **pinned** algorithm, `iss`/`aud`/`exp`/`nbf`/`jti` required, typed access/refresh separation |
| Session revocation | `token_version` per user; password change or deactivation invalidates every issued token |
| Brute force | uniform failure responses, dummy-hash verification for unknown users, account lockout, stricter auth rate limit |
| Authorisation | role hierarchy (`USER < ANALYST < ADMIN`), enforced from the **database** record, denials audited |
| Input validation | Pydantic at every boundary, `extra="forbid"` (blocks mass assignment), length/range caps, enum allowlists |
| SQL injection | SQLAlchemy expressions only; `ORDER BY` from an enum allowlist; LIKE wildcards escaped |
| XSS | Jinja autoescaping; API returns JSON; HTML sanitiser with a tag allowlist and `javascript:` URL rejection |
| SSRF | scheme/port allowlists, DNS resolution + IP classification, blocked hostnames, per-hop redirect re-validation, credential stripping across origins |
| XXE / DoS | `defusedxml` for feeds, `html.parser` for HTML, size caps on bodies, feeds and fields |
| Rate limiting | per identity (hashed token or IP), separate stricter budget for auth routes |
| Transport headers | CSP without `unsafe-inline`, HSTS (opt-in), `nosniff`, `DENY`, referrer and permissions policies |
| Request limits | body size cap enforced before any handler reads the stream |
| Logging | redaction filter over messages, args and extras; JWTs, keys, passwords and URL credentials never reach a log |
| Audit trail | append-only, anonymised client IPs (`/24`, `/48`), sensitive keys stripped |
| Supply chain | small dependency set; `bandit` and `pip-audit` in CI; a secret-shaped-string scan on every push |

**Scraping** honours `robots.txt`, applies per-host delays and bounded
concurrency, identifies itself in the user agent, and stays on the source's own
domain. Only enable a scraper source for a site whose terms permit it.

Report a suspected vulnerability privately rather than through a public issue.

---

## Performance

- Fully async I/O with connection pooling and bounded concurrency
  (`HTTP_MAX_CONCURRENCY`, `WORKER_CONCURRENCY`).
- **No N+1 queries**: `selectinload` for topics/entities/sources; per-source
  aggregates fetched in bulk; entity resolution batches many names into one
  `SELECT`.
- Deduplication loads its candidate window **once per batch** (one indexed
  query) and answers most comparisons with a dictionary hit or an integer
  Hamming distance; TF-IDF only runs on survivors.
- Aggregate statistics use one grouped query rather than eight counters.
- Pagination is capped server-side (`page_size ≤ 100`), and response bodies,
  request bodies and upstream responses all have hard limits.
- Redis caches derived analytics with a short TTL and is invalidated after each
  ingestion run.
- Retention cleanup deletes in bounded batches instead of one large statement.

---

## Architecture decisions

**Why a database-backed job queue instead of Celery?**
The queue needs exactly-once semantics with the ingest transaction, must survive
restarts, and must not require another service for a single-node deployment.
`SELECT … FOR UPDATE SKIP LOCKED` gives safe multi-worker claiming on
PostgreSQL. Celery remains the right answer at much higher throughput; the
`JobRepository` interface is the seam for that change.

**Why in-house NLP?**
spaCy and scikit-learn would add hundreds of megabytes and a model-download step
for functionality that is a few hundred readable lines here. Every component
(`SentimentAnalyzer`, `EntityRecognizer`, `TopicClassifier`) is a `Protocol`
with a swappable default, so upgrading to a transformer is a one-line change
with no pipeline edits.

**Why four deduplication levels instead of one?**
Cost. A URL or hash match is O(1) and certain; SimHash is an integer comparison
that catches rewrites; TF-IDF cosine is the only expensive step and runs on a
small candidate set. Ordering them cheapest-first keeps ingestion linear.

**Why is an identical headline not enough to call a duplicate?**
Recurring columns and numbered updates ("Market wrap 3") share headlines while
reporting different facts. The body must agree too - a lesson encoded in
`TITLE_MATCH_MIN_CONTENT` and covered by tests.

**Why manual redirect following?**
`follow_redirects=True` would let a redirect escape the SSRF check. Following by
hand means every hop is re-validated and credentials are dropped when the origin
changes.

**Why store trends instead of computing them on request?**
Dashboards must be cheap to render, trend history has to survive retention
cleanup of the underlying articles, and a stored snapshot is reproducible.

**Why SQLite *and* PostgreSQL?**
Contributors and CI get a zero-setup database; production gets real
concurrency and indexing. The repository layer is dialect-agnostic, and CI runs
the integration suite against both.

---

## Project structure

```
news-intelligence-platform/
├── app/
│   ├── api/                  # routes, dependencies, middleware
│   ├── core/                 # config, logging, security, cache, metrics, resilience
│   ├── dashboard/            # server-rendered UI (templates + local assets)
│   ├── database/             # models, repositories, session management
│   ├── ingestion/            # connectors, fetchers, parsers, pipeline
│   ├── intelligence/         # language, sentiment, entities, topics, trends, events
│   ├── processing/           # cleaning, normalisation, deduplication, validation
│   ├── schemas/              # Pydantic contracts
│   ├── services/             # business logic
│   ├── workers/              # job handlers, worker, scheduler
│   ├── cli.py
│   └── main.py
├── tests/{unit,integration,api,e2e}/
├── migrations/               # Alembic
├── configs/sources.yaml      # source catalogue
├── scripts/                  # developer helpers
├── .github/workflows/ci.yml
├── Dockerfile · docker-compose.yml
└── pyproject.toml · .env.example
```

---

## Future improvements

- PostgreSQL full-text search (`tsvector` + GIN) behind the existing search API.
- Embedding-based semantic search and recommendations (the `TfidfIndex` seam is
  already in place).
- Relation extraction to turn the co-occurrence graph into a true knowledge
  graph.
- SMTP delivery for alerts (the channel and trigger records already exist).
- OpenTelemetry traces alongside the current metrics.
- A pluggable object store for raw source payloads to support reprocessing.

---

## Licence

MIT.
