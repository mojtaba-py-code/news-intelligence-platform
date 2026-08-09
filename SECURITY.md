# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** - open a security advisory
on the repository or contact the maintainer directly. Do not open a public issue
for an unpatched flaw.

When reporting, include the affected version or commit, reproduction steps, the
impact you believe it has, and any suggested mitigation.

## Supported versions

The `main` branch is supported. Fixes are not backported to tagged releases
before 1.0.

## Threat model

The platform ingests **untrusted input by design**: feeds, API responses and
scraped pages are attacker-influenceable. The controls below assume that.

### What is defended

| Threat | Control |
|---|---|
| SSRF via source or webhook URLs | scheme/port allowlists, DNS resolution + IP classification (private, loopback, link-local, reserved, IPv4-mapped IPv6, cloud metadata), per-hop redirect re-validation, credential stripping across origins |
| XXE / XML entity expansion | `defusedxml` for every feed; external entities and expansion are refused |
| Memory exhaustion | streamed responses with a hard byte cap, bounded feed/HTML/field sizes, request body limit enforced before handlers run |
| SQL injection | SQLAlchemy expressions only; `ORDER BY` restricted to an enum allowlist; LIKE wildcards escaped |
| XSS | Jinja autoescaping in the dashboard; JSON-only API; HTML sanitiser with a tag allowlist that strips event handlers and `javascript:` URLs |
| Credential theft | Argon2id password hashing, secrets only in environment variables, log redaction, no credentials in the database or config files |
| Token forgery / replay | JWT with a pinned algorithm, required `iss`/`aud`/`exp`/`nbf`/`jti`, typed access/refresh separation, per-user `token_version` revocation |
| Brute force | uniform failure responses, constant-work verification for unknown accounts, account lockout, a stricter rate limit on auth routes |
| Privilege escalation | `extra="forbid"` on every input model, role read from the database record, self-demotion and last-admin removal blocked |
| Abuse / DoS of the API | per-identity rate limiting, capped pagination, bounded query parameters |
| Data exposure in logs | redaction filter applied to messages, args and structured extras; audit log stores anonymised IPs only |

### What is out of scope

- Attacks requiring a compromised host or database.
- Correctness of third-party news content. The platform validates *shape* and
  quality; it does not adjudicate truth.
- Denial of service from an operator misconfiguring limits deliberately.

## Deployment requirements

`ENVIRONMENT=production` refuses to start unless:

- `JWT_SECRET_KEY` is a random value of at least 32 characters,
- `DEBUG` is false,
- `CORS_ORIGINS` and `TRUSTED_HOSTS` list explicit hosts (no `*`),
- SSRF protection and rate limiting are enabled,
- the database is PostgreSQL, not SQLite.

Additionally, run behind TLS with `HSTS_ENABLED=true`, keep `/docs` disabled
(the default in production), and rotate `JWT_SECRET_KEY` on suspicion of
compromise - every issued token becomes invalid immediately.

## Scraping conduct

The scraper honours `robots.txt`, applies per-host delays and bounded
concurrency, identifies itself in the `User-Agent`, and stays on the source's
own domain. Enable a scraper source only for sites whose terms permit it.

## Dependency hygiene

CI runs `bandit` for static analysis, `pip-audit` for known CVEs, and a
credential-shaped-string scan on every push. The dependency set is deliberately
small to keep the supply-chain surface narrow.
