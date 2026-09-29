# syntax=docker/dockerfile:1.7
# =========================================================================== #
# Multi-stage build.
#
#  * builder - compiles wheels once, so the runtime image needs no toolchain
#  * runtime - slim, non-root, read-only-friendly
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Stage 1 - build wheels
# --------------------------------------------------------------------------- #
FROM python:3.14-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY app ./app

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip wheel --wheel-dir /wheels ".[postgres]"

# --------------------------------------------------------------------------- #
# Stage 2 - runtime
# --------------------------------------------------------------------------- #
FROM python:3.14-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    ENVIRONMENT=production \
    LOG_FORMAT=json

# curl is only for the container healthcheck; nothing else is added.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin --uid 10001 appuser

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels news-intelligence-platform[postgres] \
    && rm -rf /wheels

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser migrations ./migrations
COPY --chown=appuser:appuser configs ./configs
COPY --chown=appuser:appuser alembic.ini pyproject.toml README.md ./

RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Two workers by default: the workload is I/O bound, and the process-local
# caches (dedup window, robots.txt) are cheaper with fewer processes.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--no-server-header"]
