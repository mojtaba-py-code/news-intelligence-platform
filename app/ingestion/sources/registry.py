"""Connector registry and YAML source-definition loading.

The registry is the factory: given a :class:`SourceContext` it returns the right
connector. Third-party connectors register themselves with
:func:`register_source_type`, so the platform is extensible without editing the
core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.core.errors import ConfigurationError
from app.core.logging import get_logger
from app.core.url_safety import is_safe_url
from app.database.models.source import SourceKind
from app.ingestion.fetchers.http import SecureHTTPFetcher
from app.ingestion.sources.api_source import APINewsSource
from app.ingestion.sources.base import NewsSource, SourceContext
from app.ingestion.sources.json_source import JSONFeedSource
from app.ingestion.sources.rss_source import RSSNewsSource
from app.ingestion.sources.scraper_source import WebScraperSource

logger = get_logger(__name__)

MAX_SOURCES_PER_FILE = 500


@dataclass
class SourceRegistry:
    """Maps a :class:`SourceKind` to the connector class implementing it."""

    _types: dict[str, type[NewsSource]] = field(default_factory=dict)

    def register(self, kind: SourceKind | str, connector: type[NewsSource]) -> None:
        key = str(kind).lower()
        if key in self._types and self._types[key] is not connector:
            logger.warning("source_type_overridden", extra={"kind": key})
        self._types[key] = connector

    def get(self, kind: SourceKind | str) -> type[NewsSource]:
        key = str(kind).lower()
        connector = self._types.get(key)
        if connector is None:
            raise ConfigurationError(
                f"No connector registered for source kind '{key}'. "
                f"Known kinds: {', '.join(sorted(self._types))}"
            )
        return connector

    def build(
        self,
        context: SourceContext,
        *,
        fetcher: SecureHTTPFetcher | None = None,
        config: Settings | None = None,
    ) -> NewsSource:
        return self.get(context.kind)(context, fetcher=fetcher, config=config)

    @property
    def kinds(self) -> list[str]:
        return sorted(self._types)


registry = SourceRegistry()
registry.register(SourceKind.RSS, RSSNewsSource)
registry.register(SourceKind.API, APINewsSource)
registry.register(SourceKind.SCRAPER, WebScraperSource)
registry.register(SourceKind.JSON_FEED, JSONFeedSource)


def register_source_type(kind: SourceKind | str, connector: type[NewsSource]) -> None:
    """Register a custom connector implementation."""
    registry.register(kind, connector)


def build_source(
    context: SourceContext,
    *,
    fetcher: SecureHTTPFetcher | None = None,
    config: Settings | None = None,
) -> NewsSource:
    """Instantiate the connector for ``context``."""
    return registry.build(context, fetcher=fetcher, config=config)


# --------------------------------------------------------------------------- #
# YAML source definitions
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class SourceDefinition:
    """A source as declared in ``configs/sources.yaml``."""

    slug: str
    name: str
    kind: SourceKind
    url: str
    enabled: bool = True
    description: str | None = None
    homepage: str | None = None
    language: str | None = None
    country: str | None = None
    category: str | None = None
    api_key_env: str | None = None
    weight: float = 1.0
    request_delay_seconds: float = 1.0
    max_articles_per_run: int = 100
    respect_robots: bool = True
    config: dict[str, Any] = field(default_factory=dict)

    def to_context(self) -> SourceContext:
        return SourceContext(
            slug=self.slug,
            name=self.name,
            kind=self.kind,
            url=self.url,
            config=self.config,
            api_key_env=self.api_key_env,
            language=self.language,
            country=self.country,
            category=self.category,
            max_articles=self.max_articles_per_run,
            request_delay=self.request_delay_seconds,
            respect_robots=self.respect_robots,
        )

    def to_row(self) -> dict[str, Any]:
        """Values used to insert/update the ``sources`` table."""
        return {
            "slug": self.slug,
            "name": self.name,
            "kind": str(self.kind),
            "url": self.url,
            "homepage": self.homepage,
            "description": self.description,
            "language": self.language,
            "country": self.country,
            "category": self.category,
            "api_key_env": self.api_key_env,
            "config": self.config,
            "weight": self.weight,
            "request_delay_seconds": self.request_delay_seconds,
            "max_articles_per_run": self.max_articles_per_run,
            "respect_robots": self.respect_robots,
            "enabled": self.enabled,
        }


def load_source_definitions(path: str | Path | None = None) -> list[SourceDefinition]:
    """Read and validate ``configs/sources.yaml``.

    ``yaml.safe_load`` is mandatory here: ``yaml.load`` would let a config file
    instantiate arbitrary Python objects.
    """
    config = get_settings()
    target = Path(path) if path else Path(config.sources_config_path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    if not target.exists():
        logger.info("sources_config_missing", extra={"path": str(target)})
        return []

    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Could not parse {target.name}: {exc}") from exc

    entries = raw.get("sources", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ConfigurationError(f"{target.name} must contain a list under 'sources'")

    definitions: list[SourceDefinition] = []
    seen: set[str] = set()

    for index, entry in enumerate(entries[:MAX_SOURCES_PER_FILE]):
        if not isinstance(entry, dict):
            logger.warning(
                "source_entry_skipped", extra={"index": index, "reason": "not a mapping"}
            )
            continue
        try:
            definition = _build_definition(entry)
        except (ConfigurationError, ValueError) as exc:
            logger.warning(
                "source_entry_invalid",
                extra={
                    "index": index,
                    "slug": str(entry.get("slug", "?"))[:40],
                    "reason": str(exc)[:200],
                },
            )
            continue
        if definition.slug in seen:
            logger.warning("duplicate_source_slug", extra={"slug": definition.slug})
            continue
        seen.add(definition.slug)
        definitions.append(definition)

    logger.info("sources_loaded", extra={"count": len(definitions), "path": str(target)})
    return definitions


def _build_definition(entry: dict[str, Any]) -> SourceDefinition:
    slug = str(entry.get("slug", "")).strip().lower()
    if not slug or not slug.replace("-", "").replace("_", "").isalnum():
        raise ValueError("missing or invalid slug")

    url = str(entry.get("url", "")).strip()
    if not url:
        raise ValueError("missing url")
    # Config files are trusted less than code: an SSRF-unsafe endpoint here
    # would otherwise be fetched on every scheduled run.
    if not is_safe_url(url):
        raise ValueError("url rejected by the URL safety policy")

    kind_value = str(entry.get("kind", "rss")).strip().lower()
    try:
        kind = SourceKind(kind_value)
    except ValueError as exc:
        raise ValueError(f"unknown kind '{kind_value}'") from exc

    config = entry.get("config") or {}
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    for forbidden in ("api_key", "apikey", "token", "password", "secret"):
        if forbidden in {str(key).lower() for key in config}:
            raise ValueError(f"'{forbidden}' must not be inlined; use api_key_env")

    return SourceDefinition(
        slug=slug,
        name=str(entry.get("name") or slug.replace("-", " ").title())[:160],
        kind=kind,
        url=url,
        enabled=bool(entry.get("enabled", True)),
        description=_optional_str(entry.get("description"), 2000),
        homepage=_optional_str(entry.get("homepage"), 2048),
        language=_optional_str(entry.get("language"), 8),
        country=_optional_str(entry.get("country"), 8),
        category=_optional_str(entry.get("category"), 48),
        api_key_env=_optional_str(entry.get("api_key_env"), 64),
        weight=max(0.0, min(float(entry.get("weight", 1.0) or 1.0), 2.0)),
        request_delay_seconds=max(
            0.0, min(float(entry.get("request_delay_seconds", 1.0) or 1.0), 60.0)
        ),
        max_articles_per_run=max(1, min(int(entry.get("max_articles_per_run", 100) or 100), 1000)),
        respect_robots=bool(entry.get("respect_robots", True)),
        config=config,
    )


def _optional_str(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] or None


__all__ = [
    "SourceDefinition",
    "SourceRegistry",
    "build_source",
    "load_source_definitions",
    "register_source_type",
    "registry",
]
