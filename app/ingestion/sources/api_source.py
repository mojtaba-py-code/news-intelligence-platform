"""Generic REST-API connector, driven entirely by configuration.

Supports the authentication styles news APIs actually use (header key, bearer
token, query parameter), three pagination strategies, and a declarative field
mapping. Adding NewsAPI, GDELT, Guardian or an internal service is a YAML entry
rather than a new module.
"""

from __future__ import annotations

from typing import Any, Final

from app.core.errors import ConfigurationError, ParseError
from app.core.logging import get_logger
from app.database.models.source import SourceKind
from app.ingestion.parsers.json_feed import JSONFieldMapping, get_path, parse_json_items
from app.ingestion.sources.base import NewsSource

logger = get_logger(__name__)

MAX_PAGES: Final[int] = 10
JSON_ACCEPT: Final[str] = "application/json, application/feed+json;q=0.9, */*;q=0.5"

#: Placeholder used in configured header templates, replaced at request time.
KEY_PLACEHOLDER: Final[str] = "{api_key}"


class APINewsSource(NewsSource):
    """REST connector.

    Config keys
    -----------
    ``auth``
        ``"header"`` (default), ``"bearer"``, ``"query"`` or ``"none"``.
    ``auth_header``
        Header name for ``auth: header`` (default ``X-Api-Key``).
    ``auth_param``
        Query-parameter name for ``auth: query`` (default ``apiKey``).
    ``params``
        Static query parameters merged into every request.
    ``mapping``
        A :class:`JSONFieldMapping` block; see the parser module.
    ``pagination``
        ``"page"``, ``"offset"``, ``"cursor"`` or ``"none"`` (default).
    ``page_param`` / ``page_size_param`` / ``page_size`` / ``cursor_path``
        Pagination details.
    ``max_pages``
        Upper bound on requests per run (hard-capped at 10).
    """

    kind = SourceKind.API

    async def _collect(self) -> list[dict[str, Any]]:
        mapping = JSONFieldMapping.from_config(self.context.option("mapping"))
        headers, params = self._auth()
        static_params = self.context.option("params") or {}
        if not isinstance(static_params, dict):
            raise ConfigurationError(f"Source '{self.slug}': 'params' must be an object")
        params.update({str(key): value for key, value in static_params.items()})

        strategy = str(self.context.option("pagination", "none")).lower()
        max_pages = min(int(self.context.option("max_pages", 3) or 1), MAX_PAGES)
        page_size = int(self.context.option("page_size", 100) or 100)

        items: list[dict[str, Any]] = []
        cursor: str | None = None

        for page in range(max_pages):
            page_params = dict(params)
            self._apply_pagination(page_params, strategy, page, page_size, cursor)

            response = await self.get(
                self.context.url, params=page_params, headers=headers, accept=JSON_ACCEPT
            )
            try:
                payload = response.json()
            except ValueError as exc:
                raise ParseError(self.slug, "Response body is not valid JSON") from exc

            self._check_api_status(payload)
            batch = parse_json_items(
                payload, mapping, source_slug=self.slug, source_name=self.context.name
            )
            items.extend(batch)

            if len(items) >= self.context.max_articles or len(batch) < max(1, page_size // 2):
                break
            if strategy == "cursor":
                cursor = self._next_cursor(payload)
                if not cursor:
                    break
            elif strategy == "none":
                break

        return items[: self.context.max_articles]

    # ------------------------------------------------------------------ inner
    def _auth(self) -> tuple[dict[str, str], dict[str, Any]]:
        """Build the auth header/param pair without ever logging the key."""
        style = str(
            self.context.option("auth", "header" if self.context.api_key_env else "none")
        ).lower()
        headers: dict[str, str] = {}
        params: dict[str, Any] = {}

        if style == "none":
            return headers, params

        key = self.api_key()
        if not key:
            raise ConfigurationError(
                f"Source '{self.slug}' declares auth '{style}' but has no api_key_env"
            )

        if style == "bearer":
            headers["Authorization"] = f"Bearer {key}"
        elif style == "header":
            header_name = str(self.context.option("auth_header", "X-Api-Key"))
            template = str(self.context.option("auth_header_template", KEY_PLACEHOLDER))
            headers[header_name] = template.replace(KEY_PLACEHOLDER, key)
        elif style == "query":
            params[str(self.context.option("auth_param", "apiKey"))] = key
        else:
            raise ConfigurationError(f"Source '{self.slug}': unknown auth style '{style}'")

        return headers, params

    def _apply_pagination(
        self,
        params: dict[str, Any],
        strategy: str,
        page: int,
        page_size: int,
        cursor: str | None,
    ) -> None:
        if strategy == "page":
            params[str(self.context.option("page_param", "page"))] = page + 1
            params[str(self.context.option("page_size_param", "pageSize"))] = page_size
        elif strategy == "offset":
            params[str(self.context.option("page_param", "offset"))] = page * page_size
            params[str(self.context.option("page_size_param", "limit"))] = page_size
        elif strategy == "cursor" and cursor:
            params[str(self.context.option("cursor_param", "cursor"))] = cursor

    def _next_cursor(self, payload: Any) -> str | None:
        path = str(self.context.option("cursor_path", "nextCursor"))
        value = get_path(payload, path)
        return str(value) if value else None

    def _check_api_status(self, payload: Any) -> None:
        """Many APIs return HTTP 200 with an error body; surface that properly."""
        if not isinstance(payload, dict):
            return
        status = str(payload.get("status", "")).lower()
        if status in ("error", "fail", "failed"):
            message = str(payload.get("message") or payload.get("error") or "upstream error")
            raise ParseError(self.slug, f"API reported an error: {message[:200]}")
        if isinstance(payload.get("errors"), list) and payload["errors"]:
            raise ParseError(self.slug, "API returned errors in the response body")


__all__ = ["APINewsSource"]
