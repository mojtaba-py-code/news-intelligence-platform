"""Source registry queries and health accounting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import Integer, delete, func, select, update

from app.core.utils import clamp, hours_ago, utcnow
from app.database.models.source import Source, SourceHealth, SourceStatus
from app.database.repositories.base import BaseRepository, affected_rows

#: Consecutive failures after which a source is auto-paused.
FAILURE_PAUSE_THRESHOLD = 10


class SourceRepository(BaseRepository[Source]):
    """Reads and writes for configured sources."""

    model = Source

    async def get_by_slug(self, slug: str) -> Source | None:
        return await self.get_by(slug=slug.lower())

    async def active(self) -> Sequence[Source]:
        statement = (
            select(Source)
            .where(Source.enabled.is_(True))
            .where(Source.status.in_([str(SourceStatus.ACTIVE), str(SourceStatus.FAILING)]))
            .order_by(Source.slug)
        )
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def all_sources(self, *, include_disabled: bool = True) -> Sequence[Source]:
        statement = select(Source).order_by(Source.name)
        if not include_disabled:
            statement = statement.where(Source.enabled.is_(True))
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def upsert_definition(self, row: dict[str, Any]) -> Source:
        """Insert or update a source declared in ``configs/sources.yaml``.

        Operator-managed fields (``status``, ``reliability_score``) are never
        overwritten by a config reload - a paused source stays paused.
        """
        existing = await self.get_by_slug(row["slug"])
        if existing is None:
            source = Source(**row)
            await self.add(source)
            return source

        for key, value in row.items():
            if key in ("slug", "status", "reliability_score"):
                continue
            setattr(existing, key, value)
        await self.flush()
        return existing

    async def record_run(
        self,
        source: Source,
        *,
        success: bool,
        status_code: int | None = None,
        latency_ms: float | None = None,
        fetched: int = 0,
        valid: int = 0,
        duplicates: int = 0,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Persist one health sample and update the source's rolling state."""
        now = utcnow()
        self.session.add(
            SourceHealth(
                source_id=source.id,
                checked_at=now,
                success=success,
                status_code=status_code,
                latency_ms=latency_ms,
                articles_fetched=fetched,
                articles_valid=valid,
                duplicates=duplicates,
                error_type=error_type,
                error_message=(error_message or "")[:512] or None,
            )
        )

        source.last_fetched_at = now
        if success:
            source.last_success_at = now
            source.consecutive_failures = 0
            source.last_error = None
            source.total_articles += valid
            if source.status == str(SourceStatus.FAILING):
                source.status = str(SourceStatus.ACTIVE)
        else:
            source.consecutive_failures += 1
            source.last_error = (error_message or error_type or "unknown error")[:1000]
            if source.consecutive_failures >= FAILURE_PAUSE_THRESHOLD:
                # Stop hammering an endpoint that has failed ten times running.
                source.status = str(SourceStatus.PAUSED)
            elif source.status == str(SourceStatus.ACTIVE):
                source.status = str(SourceStatus.FAILING)

        await self.flush()

    async def recompute_reliability(self, source_id: int, *, days: int = 7) -> float:
        """Blend fetch success rate with content quality into ``[0, 1]``.

        A source that always responds but returns unusable articles is not
        reliable, so both halves matter.
        """
        statement = (
            select(
                func.count(SourceHealth.id),
                func.sum(func.cast(SourceHealth.success, Integer)),
                func.sum(SourceHealth.articles_fetched),
                func.sum(SourceHealth.articles_valid),
                func.sum(SourceHealth.duplicates),
            )
            .where(SourceHealth.source_id == source_id)
            .where(SourceHealth.checked_at >= hours_ago(days * 24))
        )
        total, successes, fetched, valid, duplicates = (await self.session.execute(statement)).one()

        total = int(total or 0)
        if total == 0:
            return 0.5

        success_rate = float(successes or 0) / total
        fetched = int(fetched or 0)
        valid_rate = (float(valid or 0) / fetched) if fetched else 0.5
        duplicate_rate = (float(duplicates or 0) / fetched) if fetched else 0.0

        score = clamp(0.55 * success_rate + 0.35 * valid_rate + 0.10 * (1.0 - duplicate_rate))
        await self.session.execute(
            update(Source).where(Source.id == source_id).values(reliability_score=round(score, 4))
        )
        return round(score, 4)

    async def health_history(self, source_id: int, *, limit: int = 50) -> Sequence[SourceHealth]:
        statement = (
            select(SourceHealth)
            .where(SourceHealth.source_id == source_id)
            .order_by(SourceHealth.checked_at.desc())
            .limit(min(limit, 500))
        )
        result = await self.session.execute(statement)
        return result.scalars().all()

    async def success_rates(self, *, days: int = 7) -> dict[int, float]:
        statement = (
            select(
                SourceHealth.source_id,
                func.count(SourceHealth.id),
                func.sum(func.cast(SourceHealth.success, Integer)),
            )
            .where(SourceHealth.checked_at >= hours_ago(days * 24))
            .group_by(SourceHealth.source_id)
        )
        result = await self.session.execute(statement)
        return {
            int(source_id): round(float(successes or 0) / int(total or 1), 4)
            for source_id, total, successes in result.all()
        }

    async def prune_health(self, *, days: int = 30) -> int:
        cutoff = hours_ago(days * 24)
        result = await self.session.execute(
            delete(SourceHealth).where(SourceHealth.checked_at < cutoff)
        )
        return affected_rows(result)


__all__ = ["FAILURE_PAUSE_THRESHOLD", "SourceRepository"]
