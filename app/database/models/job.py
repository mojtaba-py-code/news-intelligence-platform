"""Background jobs, alert rules and the security audit trail."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database.base import Base, TimestampMixin


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEAD_LETTER = "dead_letter"


class JobType(StrEnum):
    INGEST = "ingest"
    PROCESS = "process"
    DEDUPLICATE = "deduplicate"
    TRENDS = "trends"
    EVENTS = "events"
    ALERTS = "alerts"
    CLEANUP = "cleanup"


class ProcessingJob(Base, TimestampMixin):
    """A unit of background work, with retry and dead-letter accounting."""

    __tablename__ = "processing_jobs"
    __table_args__ = (
        Index("ix_jobs_status_type", "status", "job_type"),
        Index("ix_jobs_scheduled", "scheduled_for"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_type: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(16), default=JobStatus.QUEUED, nullable=False, index=True
    )
    target: Mapped[str | None] = mapped_column(String(160), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    scheduled_for: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)

    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.SUCCEEDED, JobStatus.CANCELLED, JobStatus.DEAD_LETTER)


class AlertChannel(StrEnum):
    IN_APP = "in_app"
    WEBHOOK = "webhook"
    EMAIL = "email"


class Alert(Base, TimestampMixin):
    """A user-defined alert rule.

    Conditions are *structured*, never an expression string: allowing arbitrary
    expressions here would be a code-injection sink in a user-facing feature.
    """

    __tablename__ = "alerts"
    __table_args__ = (Index("ix_alerts_user_active", "user_id", "is_active"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)

    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    topics: Mapped[list[str]] = mapped_column(JSON, default=list)
    entities: Mapped[list[str]] = mapped_column(JSON, default=list)
    sources: Mapped[list[str]] = mapped_column(JSON, default=list)
    languages: Mapped[list[str]] = mapped_column(JSON, default=list)

    min_articles: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    window_minutes: Mapped[int] = mapped_column(Integer, default=60, nullable=False)
    min_relevance: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    sentiment_filter: Mapped[str | None] = mapped_column(String(16))

    channel: Mapped[str] = mapped_column(String(16), default=AlertChannel.IN_APP, nullable=False)
    #: Destination for webhook/email channels. Validated against the SSRF policy.
    destination: Mapped[str | None] = mapped_column(String(2048))
    cooldown_minutes: Mapped[int] = mapped_column(Integer, default=60, nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    last_triggered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    trigger_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    triggers: Mapped[list[AlertTrigger]] = relationship(
        back_populates="alert", cascade="all, delete-orphan", passive_deletes=True
    )


class AlertTrigger(Base):
    """A recorded firing of an alert rule."""

    __tablename__ = "alert_triggers"
    __table_args__ = (Index("ix_alert_triggers_alert_time", "alert_id", "triggered_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alert_id: Mapped[int] = mapped_column(
        ForeignKey("alerts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    article_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    article_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    message: Mapped[str] = mapped_column(String(512), nullable=False)
    delivered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    delivery_error: Mapped[str | None] = mapped_column(String(512))
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    alert: Mapped[Alert] = relationship(back_populates="triggers")


class AuditAction(StrEnum):
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"  # noqa: S105 - an action name, not a secret
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    ROLE_CHANGED = "role_changed"
    PASSWORD_CHANGED = "password_changed"  # noqa: S105
    ACCOUNT_LOCKED = "account_locked"
    SOURCE_CREATED = "source_created"
    SOURCE_UPDATED = "source_updated"
    SOURCE_DELETED = "source_deleted"
    INGESTION_RUN = "ingestion_run"
    RATE_LIMITED = "rate_limited"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


class AuditLog(Base):
    """Append-only security-relevant event log.

    Deliberately free of request bodies and headers: this table must never
    become a secondary store of credentials.
    """

    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_action_time", "action", "created_at"),
        Index("ix_audit_user_time", "user_id", "created_at"),
        {"comment": "Append-only audit trail"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    action: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    actor: Mapped[str | None] = mapped_column(String(160))
    #: Truncated/anonymised client address - never a full request fingerprint.
    client_ip: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(256))
    resource: Mapped[str | None] = mapped_column(String(160))
    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)


__all__ = [
    "Alert",
    "AlertChannel",
    "AlertTrigger",
    "AuditAction",
    "AuditLog",
    "JobStatus",
    "JobType",
    "ProcessingJob",
]
