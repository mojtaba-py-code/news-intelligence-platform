"""Audit-trail persistence.

The audit log is append-only by construction: this repository exposes writes
and reads but no update or delete path.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select

from app.core.logging import request_id_ctx
from app.core.utils import utcnow
from app.database.models.job import AuditAction, AuditLog
from app.database.repositories.base import BaseRepository

#: Details keys that must never be persisted, even by accident.
_FORBIDDEN_DETAIL_KEYS = frozenset(
    {"password", "token", "access_token", "refresh_token", "secret", "api_key", "authorization"}
)


def anonymize_ip(value: str | None) -> str | None:
    """Truncate the host portion of an address.

    Full client IPs are personal data under GDPR; the network prefix is enough
    to investigate abuse patterns, so that is all this stores.
    """
    if not value:
        return None
    candidate = value.split(",")[0].strip()
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv4Address):
        octets = str(address).split(".")
        return ".".join([*octets[:3], "0"])
    network = ipaddress.IPv6Network(f"{address}/48", strict=False)
    return str(network.network_address)


class AuditRepository(BaseRepository[AuditLog]):
    """Append-only security event log."""

    model = AuditLog

    async def record(
        self,
        action: AuditAction,
        *,
        user_id: int | None = None,
        actor: str | None = None,
        client_ip: str | None = None,
        user_agent: str | None = None,
        resource: str | None = None,
        success: bool = True,
        detail: dict[str, Any] | None = None,
    ) -> AuditLog:
        """Write one audit entry with sensitive keys stripped."""
        safe_detail = {
            key: value
            for key, value in (detail or {}).items()
            if key.lower() not in _FORBIDDEN_DETAIL_KEYS
        }
        entry = AuditLog(
            created_at=utcnow(),
            action=str(action),
            user_id=user_id,
            actor=(actor or "")[:160] or None,
            client_ip=anonymize_ip(client_ip),
            user_agent=(user_agent or "")[:256] or None,
            resource=(resource or "")[:160] or None,
            success=success,
            detail=safe_detail,
            request_id=request_id_ctx.get(),
        )
        self.session.add(entry)
        await self.flush()
        return entry

    async def recent(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        action: AuditAction | None = None,
        user_id: int | None = None,
    ) -> tuple[Sequence[AuditLog], int]:
        base = select(AuditLog)
        if action is not None:
            base = base.where(AuditLog.action == str(action))
        if user_id is not None:
            base = base.where(AuditLog.user_id == user_id)

        total = await self.count(base)
        statement = (
            base.order_by(AuditLog.created_at.desc()).limit(min(limit, 500)).offset(max(0, offset))
        )
        result = await self.session.execute(statement)
        return result.scalars().all(), total


__all__ = ["AuditRepository", "anonymize_ip"]
