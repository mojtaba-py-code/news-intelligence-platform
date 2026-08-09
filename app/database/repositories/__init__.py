"""Repository layer - the only place that builds SQL.

Routes and services depend on repositories, never on the ORM session directly.
That keeps query construction (and therefore injection safety, index usage and
N+1 avoidance) in one reviewable place.
"""

from app.database.repositories.article import ArticleRepository
from app.database.repositories.audit import AuditRepository
from app.database.repositories.base import BaseRepository
from app.database.repositories.event import EventRepository
from app.database.repositories.job import AlertRepository, JobRepository
from app.database.repositories.source import SourceRepository
from app.database.repositories.taxonomy import EntityRepository, TopicRepository, TrendRepository
from app.database.repositories.user import UserRepository

__all__ = [
    "AlertRepository",
    "ArticleRepository",
    "AuditRepository",
    "BaseRepository",
    "EntityRepository",
    "EventRepository",
    "JobRepository",
    "SourceRepository",
    "TopicRepository",
    "TrendRepository",
    "UserRepository",
]
