"""ORM models. Importing this package registers every table on ``Base.metadata``."""

from app.database.models.article import Article, ArticleEntity, ArticleTopic, Author
from app.database.models.event import Event, EventArticle
from app.database.models.job import Alert, AlertTrigger, AuditLog, ProcessingJob
from app.database.models.source import Source, SourceHealth
from app.database.models.taxonomy import Entity, Topic, TrendSnapshot
from app.database.models.user import SavedArticle, User, UserPreference

__all__ = [
    "Alert",
    "AlertTrigger",
    "Article",
    "ArticleEntity",
    "ArticleTopic",
    "AuditLog",
    "Author",
    "Entity",
    "Event",
    "EventArticle",
    "ProcessingJob",
    "SavedArticle",
    "Source",
    "SourceHealth",
    "Topic",
    "TrendSnapshot",
    "User",
    "UserPreference",
]
