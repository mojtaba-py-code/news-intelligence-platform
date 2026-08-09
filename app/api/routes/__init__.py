"""Versioned API routers."""

from fastapi import APIRouter

from app.api.routes import (
    admin,
    alerts,
    analytics,
    articles,
    auth,
    health,
    intelligence,
    sources,
    users,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(articles.router)
api_router.include_router(sources.router)
api_router.include_router(intelligence.router)
api_router.include_router(analytics.router)
api_router.include_router(users.router)
api_router.include_router(alerts.router)
api_router.include_router(admin.router)

__all__ = ["api_router"]
