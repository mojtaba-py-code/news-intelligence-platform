"""``news-platform`` - the operator CLI.

Everything an operator needs without an HTTP client: bootstrap the database,
sync the source catalogue, run ingestion, inspect trends and events, create the
first administrator, and start the worker or scheduler.
"""

from __future__ import annotations

import asyncio
import getpass
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypeVar

import typer
from rich.console import Console
from rich.table import Table

from app import __version__
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.security import Role, hash_password, validate_password_strength
from app.database.models.taxonomy import TrendSubject
from app.database.repositories.article import ArticleRepository
from app.database.repositories.event import EventRepository
from app.database.repositories.job import JobRepository
from app.database.repositories.source import SourceRepository
from app.database.repositories.taxonomy import TopicRepository, TrendRepository
from app.database.repositories.user import UserRepository
from app.database.session import check_connection, create_all, dispose_engine, session_scope
from app.ingestion.pipeline import ingest_sources
from app.ingestion.sources.registry import load_source_definitions, registry
from app.intelligence.topics import default_topic_seed
from app.services.events import EventService
from app.services.trends import TrendService

logger = get_logger(__name__)
console = Console()

app = typer.Typer(
    name="news-platform",
    help="Multi-Source News Intelligence Platform - operator CLI.",
    no_args_is_help=True,
    add_completion=False,
)
sources_app = typer.Typer(help="Manage the source catalogue.", no_args_is_help=True)
users_app = typer.Typer(help="Manage platform users.", no_args_is_help=True)
app.add_typer(sources_app, name="sources")
app.add_typer(users_app, name="users")

T = TypeVar("T")


def _run(coroutine: Callable[[], Awaitable[T]]) -> T:
    """Run an async command and always dispose the engine afterwards."""

    async def wrapper() -> T:
        try:
            return await coroutine()
        finally:
            await dispose_engine()

    return asyncio.run(wrapper())


@app.callback()
def main_callback(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging")] = False,
) -> None:
    """Configure logging before any command runs."""
    config = get_settings()
    if verbose:
        object.__setattr__(config, "log_level", "DEBUG")
    configure_logging(config)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
@app.command("init-db")
def init_db(
    with_topics: Annotated[bool, typer.Option(help="Seed the default topic catalogue")] = True,
    with_sources: Annotated[bool, typer.Option(help="Import configs/sources.yaml")] = True,
) -> None:
    """Create the schema and seed reference data."""

    async def run() -> dict[str, int]:
        await create_all()
        summary = {"topics": 0, "sources": 0}
        async with session_scope() as session:
            if with_topics:
                summary["topics"] = await TopicRepository(session).seed(default_topic_seed())
            if with_sources:
                repository = SourceRepository(session)
                for definition in load_source_definitions():
                    await repository.upsert_definition(definition.to_row())
                    summary["sources"] += 1
        return summary

    result = _run(run)
    console.print("[green]Database ready.[/green]")
    console.print(f"  seeded topics: {result['topics']}   sources synced: {result['sources']}")


@app.command("create-admin")
def create_admin(
    email: Annotated[str, typer.Option(prompt=True)],
    username: Annotated[str, typer.Option(prompt=True)],
    password: Annotated[str | None, typer.Option(help="Omit to be prompted securely")] = None,
) -> None:
    """Create an administrator account."""
    secret = password or getpass.getpass("Password: ")
    strength = validate_password_strength(secret)
    if not strength.ok:
        console.print("[red]Password rejected:[/red] " + "; ".join(strength.problems))
        raise typer.Exit(code=1)

    async def run() -> str:
        async with session_scope() as session:
            repository = UserRepository(session)
            if await repository.get_by_email(email) or await repository.get_by_username(username):
                raise typer.Exit(code=1)
            user = await repository.create(
                email=email,
                username=username,
                password_hash=hash_password(secret),
                role=Role.ADMIN,
                is_verified=True,
            )
            return user.username

    created = _run(run)
    console.print(f"[green]Administrator '{created}' created.[/green]")


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
@sources_app.command("list")
def sources_list() -> None:
    """List every configured source."""

    async def run() -> list[dict[str, Any]]:
        async with session_scope() as session:
            rows = await SourceRepository(session).all_sources()
            return [
                {
                    "slug": row.slug,
                    "name": row.name,
                    "kind": row.kind,
                    "status": row.status,
                    "enabled": row.enabled,
                    "articles": row.total_articles,
                    "reliability": row.reliability_score,
                    "failures": row.consecutive_failures,
                }
                for row in rows
            ]

    rows = _run(run)
    if not rows:
        console.print("[yellow]No sources configured. Run 'news-platform sources sync'.[/yellow]")
        return

    table = Table(title=f"Sources ({len(rows)})")
    for column in ("Slug", "Name", "Kind", "Status", "Articles", "Reliability", "Fails"):
        table.add_column(column)
    for row in rows:
        status = row["status"] if row["enabled"] else "disabled"
        colour = {"active": "green", "failing": "yellow", "paused": "red"}.get(status, "white")
        table.add_row(
            row["slug"],
            row["name"][:32],
            row["kind"],
            f"[{colour}]{status}[/{colour}]",
            str(row["articles"]),
            f"{row['reliability']:.2f}",
            str(row["failures"]),
        )
    console.print(table)


@sources_app.command("sync")
def sources_sync(
    path: Annotated[str | None, typer.Option(help="Path to a sources YAML file")] = None,
) -> None:
    """Import or update sources from the YAML catalogue."""

    async def run() -> int:
        definitions = load_source_definitions(path)
        async with session_scope() as session:
            repository = SourceRepository(session)
            for definition in definitions:
                await repository.upsert_definition(definition.to_row())
        return len(definitions)

    count = _run(run)
    console.print(f"[green]Synced {count} source definition(s).[/green]")


@sources_app.command("kinds")
def sources_kinds() -> None:
    """Show the registered connector types."""
    console.print("Registered connectors: " + ", ".join(registry.kinds))


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@app.command("ingest")
def ingest(
    source: Annotated[list[str] | None, typer.Option("--source", "-s", help="Source slug")] = None,
    all_sources: Annotated[bool, typer.Option("--all", help="Ingest every active source")] = False,
) -> None:
    """Run the ingestion pipeline."""
    if not source and not all_sources:
        console.print("[red]Specify --source <slug> or --all.[/red]")
        raise typer.Exit(code=1)

    results = _run(lambda: ingest_sources(source if not all_sources else None))
    if not results:
        console.print("[yellow]Nothing to ingest.[/yellow]")
        return

    table = Table(title="Ingestion results")
    for column in ("Source", "OK", "Fetched", "Valid", "Dupes", "Stored", "ms", "Error"):
        table.add_column(column)
    for result in results:
        table.add_row(
            result.source,
            "[green]yes[/green]" if result.success else "[red]no[/red]",
            str(result.fetched),
            str(result.valid),
            str(result.duplicates),
            str(result.stored),
            f"{result.duration_ms:.0f}",
            (result.error or "")[:48],
        )
    console.print(table)
    console.print(f"Total stored: [bold]{sum(r.stored for r in results)}[/bold]")


@app.command("trends")
def trends(
    hours: Annotated[int, typer.Option(help="Window size in hours")] = 24,
    limit: Annotated[int, typer.Option(help="Rows to display")] = 15,
    subject: Annotated[str | None, typer.Option(help="topic | keyword | entity | source")] = None,
) -> None:
    """Recompute and display emerging trends."""

    async def run() -> list[dict[str, Any]]:
        async with session_scope() as session:
            service = TrendService(session)
            results = await service.compute(hours=hours)
        wanted = TrendSubject(subject) if subject else None
        filtered = [r for r in results if wanted is None or r.subject_type is wanted]
        return [
            {
                "type": str(item.subject_type),
                "label": item.subject_label,
                "current": item.current_count,
                "previous": item.previous_count,
                "growth": item.growth_percent,
                "score": item.trend_score,
                "sources": item.source_count,
                "breaking": item.is_breaking,
            }
            for item in filtered[:limit]
        ]

    rows = _run(run)
    if not rows:
        console.print("[yellow]No trends detected in this window.[/yellow]")
        return

    table = Table(title=f"Trends (last {hours}h)")
    for column in ("Type", "Subject", "Now", "Before", "Growth", "Score", "Sources", ""):
        table.add_column(column)
    for row in rows:
        growth = f"{row['growth']:+.0f}%"
        colour = "green" if row["growth"] >= 0 else "red"
        table.add_row(
            row["type"],
            row["label"][:38],
            str(row["current"]),
            str(row["previous"]),
            f"[{colour}]{growth}[/{colour}]",
            f"{row['score']:.2f}",
            str(row["sources"]),
            "[bold red]BREAKING[/bold red]" if row["breaking"] else "",
        )
    console.print(table)


@app.command("events")
def events(
    hours: Annotated[int, typer.Option(help="Clustering window in hours")] = 48,
    limit: Annotated[int, typer.Option(help="Rows to display")] = 15,
) -> None:
    """Detect and display event clusters."""

    async def run() -> list[dict[str, Any]]:
        async with session_scope() as session:
            await EventService(session).detect(hours=hours)
            rows, _ = await EventRepository(session).recent(limit=limit, offset=0)
            return [
                {
                    "title": row.title,
                    "articles": row.article_count,
                    "sources": row.source_count,
                    "importance": row.importance,
                    "sentiment": row.avg_sentiment,
                }
                for row in rows
            ]

    rows = _run(run)
    if not rows:
        console.print("[yellow]No event clusters found.[/yellow]")
        return

    table = Table(title=f"Events (last {hours}h)")
    for column in ("Title", "Articles", "Sources", "Importance", "Sentiment"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            row["title"][:64],
            str(row["articles"]),
            str(row["sources"]),
            f"{row['importance']:.2f}",
            f"{row['sentiment']:+.2f}",
        )
    console.print(table)


@app.command("process")
def process(
    job: Annotated[
        str, typer.Option(help="ingest|process|trends|events|alerts|cleanup|deduplicate")
    ] = "process",
) -> None:
    """Run one background job synchronously."""
    from app.workers.jobs import run_job

    result = _run(lambda: run_job(job))
    console.print(f"[green]Job '{job}' finished.[/green]")
    for key, value in result.items():
        console.print(f"  {key}: {value}")


@app.command("worker")
def worker() -> None:
    """Start a background worker (Ctrl-C to stop)."""
    from app.workers.worker import main as worker_main

    console.print("[green]Worker started.[/green] Press Ctrl-C to stop.")
    asyncio.run(worker_main())


@app.command("scheduler")
def scheduler() -> None:
    """Start the scheduler (Ctrl-C to stop)."""
    from app.workers.scheduler import main as scheduler_main

    console.print("[green]Scheduler started.[/green] Press Ctrl-C to stop.")
    asyncio.run(scheduler_main())


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option(help="Bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port")] = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload (development only)")] = False,
) -> None:
    """Run the API and dashboard with uvicorn."""
    import uvicorn

    config = get_settings()
    console.print(f"[green]Serving on http://{host}:{port}[/green] ({config.environment})")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_config=None,  # our structured logging is already configured
    )


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
@app.command("health")
def health() -> None:
    """Check the database, cache and source circuit breakers."""

    async def run() -> dict[str, Any]:
        from app.core.cache import get_cache
        from app.core.resilience import breakers

        database = await check_connection()
        cache_ok = await get_cache().ping()
        return {"database": database, "cache": cache_ok, "circuits": breakers.status()}

    result = _run(run)
    console.print(f"database : {'[green]ok[/green]' if result['database'] else '[red]down[/red]'}")
    console.print(
        f"cache    : {'[green]ok[/green]' if result['cache'] else '[yellow]fallback[/yellow]'}"
    )
    open_circuits = [c for c in result["circuits"] if c["state"] != "closed"]
    console.print(f"circuits : {len(open_circuits)} open/half-open of {len(result['circuits'])}")
    if not result["database"]:
        raise typer.Exit(code=1)


@app.command("stats")
def stats() -> None:
    """Show platform-wide statistics."""

    async def run() -> dict[str, Any]:
        async with session_scope() as session:
            articles = await ArticleRepository(session).stats()
            sources = await SourceRepository(session).all_sources()
            queue = await JobRepository(session).queue_depth()
            trending = await TrendRepository(session).latest(limit=5)
            return {
                "articles": articles,
                "sources": len(sources),
                "active_sources": sum(1 for s in sources if s.is_operational),
                "queue": queue,
                "trending": [(t.subject_label, t.growth_percent) for t in trending],
            }

    result = _run(run)
    articles = result["articles"]

    table = Table(title="Platform statistics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total articles", str(articles.total))
    table.add_row("Today", str(articles.today))
    table.add_row("Last 24h", str(articles.last_24h))
    table.add_row("Last 7d", str(articles.last_7d))
    table.add_row("Duplicates", str(articles.duplicates))
    table.add_row("Pending", str(articles.pending))
    table.add_row("Failed", str(articles.failed))
    table.add_row("Avg relevance", f"{articles.avg_relevance:.3f}")
    table.add_row("Avg sentiment", f"{articles.avg_sentiment:+.3f}")
    table.add_row("Sources (active)", f"{result['sources']} ({result['active_sources']})")
    table.add_row("Queued jobs", str(result["queue"]))
    console.print(table)

    if result["trending"]:
        console.print("\nTrending:")
        for label, growth in result["trending"]:
            console.print(f"  #{label}  {growth:+.0f}%")


@app.command("version")
def version() -> None:
    """Print the platform version."""
    console.print(f"news-platform {__version__}")


def main() -> None:  # pragma: no cover - console-script entrypoint
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
