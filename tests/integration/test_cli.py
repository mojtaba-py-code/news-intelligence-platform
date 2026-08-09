"""The operator CLI, exercised through Typer's runner."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from app.cli import app

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture(autouse=True)
def _database(engine: object) -> None:
    """Commands open their own sessions against the shared test engine."""
    return None


def _recreate_schema() -> None:
    """Re-create the in-memory schema.

    Each CLI command disposes the engine when it finishes - correct for a real
    process, but it also discards an in-memory SQLite database, so a test that
    runs two commands has to rebuild the schema in between.
    """
    import asyncio

    from app.database.session import create_all, dispose_engine

    async def run() -> None:
        await create_all()
        await dispose_engine()

    asyncio.run(run())


def invoke(*args: str):
    return runner.invoke(app, list(args))


class TestCLI:
    def test_version(self) -> None:
        result = invoke("version")
        assert result.exit_code == 0
        assert "news-platform" in result.stdout

    def test_help_lists_the_main_commands(self) -> None:
        result = invoke("--help")
        assert result.exit_code == 0
        for command in ("ingest", "trends", "events", "stats", "health", "sources"):
            assert command in result.stdout

    def test_sources_list_on_an_empty_catalogue(self) -> None:
        result = invoke("sources", "list")
        assert result.exit_code == 0
        assert "No sources configured" in result.stdout

    def test_sources_kinds(self) -> None:
        result = invoke("sources", "kinds")
        assert result.exit_code == 0
        for kind in ("rss", "api", "scraper", "json_feed"):
            assert kind in result.stdout

    def test_sources_sync_imports_the_catalogue(self, tmp_path) -> None:
        catalogue = tmp_path / "sources.yaml"
        catalogue.write_text(
            """
sources:
  - slug: cli-source
    name: CLI Source
    kind: rss
    url: https://93.184.216.34/feed.xml
""",
            encoding="utf-8",
        )
        result = invoke("sources", "sync", "--path", str(catalogue))
        assert result.exit_code == 0
        assert "Synced 1" in result.stdout

    def test_sources_sync_skips_unsafe_entries(self, tmp_path) -> None:
        catalogue = tmp_path / "sources.yaml"
        catalogue.write_text(
            """
sources:
  - slug: internal
    name: Internal
    kind: rss
    url: http://169.254.169.254/feed.xml
""",
            encoding="utf-8",
        )
        _recreate_schema()
        result = invoke("sources", "sync", "--path", str(catalogue))
        assert result.exit_code == 0
        assert "Synced 0" in result.stdout

    def test_ingest_requires_a_target(self) -> None:
        result = invoke("ingest")
        assert result.exit_code == 1
        assert "--source" in result.stdout

    def test_ingest_all_with_no_sources(self) -> None:
        result = invoke("ingest", "--all")
        assert result.exit_code == 0
        assert "Nothing to ingest" in result.stdout

    def test_stats_on_an_empty_database(self) -> None:
        result = invoke("stats")
        assert result.exit_code == 0
        assert "Total articles" in result.stdout

    def test_health(self) -> None:
        result = invoke("health")
        assert result.exit_code == 0
        assert "database" in result.stdout

    def test_trends_reports_emptiness(self) -> None:
        assert "No trends detected" in invoke("trends").stdout

    def test_events_reports_emptiness(self) -> None:
        assert "No event clusters" in invoke("events").stdout

    def test_process_runs_a_single_job(self) -> None:
        result = invoke("process", "--job", "cleanup")
        assert result.exit_code == 0
        assert "finished" in result.stdout

    def test_create_admin_rejects_a_weak_password(self) -> None:
        result = invoke(
            "create-admin",
            "--email",
            "cli@example.com",
            "--username",
            "cliadmin",
            "--password",
            "weak",
        )
        assert result.exit_code == 1
        assert "rejected" in result.stdout

    def test_create_admin(self) -> None:
        result = invoke(
            "create-admin",
            "--email",
            "cli@example.com",
            "--username",
            "cliadmin",
            "--password",
            "Str0ng-Passw0rd!x",
        )
        assert result.exit_code == 0
        assert "created" in result.stdout
