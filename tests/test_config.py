"""Runtime configuration: hosted database URLs and browser origins."""

from __future__ import annotations

import pytest

from docqa.config import (
    DEFAULT_CORS_ORIGINS,
    alembic_database_url,
    cors_origins_from_environment,
    get_settings,
    normalise_database_url,
)


def test_neon_url_becomes_asyncpg_url_with_asyncpg_ssl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neon hands out a libpq URL; SQLAlchemy needs an asyncpg URL and kwargs."""
    normalised = normalise_database_url(
        "postgresql://owner:secret@ep-example.eu-central-1.aws.neon.tech/docqa"
        "?sslmode=require&channel_binding=require"
    )
    assert normalised.startswith("postgresql+asyncpg://owner:secret@ep-example")
    assert "ssl=require" in normalised
    assert "sslmode" not in normalised
    assert "channel_binding" not in normalised


def test_asyncpg_url_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local URL already in the driver's dialect keeps its shape."""
    monkeypatch.setenv(
        "DOCQA_DATABASE_URL", "postgresql+asyncpg://postgres@localhost:5433/docqa"
    )
    settings = get_settings()
    assert settings.database_url == "postgresql+asyncpg://postgres@localhost:5433/docqa"


def test_cors_origins_are_parsed_and_trailing_slashes_removed() -> None:
    """The UI origin is configured explicitly; a trailing slash is not an origin."""
    origins = cors_origins_from_environment(
        " https://marginalia-ui.onrender.com/ , http://localhost:5173 "
    )
    assert origins == (
        "https://marginalia-ui.onrender.com",
        "http://localhost:5173",
    )


def test_cors_defaults_are_local_dev_origins() -> None:
    """An unset variable keeps the existing local development origins."""
    assert cors_origins_from_environment(None) == DEFAULT_CORS_ORIGINS
    assert cors_origins_from_environment("  ") == DEFAULT_CORS_ORIGINS


def test_settings_reads_cors_origins(monkeypatch: pytest.MonkeyPatch) -> None:
    """``DOCQA_CORS_ORIGINS`` reaches the settings object served by the API."""
    monkeypatch.setenv("DOCQA_CORS_ORIGINS", "https://marginalia-ui.onrender.com")
    assert get_settings().cors_origins == ("https://marginalia-ui.onrender.com",)


def test_alembic_uses_the_hosted_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The release step's bare ``alembic`` must not fall back to localhost."""
    monkeypatch.setenv(
        "DOCQA_DATABASE_URL", "postgresql://owner:secret@neon.example/docqa?sslmode=require"
    )
    resolved = alembic_database_url("postgresql+asyncpg://postgres@localhost:5433/docqa")
    assert resolved.startswith("postgresql+asyncpg://owner:secret@neon.example/docqa")
    assert "ssl=require" in resolved


def test_alembic_keeps_the_ini_url_without_an_environment_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local development keeps working without ``DOCQA_DATABASE_URL``."""
    monkeypatch.delenv("DOCQA_DATABASE_URL", raising=False)
    assert (
        alembic_database_url("postgresql+asyncpg://postgres@localhost:5433/docqa")
        == "postgresql+asyncpg://postgres@localhost:5433/docqa"
    )
