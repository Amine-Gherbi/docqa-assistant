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


def test_neon_url_without_a_port_gets_an_explicit_5432(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shell ``PGPORT`` must not redirect a hosted URL to a local server.

    asyncpg reads ``PGPORT`` when the URL omits a port; this build machine
    exports ``PGPORT=5433`` for its local database. The normaliser makes the
    PostgreSQL default explicit so the URL alone decides the target.
    """
    monkeypatch.setenv("PGPORT", "5433")
    normalised = normalise_database_url(
        "postgresql://owner:secret@ep-example.eu-central-1.aws.neon.tech/docqa"
        "?sslmode=require"
    )
    assert "@ep-example.eu-central-1.aws.neon.tech:5432/docqa" in normalised


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
    assert resolved.startswith(
        "postgresql+asyncpg://owner:secret@neon.example:5432/docqa"
    )
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


def test_rerank_memory_defaults_are_safe_for_a_512mb_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One pair at a time and no CPU arena unless a deploy opts in."""
    monkeypatch.delenv("DOCQA_RERANK_BATCH_SIZE", raising=False)
    monkeypatch.delenv("DOCQA_ONNX_MEMORY_ARENA", raising=False)
    settings = get_settings()
    assert settings.rerank_batch_size == 1
    assert settings.onnx_memory_arena is False


def test_rerank_memory_knobs_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A larger machine can raise the batch and re-enable the arena."""
    monkeypatch.setenv("DOCQA_RERANK_BATCH_SIZE", "4")
    monkeypatch.setenv("DOCQA_ONNX_MEMORY_ARENA", "1")
    settings = get_settings()
    assert settings.rerank_batch_size == 4
    assert settings.onnx_memory_arena is True


@pytest.mark.parametrize("raw", ["0", "-1", "many"])
def test_rerank_batch_size_rejects_bad_values(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """A batch size that would break the bound fails at startup, not later."""
    monkeypatch.setenv("DOCQA_RERANK_BATCH_SIZE", raw)
    with pytest.raises(ValueError, match="DOCQA_RERANK_BATCH_SIZE"):
        get_settings()
