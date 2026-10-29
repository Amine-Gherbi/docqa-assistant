"""Runtime configuration, resolved from the environment with local defaults."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATABASE_URL = "postgresql+asyncpg://postgres@localhost:5433/docqa"
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL_NAME = "claude-sonnet-5-5"
DEFAULT_MODEL_TIMEOUT_S = 180.0
DEFAULT_CALIBRATION_PATH = PROJECT_ROOT / "eval" / "calibration.json"
DEFAULT_STRUCTURED_OUTPUT_PROBE = PROJECT_ROOT / "eval" / "structured-output.json"
DEFAULT_DEMO_ANSWERS_PATH = PROJECT_ROOT / "data" / "demo-answers.json"
EMBEDDING_DIM = 384

# The keyed free tier answers one request at a time; 20 s between request
# starts leaves room for a ~15 s answer plus the provider's own pacing.
DEFAULT_MODEL_MIN_INTERVAL_S = 20.0
DEFAULT_MODEL_MAX_RETRIES = 2
DEFAULT_MODEL_RETRY_BACKOFF_S = 5.0

StructuredOutputMode = Literal["tool", "json"]
StructuredOutputRequest = Literal["auto", "tool", "json"]


@dataclass(frozen=True)
class Settings:
    """Everything the ingest and answer paths need to know about this machine."""

    database_url: str
    corpus_lock: Path
    corpus_dir: Path
    embedding_model: str
    embedder_mode: str
    hf_cache_dir: Path | None
    model_base_url: str | None
    model_api_key: str | None
    model_name: str
    model_timeout_s: float
    structured_output: StructuredOutputMode
    calibration_path: Path
    model_fallback_names: tuple[str, ...] = ()
    judge_api_key: str | None = None
    model_min_interval_s: float = DEFAULT_MODEL_MIN_INTERVAL_S
    model_max_retries: int = DEFAULT_MODEL_MAX_RETRIES
    model_retry_backoff_s: float = DEFAULT_MODEL_RETRY_BACKOFF_S
    structured_output_probe_path: Path = DEFAULT_STRUCTURED_OUTPUT_PROBE
    structured_output_source: str = "default"
    demo_answers_path: Path = DEFAULT_DEMO_ANSWERS_PATH
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    # Diagnostic only: bypasses the calibrated low-score refusal so the model's
    # own answer rate can be measured. Never use it for a gate run.
    force_generation: bool = False

    @property
    def lock_path(self) -> Path:
        """Alias used by the ingest pipeline."""
        return self.corpus_lock

    @property
    def model_configured(self) -> bool:
        """True when a model endpoint is configured; otherwise the extractive fallback runs."""
        return bool(self.model_base_url and self.model_name)

    @property
    def model_chain(self) -> tuple[str, ...]:
        """The model ids to try in order: the primary, then the documented retries."""
        return (self.model_name, *self.model_fallback_names)


def _fallback_names_from_environment(primary: str) -> tuple[str, ...]:
    """Read ``DOCQA_MODEL_FALLBACK_NAMES`` as an ordered, de-duplicated chain."""
    raw = os.environ.get("DOCQA_MODEL_FALLBACK_NAMES", "")
    names: list[str] = []
    for item in raw.split(","):
        name = item.strip()
        if name and name != primary and name not in names:
            names.append(name)
    return tuple(names)


def _timeout_from_environment() -> float:
    raw = os.environ.get("DOCQA_MODEL_TIMEOUT_S", str(DEFAULT_MODEL_TIMEOUT_S))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("DOCQA_MODEL_TIMEOUT_S must be a number of seconds") from exc
    if not value > 0:
        raise ValueError("DOCQA_MODEL_TIMEOUT_S must be positive")
    return value


def _min_interval_from_environment() -> float:
    raw = os.environ.get("DOCQA_MODEL_MIN_INTERVAL_S", str(DEFAULT_MODEL_MIN_INTERVAL_S))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("DOCQA_MODEL_MIN_INTERVAL_S must be a number of seconds") from exc
    if value < 0:
        raise ValueError("DOCQA_MODEL_MIN_INTERVAL_S must not be negative")
    return value


def _retries_from_environment() -> int:
    raw = os.environ.get("DOCQA_MODEL_MAX_RETRIES", str(DEFAULT_MODEL_MAX_RETRIES))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("DOCQA_MODEL_MAX_RETRIES must be an integer") from exc
    if value < 0:
        raise ValueError("DOCQA_MODEL_MAX_RETRIES must not be negative")
    return value


def _boolean_from_environment(name: str) -> bool:
    """Read a strict boolean flag; reject ambiguous values instead of guessing."""
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"{name} must be a boolean (1/0, true/false, yes/no)")


def normalise_base_url(url: str | None) -> str | None:
    """Compare endpoints without a trailing slash."""
    return url.rstrip("/") if url else None


def normalise_database_url(url: str) -> str:
    """Convert a libpq/Neon URL into a SQLAlchemy asyncpg URL.

    Neon hands out ``postgresql://…?sslmode=require&channel_binding=require``.
    SQLAlchemy's asyncpg dialect forwards query parameters as ``asyncpg.connect``
    keyword arguments, where libpq-only names such as ``sslmode`` and
    ``channel_binding`` are rejected. Map ``sslmode`` to asyncpg's ``ssl`` and
    drop the channel-binding hint, which asyncpg does not implement.
    """
    parts = urlsplit(url)
    scheme = parts.scheme
    if scheme in {"postgres", "postgresql"}:
        scheme = "postgresql+asyncpg"
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    has_ssl = any(key == "ssl" for key, _ in pairs)
    forwarded: list[tuple[str, str]] = []
    for key, value in pairs:
        if key == "channel_binding":
            continue
        if key == "sslmode":
            if not has_ssl:
                forwarded.append(("ssl", value))
            continue
        if key == "connect_timeout":
            forwarded.append(("timeout", value))
            continue
        forwarded.append((key, value))
    return urlunsplit(
        (scheme, parts.netloc, parts.path, urlencode(forwarded), parts.fragment)
    )


def cors_origins_from_environment(raw: str | None) -> tuple[str, ...]:
    """Parse ``DOCQA_CORS_ORIGINS`` as explicit browser origins.

    Origins are stripped of whitespace and a trailing slash, because the browser
    sends neither. An unset or empty value keeps the local development origins.
    """
    if raw is None:
        return DEFAULT_CORS_ORIGINS
    origins = tuple(
        origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()
    )
    return origins or DEFAULT_CORS_ORIGINS


def alembic_database_url(default: str) -> str:
    """Resolve the URL used by a bare ``alembic`` invocation.

    The hosted release step runs ``alembic upgrade head`` before the API boots,
    so the environment must win over the local URL committed in ``alembic.ini``.
    """
    raw = os.environ.get("DOCQA_DATABASE_URL")
    return normalise_database_url(raw) if raw else default


def resolve_structured_output(
    probe_path: Path, base_url: str | None, model_name: str
) -> tuple[StructuredOutputMode, str]:
    """Resolve ``DOCQA_STRUCTURED_OUTPUT``; ``auto`` reads the recorded probe.

    An explicit environment value always wins. ``auto`` trusts a probe artifact
    only when it names the same endpoint and model, so a stale artifact cannot
    silently switch a different deployment's mode.
    """
    requested = os.environ.get("DOCQA_STRUCTURED_OUTPUT", "auto").strip().lower()
    if requested in {"tool", "json"}:
        return cast(StructuredOutputMode, requested), "environment"
    if requested not in {"auto", ""}:
        raise ValueError("DOCQA_STRUCTURED_OUTPUT must be one of: auto, tool, json")
    if base_url:
        try:
            raw: object = json.loads(probe_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "tool", "default (no probe artifact)"
        if (
            isinstance(raw, dict)
            and normalise_base_url(str(raw.get("endpoint", ""))) == base_url
            and raw.get("model") == model_name
            and raw.get("mode") in {"tool", "json"}
        ):
            return cast(StructuredOutputMode, raw["mode"]), f"probe:{probe_path.name}"
        return "tool", "default (probe artifact is for another endpoint)"
    return "tool", "default"


def get_settings() -> Settings:
    """Read settings from environment variables."""
    mode = os.environ.get("DOCQA_EMBEDDER", "auto").strip().lower()
    if mode not in {"auto", "fastembed", "hash"}:
        raise ValueError("DOCQA_EMBEDDER must be one of: auto, fastembed, hash")
    cache = os.environ.get("DOCQA_HF_CACHE")
    base_url = normalise_base_url(
        os.environ.get("DOCQA_MODEL_BASE_URL", "").strip() or None
    )
    api_key = os.environ.get("DOCQA_MODEL_API_KEY", "").strip() or None
    judge_api_key = os.environ.get("DOCQA_JUDGE_API_KEY", "").strip() or api_key
    model_name = os.environ.get("DOCQA_MODEL_NAME", DEFAULT_MODEL_NAME).strip() or (
        DEFAULT_MODEL_NAME
    )
    probe_path = Path(
        os.environ.get(
            "DOCQA_STRUCTURED_OUTPUT_PROBE", str(DEFAULT_STRUCTURED_OUTPUT_PROBE)
        )
    )
    structured, structured_source = resolve_structured_output(
        probe_path, base_url, model_name
    )
    backoff_raw = os.environ.get(
        "DOCQA_MODEL_RETRY_BACKOFF_S", str(DEFAULT_MODEL_RETRY_BACKOFF_S)
    )
    try:
        backoff = float(backoff_raw)
    except ValueError as exc:
        raise ValueError("DOCQA_MODEL_RETRY_BACKOFF_S must be a number") from exc
    if backoff < 0:
        raise ValueError("DOCQA_MODEL_RETRY_BACKOFF_S must not be negative")
    return Settings(
        database_url=normalise_database_url(
            os.environ.get("DOCQA_DATABASE_URL", DEFAULT_DATABASE_URL)
        ),
        corpus_lock=Path(
            os.environ.get("DOCQA_CORPUS_LOCK", str(PROJECT_ROOT / "corpus.lock.json"))
        ),
        corpus_dir=Path(
            os.environ.get("DOCQA_CORPUS_DIR", str(PROJECT_ROOT / "data" / "corpus"))
        ),
        embedding_model=os.environ.get(
            "DOCQA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        ),
        embedder_mode=mode,
        hf_cache_dir=Path(cache) if cache else None,
        model_base_url=base_url,
        model_api_key=api_key,
        model_name=model_name,
        model_timeout_s=_timeout_from_environment(),
        structured_output=structured,
        calibration_path=Path(
            os.environ.get("DOCQA_CALIBRATION", str(DEFAULT_CALIBRATION_PATH))
        ),
        model_fallback_names=_fallback_names_from_environment(model_name),
        judge_api_key=judge_api_key,
        model_min_interval_s=_min_interval_from_environment(),
        model_max_retries=_retries_from_environment(),
        model_retry_backoff_s=backoff,
        structured_output_probe_path=probe_path,
        structured_output_source=structured_source,
        demo_answers_path=Path(
            os.environ.get("DOCQA_DEMO_ANSWERS", str(DEFAULT_DEMO_ANSWERS_PATH))
        ),
        cors_origins=cors_origins_from_environment(
            os.environ.get("DOCQA_CORS_ORIGINS")
        ),
        force_generation=_boolean_from_environment("DOCQA_FORCE_GENERATION"),
    )
