"""Runtime configuration, resolved from the environment with local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATABASE_URL = "postgresql+asyncpg://postgres@localhost:5433/docqa"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL_NAME = "claude-sonnet-5-5"
DEFAULT_CALIBRATION_PATH = PROJECT_ROOT / "eval" / "calibration.json"
EMBEDDING_DIM = 384

StructuredOutputMode = Literal["tool", "json"]


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

    @property
    def lock_path(self) -> Path:
        """Alias used by the ingest pipeline."""
        return self.corpus_lock

    @property
    def model_configured(self) -> bool:
        """True when a model endpoint is configured; otherwise the extractive fallback runs."""
        return bool(self.model_base_url and self.model_name)


def _timeout_from_environment() -> float:
    raw = os.environ.get("DOCQA_MODEL_TIMEOUT_S", "60")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("DOCQA_MODEL_TIMEOUT_S must be a number of seconds") from exc
    if not value > 0:
        raise ValueError("DOCQA_MODEL_TIMEOUT_S must be positive")
    return value


def get_settings() -> Settings:
    """Read settings from environment variables."""
    mode = os.environ.get("DOCQA_EMBEDDER", "auto").strip().lower()
    if mode not in {"auto", "fastembed", "hash"}:
        raise ValueError("DOCQA_EMBEDDER must be one of: auto, fastembed, hash")
    structured = os.environ.get("DOCQA_STRUCTURED_OUTPUT", "tool").strip().lower()
    if structured not in {"tool", "json"}:
        raise ValueError("DOCQA_STRUCTURED_OUTPUT must be one of: tool, json")
    cache = os.environ.get("DOCQA_HF_CACHE")
    base_url = os.environ.get("DOCQA_MODEL_BASE_URL", "").strip() or None
    api_key = os.environ.get("DOCQA_MODEL_API_KEY", "").strip() or None
    return Settings(
        database_url=os.environ.get("DOCQA_DATABASE_URL", DEFAULT_DATABASE_URL),
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
        model_name=os.environ.get("DOCQA_MODEL_NAME", DEFAULT_MODEL_NAME).strip()
        or DEFAULT_MODEL_NAME,
        model_timeout_s=_timeout_from_environment(),
        structured_output="json" if structured == "json" else "tool",
        calibration_path=Path(
            os.environ.get("DOCQA_CALIBRATION", str(DEFAULT_CALIBRATION_PATH))
        ),
    )
