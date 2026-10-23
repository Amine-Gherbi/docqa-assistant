"""Runtime configuration, resolved from the environment with local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATABASE_URL = "postgresql+asyncpg://postgres@localhost:5433/docqa"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


@dataclass(frozen=True)
class Settings:
    """Everything the ingest path needs to know about this machine."""

    database_url: str
    corpus_lock: Path
    corpus_dir: Path
    embedding_model: str
    embedder_mode: str
    hf_cache_dir: Path | None

    @property
    def lock_path(self) -> Path:
        """Alias used by the ingest pipeline."""
        return self.corpus_lock


def get_settings() -> Settings:
    """Read settings from environment variables."""
    mode = os.environ.get("DOCQA_EMBEDDER", "auto").strip().lower()
    if mode not in {"auto", "fastembed", "hash"}:
        raise ValueError("DOCQA_EMBEDDER must be one of: auto, fastembed, hash")
    cache = os.environ.get("DOCQA_HF_CACHE")
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
    )
