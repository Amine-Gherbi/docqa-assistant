# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

# The model key is a runtime environment variable. It is never copied, passed
# as a build argument, or written into this image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    DOCQA_HF_CACHE=/app/.cache/fastembed

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.13.5

WORKDIR /app

# Install dependencies first so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./
COPY eval ./eval
COPY eval-report.json ./
COPY corpus.lock.json ./
COPY data/demo-answers.json data/demo-questions.json ./data/
COPY scripts/fetch-corpus ./scripts/fetch-corpus
RUN uv sync --frozen --no-dev

# The corpus is never committed. Fetch and hash-check it at build time so the
# release step's `docqa ingest --check` fails loudly if anything is missing
# instead of serving an empty collection.
RUN uv run python scripts/fetch-corpus \
    && uv run docqa ingest --check

# Prefetch the ONNX weights. A free instance must not download models before it
# can answer its first question.
ENV DOCQA_EMBEDDER=fastembed \
    DOCQA_RERANKER=fastembed
RUN uv run python -c "\
from docqa.config import get_settings; \
from docqa.ingest.embedder import build_embedder; \
from docqa.rerank import build_reranker; \
s = get_settings(); \
build_embedder(s); \
build_reranker(mode='fastembed', cache_dir=str(s.hf_cache_dir) if s.hf_cache_dir else None)"

RUN useradd --create-home --uid 10001 app \
    && chown -R app:app /app
USER app

EXPOSE 8788
CMD ["docqa", "serve", "--host", "0.0.0.0", "--port", "8788"]
