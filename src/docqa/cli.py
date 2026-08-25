"""Command-line entry point: ``uv run docqa <command>``."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
from pathlib import Path

from docqa.config import Settings, get_settings
from docqa.db import upgrade_to_head
from docqa.ingest.chunking import TokenCapViolation
from docqa.ingest.embedder import EmbedderChoice, build_embedder
from docqa.ingest.lock import CorpusLockError, LockMismatch, load_lock, verify_file
from docqa.ingest.pipeline import IngestResult, ingest_files

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_LOCK = 2
EXIT_CAP = 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docqa", description="Ingest and query a pinned PDF corpus."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="ingest one or more pinned PDFs")
    ingest.add_argument("pdf", nargs="+", type=Path, help="PDF files listed in corpus.lock.json")
    ingest.add_argument(
        "--batch-size", type=int, default=32, help="embedding batch size (default 32)"
    )
    ingest.add_argument(
        "--embedder",
        choices=["auto", "fastembed", "hash"],
        default=None,
        help="override DOCQA_EMBEDDER for this run",
    )
    ingest.add_argument(
        "--no-migrate", action="store_true", help="skip 'alembic upgrade head'"
    )

    db = sub.add_parser("db", help="database maintenance")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("upgrade", help="apply alembic migrations")

    verify = sub.add_parser("verify", help="verify fetched PDFs against corpus.lock.json")
    verify.add_argument(
        "pdf", nargs="*", type=Path, help="files to verify (default: every locked file)"
    )
    return parser


def _print_banner(choice: EmbedderChoice) -> None:
    print(choice.banner)
    if choice.fallback_reason:
        print(f"  fallback reason: {choice.fallback_reason}")


def _print_result(result: IngestResult) -> None:
    header = (
        f"{'file':34s} {'pages':>5s} {'chunks':>6s} {'p50':>5s} "
        f"{'p95':>5s} {'max':>5s} {'clipped':>7s}  state"
    )
    print()
    print(header)
    print("-" * len(header))
    for stats in result.files:
        state = "updated" if stats.updated else "new"
        print(
            f"{stats.path.name:34s} {stats.pages:5d} {stats.chunks:6d} "
            f"{stats.p50_tokens:5d} {stats.p95_tokens:5d} {stats.max_tokens:5d} "
            f"{stats.clipped_vectors:7d}  {state}"
        )
    total_chunks = sum(stats.chunks for stats in result.files)
    total_pages = sum(stats.pages for stats in result.files)
    print("-" * len(header))
    print(
        f"{'TOTAL':34s} {total_pages:5d} {total_chunks:6d}"
        f"   embedding: {result.embedder_name}"
    )
    if result.dummy_embeddings:
        print("\nWARNING: this run used DUMMY embeddings; vectors are not semantic.")


def _override_embedder(settings: Settings, requested: str | None) -> Settings:
    if requested is None:
        return settings
    return dataclasses.replace(settings, embedder_mode=requested)


def _verify_inputs(settings: Settings, paths: list[Path]) -> None:
    lock = load_lock(settings.corpus_lock)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")
        verify_file(path, lock)


def _verify_all(settings: Settings, paths: list[Path]) -> None:
    lock = load_lock(settings.corpus_lock)
    targets = paths or [settings.corpus_dir / entry.filename for entry in lock.files]
    for path in targets:
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; run scripts/fetch-corpus")
        verify_file(path, lock)
        print(f"OK {path.name}")


def _run_ingest(args: argparse.Namespace, settings: Settings) -> int:
    paths: list[Path] = list(args.pdf)
    _verify_inputs(settings, paths)
    choice = build_embedder(settings)
    _print_banner(choice)
    if not args.no_migrate:
        upgrade_to_head(settings.database_url)
    result = asyncio.run(
        ingest_files(paths, settings, choice, batch_size=args.batch_size)
    )
    _print_result(result)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch."""
    args = _parser().parse_args(argv)
    settings = _override_embedder(get_settings(), getattr(args, "embedder", None))
    try:
        if args.command == "ingest":
            return _run_ingest(args, settings)
        if args.command == "db":
            upgrade_to_head(settings.database_url)
            print("database is at migration head")
            return EXIT_OK
        if args.command == "verify":
            _verify_all(settings, list(args.pdf))
            return EXIT_OK
    except LockMismatch as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_LOCK
    except TokenCapViolation as exc:
        print(f"TOKEN CAP VIOLATION: {exc}", file=sys.stderr)
        return EXIT_CAP
    except (CorpusLockError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"unknown command: {args.command}", file=sys.stderr)
    return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
