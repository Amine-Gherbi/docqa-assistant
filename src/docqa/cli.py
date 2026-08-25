"""Command-line entry point: ``uv run docqa <command>``."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from pathlib import Path

from docqa.answer import CalibrationUnavailable
from docqa.config import Settings, get_settings
from docqa.db import upgrade_to_head
from docqa.eval import EvalError
from docqa.ingest.chunking import TokenCapViolation
from docqa.ingest.embedder import EmbedderChoice, build_embedder
from docqa.ingest.lock import CorpusLockError, LockMismatch, load_lock, verify_file
from docqa.ingest.pipeline import IngestResult, ingest_files

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_LOCK = 2
EXIT_CAP = 3
EXIT_CALIBRATION = 4
EXIT_PROBE_REWRITTEN = 5


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

    serve = sub.add_parser("serve", help="run the answer API")
    serve.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8100, help="port (default 8100)")
    serve.add_argument(
        "--calibration", type=Path, default=None, help="override eval/calibration.json"
    )
    serve.add_argument(
        "--uncalibrated",
        action="store_true",
        help="boot without eval/calibration.json; recorded in every response",
    )
    serve.add_argument(
        "--structured-output",
        choices=["tool", "json"],
        default=None,
        help="override DOCQA_STRUCTURED_OUTPUT",
    )

    probe = sub.add_parser(
        "probe-tool-choice",
        help="check whether the configured gateway honours a forced tool call",
    )
    probe.add_argument("--json", action="store_true", help="print the verdict as JSON")

    evaluate = sub.add_parser(
        "eval", help="run the fixed gold set, print Wilson intervals, enforce gates"
    )
    evaluate.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="gold JSONL (default: eval/gold.jsonl)",
    )
    evaluate.add_argument(
        "--out",
        type=Path,
        default=None,
        help="report path (default: eval-report.json, or eval-report-ablation.json)",
    )
    evaluate.add_argument(
        "--ablate",
        choices=["vector-only"],
        default=None,
        help="weakened retriever: vector-only top-3 (writes a separate report)",
    )
    evaluate.add_argument(
        "--k", type=int, default=None, help="final chunks kept (default 8; ablation 3)"
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


def _run_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from docqa.api import create_app

    if args.structured_output is not None:
        settings = dataclasses.replace(
            settings, structured_output=args.structured_output
        )
    app = create_app(
        settings=settings,
        calibration_path=args.calibration,
        allow_uncalibrated=args.uncalibrated,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return EXIT_OK


def _run_probe(args: argparse.Namespace, settings: Settings) -> int:
    from docqa.answer import (
        OpenAICompatibleModel,
        probe_local_client,
        probe_tool_choice,
    )

    if settings.model_base_url is not None:
        model = OpenAICompatibleModel(
            settings.model_base_url,
            settings.model_name,
            structured_output="tool",
            api_key=settings.model_api_key,
            timeout_s=settings.model_timeout_s,
        )
        verdict = asyncio.run(probe_tool_choice(model))
    else:
        verdict = asyncio.run(probe_local_client())
    record = {
        **dataclasses.asdict(verdict),
        "honoured": verdict.honoured,
    }
    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        print(f"endpoint: {verdict.endpoint}")
        print(f"request made: {verdict.made_request}")
        print(f"tool_choice sent: {verdict.tool_choice_sent}")
        print(f"submit_answer returned: {verdict.tool_call_returned}")
        print(f"verdict: {verdict.honoured}")
        print(f"detail: {verdict.detail}")
    return EXIT_OK if verdict.honoured is not False else EXIT_PROBE_REWRITTEN


def _run_eval(args: argparse.Namespace, settings: Settings) -> int:
    from docqa.eval import (
        DEFAULT_ABLATION_REPORT_PATH,
        DEFAULT_GOLD_PATH,
        DEFAULT_REPORT_PATH,
        EvalConfig,
        evaluate,
        render,
        write_report,
    )

    ablated = args.ablate == "vector-only"
    k = args.k if args.k is not None else (3 if ablated else 8)
    config = EvalConfig(
        mode="vector" if ablated else "hybrid_rerank",
        k=k,
        label="ablate:vector-only-top3" if ablated else "production",
        ablated=ablated,
    )
    out = args.out or (DEFAULT_ABLATION_REPORT_PATH if ablated else DEFAULT_REPORT_PATH)
    report = asyncio.run(
        evaluate(
            config,
            settings=settings,
            gold_path=args.gold or DEFAULT_GOLD_PATH,
        )
    )
    write_report(report, out)
    print(render(report))
    print(f"Report: {out}")
    if not report.passed:
        print(
            f"GATE FAILURE: exit {6}; see {out}",
            file=sys.stderr,
        )
        return 6
    return EXIT_OK


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
        if args.command == "serve":
            return _run_serve(args, settings)
        if args.command == "probe-tool-choice":
            return _run_probe(args, settings)
        if args.command == "eval":
            return _run_eval(args, settings)
    except CalibrationUnavailable as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_CALIBRATION
    except LockMismatch as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_LOCK
    except TokenCapViolation as exc:
        print(f"TOKEN CAP VIOLATION: {exc}", file=sys.stderr)
        return EXIT_CAP
    except (CorpusLockError, EvalError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"unknown command: {args.command}", file=sys.stderr)
    return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
