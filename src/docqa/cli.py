"""Command-line entry point: ``uv run docqa <command>``."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from docqa.answer import CalibrationUnavailable
from docqa.config import PROJECT_ROOT, Settings, get_settings
from docqa.db import upgrade_to_head
from docqa.eval import EvalError
from docqa.ingest.chunking import TokenCapViolation
from docqa.ingest.embedder import EmbedderChoice, build_embedder
from docqa.ingest.lock import CorpusLockError, LockMismatch, load_lock, verify_file
from docqa.ingest.pipeline import IngestResult, ingest_files

if TYPE_CHECKING:
    from docqa.answer import AnswerResponse
    from docqa.cache import CachedAnswer

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_LOCK = 2
EXIT_CAP = 3
EXIT_CALIBRATION = 4
EXIT_PROBE_REWRITTEN = 5
EXIT_JUDGE_INVALID = 7
EXIT_RATE_LIMITED = 8

JUDGE_REQUIRED = ("model", "prompt_sha256", "rubric_version", "temperature")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docqa", description="Ingest and query a pinned PDF corpus."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="ingest or verify pinned PDFs")
    ingest.add_argument(
        "pdf", nargs="*", type=Path, help="PDF files listed in corpus.lock.json"
    )
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
    ingest.add_argument(
        "--check",
        action="store_true",
        help="verify files against corpus.lock.json without storing anything",
    )

    serve = sub.add_parser("serve", help="run the answer API")
    serve.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8788, help="port (default 8788)")
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

    ask = sub.add_parser("ask", help="ask one question against the ingested corpus")
    ask.add_argument("--question", required=True, help="the exact question text")
    ask.add_argument("--k", type=int, default=8, help="final chunks kept (default 8)")
    ask.add_argument("--json", action="store_true", help="print the full response JSON")
    ask.add_argument(
        "--force-generation",
        action="store_true",
        help="diagnostic: bypass the low-score refusal and call the model",
    )
    ask.add_argument(
        "--uncalibrated",
        action="store_true",
        help="boot without eval/calibration.json; recorded in the response",
    )
    ask.add_argument(
        "--structured-output",
        choices=["tool", "json"],
        default=None,
        help="override DOCQA_STRUCTURED_OUTPUT for this run",
    )

    warm = sub.add_parser(
        "warm", help="precompute the six demo answers into data/demo-answers.json"
    )
    warm.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data" / "demo-questions.json",
        help="fixed demo question file",
    )
    warm.add_argument(
        "--out",
        type=Path,
        default=None,
        help="cache path (default: data/demo-answers.json)",
    )
    warm.add_argument(
        "--force",
        action="store_true",
        help="regenerate every answer instead of reusing recorded ones",
    )
    warm.add_argument("--k", type=int, default=8, help="final chunks kept (default 8)")
    warm.add_argument(
        "--uncalibrated",
        action="store_true",
        help="boot without eval/calibration.json; recorded in every answer",
    )

    probe = sub.add_parser(
        "probe",
        help="probe tool_choice, fall back to JSON, and record the chosen mode",
    )
    probe.add_argument("--json", action="store_true", help="print the verdict as JSON")
    probe.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="probe artifact path (default: eval/structured-output.json)",
    )

    judge = sub.add_parser(
        "judge-check",
        help="validate the pinned judge protocol and report its measurement status",
    )
    judge.add_argument("--json", action="store_true", help="print the verdict as JSON")
    judge.add_argument(
        "--probe",
        action="store_true",
        help=(
            "ask the pinned judge id one real question and record every HTTP "
            "status; never substitutes the generator"
        ),
    )
    judge.add_argument(
        "--probe-tool-choice",
        action="store_true",
        help="also probe whether the configured gateway honours a forced tool call",
    )

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
    evaluate.add_argument(
        "--force-generation",
        action="store_true",
        help=(
            "diagnostic: bypass the calibrated refusal for every retrievable "
            "question; writes eval-report-forced.json and does not decide a gate"
        ),
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
            raise FileNotFoundError(
                f"{path} is missing; run scripts/fetch-corpus or "
                "`uv run docqa ingest --check` after fetching"
            )
        verify_file(path, lock)
        print(f"OK {path.name}")


def _run_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from docqa.api import create_app

    if args.structured_output is not None:
        settings = dataclasses.replace(
            settings,
            structured_output=args.structured_output,
            structured_output_source="command line",
        )
    app = create_app(
        settings=settings,
        calibration_path=args.calibration,
        allow_uncalibrated=args.uncalibrated,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return EXIT_OK


def _print_answer(
    question: str, response: AnswerResponse, trace_payload: dict[str, object] | None
) -> None:
    print(f"Question: {question}")
    print(f"Model: {response.model} ({response.structured_output})")
    calibration = response.calibration
    print(
        "Calibration: "
        f"threshold={calibration.threshold} "
        f"uncalibrated={calibration.uncalibrated}"
    )
    if response.forced_generation:
        print("Forced generation: DIAGNOSTIC (low-score refusal bypassed)")
    if response.insufficient_evidence:
        print(f"Result: insufficient_evidence ({response.reason})")
        if response.model_reason:
            print(f"Model reason: {response.model_reason}")
        for passage in response.passages:
            print(
                f"  closest: {passage.doc_id} p{passage.page_no} "
                f"chunk {passage.chunk_id}"
            )
        return
    print(f"Result: answered with {len(response.claims)} validated claim(s)")
    for number, claim in enumerate(response.claims, start=1):
        print(f"\nClaim {number}: {claim.text}")
        for citation in claim.citations:
            print(
                f"  citation: {citation.doc_id} p{citation.page_no} "
                f"chars {citation.char_start}-{citation.char_end}"
            )
            print(f"            sha256={citation.page_text_sha256}")
            print(f"            quote={citation.quote!r}")
    report = response.validation
    print(
        "\nValidation: "
        f"returned={report.claims_returned} kept={report.claims_kept} "
        f"dropped={report.claims_dropped} "
        f"schema_failures={report.schema_failures}"
    )
    if trace_payload is not None:
        refusal = trace_payload.get("refusal")
        print(f"Refusal trace: {refusal}")


def _run_ask(args: argparse.Namespace, settings: Settings) -> int:
    from docqa.answer import load_calibration
    from docqa.cache import trace_payload
    from docqa.run import ProductionAnswerer

    calibration = load_calibration(
        settings.calibration_path, allow_uncalibrated=args.uncalibrated
    )
    answerer = ProductionAnswerer(
        settings,
        calibration,
        k=args.k,
        force_generation=args.force_generation or settings.force_generation,
    )

    async def run() -> tuple[AnswerResponse, dict[str, object]]:
        try:
            response, trace = await answerer.answer(args.question)
            return response, trace_payload(trace, response, calibration)
        finally:
            await answerer.close()

    response, trace = asyncio.run(run())
    if args.json:
        body = {
            "response": _asdict(response),
            "trace": trace,
        }
        print(json.dumps(body, indent=2, sort_keys=True))
    else:
        print(answerer.embedder_banner)
        print(answerer.reranker_banner)
        _print_answer(args.question, response, trace)
    return EXIT_OK


def _asdict(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


def _run_warm(args: argparse.Namespace, settings: Settings) -> int:
    from docqa.answer import RateLimitedError, load_calibration
    from docqa.cache import load_demo_questions, warm_demo_answers
    from docqa.config import PROJECT_ROOT
    from docqa.run import ProductionAnswerer

    questions_path = args.questions or (PROJECT_ROOT / "data" / "demo-questions.json")
    questions = load_demo_questions(questions_path)
    out_path = args.out or settings.demo_answers_path
    calibration = load_calibration(
        settings.calibration_path, allow_uncalibrated=args.uncalibrated
    )
    answerer = ProductionAnswerer(
        settings,
        calibration,
        k=args.k,
        force_generation=settings.force_generation,
    )
    print(answerer.embedder_banner)
    print(answerer.reranker_banner)
    print(f"Model: {answerer.model.name} ({answerer.model.structured_output})")
    print(f"Questions: {len(questions)} from {questions_path}")

    def progress(position: int, total: int, entry: CachedAnswer) -> None:
        response = entry.response
        refused = bool(response.get("insufficient_evidence"))
        claims_raw = response.get("claims")
        claims = len(claims_raw) if isinstance(claims_raw, (list, tuple)) else 0
        state = "refused" if refused else f"answered ({claims} claims)"
        print(f"[{position}/{total}] {state}: {entry.question}")

    async def run() -> None:
        try:
            await warm_demo_answers(
                questions,
                answer=answerer.answer,
                out_path=out_path,
                model_name=answerer.model.name,
                calibration=calibration,
                reuse=not args.force,
                progress=progress,
            )
        finally:
            await answerer.close()

    try:
        asyncio.run(run())
    except RateLimitedError as exc:
        print(f"RATE LIMITED: {exc}", file=sys.stderr)
        print(
            f"Partial cache kept at {out_path}; rerun `docqa warm` when the "
            "window reopens (completed answers are reused).",
            file=sys.stderr,
        )
        return EXIT_RATE_LIMITED
    print(f"Demo answers: {out_path}")
    return EXIT_OK


def _run_probe_command(args: argparse.Namespace, settings: Settings) -> int:
    from docqa.answer import ModelCallError, probe_local_client, probe_structured_output

    if settings.model_base_url is None:
        verdict = asyncio.run(probe_local_client())
        print(f"endpoint: {verdict.endpoint}")
        print(f"tool_choice sent: {verdict.tool_choice_sent}")
        print(f"submit_answer returned: {verdict.tool_call_returned}")
        print(f"detail: {verdict.detail}")
        print(
            "No DOCQA_MODEL_BASE_URL configured; no hosted probe artifact was "
            "written."
        )
        return EXIT_OK
    try:
        result = asyncio.run(
            probe_structured_output(settings, artifact_path=args.artifact)
        )
    except ModelCallError as exc:
        print(f"PROBE FAILED: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    else:
        print(f"endpoint: {result.endpoint}")
        print(f"model: {result.model}")
        print(f"tool_choice sent: {result.tool_choice_sent}")
        print(f"submit_answer returned: {result.tool_call_returned}")
        print(f"JSON mode usable: {result.json_mode_ok}")
        print(f"recorded mode: {result.mode}")
        print(f"detail: {result.detail}")
        print(f"artifact: {args.artifact or settings.structured_output_probe_path}")
    return EXIT_OK if result.mode == "tool" else EXIT_PROBE_REWRITTEN


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


def _run_judge_check(args: argparse.Namespace, settings: Settings) -> int:
    """Validate the judge protocol; never claim a reliability it did not measure."""
    from docqa.judge import probe_judge, read_flat_yaml, write_judge_status

    judge_path = settings.calibration_path.parent / "judge.yaml"
    prompt_path = settings.calibration_path.parent / "judge-prompt.txt"
    record: dict[str, object] = {
        "judge_yaml": str(judge_path),
        "configured": judge_path.is_file(),
        "endpoint_configured": settings.model_base_url is not None,
        "generator_model": settings.model_name,
        "status": "unverified",
        "reason": "",
    }
    exit_code = EXIT_OK

    if not judge_path.is_file():
        record["reason"] = (
            "eval/judge.yaml is absent; citation precision and faithfulness use "
            "the deterministic rule-based checker, not a model judge"
        )
    else:
        try:
            protocol = read_flat_yaml(judge_path)
            missing = [name for name in JUDGE_REQUIRED if not protocol.get(name)]
            if missing:
                raise ValueError(f"missing required field(s): {', '.join(missing)}")
            model = protocol["model"]
            prompt_sha = protocol["prompt_sha256"].lower()
            if len(prompt_sha) != 64 or any(
                char not in "0123456789abcdef" for char in prompt_sha
            ):
                raise ValueError("prompt_sha256 must be 64 lower-case hex characters")
            temperature = float(protocol["temperature"])
            if temperature != 0.0:
                raise ValueError("temperature must be 0 for a pinned judge")
            if model == settings.model_name:
                raise ValueError(
                    f"judge model {model!r} equals the generator model; "
                    "invariant 6 requires a different model"
                )
            prompt_matches: bool | None = None
            if prompt_path.is_file():
                digest = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
                prompt_matches = digest == prompt_sha
                if not prompt_matches:
                    raise ValueError(
                        f"prompt_sha256 does not match {prompt_path.name} "
                        f"(file {digest[:12]}…, protocol {prompt_sha[:12]}…)"
                    )
            record.update(
                {
                    "model": model,
                    "prompt_sha256": prompt_sha,
                    "prompt_file_matches": prompt_matches,
                    "rubric_version": protocol["rubric_version"],
                    "temperature": temperature,
                    "model_differs_from_generator": True,
                }
            )
            if settings.model_base_url is None:
                record["reason"] = (
                    "judge protocol is pinned and valid, but no model endpoint is "
                    "configured; no reliability (Cohen's kappa) was measured"
                )
            elif not prompt_path.is_file():
                record["reason"] = (
                    "a model endpoint is configured, but eval/judge-prompt.txt is "
                    "absent so the pinned prompt cannot be verified"
                )
            elif getattr(args, "probe", False):
                status = asyncio.run(
                    probe_judge(
                        settings,
                        model,
                        system_prompt=prompt_path.read_text(encoding="utf-8"),
                    )
                )
                write_judge_status(
                    settings.calibration_path.parent / "judge-status.json", status
                )
                record.update(
                    {
                        "status": status.status,
                        "reason": status.reason,
                        "attempts": [
                            attempt.as_dict() for attempt in status.attempts
                        ],
                        "retries": status.retries,
                        "judge_status_artifact": str(
                            settings.calibration_path.parent / "judge-status.json"
                        ),
                    }
                )
            else:
                record["reason"] = (
                    "judge protocol and endpoint are configured; run "
                    "`docqa judge-check --probe` to ask the judge id one real "
                    "question. Reliability is unmeasured either way: no "
                    "human-labelled per-claim decisions exist, so no Cohen's "
                    "kappa is computed."
                )
        except (OSError, ValueError) as exc:
            record["status"] = "invalid"
            record["reason"] = str(exc)
            exit_code = EXIT_JUDGE_INVALID

    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        print(f"judge protocol: {judge_path}")
        if record["configured"]:
            print(f"  model: {record.get('model', '—')} (generator: {settings.model_name})")
            print(f"  prompt_sha256: {record.get('prompt_sha256', '—')}")
            print(f"  rubric_version: {record.get('rubric_version', '—')}")
            print(f"  temperature: {record.get('temperature', '—')}")
        print(
            "  endpoint: "
            + ("set" if settings.model_base_url else "unset (DOCQA_MODEL_BASE_URL)")
        )
        print(f"  reason: {record['reason']}")
        print(f"status: {str(record['status']).upper()}")

    if getattr(args, "probe_tool_choice", False):
        probe_exit = _run_probe(args, settings)
        if probe_exit != EXIT_OK:
            return probe_exit
    return exit_code


def _run_eval(args: argparse.Namespace, settings: Settings) -> int:
    from docqa.eval import (
        DEFAULT_ABLATION_REPORT_PATH,
        DEFAULT_FORCED_REPORT_PATH,
        DEFAULT_GOLD_PATH,
        DEFAULT_REPORT_PATH,
        EvalConfig,
        evaluate,
        render,
        write_report,
    )

    ablated = args.ablate == "vector-only"
    force_generation = args.force_generation or settings.force_generation
    k = args.k if args.k is not None else (3 if ablated else 8)
    if force_generation:
        label = "diagnostic:forced-generation"
    else:
        label = "ablate:vector-only-top3" if ablated else "production"
    config = EvalConfig(
        mode="vector" if ablated else "hybrid_rerank",
        k=k,
        label=label,
        ablated=ablated,
        force_generation=force_generation,
    )
    if args.out is not None:
        out = args.out
    elif force_generation:
        out = DEFAULT_FORCED_REPORT_PATH
    else:
        out = DEFAULT_ABLATION_REPORT_PATH if ablated else DEFAULT_REPORT_PATH
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
    if force_generation:
        print(
            "DIAGNOSTIC: the refusal threshold was bypassed; this is not a "
            "production gate run, so the exit code is 0 by design.",
            file=sys.stderr,
        )
        return EXIT_OK
    if not report.passed:
        print(
            f"GATE FAILURE: exit {6}; see {out}",
            file=sys.stderr,
        )
        return 6
    return EXIT_OK


def _run_ingest(args: argparse.Namespace, settings: Settings) -> int:
    paths: list[Path] = list(args.pdf)
    if args.check:
        _verify_all(settings, paths)
        return EXIT_OK
    if not paths:
        raise ValueError("provide at least one PDF, or pass --check to verify the corpus")
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
    if getattr(args, "structured_output", None) is not None:
        settings = dataclasses.replace(
            settings,
            structured_output=args.structured_output,
            structured_output_source="command line",
        )
    from docqa.answer import ModelCallError, RateLimitedError

    try:
        if args.command == "ingest":
            return _run_ingest(args, settings)
        if args.command == "serve":
            return _run_serve(args, settings)
        if args.command == "ask":
            return _run_ask(args, settings)
        if args.command == "warm":
            return _run_warm(args, settings)
        if args.command == "probe":
            return _run_probe_command(args, settings)
        if args.command == "eval":
            return _run_eval(args, settings)
        if args.command == "judge-check":
            return _run_judge_check(args, settings)
    except CalibrationUnavailable as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_CALIBRATION
    except RateLimitedError as exc:
        print(f"RATE LIMITED: {exc}", file=sys.stderr)
        return EXIT_RATE_LIMITED
    except ModelCallError as exc:
        print(f"MODEL ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
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
