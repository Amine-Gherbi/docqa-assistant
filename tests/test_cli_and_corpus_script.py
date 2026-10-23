from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from docqa.cli import main
from docqa.config import PROJECT_ROOT, get_settings
from docqa.ingest.lock import load_lock


def _require_corpus() -> Path:
    settings = get_settings()
    lock = load_lock(settings.corpus_lock)
    missing = [
        entry.filename
        for entry in lock.files
        if not (settings.corpus_dir / entry.filename).is_file()
    ]
    if missing:
        pytest.skip(f"corpus not fetched: {missing}")
    return settings.corpus_dir / lock.files[0].filename


def test_ingest_refuses_tampered_file_with_nonzero_exit(tmp_path: Path) -> None:
    source = _require_corpus()
    raw = bytearray(source.read_bytes())
    raw[len(raw) // 2] ^= 0xFF  # same size, different hash
    target = tmp_path / source.name
    target.write_bytes(bytes(raw))

    assert main(["ingest", str(target)]) == 2


def test_ingest_refuses_unlisted_file_with_nonzero_exit(tmp_path: Path) -> None:
    target = tmp_path / "not-in-the-lock.pdf"
    target.write_bytes(b"%PDF-1.4\n")
    assert main(["ingest", str(target)]) == 2


def test_ingest_refuses_missing_file(tmp_path: Path) -> None:
    assert main(["ingest", str(tmp_path / "absent.pdf")]) == 1


def test_ingest_check_accepts_the_fetched_corpus() -> None:
    _require_corpus()
    assert main(["ingest", "--check"]) == 0


def test_help_lists_exactly_the_supported_verbs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    verbs = output.split("{", 1)[1].split("}", 1)[0].split(",")
    assert set(verbs) == {
        "ingest",
        "serve",
        "ask",
        "warm",
        "probe",
        "eval",
        "judge-check",
    }
    assert "probe-tool-choice" not in output


def test_live_script_forwards_to_the_docqa_command() -> None:
    script = (PROJECT_ROOT / "scripts" / "live.sh").read_text(encoding="utf-8")
    assert 'exec uv run docqa "$@"' in script
    assert "bash scripts/live.sh docqa" not in script
    assert "bash scripts/live.sh probe --json" in script
    assert 'bash scripts/live.sh ask --question "How many identical layers ...?"' in script


def _run_fetch_corpus(*args: str) -> subprocess.CompletedProcess[str]:
    script = PROJECT_ROOT / "scripts" / "fetch-corpus"
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_fetch_corpus_verifies_existing_files() -> None:
    _require_corpus()
    result = _run_fetch_corpus("--check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All 5 corpus files present and hash-verified." in result.stdout


def test_fetch_corpus_reports_a_corrupt_file(tmp_path: Path) -> None:
    source = _require_corpus()
    raw = bytearray(source.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    (tmp_path / source.name).write_bytes(bytes(raw))

    result = _run_fetch_corpus(
        "--check", "--lock", str(PROJECT_ROOT / "corpus.lock.json"), "--dest", str(tmp_path)
    )
    assert result.returncode == 1
    assert "MISMATCH" in result.stdout
    assert "missing or corrupt" in result.stderr


def test_fetch_corpus_reports_missing_files(tmp_path: Path) -> None:
    result = _run_fetch_corpus(
        "--check", "--lock", str(PROJECT_ROOT / "corpus.lock.json"), "--dest", str(tmp_path)
    )
    assert result.returncode == 1
    assert "MISSING" in result.stdout
