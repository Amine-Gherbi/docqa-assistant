from __future__ import annotations

import json
from pathlib import Path

import pytest

from docqa.cli import EXIT_JUDGE_INVALID, EXIT_OK, main
from docqa.config import PROJECT_ROOT


def test_pinned_judge_is_valid_but_reliability_is_unverified(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DOCQA_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("DOCQA_MODEL_NAME", raising=False)
    assert main(["judge-check", "--json"]) == EXIT_OK
    record = json.loads(capsys.readouterr().out)
    assert record["configured"] is True
    assert record["status"] == "unverified"
    assert record["model"] == "gemini-3.8-flash"
    assert record["model_differs_from_generator"] is True
    assert record["prompt_file_matches"] is True
    assert record["temperature"] == 0.0
    assert "unmeasured" in record["reason"] or "no reliability" in record["reason"]


def test_judge_equal_to_generator_is_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    judge = tmp_path / "judge.yaml"
    source = (PROJECT_ROOT / "eval" / "judge.yaml").read_text(encoding="utf-8")
    judge.write_text(source, encoding="utf-8")
    (tmp_path / "judge-prompt.txt").write_text(
        (PROJECT_ROOT / "eval" / "judge-prompt.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    model = next(
        line.split(":", 1)[1].strip()
        for line in source.splitlines()
        if line.startswith("model:")
    )
    monkeypatch.setenv("DOCQA_CALIBRATION", str(tmp_path / "calibration.json"))
    monkeypatch.setenv("DOCQA_MODEL_NAME", model)
    assert main(["judge-check", "--json"]) == EXIT_JUDGE_INVALID
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "invalid"
    assert "equals the generator" in record["reason"]


def test_judge_missing_file_is_reported_as_unverified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DOCQA_CALIBRATION", str(tmp_path / "calibration.json"))
    assert main(["judge-check", "--json"]) == EXIT_OK
    record = json.loads(capsys.readouterr().out)
    assert record["configured"] is False
    assert record["status"] == "unverified"
    assert "rule-based" in record["reason"]
