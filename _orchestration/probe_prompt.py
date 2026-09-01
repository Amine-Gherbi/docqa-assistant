"""Run the documented production path for the questions that lost a quote."""

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_orchestration" / "runs" / "probe"
OUT.mkdir(parents=True, exist_ok=True)

report = json.loads((ROOT / "eval-report.json").read_text(encoding="utf-8"))
failing = [
    row
    for row in report["questions"]
    if row.get("stage_failed") in {"schema", "validation"}
]

for row in failing:
    question_id = row["id"]
    destination = OUT / f"{question_id}.json"
    if destination.exists():
        print(question_id, "cached")
        continue
    started = time.monotonic()
    completed = subprocess.run(
        [
            "bash",
            "scripts/live.sh",
            "ask",
            "--question",
            row["question"],
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        print(question_id, "EXIT", completed.returncode, completed.stderr[-200:])
        continue
    destination.write_text(completed.stdout, encoding="utf-8")
    print(question_id, f"{elapsed:.1f}s", "->", destination.name)
    time.sleep(5)
