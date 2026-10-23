"""Temporary gold rewriter for the last two column-interleaved spans."""

import json
from pathlib import Path

UPDATES = {
    "S1": {
        "question": "What objective function does Sentence-BERT use when fine-tuning on NLI data?",
        "quote": "we fine-tune SBERT with a 3-way softmax-",
    },
    "D2": {
        "question": "Which similarity function does DPR choose for its retriever?",
        "quote": "the simpler inner product function and improve the",
    },
}


def main() -> None:
    path = Path("eval/gold.jsonl")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        update = UPDATES.get(row["id"])
        if update is None:
            continue
        row["question"] = update["question"]
        row["gold"]["quote"] = update["quote"]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"updated {len(UPDATES)} rows")


main()
