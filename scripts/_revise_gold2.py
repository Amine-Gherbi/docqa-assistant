"""Temporary gold rewriter for the remaining interleaved-column spans."""

import json
from pathlib import Path

UPDATES = {
    "B2": {
        "question": "What proportion of chosen BERT tokens is replaced with a random token?",
        "quote": "a random token 10% of the time (3)",
    },
    "B3": {
        "question": "How many layers and hidden units does BERT-Large have?",
        "quote": "BERTLARGE (L=24, H=1024,",
    },
    "S1": {
        "question": "What objective function did Sentence-BERT use to fine-tune on NLI data?",
        "quote": "We ﬁne-tune SBERT with a 3-way softmax- classiﬁer objective function for one epoch.",
        "chunk_index": 11,
        "page_no": 4,
    },
    "S2": {
        "question": "What batch size did Sentence-BERT use for fine-tuning?",
        "quote": "used a batch-size of 16, Adam optimizer with",
        "chunk_index": 11,
        "page_no": 4,
    },
    "D2": {
        "question": "Which similarity function does DPR choose for its retriever?",
        "quote": "we thus choose the simpler inner product function and improve the dense passage retriever by learning better encoders.",
        "chunk_index": 9,
        "page_no": 3,
    },
    "D4": {
        "question": "What is the output dimension of DPR's encoders?",
        "quote": "token as the output, so d = 768.",
    },
    "D5": {
        "question": "What does DPR's in-batch negative scheme train on in each batch?",
        "quote": "train on B2 (qi, pj) question/passage pairs in each",
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
        if "chunk_index" in update:
            row["gold"]["chunk_index"] = update["chunk_index"]
        if "page_no" in update:
            row["gold"]["page_no"] = update["page_no"]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"updated {len(UPDATES)} rows")


main()
