"""Temporary gold rewriter used to replace spans that crossed PDF columns."""

import json
from pathlib import Path

UPDATES = {
    "B1": (
        "What percentage of token positions does BERT's training data generator "
        "choose for prediction?",
        "chooses 15% of the token positions at random for",
    ),
    "B2": (
        "What proportion of chosen BERT tokens is replaced with a random token?",
        "a random token 10% of the time (3) the unchanged i-th token 10% of the time.",
    ),
    "B3": (
        "How many layers, hidden units, attention heads and parameters does "
        "BERT-Large have?",
        "BERTLARGE (L=24, H=1024, A=16, Total Parameters=340M).",
    ),
    "B4": (
        "How many words from English Wikipedia were used in BERT's pre-training corpus?",
        "English Wikipedia (2,500M words).",
    ),
    "B5": (
        "How many TPU chips in total did BERT-Base pre-training use?",
        "Cloud TPUs in Pod conﬁguration (16 TPU chips total).",
    ),
    "S1": (
        "What objective function did Sentence-BERT use to fine-tune on NLI data?",
        "We ﬁne-tune SBERT with a 3-way softmax- classiﬁer objective function "
        "for one epoch.",
    ),
    "S2": ("What batch size did Sentence-BERT use for fine-tuning?", "a batch-size of 16, Adam optimizer with"),
    "S3": (
        "How long does Sentence-BERT take to compute 10,000 sentence embeddings?",
        "(~5 seconds with SBERT)",
    ),
    "S4": ("What is Sentence-BERT's default pooling strategy?", "pooling strategy is MEAN."),
    "D1": (
        "How does DPR compute the similarity between a question and a passage?",
        "using the dot product of their vectors:",
    ),
    "D2": (
        "Which similarity function does DPR choose for its retriever?",
        "we thus choose the simpler inner product function and improve the dense "
        "passage retriever by learning better encoders.",
    ),
    "D4": (
        "What output representation do DPR's encoders use and what is its dimension?",
        "and take the representation at the [CLS] token as the output, so d = 768.",
    ),
    "D5": (
        "Why does DPR's in-batch negative scheme reuse computation?",
        "we reuse computation and effectively train on B2 (qi, pj) question/passage "
        "pairs in each batch.",
    ),
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
        row["question"], row["gold"]["quote"] = update
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"updated {len(UPDATES)} rows")


main()
